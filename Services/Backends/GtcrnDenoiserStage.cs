using System.IO;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using Microsoft.ML.OnnxRuntime;
using VoiceLockStudio.Models;
using VoiceLockStudio.Services.Audio;

namespace VoiceLockStudio.Services.Backends;

/// <summary>
/// Streaming GTCRN speech-enhancement denoiser operating in the STFT domain
/// (<c>n_fft = 512</c>, <c>hop = 256</c> / 16 ms at 16 kHz, periodic <c>hann_sqrt</c> window)
/// with preallocated radix-2 FFT/IFFT and ping-pong recurrent/convolution caches.
/// </summary>
public sealed class GtcrnDenoiserStage : IAudioFilterStage
{
    private const string DefaultModelKey = "gtcrn_simple.onnx";

    private readonly object _lifecycleLock = new();

    private OnnxSessionResult? _sessionResult;
    private RunOptions? _runOptions;

    private int _sampleRate = 16000;
    private int _nFft = 512;
    private int _hopLength = 256;
    private int _freqBins = 257;

    private float[] _window = [];
    private float[] _analysisBuffer = [];
    private float[] _synthesisOla = [];

    private int _fftHalf;
    private int[] _bitRev = [];
    private float[] _twiddleCos = [];
    private float[] _twiddleSin = [];
    private float[][] _stageCos = [];
    private float[][] _stageSin = [];
    private float[] _realTwiddleCos = [];
    private float[] _realTwiddleSin = [];
    private float[] _fftReal = [];
    private float[] _fftImag = [];
    private float[] _windowedScratch = [];

    private float[] _mixBuffer = [];
    private float[] _enhBuffer = [];
    private OrtValue? _ortMix;
    private OrtValue? _ortEnh;

    private float[][] _cacheBuffersA = [];
    private float[][] _cacheBuffersB = [];
    private OrtValue[] _ortCachesA = [];
    private OrtValue[] _ortCachesB = [];
    private int[] _cacheInputSlots = [];
    private int[] _cacheOutputSlots = [];

    private string[] _inputNames = [];
    private OrtValue[] _inputValues = [];
    private string[] _outputNames = [];
    private OrtValue[] _outputValues = [];
    private bool _cacheIsA = true;

    private bool _isInitialized;
    private bool _disposed;

    public string Name => "GTCRN Denoiser";
    public int FrameSizeSamples => _hopLength;
    public int SampleRate => _sampleRate;
    public bool IsInitialized => Volatile.Read(ref _isInitialized);

    public string ActiveProviderName => _sessionResult?.ActiveProviderName ?? string.Empty;
    public string? ProviderFallbackMessage => _sessionResult?.FallbackMessage;

    public void Initialize(AudioStageOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        ObjectDisposedException.ThrowIf(_disposed, this);

        lock (_lifecycleLock)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            ReleaseResources();

            ModelManifest manifest = OnnxSessionFactory.ResolveManifest(options);
            ModelEntry entry = ResolveGtcrnModelEntry(manifest);

            ModelAudioParams audio = entry.AudioParams ?? throw new InvalidDataException(
                $"Model '{entry.Filename}' is missing audio_params in model_manifest.json.");

            _sampleRate = audio.SampleRate > 0
                ? audio.SampleRate
                : throw new InvalidDataException($"Model '{entry.Filename}' has invalid sample_rate.");

            _nFft = audio.NFft
                ?? throw new InvalidDataException($"Model '{entry.Filename}' is missing n_fft.");

            _hopLength = audio.HopLength
                ?? throw new InvalidDataException($"Model '{entry.Filename}' is missing hop_length.");

            int windowLength = audio.WindowLength ?? _nFft;
            if (windowLength != _nFft || (_nFft & (_nFft - 1)) != 0 || _hopLength * 2 != _nFft)
            {
                throw new InvalidDataException(
                    $"GTCRN expects power-of-2 n_fft=window_length=2*hop_length; got n_fft={_nFft}, win={windowLength}, hop={_hopLength}.");
            }

            _freqBins = audio.FreqBins ?? (_nFft / 2 + 1);

            IReadOnlyList<TensorDescriptor> inputs = entry.GetEffectiveInputs();
            IReadOnlyList<TensorDescriptor> outputs = entry.GetEffectiveOutputs();
            if (inputs.Count < 4 || outputs.Count < 4 || inputs.Count != outputs.Count)
            {
                throw new InvalidDataException(
                    $"Model '{entry.Filename}' has unexpected I/O count ({inputs.Count} inputs, {outputs.Count} outputs).");
            }

            _sessionResult = OnnxSessionFactory.CreateSession(
                manifest,
                entry.Filename,
                options.ModelsRootDirectory,
                options.Backend);

            _fftHalf = _nFft >> 1;
            BuildFftTables(_nFft);
            BuildHannSqrtWindow(_nFft, audio.WindowType);

            _analysisBuffer = GC.AllocateArray<float>(_nFft, pinned: false);
            _synthesisOla = GC.AllocateArray<float>(_nFft, pinned: false);
            _fftReal = GC.AllocateArray<float>(_fftHalf, pinned: false);
            _fftImag = GC.AllocateArray<float>(_fftHalf, pinned: false);
            _windowedScratch = GC.AllocateArray<float>(_nFft, pinned: false);

            // Identify spectrum tensor ("mix" / "enh": rank 4 with freqBins & 2) vs state cache tensors.
            int mixInIdx = FindSpectrumTensorIndex(inputs, _freqBins);
            int enhOutIdx = FindSpectrumTensorIndex(outputs, _freqBins);

            TensorDescriptor mixDesc = inputs[mixInIdx];
            TensorDescriptor enhDesc = outputs[enhOutIdx];
            long[] mixShape = OnnxSessionFactory.ResolveTensorShape(mixDesc);
            long[] enhShape = OnnxSessionFactory.ResolveTensorShape(enhDesc);

            _mixBuffer = GC.AllocateArray<float>(OnnxSessionFactory.ComputeElementCount(mixShape), pinned: true);
            _enhBuffer = GC.AllocateArray<float>(OnnxSessionFactory.ComputeElementCount(enhShape), pinned: true);
            _ortMix = OrtValue.CreateTensorValueFromMemory(_mixBuffer, mixShape);
            _ortEnh = OrtValue.CreateTensorValueFromMemory(_enhBuffer, enhShape);

            _inputNames = new string[inputs.Count];
            _inputValues = new OrtValue[inputs.Count];
            _outputNames = new string[outputs.Count];
            _outputValues = new OrtValue[outputs.Count];

            for (int i = 0; i < inputs.Count; i++)
            {
                _inputNames[i] = inputs[i].Name;
            }
            for (int i = 0; i < outputs.Count; i++)
            {
                _outputNames[i] = outputs[i].Name;
            }

            _inputValues[mixInIdx] = _ortMix;
            _outputValues[enhOutIdx] = _ortEnh;

            int cacheCount = inputs.Count - 1;
            _cacheBuffersA = new float[cacheCount][];
            _cacheBuffersB = new float[cacheCount][];
            _ortCachesA = new OrtValue[cacheCount];
            _ortCachesB = new OrtValue[cacheCount];
            _cacheInputSlots = new int[cacheCount];
            _cacheOutputSlots = new int[cacheCount];

            int cIdx = 0;
            for (int i = 0; i < inputs.Count; i++)
            {
                if (i == mixInIdx)
                {
                    continue;
                }

                TensorDescriptor inCache = inputs[i];
                int outSlot = FindMatchingCacheOutputIndex(outputs, inCache, enhOutIdx);
                long[] shape = OnnxSessionFactory.ResolveTensorShape(inCache);
                int elemCount = OnnxSessionFactory.ComputeElementCount(shape);

                float[] bufA = GC.AllocateArray<float>(elemCount, pinned: true);
                float[] bufB = GC.AllocateArray<float>(elemCount, pinned: true);
                OrtValue ortA = OrtValue.CreateTensorValueFromMemory(bufA, shape);
                OrtValue ortB = OrtValue.CreateTensorValueFromMemory(bufB, shape);

                _cacheBuffersA[cIdx] = bufA;
                _cacheBuffersB[cIdx] = bufB;
                _ortCachesA[cIdx] = ortA;
                _ortCachesB[cIdx] = ortB;
                _cacheInputSlots[cIdx] = i;
                _cacheOutputSlots[cIdx] = outSlot;

                _inputValues[i] = ortA;
                _outputValues[outSlot] = ortB;
                cIdx++;
            }

            _runOptions = new RunOptions();
            _cacheIsA = true;
            Volatile.Write(ref _isInitialized, true);

            Warmup();
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public void ProcessFrame(ReadOnlySpan<float> input, Span<float> output)
    {
        if (!Volatile.Read(ref _isInitialized) || _sessionResult is null || _runOptions is null)
        {
            throw new InvalidOperationException($"{Name} must be initialized before calling ProcessFrame.");
        }

        if (input.Length != _hopLength)
        {
            throw new ArgumentException($"Expected {_hopLength} input samples, got {input.Length}.", nameof(input));
        }

        if (output.Length != _hopLength)
        {
            throw new ArgumentException($"Expected {_hopLength} output samples, got {output.Length}.", nameof(output));
        }

        int nFft = _nFft;
        int hop = _hopLength;

        // 1. Shift 512-sample analysis buffer left by 256 and append new 256-sample hop.
        _analysisBuffer.AsSpan(hop, nFft - hop).CopyTo(_analysisBuffer.AsSpan(0, nFft - hop));
        input.CopyTo(_analysisBuffer.AsSpan(nFft - hop, hop));

        // 2. Apply periodic hann_sqrt analysis window and pack even/odd samples into N/2 complex bins.
        float[] win = _window;
        float[] analysis = _analysisBuffer;
        float[] windowed = _windowedScratch;
        WasapiPcmConverter.Multiply(analysis, win, windowed);

        float[] fftReal = _fftReal;
        float[] fftImag = _fftImag;
        int halfFft = _fftHalf;
        for (int i = 0; i < halfFft; i++)
        {
            int evenIdx = i << 1;
            fftReal[i] = windowed[evenIdx];
            fftImag[i] = windowed[evenIdx + 1];
        }

        // 3. Unscaled 256-pt complex FFT + real-to-Hermitian unpack into mix: [1, 257, 1, 2].
        TransformComplexInPlace(fftReal, fftImag, inverse: false);

        float[] mix = _mixBuffer;
        float dcSumRe = fftReal[0];
        float dcSumIm = fftImag[0];
        mix[0] = dcSumRe + dcSumIm;
        mix[1] = 0f;
        mix[halfFft << 1] = dcSumRe - dcSumIm;
        mix[(halfFft << 1) + 1] = 0f;

        int quarter = halfFft >> 1;
        float[] realCos = _realTwiddleCos;
        float[] realSin = _realTwiddleSin;

        for (int k = 1; k < quarter; k++)
        {
            int mk = halfFft - k;
            float ar = fftReal[k];
            float ai = fftImag[k];
            float br = fftReal[mk];
            float bi = fftImag[mk];

            float er = 0.5f * (ar + br);
            float ei = 0.5f * (ai - bi);
            float or_ = 0.5f * (ai + bi);
            float oi = 0.5f * (br - ar);

            float c = realCos[k];
            float s = realSin[k];
            float tr = or_ * c + oi * s;
            float ti = oi * c - or_ * s;

            int kIdx = k << 1;
            mix[kIdx] = er + tr;
            mix[kIdx + 1] = ei + ti;

            int mkIdx = mk << 1;
            mix[mkIdx] = er - tr;
            mix[mkIdx + 1] = ti - ei;
        }

        int qIdx = quarter << 1;
        mix[qIdx] = fftReal[quarter];
        mix[qIdx + 1] = -fftImag[quarter];

        // 4. Run GTCRN inference and ping-pong caches with zero heap allocations.
        _sessionResult.Session.Run(_runOptions, _inputNames, _inputValues, _outputNames, _outputValues);

        _cacheIsA = !_cacheIsA;
        if (_cacheIsA)
        {
            for (int c = 0; c < _cacheInputSlots.Length; c++)
            {
                _inputValues[_cacheInputSlots[c]] = _ortCachesA[c];
                _outputValues[_cacheOutputSlots[c]] = _ortCachesB[c];
            }
        }
        else
        {
            for (int c = 0; c < _cacheInputSlots.Length; c++)
            {
                _inputValues[_cacheInputSlots[c]] = _ortCachesB[c];
                _outputValues[_cacheOutputSlots[c]] = _ortCachesA[c];
            }
        }

        // 5. Pack Hermitian spectrum enh: [1, 257, 1, 2] into N/2 complex bins and run 1/(N/2)-scaled IFFT.
        float[] enh = _enhBuffer;
        fftReal[0] = 0.5f * (enh[0] + enh[halfFft << 1]);
        fftImag[0] = 0.5f * (enh[0] - enh[halfFft << 1]);

        for (int k = 1; k < quarter; k++)
        {
            int mk = halfFft - k;
            int kIdx = k << 1;
            int mkIdx = mk << 1;

            float ar = enh[kIdx];
            float ai = enh[kIdx + 1];
            float br = enh[mkIdx];
            float bi = enh[mkIdx + 1];

            float pr = ar + br;
            float pi = ai - bi;
            float dr = ar - br;
            float di = ai + bi;

            float c = realCos[k];
            float s = realSin[k];
            float qr = dr * c - di * s;
            float qi = dr * s + di * c;

            fftReal[k] = 0.5f * (pr - qi);
            fftImag[k] = 0.5f * (pi + qr);
            fftReal[mk] = 0.5f * (pr + qi);
            fftImag[mk] = 0.5f * (qr - pi);
        }

        fftReal[quarter] = enh[qIdx];
        fftImag[quarter] = -enh[qIdx + 1];

        TransformComplexInPlace(fftReal, fftImag, inverse: true);

        // 6. Unpack real time-domain frame, apply synthesis hann_sqrt window, and overlap-add.
        for (int i = 0; i < halfFft; i++)
        {
            int evenIdx = i << 1;
            windowed[evenIdx] = fftReal[i];
            windowed[evenIdx + 1] = fftImag[i];
        }

        float[] ola = _synthesisOla;
        WasapiPcmConverter.MultiplyAddInPlace(windowed, win, ola);

        ola.AsSpan(0, hop).CopyTo(output);
        ola.AsSpan(hop, nFft - hop).CopyTo(ola.AsSpan(0, nFft - hop));
        ola.AsSpan(nFft - hop, hop).Clear();
    }

    public void ResetState()
    {
        if (!Volatile.Read(ref _isInitialized))
        {
            return;
        }

        Array.Clear(_analysisBuffer);
        Array.Clear(_synthesisOla);
        Array.Clear(_mixBuffer);
        Array.Clear(_enhBuffer);

        for (int c = 0; c < _cacheBuffersA.Length; c++)
        {
            Array.Clear(_cacheBuffersA[c]);
            Array.Clear(_cacheBuffersB[c]);
            _inputValues[_cacheInputSlots[c]] = _ortCachesA[c];
            _outputValues[_cacheOutputSlots[c]] = _ortCachesB[c];
        }

        _cacheIsA = true;
    }

    public void Warmup()
    {
        if (!Volatile.Read(ref _isInitialized) || _sessionResult is null || _runOptions is null)
        {
            return;
        }

        Array.Clear(_mixBuffer);
        for (int i = 0; i < 2; i++)
        {
            _sessionResult.Session.Run(_runOptions, _inputNames, _inputValues, _outputNames, _outputValues);
        }
        ResetState();
    }

    public void Dispose()
    {
        lock (_lifecycleLock)
        {
            if (_disposed)
            {
                return;
            }

            _disposed = true;
            ReleaseResources();
        }
    }

    private void BuildHannSqrtWindow(int nFft, string? windowType)
    {
        if (!string.IsNullOrWhiteSpace(windowType) &&
            !string.Equals(windowType, "hann_sqrt", StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidDataException($"Unsupported GTCRN window_type '{windowType}'; expected 'hann_sqrt'.");
        }

        _window = GC.AllocateArray<float>(nFft, pinned: false);
        double twoPiOverN = 2.0 * Math.PI / nFft;
        for (int n = 0; n < nFft; n++)
        {
            double hann = 0.5 * (1.0 - Math.Cos(twoPiOverN * n));
            _window[n] = (float)Math.Sqrt(hann);
        }
    }

    private void BuildFftTables(int nFft)
    {
        int m = nFft >> 1;
        int bits = 0;
        for (int v = m; v > 1; v >>= 1)
        {
            bits++;
        }

        _bitRev = new int[m];
        for (int i = 0; i < m; i++)
        {
            int rev = 0;
            int x = i;
            for (int b = 0; b < bits; b++)
            {
                rev = (rev << 1) | (x & 1);
                x >>= 1;
            }

            _bitRev[i] = rev;
        }

        int halfM = m >> 1;
        _twiddleCos = new float[halfM];
        _twiddleSin = new float[halfM];
        double twoPiOverM = 2.0 * Math.PI / m;
        for (int k = 0; k < halfM; k++)
        {
            double angle = twoPiOverM * k;
            _twiddleCos[k] = (float)Math.Cos(angle);
            _twiddleSin[k] = (float)Math.Sin(angle);
        }

        _realTwiddleCos = new float[halfM];
        _realTwiddleSin = new float[halfM];
        double twoPiOverN = 2.0 * Math.PI / nFft;
        for (int k = 0; k < halfM; k++)
        {
            double angle = twoPiOverN * k;
            _realTwiddleCos[k] = (float)Math.Cos(angle);
            _realTwiddleSin[k] = (float)Math.Sin(angle);
        }

        int stageCount = 0;
        for (int len = 8; len <= m; len <<= 1)
        {
            stageCount++;
        }

        _stageCos = new float[stageCount][];
        _stageSin = new float[stageCount][];
        int sIdx = 0;
        for (int len = 8; len <= m; len <<= 1)
        {
            int half = len >> 1;
            int step = m / len;
            var sc = new float[half];
            var ss = new float[half];
            for (int j = 0; j < half; j++)
            {
                int tw = j * step;
                sc[j] = _twiddleCos[tw];
                ss[j] = _twiddleSin[tw];
            }
            _stageCos[sIdx] = sc;
            _stageSin[sIdx] = ss;
            sIdx++;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void TransformComplexInPlace(float[] real, float[] imag, bool inverse)
    {
        int n = _fftHalf;
        int[] bitRev = _bitRev;

        for (int i = 0; i < n; i++)
        {
            int j = bitRev[i];
            if (i < j)
            {
                (real[i], real[j]) = (real[j], real[i]);
                (imag[i], imag[j]) = (imag[j], imag[i]);
            }
        }

        float sinSign = inverse ? 1f : -1f;

        // Unrolled stage len = 2 (half = 1, twiddle = 1 + 0i)
        if (n >= 2)
        {
            for (int i = 0; i < n; i += 2)
            {
                float ur = real[i];
                float ui = imag[i];
                float vr = real[i + 1];
                float vi = imag[i + 1];

                real[i] = ur + vr;
                imag[i] = ui + vi;
                real[i + 1] = ur - vr;
                imag[i + 1] = ui - vi;
            }
        }

        // Unrolled stage len = 4 (half = 2, j=0 is 1+0i, j=1 is 0 +/- 1i)
        if (n >= 4)
        {
            for (int i = 0; i < n; i += 4)
            {
                float ur0 = real[i];
                float ui0 = imag[i];
                float vr0 = real[i + 2];
                float vi0 = imag[i + 2];

                real[i] = ur0 + vr0;
                imag[i] = ui0 + vi0;
                real[i + 2] = ur0 - vr0;
                imag[i + 2] = ui0 - vi0;

                float ur1 = real[i + 1];
                float ui1 = imag[i + 1];
                float tr1 = -sinSign * imag[i + 3];
                float ti1 = sinSign * real[i + 3];

                real[i + 1] = ur1 + tr1;
                imag[i + 1] = ui1 + ti1;
                real[i + 3] = ur1 - tr1;
                imag[i + 3] = ui1 - ti1;
            }
        }

        ref float rRef = ref MemoryMarshal.GetArrayDataReference(real);
        ref float iRef = ref MemoryMarshal.GetArrayDataReference(imag);
        Vector256<float> vSinSign256 = Vector256.Create(sinSign);
        Vector128<float> vSinSign128 = Vector128.Create(sinSign);

        int sIdx = 0;
        for (int len = 8; len <= n; len <<= 1, sIdx++)
        {
            int half = len >> 1;
            float[] stageCos = _stageCos[sIdx];
            float[] stageSin = _stageSin[sIdx];
            ref float scRef = ref MemoryMarshal.GetArrayDataReference(stageCos);
            ref float ssRef = ref MemoryMarshal.GetArrayDataReference(stageSin);

            for (int i = 0; i < n; i += len)
            {
                int j = 0;
                if (Vector256.IsHardwareAccelerated && half >= 8)
                {
                    for (; j <= half - 8; j += 8)
                    {
                        nuint uOff = (nuint)(i + j);
                        nuint vOff = (nuint)(i + j + half);
                        nuint twOff = (nuint)j;

                        Vector256<float> wr = Vector256.LoadUnsafe(ref scRef, twOff);
                        Vector256<float> wi = Vector256.LoadUnsafe(ref ssRef, twOff) * vSinSign256;

                        Vector256<float> vr = Vector256.LoadUnsafe(ref rRef, vOff);
                        Vector256<float> vi = Vector256.LoadUnsafe(ref iRef, vOff);
                        Vector256<float> ur = Vector256.LoadUnsafe(ref rRef, uOff);
                        Vector256<float> ui = Vector256.LoadUnsafe(ref iRef, uOff);

                        Vector256<float> tr = (wr * vr) - (wi * vi);
                        Vector256<float> ti = (wr * vi) + (wi * vr);

                        (ur + tr).StoreUnsafe(ref rRef, uOff);
                        (ui + ti).StoreUnsafe(ref iRef, uOff);
                        (ur - tr).StoreUnsafe(ref rRef, vOff);
                        (ui - ti).StoreUnsafe(ref iRef, vOff);
                    }
                }
                else if (Vector128.IsHardwareAccelerated && half >= 4)
                {
                    for (; j <= half - 4; j += 4)
                    {
                        nuint uOff = (nuint)(i + j);
                        nuint vOff = (nuint)(i + j + half);
                        nuint twOff = (nuint)j;

                        Vector128<float> wr = Vector128.LoadUnsafe(ref scRef, twOff);
                        Vector128<float> wi = Vector128.LoadUnsafe(ref ssRef, twOff) * vSinSign128;

                        Vector128<float> vr = Vector128.LoadUnsafe(ref rRef, vOff);
                        Vector128<float> vi = Vector128.LoadUnsafe(ref iRef, vOff);
                        Vector128<float> ur = Vector128.LoadUnsafe(ref rRef, uOff);
                        Vector128<float> ui = Vector128.LoadUnsafe(ref iRef, uOff);

                        Vector128<float> tr = (wr * vr) - (wi * vi);
                        Vector128<float> ti = (wr * vi) + (wi * vr);

                        (ur + tr).StoreUnsafe(ref rRef, uOff);
                        (ui + ti).StoreUnsafe(ref iRef, uOff);
                        (ur - tr).StoreUnsafe(ref rRef, vOff);
                        (ui - ti).StoreUnsafe(ref iRef, vOff);
                    }
                }

                for (; j < half; j++)
                {
                    float wr = stageCos[j];
                    float wi = sinSign * stageSin[j];

                    int u = i + j;
                    int v = u + half;

                    float tr = wr * real[v] - wi * imag[v];
                    float ti = wr * imag[v] + wi * real[v];

                    float ur = real[u];
                    float ui = imag[u];

                    real[u] = ur + tr;
                    imag[u] = ui + ti;
                    real[v] = ur - tr;
                    imag[v] = ui - ti;
                }
            }
        }

        if (inverse)
        {
            float invM = 1f / n;
            WasapiPcmConverter.MultiplyScalar(real.AsSpan(0, n), invM, real.AsSpan(0, n));
            WasapiPcmConverter.MultiplyScalar(imag.AsSpan(0, n), invM, imag.AsSpan(0, n));
        }
    }

    private static ModelEntry ResolveGtcrnModelEntry(ModelManifest manifest)
    {
        return manifest.FindByFilename(DefaultModelKey)
            ?? manifest.Models.FirstOrDefault(m =>
                m.Filename.Contains("gtcrn", StringComparison.OrdinalIgnoreCase))
            ?? throw new FileNotFoundException("GTCRN model entry not found in model_manifest.json.", DefaultModelKey);
    }

    private static int FindSpectrumTensorIndex(IReadOnlyList<TensorDescriptor> descriptors, int freqBins)
    {
        for (int i = 0; i < descriptors.Count; i++)
        {
            TensorDescriptor d = descriptors[i];
            if (d.Shape.Count == 4 &&
                d.Shape[1].StaticValue == freqBins &&
                d.Shape[3].StaticValue == 2)
            {
                return i;
            }
        }

        throw new InvalidDataException("GTCRN spectrum tensor matching [1, freqBins, 1, 2] not found in manifest.");
    }

    private static int FindMatchingCacheOutputIndex(
        IReadOnlyList<TensorDescriptor> outputs,
        TensorDescriptor inputCache,
        int spectrumOutIdx)
    {
        // Match by name prefix (e.g. conv_cache -> conv_cache_out) and exact static shape.
        for (int i = 0; i < outputs.Count; i++)
        {
            if (i == spectrumOutIdx)
            {
                continue;
            }

            TensorDescriptor candidate = outputs[i];
            if (candidate.Name.StartsWith(inputCache.Name, StringComparison.OrdinalIgnoreCase) &&
                ShapesEqual(inputCache.Shape, candidate.Shape))
            {
                return i;
            }
        }

        // Fallback: match by exact shape if naming differs.
        for (int i = 0; i < outputs.Count; i++)
        {
            if (i != spectrumOutIdx && ShapesEqual(inputCache.Shape, outputs[i].Shape))
            {
                return i;
            }
        }

        throw new InvalidDataException(
            $"Matching output tensor for GTCRN cache '{inputCache.Name}' not found in manifest.");
    }

    private static bool ShapesEqual(IReadOnlyList<TensorDimension> a, IReadOnlyList<TensorDimension> b)
    {
        if (a.Count != b.Count)
        {
            return false;
        }

        for (int i = 0; i < a.Count; i++)
        {
            if (a[i].StaticValue != b[i].StaticValue)
            {
                return false;
            }
        }

        return true;
    }

    private void ReleaseResources()
    {
        Volatile.Write(ref _isInitialized, false);

        _ortMix?.Dispose();
        _ortEnh?.Dispose();
        _ortMix = null;
        _ortEnh = null;

        for (int i = 0; i < _ortCachesA.Length; i++)
        {
            _ortCachesA[i]?.Dispose();
        }
        for (int i = 0; i < _ortCachesB.Length; i++)
        {
            _ortCachesB[i]?.Dispose();
        }

        _ortCachesA = [];
        _ortCachesB = [];
        _cacheBuffersA = [];
        _cacheBuffersB = [];
        _cacheInputSlots = [];
        _cacheOutputSlots = [];

        _runOptions?.Dispose();
        _runOptions = null;

        _sessionResult?.Dispose();
        _sessionResult = null;

        _inputNames = [];
        _inputValues = [];
        _outputNames = [];
        _outputValues = [];
    }
}
