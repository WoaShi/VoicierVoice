using System.IO;
using System.Runtime.CompilerServices;
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

    private int[] _bitRev = [];
    private float[] _twiddleCos = [];
    private float[] _twiddleSin = [];
    private float[] _fftReal = [];
    private float[] _fftImag = [];

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

            BuildFftTables(_nFft);
            BuildHannSqrtWindow(_nFft, audio.WindowType);

            _analysisBuffer = GC.AllocateArray<float>(_nFft, pinned: false);
            _synthesisOla = GC.AllocateArray<float>(_nFft, pinned: false);
            _fftReal = GC.AllocateArray<float>(_nFft, pinned: false);
            _fftImag = GC.AllocateArray<float>(_nFft, pinned: false);

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

        // 2. Apply periodic hann_sqrt analysis window and prepare complex buffer for FFT.
        float[] win = _window;
        float[] analysis = _analysisBuffer;
        float[] fftReal = _fftReal;
        float[] fftImag = _fftImag;

        for (int i = 0; i < nFft; i++)
        {
            fftReal[i] = analysis[i] * win[i];
            fftImag[i] = 0f;
        }

        // 3. Unscaled 512-pt real->complex FFT, populate mix: [1, 257, 1, 2].
        TransformInPlace(fftReal, fftImag, inverse: false);

        float[] mix = _mixBuffer;
        int bins = _freqBins;
        for (int k = 0; k < bins; k++)
        {
            int idx = k << 1;
            mix[idx] = fftReal[k];
            mix[idx + 1] = fftImag[k];
        }

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

        // 5. Reconstruct Hermitian spectrum from enh: [1, 257, 1, 2] and run 1/512-scaled IFFT.
        float[] enh = _enhBuffer;
        fftReal[0] = enh[0];
        fftImag[0] = 0f;

        int half = nFft >> 1;
        for (int k = 1; k < half; k++)
        {
            int idx = k << 1;
            float re = enh[idx];
            float im = enh[idx + 1];
            fftReal[k] = re;
            fftImag[k] = im;
            fftReal[nFft - k] = re;
            fftImag[nFft - k] = -im;
        }

        fftReal[half] = enh[half << 1];
        fftImag[half] = 0f;

        TransformInPlace(fftReal, fftImag, inverse: true);

        // 6. Apply synthesis hann_sqrt window and overlap-add (w[n]^2 + w[n+256]^2 == 1.0 -> 0 dB unity).
        float[] ola = _synthesisOla;
        for (int i = 0; i < nFft; i++)
        {
            ola[i] += fftReal[i] * win[i];
        }

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
        int bits = 0;
        for (int v = nFft; v > 1; v >>= 1)
        {
            bits++;
        }

        _bitRev = new int[nFft];
        for (int i = 0; i < nFft; i++)
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

        int half = nFft >> 1;
        _twiddleCos = new float[half];
        _twiddleSin = new float[half];
        double twoPiOverN = 2.0 * Math.PI / nFft;
        for (int k = 0; k < half; k++)
        {
            double angle = twoPiOverN * k;
            _twiddleCos[k] = (float)Math.Cos(angle);
            _twiddleSin[k] = (float)Math.Sin(angle);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void TransformInPlace(float[] real, float[] imag, bool inverse)
    {
        int n = _nFft;
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

        float[] cosTable = _twiddleCos;
        float[] sinTable = _twiddleSin;
        float sinSign = inverse ? 1f : -1f;

        for (int len = 2; len <= n; len <<= 1)
        {
            int half = len >> 1;
            int step = n / len;

            for (int i = 0; i < n; i += len)
            {
                int twIdx = 0;
                for (int j = 0; j < half; j++)
                {
                    float wr = cosTable[twIdx];
                    float wi = sinSign * sinTable[twIdx];
                    twIdx += step;

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
            float invN = 1f / n;
            for (int i = 0; i < n; i++)
            {
                real[i] *= invN;
                imag[i] *= invN;
            }
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
