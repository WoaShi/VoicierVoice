using System.IO;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using Microsoft.ML.OnnxRuntime;
using NAudio.Wave;
using VoiceLockStudio.Models;
using VoiceLockStudio.Services.Audio;
using VoiceLockStudio.Services.System;

namespace VoiceLockStudio.Services.Backends;

/// <summary>
/// Target-speaker voice isolation backend powered by VoxLock v2 ONNX models.
/// Encapsulates all VoxLock model selection, enrollment, binary voiceprint caching,
/// reflect-padded STFT chunking (<c>K = 4</c>, <c>S = 896</c>, 32 ms chunk / 24 ms overlap),
/// and squared-Hann OLA synthesis strictly from <see cref="ModelManifest"/>.
/// </summary>
public sealed class VoxLockOnnxBackend : IVoiceIsolatorBackend
{
    private const int HopsPerChunk = 4; // K = 4 -> FrameSizeSamples = 512 (32 ms @ 16 kHz, 2x GTCRN hops), S = 896
    private const int MaxEnrollSearchSeconds = 30;
    private const string CacheMagic = "VXLCACH2";
    private const uint CacheFormatVersion = 1;

    private readonly object _lifecycleLock = new();

    private ModelManifest? _manifest;
    private string _modelsRootDirectory = AppContext.BaseDirectory;
    private ComputeBackend _requestedBackend = ComputeBackend.Cpu;
    private VoxLockModelVariant _variant = VoxLockModelVariant.Int8;

    private ModelEntry? _chunkEntry;
    private ModelEntry? _enrollEntry;
    private string _combinedModelSha256 = string.Empty;

    private OnnxSessionResult? _chunkSessionResult;
    private RunOptions? _runOptions;

    private int _sampleRate = 16000;
    private int _nFft = 512;
    private int _hopLength = 128;
    private int _reflectPadSamples = 256;
    private int _chunkInputSamples = 896; // S = nFft + (K - 1) * hop
    private int _doneSamples = 512;       // K * hop
    private float _outputGain = -0.5f;
    private double _algorithmicLatencyMs = 24.0;

    // Precomputed OLA window (periodic Hann squared) and streaming buffers
    private float[] _win2 = [];
    private float[] _inBuf = [];
    private int _inLen;
    private bool _didStartupPad;
    private int _droppedRemaining;
    private float[] _olaBuffer = [];
    private float[] _envBuffer = [];

    // Preallocated ONNX I/O buffers and pinned OrtValues for chunk inference
    private float[] _wavBuffer = [];
    private float[] _enrollNBuffer = [];
    private float[] _kBuffer = [];
    private float[] _vBuffer = [];
    private float[] _hBufferA = [];
    private float[] _hBufferB = [];
    private float[] _cBufferA = [];
    private float[] _cBufferB = [];
    private float[] _framesBuffer = [];

    private long[] _enrollNShape = [];
    private long[] _kShape = [];
    private long[] _vShape = [];

    private OrtValue? _ortWav;
    private OrtValue? _ortEnrollN;
    private OrtValue? _ortK;
    private OrtValue? _ortV;
    private OrtValue? _ortHA;
    private OrtValue? _ortHB;
    private OrtValue? _ortCA;
    private OrtValue? _ortCB;
    private OrtValue? _ortFrames;

    private string[] _inputNames = [];
    private OrtValue[] _inputValues = [];
    private string[] _outputNames = [];
    private OrtValue[] _outputValues = [];
    private int _hInputSlot;
    private int _cInputSlot;
    private int _hOutputSlot;
    private int _cOutputSlot;
    private bool _stateIsA = true;

    private VoiceprintProfileInfo? _currentVoiceprint;
    private bool _hasEnrolledVoiceprint;
    private bool _isInitialized;
    private bool _disposed;

    public string Name => $"VoxLock v2 ({_variant})";
    public int FrameSizeSamples => _doneSamples;
    public int SampleRate => _sampleRate;
    public double AlgorithmicLatencyMs => _algorithmicLatencyMs;
    public bool IsInitialized => Volatile.Read(ref _isInitialized);
    public bool HasEnrolledVoiceprint => Volatile.Read(ref _hasEnrolledVoiceprint);
    public VoiceprintProfileInfo? CurrentVoiceprint => Volatile.Read(ref _currentVoiceprint);

    public string ActiveProviderName => _chunkSessionResult?.ActiveProviderName ?? string.Empty;
    public string? ProviderFallbackMessage => _chunkSessionResult?.FallbackMessage;

    public void Initialize(AudioStageOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        ObjectDisposedException.ThrowIf(_disposed, this);

        lock (_lifecycleLock)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            ReleaseResources();

            _manifest = OnnxSessionFactory.ResolveManifest(options);
            _modelsRootDirectory = string.IsNullOrWhiteSpace(options.ModelsRootDirectory)
                ? AppContext.BaseDirectory
                : options.ModelsRootDirectory;
            _requestedBackend = options.Backend;
            _variant = options.VoxLockVariant;

            (string chunkModelKey, string enrollModelKey) = GetModelFilenamesForVariant(_variant);
            _chunkEntry = OnnxSessionFactory.ResolveModelEntry(_manifest, chunkModelKey);
            _enrollEntry = OnnxSessionFactory.ResolveModelEntry(_manifest, enrollModelKey);
            _combinedModelSha256 = ComputeCombinedModelSha256(_enrollEntry.Sha256, _chunkEntry.Sha256);

            ModelAudioParams audio = _chunkEntry.AudioParams ?? throw new InvalidDataException(
                $"Model '{_chunkEntry.Filename}' is missing audio_params in model_manifest.json.");

            _sampleRate = audio.SampleRate > 0
                ? audio.SampleRate
                : throw new InvalidDataException($"Model '{_chunkEntry.Filename}' has invalid sample_rate.");
            _nFft = audio.NFft
                ?? throw new InvalidDataException($"Model '{_chunkEntry.Filename}' is missing n_fft.");
            _hopLength = audio.HopLength
                ?? throw new InvalidDataException($"Model '{_chunkEntry.Filename}' is missing hop_length.");
            _reflectPadSamples = audio.ReflectPadSamples
                ?? throw new InvalidDataException($"Model '{_chunkEntry.Filename}' is missing reflect_pad_samples.");
            _outputGain = (float)(audio.OutputGainMultiplier ?? -0.5);
            _algorithmicLatencyMs = audio.AlgorithmicLatencyMs ?? 24.0;

            _doneSamples = HopsPerChunk * _hopLength;
            _chunkInputSamples = _nFft + (HopsPerChunk - 1) * _hopLength;

            int intraOpThreads = ResolveIntraOpThreads(options);
            HashSet<int>? baselineThreads = (options.Backend == ComputeBackend.Cpu && options.CpuAffinityMask != 0 && intraOpThreads > 1)
                ? CpuAffinityManager.CaptureCurrentOsThreadIds()
                : null;

            _chunkSessionResult = OnnxSessionFactory.CreateSession(
                _manifest,
                _chunkEntry.Filename,
                _modelsRootDirectory,
                _requestedBackend,
                intraOpThreads);

            BuildPeriodicHannSquaredWindow(_nFft, audio.WindowType);

            _inBuf = GC.AllocateArray<float>(_reflectPadSamples + _chunkInputSamples + _doneSamples, pinned: true);
            _olaBuffer = GC.AllocateArray<float>(_nFft + _doneSamples, pinned: true);
            _envBuffer = GC.AllocateArray<float>(_nFft + _doneSamples, pinned: true);

            AllocateChunkTensors(_chunkEntry, _enrollEntry);

            _runOptions = new RunOptions();
            _inLen = 0;
            _didStartupPad = false;
            _droppedRemaining = 0;
            _stateIsA = true;
            Volatile.Write(ref _hasEnrolledVoiceprint, false);
            Volatile.Write(ref _currentVoiceprint, null);
            Volatile.Write(ref _isInitialized, true);

            // Warm up chunk session immediately so lazy ORT arenas/threads are initialized before audio capture starts.
            WarmupCore();

            if (baselineThreads is not null && options.CpuAffinityMask != 0)
            {
                new CpuAffinityManager().BindNewProcessThreads(
                    baselineThreads,
                    options.CpuAffinityMask,
                    options.ProcessorGroup);
            }

            // Auto-load or rebuild cached voiceprint if the default raw voice WAV already exists.
            if (File.Exists(VoiceprintStorage.DefaultUserRawVoiceWavPath))
            {
                LoadOrRebuildVoiceprintCache(VoiceprintStorage.DefaultUserRawVoiceWavPath, forceRebuild: false);
            }
        }
    }

    public VoiceprintProfileInfo EnrollVoiceprint(string wavFilePath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(wavFilePath);
        ObjectDisposedException.ThrowIf(_disposed, this);

        if (!File.Exists(wavFilePath))
        {
            throw new FileNotFoundException($"Voiceprint source WAV file '{wavFilePath}' was not found.", wavFilePath);
        }

        lock (_lifecycleLock)
        {
            EnsureInitializedCore();
            return EnrollVoiceprintCore(wavFilePath);
        }
    }

    public VoiceprintProfileInfo? LoadOrRebuildVoiceprintCache(string? wavFilePath = null, bool forceRebuild = false)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        lock (_lifecycleLock)
        {
            EnsureInitializedCore();

            string targetWav = string.IsNullOrWhiteSpace(wavFilePath)
                ? VoiceprintStorage.DefaultUserRawVoiceWavPath
                : wavFilePath;

            string cachePath = VoiceprintStorage.GetCacheFilePath(_combinedModelSha256);

            if (!forceRebuild && File.Exists(cachePath))
            {
                string? currentWavHash = File.Exists(targetWav)
                    ? VoiceprintStorage.ComputeFileSha256Hex(targetWav)
                    : null;

                if (TryLoadCacheFile(cachePath, currentWavHash, targetWav, out VoiceprintProfileInfo? cachedProfile))
                {
                    Volatile.Write(ref _currentVoiceprint, cachedProfile);
                    Volatile.Write(ref _hasEnrolledVoiceprint, true);
                    return cachedProfile;
                }
            }

            if (File.Exists(targetWav))
            {
                return EnrollVoiceprintCore(targetWav);
            }

            Volatile.Write(ref _hasEnrolledVoiceprint, false);
            Volatile.Write(ref _currentVoiceprint, null);
            return null;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public void ProcessFrame(ReadOnlySpan<float> input, Span<float> output)
    {
        if (!Volatile.Read(ref _isInitialized) || _chunkSessionResult is null || _runOptions is null)
        {
            throw new InvalidOperationException($"{Name} must be initialized before calling ProcessFrame.");
        }

        if (input.Length != _doneSamples)
        {
            throw new ArgumentException($"Expected {_doneSamples} input samples, got {input.Length}.", nameof(input));
        }

        if (output.Length != _doneSamples)
        {
            throw new ArgumentException($"Expected {_doneSamples} output samples, got {output.Length}.", nameof(output));
        }

        if (!Volatile.Read(ref _hasEnrolledVoiceprint))
        {
            if (!Unsafe.AreSame(
                ref MemoryMarshal.GetReference(input),
                ref MemoryMarshal.GetReference(output)))
            {
                input.CopyTo(output);
            }
            return;
        }

        input.CopyTo(_inBuf.AsSpan(_inLen, _doneSamples));
        _inLen += _doneSamples;

        if (!_didStartupPad)
        {
            if (_inLen <= _reflectPadSamples)
            {
                output.Clear();
                return;
            }

            // Prepend 256-sample reflect pad: head[1..256] reversed before head[0].
            _inBuf.AsSpan(0, _inLen).CopyTo(_inBuf.AsSpan(_reflectPadSamples, _inLen));
            for (int p = 0; p < _reflectPadSamples; p++)
            {
                _inBuf[p] = _inBuf[_reflectPadSamples + (_reflectPadSamples - p)];
            }

            _inLen += _reflectPadSamples;
            _didStartupPad = true;
            _droppedRemaining = _reflectPadSamples;
        }

        bool emittedOutput = false;
        while (_inLen >= _chunkInputSamples)
        {
            _inBuf.AsSpan(0, _chunkInputSamples).CopyTo(_wavBuffer);

            int remainingIn = _inLen - _doneSamples;
            if (remainingIn > 0)
            {
                _inBuf.AsSpan(_doneSamples, remainingIn).CopyTo(_inBuf.AsSpan(0, remainingIn));
            }
            _inLen = remainingIn;

            _chunkSessionResult.Session.Run(_runOptions, _inputNames, _inputValues, _outputNames, _outputValues);

            _stateIsA = !_stateIsA;
            if (_stateIsA)
            {
                _inputValues[_hInputSlot] = _ortHA!;
                _inputValues[_cInputSlot] = _ortCA!;
                _outputValues[_hOutputSlot] = _ortHB!;
                _outputValues[_cOutputSlot] = _ortCB!;
            }
            else
            {
                _inputValues[_hInputSlot] = _ortHB!;
                _inputValues[_cInputSlot] = _ortCB!;
                _outputValues[_hOutputSlot] = _ortHA!;
                _outputValues[_cOutputSlot] = _ortCA!;
            }

            // Overlap-add K = 2 frames of length nFft (512) at stride hopLength (128).
            float[] frames = _framesBuffer;
            float[] ola = _olaBuffer;
            float[] env = _envBuffer;
            float[] win2 = _win2;
            int nFft = _nFft;
            int hop = _hopLength;

            for (int f = 0; f < HopsPerChunk; f++)
            {
                int srcOffset = f * nFft;
                int dstOffset = f * hop;
                for (int n = 0; n < nFft; n++)
                {
                    ola[dstOffset + n] += frames[srcOffset + n];
                    env[dstOffset + n] += win2[n];
                }
            }

            int done = _doneSamples;
            if (_droppedRemaining >= done)
            {
                _droppedRemaining -= done;
            }
            else
            {
                int start = _droppedRemaining;
                _droppedRemaining = 0;

                if (start > 0)
                {
                    output.Slice(0, start).Clear();
                }

                float gain = _outputGain;
                for (int i = start; i < done; i++)
                {
                    float denom = env[i];
                    output[i] = (ola[i] / (denom > 1e-11f ? denom : 1f)) * gain;
                }

                emittedOutput = true;
            }

            int olaTail = ola.Length - done;
            ola.AsSpan(done, olaTail).CopyTo(ola.AsSpan(0, olaTail));
            ola.AsSpan(olaTail, done).Clear();

            env.AsSpan(done, olaTail).CopyTo(env.AsSpan(0, olaTail));
            env.AsSpan(olaTail, done).Clear();
        }

        if (!emittedOutput)
        {
            output.Clear();
        }
    }

    public void ResetState()
    {
        if (!Volatile.Read(ref _isInitialized))
        {
            return;
        }

        Array.Clear(_inBuf);
        Array.Clear(_olaBuffer);
        Array.Clear(_envBuffer);
        Array.Clear(_wavBuffer);
        Array.Clear(_framesBuffer);
        Array.Clear(_hBufferA);
        Array.Clear(_hBufferB);
        Array.Clear(_cBufferA);
        Array.Clear(_cBufferB);

        _inLen = 0;
        _didStartupPad = false;
        _droppedRemaining = 0;
        _stateIsA = true;

        if (_ortHA is not null && _ortHB is not null && _ortCA is not null && _ortCB is not null)
        {
            _inputValues[_hInputSlot] = _ortHA;
            _inputValues[_cInputSlot] = _ortCA;
            _outputValues[_hOutputSlot] = _ortHB;
            _outputValues[_cOutputSlot] = _ortCB;
        }
    }

    public void Warmup()
    {
        if (!Volatile.Read(ref _isInitialized) || _chunkSessionResult is null || _runOptions is null)
        {
            return;
        }

        lock (_lifecycleLock)
        {
            WarmupCore();
        }
    }

    private void WarmupCore()
    {
        if (_chunkSessionResult is null || _runOptions is null)
        {
            return;
        }

        Array.Clear(_wavBuffer);
        for (int i = 0; i < 2; i++)
        {
            _chunkSessionResult.Session.Run(_runOptions, _inputNames, _inputValues, _outputNames, _outputValues);
        }
        ResetState();
    }

    private static int ResolveIntraOpThreads(AudioStageOptions options)
    {
        if (options.OrtIntraOpThreads > 1)
        {
            return Math.Clamp(options.OrtIntraOpThreads, 1, 8);
        }

        if (options.Backend == ComputeBackend.Cpu && options.CpuAffinityMask != 0)
        {
            // A 27.3M BSRNN at 40 ms chunk takes ~51 ms on 1 E-Core (RTF=1.28) vs ~16-21 ms on 2-3 E-Cores (RTF=0.41-0.53).
            int coreCount = BitOperations.PopCount(options.CpuAffinityMask);
            return Math.Clamp(coreCount, 1, 3);
        }

        return 1;
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

    private void EnsureInitializedCore()
    {
        if (!Volatile.Read(ref _isInitialized) ||
            _manifest is null ||
            _chunkEntry is null ||
            _enrollEntry is null)
        {
            throw new InvalidOperationException($"{Name} must be initialized first.");
        }
    }

    private VoiceprintProfileInfo EnrollVoiceprintCore(string wavFilePath)
    {
        ModelManifest manifest = _manifest!;
        ModelEntry enrollEntry = _enrollEntry!;
        ModelEntry chunkEntry = _chunkEntry!;

        ModelAudioParams enrollAudio = enrollEntry.AudioParams ?? throw new InvalidDataException(
            $"Model '{enrollEntry.Filename}' is missing audio_params.");

        int enrollSampleRate = enrollAudio.SampleRate > 0 ? enrollAudio.SampleRate : _sampleRate;
        int minEnrollSamples = enrollAudio.MinEnrollSamples ?? 48000;
        int fbankHop = enrollAudio.FbankHopLengthSamples ?? 160;

        float[] monoAtEnrollRate = LoadMonoAudioAtSampleRate(wavFilePath, enrollSampleRate, out double sourceDurationSeconds);
        if (monoAtEnrollRate.Length == 0)
        {
            throw new InvalidDataException($"WAV file '{wavFilePath}' contains no audio samples.");
        }

        // Persist raw 16 kHz mono WAV in LocalAppData if enrollment came from another path.
        string fullSourcePath = Path.GetFullPath(wavFilePath);
        string fullDefaultPath = Path.GetFullPath(VoiceprintStorage.DefaultUserRawVoiceWavPath);
        string effectiveWavPath = fullDefaultPath;

        if (!string.Equals(fullSourcePath, fullDefaultPath, StringComparison.OrdinalIgnoreCase))
        {
            SaveMonoWav16kFloat(fullDefaultPath, monoAtEnrollRate, enrollSampleRate);
        }
        else
        {
            effectiveWavPath = fullSourcePath;
        }

        string wavSha256 = VoiceprintStorage.ComputeFileSha256Hex(effectiveWavPath);
        float[] enrollWindow = SelectLoudestOrPadWindow(monoAtEnrollRate, minEnrollSamples, fbankHop, enrollSampleRate);

        RunEnrollmentModel(manifest, enrollEntry, enrollWindow);

        DateTimeOffset enrolledAtUtc = DateTimeOffset.UtcNow;
        string cachePath = VoiceprintStorage.GetCacheFilePath(_combinedModelSha256);
        WriteCacheFile(
            cachePath,
            enrollEntry.Sha256,
            chunkEntry.Sha256,
            _combinedModelSha256,
            wavSha256,
            enrolledAtUtc,
            sourceDurationSeconds);

        var profile = new VoiceprintProfileInfo(
            SourceWavPath: effectiveWavPath,
            CacheFilePath: cachePath,
            ContentSha256: wavSha256,
            EnrolledAtUtc: enrolledAtUtc,
            SourceDurationSeconds: sourceDurationSeconds,
            LoadedFromCache: false);

        Volatile.Write(ref _currentVoiceprint, profile);
        Volatile.Write(ref _hasEnrolledVoiceprint, true);
        return profile;
    }

    private void RunEnrollmentModel(ModelManifest manifest, ModelEntry enrollEntry, float[] enrollWindow)
    {
        IReadOnlyList<TensorDescriptor> enrollInputs = enrollEntry.GetEffectiveInputs();
        IReadOnlyList<TensorDescriptor> enrollOutputs = enrollEntry.GetEffectiveOutputs();
        if (enrollInputs.Count < 1 || enrollOutputs.Count < 3)
        {
            throw new InvalidDataException($"Model '{enrollEntry.Filename}' has unexpected I/O count.");
        }

        TensorDescriptor enrollInDesc = enrollInputs[0];
        long[] enrollInShape = OnnxSessionFactory.ResolveTensorShape(enrollInDesc);

        TensorDescriptor enrollNOutDesc = FindDescriptorMatchingShape(enrollOutputs, _enrollNShape);
        TensorDescriptor kOutDesc = FindDescriptorMatchingShape(enrollOutputs, _kShape);
        TensorDescriptor vOutDesc = FindDescriptorMatchingShape(enrollOutputs, _vShape);

        using OnnxSessionResult enrollSession = OnnxSessionFactory.CreateSession(
            manifest,
            enrollEntry.Filename,
            _modelsRootDirectory,
            _requestedBackend);

        using var runOptions = new RunOptions();
        using OrtValue ovEnrollIn = OrtValue.CreateTensorValueFromMemory(enrollWindow, enrollInShape);
        using OrtValue ovEnrollNOut = OrtValue.CreateTensorValueFromMemory(_enrollNBuffer, _enrollNShape);
        using OrtValue ovKOut = OrtValue.CreateTensorValueFromMemory(_kBuffer, _kShape);
        using OrtValue ovVOut = OrtValue.CreateTensorValueFromMemory(_vBuffer, _vShape);

        enrollSession.Session.Run(
            runOptions,
            [enrollInDesc.Name],
            [ovEnrollIn],
            [enrollNOutDesc.Name, kOutDesc.Name, vOutDesc.Name],
            [ovEnrollNOut, ovKOut, ovVOut]);
    }

    private void AllocateChunkTensors(ModelEntry chunkEntry, ModelEntry enrollEntry)
    {
        IReadOnlyList<TensorDescriptor> inputs = chunkEntry.GetEffectiveInputs();
        IReadOnlyList<TensorDescriptor> outputs = chunkEntry.GetEffectiveOutputs();
        IReadOnlyList<TensorDescriptor> enrollOutputs = enrollEntry.GetEffectiveOutputs();

        if (inputs.Count != 6 || outputs.Count != 3 || enrollOutputs.Count != 3)
        {
            throw new InvalidDataException(
                $"VoxLock manifest I/O mismatch: chunk({inputs.Count},{outputs.Count}), enroll_out({enrollOutputs.Count}).");
        }

        // Identify enroll_n, k, v from enroll model outputs (by rank & relative dimensions).
        TensorDescriptor enrollNDesc = enrollOutputs.First(d => d.Shape.Count == 2);
        TensorDescriptor kEnrollDesc = enrollOutputs.First(d =>
            d.Shape.Count == 4 && d.Shape[2].StaticValue < d.Shape[3].StaticValue);
        TensorDescriptor vEnrollDesc = enrollOutputs.First(d =>
            d.Shape.Count == 4 && d.Shape[2].StaticValue > d.Shape[3].StaticValue);

        _enrollNShape = OnnxSessionFactory.ResolveTensorShape(enrollNDesc);
        _kShape = OnnxSessionFactory.ResolveTensorShape(kEnrollDesc);
        _vShape = OnnxSessionFactory.ResolveTensorShape(vEnrollDesc);

        _enrollNBuffer = GC.AllocateArray<float>(OnnxSessionFactory.ComputeElementCount(_enrollNShape), pinned: true);
        _kBuffer = GC.AllocateArray<float>(OnnxSessionFactory.ComputeElementCount(_kShape), pinned: true);
        _vBuffer = GC.AllocateArray<float>(OnnxSessionFactory.ComputeElementCount(_vShape), pinned: true);

        _ortEnrollN = OrtValue.CreateTensorValueFromMemory(_enrollNBuffer, _enrollNShape);
        _ortK = OrtValue.CreateTensorValueFromMemory(_kBuffer, _kShape);
        _ortV = OrtValue.CreateTensorValueFromMemory(_vBuffer, _vShape);

        // Identify chunk inputs: wav (rank 2 with symbolic dim), enroll_n, k, v, h, c.
        TensorDescriptor wavInDesc = inputs.First(d => d.Shape.Count == 2 && d.Shape.Any(dim => dim.IsDynamic));
        var wavOverrides = new Dictionary<string, long>(StringComparer.Ordinal);
        foreach (TensorDimension dim in wavInDesc.Shape)
        {
            if (!string.IsNullOrEmpty(dim.SymbolicName))
            {
                wavOverrides[dim.SymbolicName] = _chunkInputSamples;
            }
        }
        long[] wavShape = OnnxSessionFactory.ResolveTensorShape(wavInDesc, wavOverrides, _chunkInputSamples);
        _wavBuffer = GC.AllocateArray<float>(OnnxSessionFactory.ComputeElementCount(wavShape), pinned: true);
        _ortWav = OrtValue.CreateTensorValueFromMemory(_wavBuffer, wavShape);

        // Identify chunk outputs: frames (rank 3 with dynamic K) and h_out, c_out (rank 3 static).
        TensorDescriptor framesOutDesc = outputs.First(d => d.Shape.Count == 3 && d.Shape.Any(dim => dim.IsDynamic));
        var framesOverrides = new Dictionary<string, long>(StringComparer.Ordinal);
        foreach (TensorDimension dim in framesOutDesc.Shape)
        {
            if (!string.IsNullOrEmpty(dim.SymbolicName))
            {
                framesOverrides[dim.SymbolicName] = HopsPerChunk;
            }
        }
        long[] framesShape = OnnxSessionFactory.ResolveTensorShape(framesOutDesc, framesOverrides, HopsPerChunk);
        _framesBuffer = GC.AllocateArray<float>(OnnxSessionFactory.ComputeElementCount(framesShape), pinned: true);
        _ortFrames = OrtValue.CreateTensorValueFromMemory(_framesBuffer, framesShape);

        // Recurrent state tensors h & c (rank 3 static).
        List<TensorDescriptor> stateInputs = inputs.Where(d => d.Shape.Count == 3 && d.Shape.All(dim => dim.IsStatic)).ToList();
        List<TensorDescriptor> stateOutputs = outputs.Where(d => d.Shape.Count == 3 && d.Shape.All(dim => dim.IsStatic)).ToList();
        if (stateInputs.Count != 2 || stateOutputs.Count != 2)
        {
            throw new InvalidDataException("Expected 2 recurrent state input/output pairs in VoxLock chunk model.");
        }

        TensorDescriptor hInDesc = stateInputs[0];
        TensorDescriptor cInDesc = stateInputs[1];
        TensorDescriptor hOutDesc = stateOutputs.FirstOrDefault(o =>
            o.Name.StartsWith(hInDesc.Name, StringComparison.OrdinalIgnoreCase)) ?? stateOutputs[0];
        TensorDescriptor cOutDesc = stateOutputs.FirstOrDefault(o =>
            o != hOutDesc && o.Name.StartsWith(cInDesc.Name, StringComparison.OrdinalIgnoreCase)) ?? stateOutputs[1];

        long[] hShape = OnnxSessionFactory.ResolveTensorShape(hInDesc);
        long[] cShape = OnnxSessionFactory.ResolveTensorShape(cInDesc);
        int hCount = OnnxSessionFactory.ComputeElementCount(hShape);
        int cCount = OnnxSessionFactory.ComputeElementCount(cShape);

        _hBufferA = GC.AllocateArray<float>(hCount, pinned: true);
        _hBufferB = GC.AllocateArray<float>(hCount, pinned: true);
        _cBufferA = GC.AllocateArray<float>(cCount, pinned: true);
        _cBufferB = GC.AllocateArray<float>(cCount, pinned: true);

        _ortHA = OrtValue.CreateTensorValueFromMemory(_hBufferA, hShape);
        _ortHB = OrtValue.CreateTensorValueFromMemory(_hBufferB, hShape);
        _ortCA = OrtValue.CreateTensorValueFromMemory(_cBufferA, cShape);
        _ortCB = OrtValue.CreateTensorValueFromMemory(_cBufferB, cShape);

        TensorDescriptor chunkEnrollNDesc = FindDescriptorMatchingShape(inputs, _enrollNShape, wavInDesc);
        TensorDescriptor chunkKDesc = FindDescriptorMatchingShape(inputs, _kShape);
        TensorDescriptor chunkVDesc = FindDescriptorMatchingShape(inputs, _vShape);

        _inputNames =
        [
            wavInDesc.Name,
            chunkEnrollNDesc.Name,
            chunkKDesc.Name,
            chunkVDesc.Name,
            hInDesc.Name,
            cInDesc.Name
        ];

        _inputValues =
        [
            _ortWav,
            _ortEnrollN,
            _ortK,
            _ortV,
            _ortHA,
            _ortCA
        ];

        _hInputSlot = 4;
        _cInputSlot = 5;

        _outputNames = [framesOutDesc.Name, hOutDesc.Name, cOutDesc.Name];
        _outputValues = [_ortFrames, _ortHB, _ortCB];
        _hOutputSlot = 1;
        _cOutputSlot = 2;
    }

    private void BuildPeriodicHannSquaredWindow(int nFft, string? windowType)
    {
        if (!string.IsNullOrWhiteSpace(windowType) &&
            !windowType.Contains("hann", StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidDataException($"Unsupported VoxLock window_type '{windowType}'.");
        }

        _win2 = GC.AllocateArray<float>(nFft, pinned: false);
        double twoPiOverN = 2.0 * Math.PI / nFft;
        for (int n = 0; n < nFft; n++)
        {
            double w = 0.5 - 0.5 * Math.Cos(twoPiOverN * n);
            _win2[n] = (float)(w * w);
        }
    }

    private static (string ChunkFilename, string EnrollFilename) GetModelFilenamesForVariant(VoxLockModelVariant variant) =>
        variant switch
        {
            VoxLockModelVariant.Int8 => ("chunk_int8.onnx", "enroll.onnx"),
            VoxLockModelVariant.Int8Full => ("chunk_int8_full.onnx", "enroll.onnx"),
            VoxLockModelVariant.Fp16 => ("chunk_fp16.onnx", "enroll_fp16.onnx"),
            _ => throw new ArgumentOutOfRangeException(nameof(variant), variant, "Unsupported VoxLock model variant.")
        };

    private static string ComputeCombinedModelSha256(string enrollSha256, string chunkSha256)
    {
        byte[] payload = Encoding.UTF8.GetBytes($"{enrollSha256.ToLowerInvariant()}:{chunkSha256.ToLowerInvariant()}");
        return Convert.ToHexStringLower(SHA256.HashData(payload));
    }

    private static float[] LoadMonoAudioAtSampleRate(
        string wavFilePath,
        int targetSampleRate,
        out double sourceDurationSeconds)
    {
        using var reader = new WaveFileReader(wavFilePath);
        int sourceRate = reader.WaveFormat.SampleRate;
        int channels = reader.WaveFormat.Channels;
        if (sourceRate <= 0 || channels <= 0)
        {
            throw new InvalidDataException($"Invalid WAV format in '{wavFilePath}'.");
        }

        ISampleProvider sampleProvider = reader.ToSampleProvider();
        var monoList = new List<float>((int)Math.Min(int.MaxValue, reader.SampleCount + 1024));
        float[] readChunk = new float[4096 * channels];

        while (true)
        {
            int samplesRead = sampleProvider.Read(readChunk, 0, readChunk.Length);
            if (samplesRead <= 0)
            {
                break;
            }

            if (channels == 1)
            {
                for (int i = 0; i < samplesRead; i++)
                {
                    monoList.Add(readChunk[i]);
                }
            }
            else
            {
                float invCh = 1f / channels;
                int frames = samplesRead / channels;
                for (int f = 0; f < frames; f++)
                {
                    float sum = 0f;
                    int baseIdx = f * channels;
                    for (int ch = 0; ch < channels; ch++)
                    {
                        sum += readChunk[baseIdx + ch];
                    }
                    monoList.Add(sum * invCh);
                }
            }
        }

        sourceDurationSeconds = (double)monoList.Count / sourceRate;
        if (monoList.Count == 0)
        {
            return [];
        }

        float[] monoSource = monoList.ToArray();
        if (sourceRate == targetSampleRate)
        {
            return monoSource;
        }

        var resampler = new StreamingResampler(sourceRate, targetSampleRate);
        int estimatedOut = (int)Math.Ceiling((double)monoSource.Length * targetSampleRate / sourceRate) + 64;
        float[] resampled = new float[estimatedOut];

        int totalOut = 0;
        int offset = 0;
        const int blockSize = 2048;

        while (offset < monoSource.Length)
        {
            int take = Math.Min(blockSize, monoSource.Length - offset);
            resampler.Process(
                monoSource.AsSpan(offset, take),
                resampled.AsSpan(totalOut),
                out int consumed,
                out int written);
            offset += consumed;
            totalOut += written;
            if (consumed == 0 && written == 0)
            {
                break;
            }
        }

        return resampled.AsSpan(0, totalOut).ToArray();
    }

    private static void SaveMonoWav16kFloat(string destinationPath, float[] monoSamples, int sampleRate)
    {
        string? dir = Path.GetDirectoryName(destinationPath);
        if (!string.IsNullOrEmpty(dir))
        {
            Directory.CreateDirectory(dir);
        }

        WaveFormat format = WaveFormat.CreateIeeeFloatWaveFormat(sampleRate, 1);
        using var writer = new WaveFileWriter(destinationPath, format);
        writer.WriteSamples(monoSamples, 0, monoSamples.Length);
        writer.Flush();
    }

    private static float[] SelectLoudestOrPadWindow(
        float[] mono16k,
        int targetSamples,
        int frameHop,
        int sampleRate)
    {
        float[] window = new float[targetSamples];
        if (mono16k.Length <= targetSamples)
        {
            mono16k.AsSpan().CopyTo(window);
            return window;
        }

        int searchSamples = Math.Min(mono16k.Length, MaxEnrollSearchSeconds * sampleRate);
        int numFrames = searchSamples / frameHop;
        int framesInWindow = targetSamples / frameHop;

        if (numFrames <= framesInWindow)
        {
            mono16k.AsSpan(0, targetSamples).CopyTo(window);
            return window;
        }

        double[] frameEnergy = new double[numFrames];
        for (int f = 0; f < numFrames; f++)
        {
            int start = f * frameHop;
            double energy = 0.0;
            for (int i = 0; i < frameHop; i++)
            {
                double s = mono16k[start + i];
                energy += s * s;
            }
            frameEnergy[f] = energy;
        }

        double currentSum = 0.0;
        for (int f = 0; f < framesInWindow; f++)
        {
            currentSum += frameEnergy[f];
        }

        double bestSum = currentSum;
        int bestFrameStart = 0;
        for (int f = framesInWindow; f < numFrames; f++)
        {
            currentSum += frameEnergy[f] - frameEnergy[f - framesInWindow];
            if (currentSum > bestSum)
            {
                bestSum = currentSum;
                bestFrameStart = f - framesInWindow + 1;
            }
        }

        int sampleStart = bestFrameStart * frameHop;
        mono16k.AsSpan(sampleStart, targetSamples).CopyTo(window);
        return window;
    }

    private void WriteCacheFile(
        string cachePath,
        string enrollModelSha256,
        string chunkModelSha256,
        string combinedModelSha256,
        string sourceWavSha256,
        DateTimeOffset enrolledAtUtc,
        double sourceDurationSeconds)
    {
        string? dir = Path.GetDirectoryName(cachePath);
        if (!string.IsNullOrEmpty(dir))
        {
            Directory.CreateDirectory(dir);
        }

        string tempPath = $"{cachePath}.{Environment.ProcessId}.tmp";
        using (var fs = new FileStream(tempPath, FileMode.Create, FileAccess.Write, FileShare.None))
        using (var writer = new BinaryWriter(fs, Encoding.UTF8, leaveOpen: false))
        {
            writer.Write(Encoding.ASCII.GetBytes(CacheMagic));
            writer.Write(CacheFormatVersion);
            writer.Write(enrollModelSha256);
            writer.Write(chunkModelSha256);
            writer.Write(combinedModelSha256);
            writer.Write(sourceWavSha256);
            writer.Write(enrolledAtUtc.UtcTicks);
            writer.Write(sourceDurationSeconds);

            WriteTensorPayload(writer, _enrollNShape, _enrollNBuffer);
            WriteTensorPayload(writer, _kShape, _kBuffer);
            WriteTensorPayload(writer, _vShape, _vBuffer);
            writer.Flush();
        }

        File.Move(tempPath, cachePath, overwrite: true);
    }

    private bool TryLoadCacheFile(
        string cachePath,
        string? expectedWavSha256,
        string sourceWavPath,
        out VoiceprintProfileInfo? profile)
    {
        profile = null;
        try
        {
            using var fs = new FileStream(cachePath, FileMode.Open, FileAccess.Read, FileShare.Read);
            using var reader = new BinaryReader(fs, Encoding.UTF8, leaveOpen: false);

            byte[] magic = reader.ReadBytes(CacheMagic.Length);
            if (magic.Length != CacheMagic.Length ||
                !Encoding.ASCII.GetString(magic).Equals(CacheMagic, StringComparison.Ordinal))
            {
                return false;
            }

            uint version = reader.ReadUInt32();
            if (version != CacheFormatVersion)
            {
                return false;
            }

            string enrollSha = reader.ReadString();
            string chunkSha = reader.ReadString();
            string combinedSha = reader.ReadString();
            string wavSha = reader.ReadString();
            long enrolledTicks = reader.ReadInt64();
            double durationSeconds = reader.ReadDouble();

            if (!string.Equals(enrollSha, _enrollEntry!.Sha256, StringComparison.OrdinalIgnoreCase) ||
                !string.Equals(chunkSha, _chunkEntry!.Sha256, StringComparison.OrdinalIgnoreCase) ||
                !string.Equals(combinedSha, _combinedModelSha256, StringComparison.OrdinalIgnoreCase))
            {
                return false;
            }

            if (!string.IsNullOrEmpty(expectedWavSha256) &&
                !string.Equals(wavSha, expectedWavSha256, StringComparison.OrdinalIgnoreCase))
            {
                return false;
            }

            if (!TryReadTensorPayload(reader, _enrollNShape, _enrollNBuffer) ||
                !TryReadTensorPayload(reader, _kShape, _kBuffer) ||
                !TryReadTensorPayload(reader, _vShape, _vBuffer))
            {
                return false;
            }

            profile = new VoiceprintProfileInfo(
                SourceWavPath: sourceWavPath,
                CacheFilePath: cachePath,
                ContentSha256: wavSha,
                EnrolledAtUtc: new DateTimeOffset(enrolledTicks, TimeSpan.Zero),
                SourceDurationSeconds: durationSeconds,
                LoadedFromCache: true);

            return true;
        }
        catch
        {
            return false;
        }
    }

    private static void WriteTensorPayload(BinaryWriter writer, long[] shape, float[] data)
    {
        writer.Write(shape.Length);
        for (int i = 0; i < shape.Length; i++)
        {
            writer.Write(shape[i]);
        }

        writer.Write(data.Length);
        writer.Write(MemoryMarshal.AsBytes(data.AsSpan()));
    }

    private static bool TryReadTensorPayload(BinaryReader reader, long[] expectedShape, float[] destination)
    {
        int rank = reader.ReadInt32();
        if (rank != expectedShape.Length)
        {
            return false;
        }

        for (int i = 0; i < rank; i++)
        {
            if (reader.ReadInt64() != expectedShape[i])
            {
                return false;
            }
        }

        int elementCount = reader.ReadInt32();
        if (elementCount != destination.Length)
        {
            return false;
        }

        Span<byte> destBytes = MemoryMarshal.AsBytes(destination.AsSpan());
        int totalRead = 0;
        while (totalRead < destBytes.Length)
        {
            int n = reader.Read(destBytes.Slice(totalRead));
            if (n <= 0)
            {
                return false;
            }
            totalRead += n;
        }

        return true;
    }

    private static TensorDescriptor FindDescriptorMatchingShape(
        IReadOnlyList<TensorDescriptor> descriptors,
        long[] targetShape,
        TensorDescriptor? exclude = null)
    {
        for (int i = 0; i < descriptors.Count; i++)
        {
            TensorDescriptor d = descriptors[i];
            if (ReferenceEquals(d, exclude) || d.Shape.Count != targetShape.Length)
            {
                continue;
            }

            bool match = true;
            for (int axis = 0; axis < targetShape.Length; axis++)
            {
                if (!d.Shape[axis].IsStatic || d.Shape[axis].StaticValue != targetShape[axis])
                {
                    match = false;
                    break;
                }
            }

            if (match)
            {
                return d;
            }
        }

        throw new InvalidDataException(
            $"No tensor descriptor matching shape [{string.Join(", ", targetShape)}] found in model_manifest.json.");
    }

    private void ReleaseResources()
    {
        Volatile.Write(ref _isInitialized, false);
        Volatile.Write(ref _hasEnrolledVoiceprint, false);
        Volatile.Write(ref _currentVoiceprint, null);

        _ortWav?.Dispose();
        _ortEnrollN?.Dispose();
        _ortK?.Dispose();
        _ortV?.Dispose();
        _ortHA?.Dispose();
        _ortHB?.Dispose();
        _ortCA?.Dispose();
        _ortCB?.Dispose();
        _ortFrames?.Dispose();

        _ortWav = null;
        _ortEnrollN = null;
        _ortK = null;
        _ortV = null;
        _ortHA = null;
        _ortHB = null;
        _ortCA = null;
        _ortCB = null;
        _ortFrames = null;

        _runOptions?.Dispose();
        _runOptions = null;

        _chunkSessionResult?.Dispose();
        _chunkSessionResult = null;

        _inputNames = [];
        _inputValues = [];
        _outputNames = [];
        _outputValues = [];
    }
}
