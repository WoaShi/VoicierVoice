using System.Diagnostics;
using System.IO;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NAudio.Wave;
using VoiceLockStudio.Models;
using VoiceLockStudio.Services.Audio;
using VoiceLockStudio.Services.System;

namespace VoiceLockStudio.Services.Backends;

/// <summary>
/// Target-speaker voice isolation backend powered by VoxLock v2 ONNX models.
/// Encapsulates all VoxLock model selection, enrollment, binary voiceprint caching,
/// STFT chunking (<c>K = 2</c> / 16 ms or <c>K = 4</c> / 32 ms),
/// squared-Hann OLA synthesis strictly from <see cref="ModelManifest"/>, and optional CUDA IO binding.
/// </summary>
public sealed class VoxLockOnnxBackend : IVoiceIsolatorBackend
{
    private const int DefaultHopsPerChunk = 4; // K = 4 -> 512 samples (32 ms @ 16 kHz); K = 2 -> 256 samples (16 ms)
    private const int MaxEnrollSearchSeconds = 30;
    private const string CacheMagic = "VXLCACH2";
    private const uint CacheFormatVersion = 1;

    private readonly object _lifecycleLock = new();
    private readonly object _inferenceLock = new(); // Always acquire _lifecycleLock before _inferenceLock.

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
    private int _overlapSamples = 384;    // nFft - hop
    private int _hopsPerChunk = DefaultHopsPerChunk;
    private int _reflectPadSamples = 256;
    private int _chunkInputSamples = 896; // S = nFft + (K - 1) * hop
    private int _doneSamples = 512;       // K * hop
    private float _outputGain = -0.5f;
    private float _olaSumNorm = 1.5f;
    private double _algorithmicLatencyMs = 24.0;

    // Precomputed OLA window (periodic Hann squared) and streaming buffers
    private float[] _win2 = [];
    private float[] _inBuf = [];
    private int _inLen;
    private float[] _olaBuffer = [];

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

    private long[] _wavShape = [];
    private long[] _framesShape = [];
    private long[] _enrollNShape = [];
    private long[] _kShape = [];
    private long[] _vShape = [];
    private long[] _hShape = [];
    private long[] _cShape = [];

    private OrtValue? _ortWav;
    private OrtValue? _ortEnrollN;
    private OrtValue? _ortK;
    private OrtValue? _ortV;
    private OrtValue? _ortHA;
    private OrtValue? _ortHB;
    private OrtValue? _ortCA;
    private OrtValue? _ortCB;
    private OrtValue? _ortFrames;

    private string _wavInputName = string.Empty;
    private string _enrollNInputName = string.Empty;
    private string _kInputName = string.Empty;
    private string _vInputName = string.Empty;
    private string _hInputName = string.Empty;
    private string _cInputName = string.Empty;
    private string _framesOutputName = string.Empty;
    private string _hOutputName = string.Empty;
    private string _cOutputName = string.Empty;

    private OrtMemoryInfo? _cudaMemInfo;
    private OrtAllocator? _cudaAllocator;
    private OrtValue? _cudaHA;
    private OrtValue? _cudaHB;
    private OrtValue? _cudaCA;
    private OrtValue? _cudaCB;
    private OrtMemoryAllocation? _cudaEnrollNAlloc;
    private OrtMemoryAllocation? _cudaKAlloc;
    private OrtMemoryAllocation? _cudaVAlloc;
    private OrtValue? _cudaEnrollN;
    private OrtValue? _cudaK;
    private OrtValue? _cudaV;
    private bool _hasDeviceVoiceprintBuffers;

    private OrtMemoryInfo? _cudaPinnedMemInfo;
    private OrtAllocator? _cudaPinnedAllocator;
    private OrtValue? _cudaPinnedWav;
    private OrtValue? _cudaPinnedFrames;

    private static IntPtr _cudaMemcpyFnPtr;
    private static int _cudaMemcpyLookupState; // 0 = unresolved, 1 = found, -1 = unavailable

    private OrtIoBinding? _ioBindingReset;
    private OrtIoBinding? _ioBindingA;
    private OrtIoBinding? _ioBindingB;
    private bool _useCudaIoBinding;
    private bool _needsCudaStateInit = true;

    private string[] _inputNames = [];
    private OrtValue[] _inputValues = [];
    private string[] _outputNames = [];
    private OrtValue[] _outputValues = [];
    private int _hInputSlot;
    private int _cInputSlot;
    private int _hOutputSlot;
    private int _cOutputSlot;
    private bool _stateIsA = true;

    private List<int> _ortWorkerThreadIds = [];
    private ulong _lastBoundCoreMask;
    private ushort _lastBoundProcessorGroup;

    private AudioStageOptions? _lastOptions;
    private string _lastActiveProviderName = string.Empty;
    private string? _lastProviderFallbackMessage;
    private VoiceprintProfileInfo? _currentVoiceprint;
    private bool _hasEnrolledVoiceprint;
    private bool _isSessionLoaded;
    private bool _isInitialized;
    private bool _disposed;

    public string Name => $"VoxLock v2 ({_variant})";
    public int FrameSizeSamples => _doneSamples;
    public int SampleRate => _sampleRate;
    public double AlgorithmicLatencyMs => _algorithmicLatencyMs;
    public bool IsInitialized => Volatile.Read(ref _isInitialized);
    public bool IsSessionLoaded => Volatile.Read(ref _isSessionLoaded);
    public bool HasEnrolledVoiceprint => Volatile.Read(ref _hasEnrolledVoiceprint);
    public VoiceprintProfileInfo? CurrentVoiceprint => Volatile.Read(ref _currentVoiceprint);

    public string ActiveProviderName => _chunkSessionResult?.ActiveProviderName ?? _lastActiveProviderName;
    public string? ProviderFallbackMessage => _chunkSessionResult?.FallbackMessage ?? _lastProviderFallbackMessage;

    public void Initialize(AudioStageOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        ObjectDisposedException.ThrowIf(_disposed, this);

        lock (_lifecycleLock)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            ReleaseResources();
            _lastOptions = options;

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

            _hopsPerChunk = options.VoxLockHopsPerChunk == 2 ? 2 : DefaultHopsPerChunk;
            _overlapSamples = Math.Max(0, _nFft - _hopLength);
            _doneSamples = _hopsPerChunk * _hopLength;
            _chunkInputSamples = _nFft + (_hopsPerChunk - 1) * _hopLength;

            BuildPeriodicHannSquaredWindow(_nFft, audio.WindowType);
            LoadChunkSessionLocked(options);

            Volatile.Write(ref _hasEnrolledVoiceprint, false);
            Volatile.Write(ref _currentVoiceprint, null);
            Volatile.Write(ref _isInitialized, true);

            // Auto-load or rebuild cached voiceprint if the default raw voice WAV already exists.
            if (File.Exists(VoiceprintStorage.DefaultUserRawVoiceWavPath))
            {
                LoadOrRebuildVoiceprintCache(VoiceprintStorage.DefaultUserRawVoiceWavPath, forceRebuild: false);
            }
        }
    }

    /// <summary>
    /// Releases the heavy ONNX Runtime chunk session, CUDA VRAM, and pinned staging arrays
    /// while retaining the enrolled profile metadata so the backend can resume cleanly from disk cache.
    /// </summary>
    public void UnloadSessionKeepProfile()
    {
        if (_disposed || !Volatile.Read(ref _isInitialized))
        {
            return;
        }

        lock (_lifecycleLock)
        {
            if (_disposed || !Volatile.Read(ref _isInitialized) || !Volatile.Read(ref _isSessionLoaded))
            {
                return;
            }

            lock (_inferenceLock)
            {
                bool hasDiskCache = _currentVoiceprint is not null && File.Exists(_currentVoiceprint.CacheFilePath);
                ReleaseSessionResourcesLocked(clearProfile: false, releaseVoiceprintArrays: hasDiskCache);
            }
        }
    }

    /// <summary>
    /// Reloads the ONNX Runtime chunk session and restores the enrolled voiceprint tensors if
    /// the session was previously unloaded while idle.
    /// </summary>
    public void EnsureSessionLoaded()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (Volatile.Read(ref _isSessionLoaded) && _chunkSessionResult is not null)
        {
            return;
        }

        lock (_lifecycleLock)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            EnsureSessionLoadedLocked();
        }
    }

    private void EnsureSessionLoadedLocked()
    {
        if (Volatile.Read(ref _isSessionLoaded) && _chunkSessionResult is not null)
        {
            return;
        }

        EnsureInitializedCore();
        AudioStageOptions options = _lastOptions ?? throw new InvalidOperationException($"{Name} has no cached initialization options.");

        bool hadVoiceprint = Volatile.Read(ref _hasEnrolledVoiceprint);
        VoiceprintProfileInfo? prevProfile = Volatile.Read(ref _currentVoiceprint);
        float[]? preservedN = hadVoiceprint && _enrollNBuffer.Length > 0 ? _enrollNBuffer : null;
        float[]? preservedK = hadVoiceprint && _kBuffer.Length > 0 ? _kBuffer : null;
        float[]? preservedV = hadVoiceprint && _vBuffer.Length > 0 ? _vBuffer : null;

        LoadChunkSessionLocked(options);

        if (!hadVoiceprint)
        {
            return;
        }

        if (prevProfile is not null &&
            File.Exists(prevProfile.CacheFilePath) &&
            TryLoadCacheFile(prevProfile.CacheFilePath, expectedWavSha256: null, prevProfile.SourceWavPath, out VoiceprintProfileInfo? cached))
        {
            Volatile.Write(ref _currentVoiceprint, cached);
            Volatile.Write(ref _hasEnrolledVoiceprint, true);
            return;
        }

        if (preservedN is not null && preservedN.Length == _enrollNBuffer.Length &&
            preservedK is not null && preservedK.Length == _kBuffer.Length &&
            preservedV is not null && preservedV.Length == _vBuffer.Length)
        {
            lock (_inferenceLock)
            {
                preservedN.AsSpan().CopyTo(_enrollNBuffer);
                preservedK.AsSpan().CopyTo(_kBuffer);
                preservedV.AsSpan().CopyTo(_vBuffer);
                RefreshCudaIoBindings();
                Volatile.Write(ref _currentVoiceprint, prevProfile);
                Volatile.Write(ref _hasEnrolledVoiceprint, true);
            }
            return;
        }

        if (File.Exists(VoiceprintStorage.DefaultUserRawVoiceWavPath))
        {
            LoadOrRebuildVoiceprintCache(VoiceprintStorage.DefaultUserRawVoiceWavPath, forceRebuild: false);
        }
        else
        {
            Volatile.Write(ref _hasEnrolledVoiceprint, false);
            Volatile.Write(ref _currentVoiceprint, null);
        }
    }

    private void LoadChunkSessionLocked(AudioStageOptions options)
    {
        int intraOpThreads = ResolveIntraOpThreads(options);
        HashSet<int>? baselineThreads = (options.Backend == ComputeBackend.Cpu && intraOpThreads > 1)
            ? CpuAffinityManager.CaptureCurrentOsThreadIds()
            : null;

        bool allowIntraOpSpinning = options.Backend == ComputeBackend.Cpu && intraOpThreads > 1;
        _chunkSessionResult = OnnxSessionFactory.CreateSession(
            _manifest!,
            _chunkEntry!.Filename,
            _modelsRootDirectory,
            _requestedBackend,
            intraOpThreads,
            allowSpinning: allowIntraOpSpinning);

        _lastActiveProviderName = _chunkSessionResult.ActiveProviderName;
        _lastProviderFallbackMessage = _chunkSessionResult.FallbackMessage;

        _inBuf = GC.AllocateArray<float>(_chunkInputSamples + _doneSamples, pinned: true);
        _olaBuffer = GC.AllocateArray<float>(_nFft + _doneSamples, pinned: true);

        AllocateChunkTensors(_chunkEntry, _enrollEntry!);

        _runOptions = new RunOptions();
        _inLen = _overlapSamples;
        _stateIsA = true;
        _needsCudaStateInit = true;
        _useCudaIoBinding = _chunkSessionResult.ActiveBackend == ComputeBackend.Cuda && TryInitializeCudaIoBindings();
        Volatile.Write(ref _isSessionLoaded, true);

        // Warm up chunk session immediately so lazy ORT arenas/threads are initialized before audio capture starts.
        WarmupCore();

        if (baselineThreads is not null)
        {
            _ortWorkerThreadIds = CpuAffinityManager.GetNewProcessThreadIds(baselineThreads, intraOpThreads);
            if (_ortWorkerThreadIds.Count > 0)
            {
                new CpuAffinityManager().BindSpecificThreads(
                    _ortWorkerThreadIds,
                    options.CpuAffinityMask,
                    options.ProcessorGroup,
                    ThreadPriority.Highest);
                if (options.CpuAffinityMask != 0)
                {
                    _lastBoundCoreMask = options.CpuAffinityMask;
                    _lastBoundProcessorGroup = options.ProcessorGroup;
                }
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
            EnsureSessionLoadedLocked();
            return EnrollVoiceprintCore(wavFilePath);
        }
    }

    public VoiceprintProfileInfo? LoadOrRebuildVoiceprintCache(string? wavFilePath = null, bool forceRebuild = false)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        lock (_lifecycleLock)
        {
            EnsureSessionLoadedLocked();

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
        if (!Volatile.Read(ref _isInitialized))
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

        if (!Volatile.Read(ref _isSessionLoaded) || _chunkSessionResult is null || _runOptions is null)
        {
            EnsureSessionLoaded();
        }

        lock (_inferenceLock)
        {
            if (_disposed || !Volatile.Read(ref _isInitialized) || _chunkSessionResult is null || _runOptions is null)
            {
                output.Clear();
                return;
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

            bool emittedOutput = false;
            while (_inLen >= _chunkInputSamples)
            {
                Span<float> wavDst = (_useCudaIoBinding && _cudaPinnedWav is not null)
                    ? _cudaPinnedWav.GetTensorMutableDataAsSpan<float>()
                    : _wavBuffer.AsSpan();
                _inBuf.AsSpan(0, _chunkInputSamples).CopyTo(wavDst);

                int remainingIn = _inLen - _doneSamples;
                if (remainingIn > 0)
                {
                    _inBuf.AsSpan(_doneSamples, remainingIn).CopyTo(_inBuf.AsSpan(0, remainingIn));
                }
                _inLen = remainingIn;

                RunChunkInference();

                // Overlap-add K frames of length nFft (512) at stride hopLength (128).
                ReadOnlySpan<float> frames = (_useCudaIoBinding && _cudaPinnedFrames is not null)
                    ? _cudaPinnedFrames.GetTensorDataAsSpan<float>()
                    : _framesBuffer.AsSpan();
                float[] ola = _olaBuffer;
                int nFft = _nFft;
                int hop = _hopLength;
                int hopsPerChunk = _hopsPerChunk;

                for (int f = 0; f < hopsPerChunk; f++)
                {
                    WasapiPcmConverter.AddInPlace(
                        frames.Slice(f * nFft, nFft),
                        ola.AsSpan(f * hop, nFft));
                }

                int done = _doneSamples;
                float scale = _outputGain / _olaSumNorm;
                WasapiPcmConverter.MultiplyScalar(ola.AsSpan(0, done), scale, output);
                emittedOutput = true;

                int olaTail = ola.Length - done;
                ola.AsSpan(done, olaTail).CopyTo(ola.AsSpan(0, olaTail));
                ola.AsSpan(olaTail, done).Clear();
            }

            if (!emittedOutput)
            {
                output.Clear();
            }
        }
    }

    public void ResetState()
    {
        lock (_inferenceLock)
        {
            if (_disposed || !Volatile.Read(ref _isInitialized) || !Volatile.Read(ref _isSessionLoaded))
            {
                return;
            }

            Array.Clear(_inBuf);
            Array.Clear(_olaBuffer);
            Array.Clear(_wavBuffer);
            Array.Clear(_framesBuffer);
            _cudaPinnedWav?.GetTensorMutableDataAsSpan<float>().Clear();
            _cudaPinnedFrames?.GetTensorMutableDataAsSpan<float>().Clear();
            Array.Clear(_hBufferA);
            Array.Clear(_hBufferB);
            Array.Clear(_cBufferA);
            Array.Clear(_cBufferB);

            // Pre-seed overlap tail (384 zeros) so the very first chunk has S = 384 + K*128 samples with zero cold stall.
            _inLen = _overlapSamples;
            _stateIsA = true;
            _needsCudaStateInit = true;

            if (_ortHA is not null && _ortHB is not null && _ortCA is not null && _ortCB is not null)
            {
                _inputValues[_hInputSlot] = _ortHA;
                _inputValues[_cInputSlot] = _ortCA;
                _outputValues[_hOutputSlot] = _ortHB;
                _outputValues[_cOutputSlot] = _ortCB;
            }
        }
    }

    public void SoftResetStreamingBuffers()
    {
        lock (_inferenceLock)
        {
            if (_disposed || !Volatile.Read(ref _isInitialized) || !Volatile.Read(ref _isSessionLoaded))
            {
                return;
            }

            Array.Clear(_inBuf);
            Array.Clear(_olaBuffer);
            Array.Clear(_wavBuffer);
            Array.Clear(_framesBuffer);
            _cudaPinnedWav?.GetTensorMutableDataAsSpan<float>().Clear();
            _cudaPinnedFrames?.GetTensorMutableDataAsSpan<float>().Clear();
            _inLen = _overlapSamples;
        }
    }

    public void UpdateCpuAffinity(ulong coreMask, ushort processorGroup = 0)
    {
        if (coreMask == 0)
        {
            return;
        }

        lock (_lifecycleLock)
        {
            if (_disposed ||
                !Volatile.Read(ref _isInitialized) ||
                !Volatile.Read(ref _isSessionLoaded) ||
                _requestedBackend != ComputeBackend.Cpu ||
                _ortWorkerThreadIds.Count == 0 ||
                (_lastBoundCoreMask == coreMask && _lastBoundProcessorGroup == processorGroup))
            {
                return;
            }

            new CpuAffinityManager().BindSpecificThreads(
                _ortWorkerThreadIds,
                coreMask,
                processorGroup,
                ThreadPriority.Highest);
            _lastBoundCoreMask = coreMask;
            _lastBoundProcessorGroup = processorGroup;
        }
    }

    public void Warmup()
    {
        if (!Volatile.Read(ref _isInitialized))
        {
            return;
        }

        lock (_lifecycleLock)
        {
            if (_disposed || !Volatile.Read(ref _isInitialized))
            {
                return;
            }

            if (!Volatile.Read(ref _isSessionLoaded) || _chunkSessionResult is null || _runOptions is null)
            {
                EnsureSessionLoadedLocked();
            }

            WarmupCore();
        }
    }

    private void WarmupCore()
    {
        lock (_inferenceLock)
        {
            if (_disposed || _chunkSessionResult is null || _runOptions is null)
            {
                return;
            }

            if (_cudaPinnedWav is not null)
            {
                _cudaPinnedWav.GetTensorMutableDataAsSpan<float>().Clear();
            }
            else
            {
                Array.Clear(_wavBuffer);
            }

            for (int i = 0; i < 2; i++)
            {
                RunChunkInference();
            }
            ResetState();
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void RunChunkInference()
    {
        if (_useCudaIoBinding && _runOptions is not null && _chunkSessionResult is not null)
        {
            OrtIoBinding? binding;
            if (_needsCudaStateInit)
            {
                binding = _ioBindingReset;
                _needsCudaStateInit = false;
                _stateIsA = false;
            }
            else if (_stateIsA)
            {
                binding = _ioBindingA;
                _stateIsA = false;
            }
            else
            {
                binding = _ioBindingB;
                _stateIsA = true;
            }

            if (binding is not null)
            {
                // Rebind wav before each run so ORT refreshes H2D input on the active buffer.
                binding.BindInput(_wavInputName, _cudaPinnedWav ?? _ortWav!);
                _chunkSessionResult.Session.RunWithBinding(_runOptions, binding);
                binding.SynchronizeBoundOutputs();
                return;
            }
        }

        _chunkSessionResult!.Session.Run(_runOptions!, _inputNames, _inputValues, _outputNames, _outputValues);

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
    }

    private static int ResolveIntraOpThreads(AudioStageOptions options)
    {
        if (options.OrtIntraOpThreads > 1)
        {
            return Math.Clamp(options.OrtIntraOpThreads, 1, 8);
        }

        if (options.Backend == ComputeBackend.Cuda)
        {
            return 1;
        }

        if (options.Backend == ComputeBackend.Cpu && options.CpuAffinityMask != 0)
        {
            // A 27.3M BSRNN at 32 ms chunk runs in ~20 ms on 3 E-Cores, whereas 16 ms low-latency mode benefits from up to 4 P-Core threads.
            int coreCount = BitOperations.PopCount(options.CpuAffinityMask);
            int maxThreads = options.VoxLockHopsPerChunk <= 2 ? 4 : 3;
            return Math.Clamp(coreCount, 1, maxThreads);
        }

        return options.VoxLockHopsPerChunk <= 2 ? 4 : 3;
    }

    public void Dispose()
    {
        lock (_lifecycleLock)
        {
            if (_disposed)
            {
                return;
            }

            lock (_inferenceLock)
            {
                _disposed = true;
                ReleaseResources();
            }
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

        (float[] enrollNData, float[] kData, float[] vData) = RunEnrollmentModel(manifest, enrollEntry, enrollWindow);

        DateTimeOffset enrolledAtUtc = DateTimeOffset.UtcNow;
        string cachePath = VoiceprintStorage.GetCacheFilePath(_combinedModelSha256);
        WriteCacheFile(
            cachePath,
            enrollEntry.Sha256,
            chunkEntry.Sha256,
            _combinedModelSha256,
            wavSha256,
            enrolledAtUtc,
            sourceDurationSeconds,
            enrollNData,
            kData,
            vData);

        var profile = new VoiceprintProfileInfo(
            SourceWavPath: effectiveWavPath,
            CacheFilePath: cachePath,
            ContentSha256: wavSha256,
            EnrolledAtUtc: enrolledAtUtc,
            SourceDurationSeconds: sourceDurationSeconds,
            LoadedFromCache: false);

        lock (_inferenceLock)
        {
            enrollNData.AsSpan().CopyTo(_enrollNBuffer);
            kData.AsSpan().CopyTo(_kBuffer);
            vData.AsSpan().CopyTo(_vBuffer);
            RefreshCudaIoBindings();
            Volatile.Write(ref _currentVoiceprint, profile);
            Volatile.Write(ref _hasEnrolledVoiceprint, true);
        }

        return profile;
    }

    private (float[] EnrollN, float[] K, float[] V) RunEnrollmentModel(ModelManifest manifest, ModelEntry enrollEntry, float[] enrollWindow)
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

        float[] tempEnrollN = new float[_enrollNBuffer.Length];
        float[] tempK = new float[_kBuffer.Length];
        float[] tempV = new float[_vBuffer.Length];

        using var runOptions = new RunOptions();
        using OrtValue ovEnrollIn = OrtValue.CreateTensorValueFromMemory(enrollWindow, enrollInShape);
        using OrtValue ovEnrollNOut = OrtValue.CreateTensorValueFromMemory(tempEnrollN, _enrollNShape);
        using OrtValue ovKOut = OrtValue.CreateTensorValueFromMemory(tempK, _kShape);
        using OrtValue ovVOut = OrtValue.CreateTensorValueFromMemory(tempV, _vShape);

        enrollSession.Session.Run(
            runOptions,
            [enrollInDesc.Name],
            [ovEnrollIn],
            [enrollNOutDesc.Name, kOutDesc.Name, vOutDesc.Name],
            [ovEnrollNOut, ovKOut, ovVOut]);

        return (tempEnrollN, tempK, tempV);
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
        _wavShape = wavShape;
        _wavBuffer = GC.AllocateArray<float>(OnnxSessionFactory.ComputeElementCount(wavShape), pinned: true);
        _ortWav = OrtValue.CreateTensorValueFromMemory(_wavBuffer, wavShape);

        // Identify chunk outputs: frames (rank 3 with dynamic K) and h_out, c_out (rank 3 static).
        TensorDescriptor framesOutDesc = outputs.First(d => d.Shape.Count == 3 && d.Shape.Any(dim => dim.IsDynamic));
        var framesOverrides = new Dictionary<string, long>(StringComparer.Ordinal);
        foreach (TensorDimension dim in framesOutDesc.Shape)
        {
            if (!string.IsNullOrEmpty(dim.SymbolicName))
            {
                framesOverrides[dim.SymbolicName] = _hopsPerChunk;
            }
        }
        long[] framesShape = OnnxSessionFactory.ResolveTensorShape(framesOutDesc, framesOverrides, _hopsPerChunk);
        _framesShape = framesShape;
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
        _hShape = hShape;
        _cShape = cShape;
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

        _wavInputName = wavInDesc.Name;
        _enrollNInputName = chunkEnrollNDesc.Name;
        _kInputName = chunkKDesc.Name;
        _vInputName = chunkVDesc.Name;
        _hInputName = hInDesc.Name;
        _cInputName = cInDesc.Name;
        _framesOutputName = framesOutDesc.Name;
        _hOutputName = hOutDesc.Name;
        _cOutputName = cOutDesc.Name;

        _inputNames =
        [
            _wavInputName,
            _enrollNInputName,
            _kInputName,
            _vInputName,
            _hInputName,
            _cInputName
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

        _outputNames = [_framesOutputName, _hOutputName, _cOutputName];
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

        double overlapSum = 0.0;
        for (int i = 0; i < nFft; i += _hopLength)
        {
            overlapSum += _win2[i];
        }
        _olaSumNorm = overlapSum > 1e-6 ? (float)overlapSum : 1.5f;
    }

    private bool TryInitializeCudaIoBindings()
    {
        if (_chunkSessionResult is null ||
            _ortWav is null || _ortEnrollN is null || _ortK is null || _ortV is null ||
            _ortHA is null || _ortCA is null || _ortFrames is null ||
            _hShape.Length == 0 || _cShape.Length == 0)
        {
            return false;
        }

        try
        {
            InferenceSession session = _chunkSessionResult.Session;
            _cudaMemInfo = new OrtMemoryInfo("Cuda", OrtAllocatorType.DeviceAllocator, 0, OrtMemType.Default);
            _cudaAllocator = new OrtAllocator(session, _cudaMemInfo);

            _cudaHA = OrtValue.CreateAllocatedTensorValue(_cudaAllocator, TensorElementType.Float, _hShape);
            _cudaHB = OrtValue.CreateAllocatedTensorValue(_cudaAllocator, TensorElementType.Float, _hShape);
            _cudaCA = OrtValue.CreateAllocatedTensorValue(_cudaAllocator, TensorElementType.Float, _cShape);
            _cudaCB = OrtValue.CreateAllocatedTensorValue(_cudaAllocator, TensorElementType.Float, _cShape);

            TryAllocateDeviceVoiceprintTensors();
            TryInitializeCudaPinnedDynamicBuffers(session);

            _ioBindingReset = session.CreateIoBinding();
            _ioBindingA = session.CreateIoBinding();
            _ioBindingB = session.CreateIoBinding();

            RefreshCudaIoBindings();
            return true;
        }
        catch
        {
            DisposeCudaIoBindings();
            return false;
        }
    }

    private void TryAllocateDeviceVoiceprintTensors()
    {
        if (_cudaAllocator is null || _cudaMemInfo is null ||
            _enrollNBuffer.Length == 0 || _kBuffer.Length == 0 || _vBuffer.Length == 0)
        {
            return;
        }

        try
        {
            long enrollBytes = checked((long)_enrollNBuffer.Length * sizeof(float));
            long kBytes = checked((long)_kBuffer.Length * sizeof(float));
            long vBytes = checked((long)_vBuffer.Length * sizeof(float));

            _cudaEnrollNAlloc = _cudaAllocator.Allocate(checked((uint)enrollBytes));
            _cudaKAlloc = _cudaAllocator.Allocate(checked((uint)kBytes));
            _cudaVAlloc = _cudaAllocator.Allocate(checked((uint)vBytes));

            _cudaEnrollN = OrtValue.CreateTensorValueWithData(
                _cudaMemInfo, TensorElementType.Float, _enrollNShape, _cudaEnrollNAlloc.DangerousGetHandle(), enrollBytes);
            _cudaK = OrtValue.CreateTensorValueWithData(
                _cudaMemInfo, TensorElementType.Float, _kShape, _cudaKAlloc.DangerousGetHandle(), kBytes);
            _cudaV = OrtValue.CreateTensorValueWithData(
                _cudaMemInfo, TensorElementType.Float, _vShape, _cudaVAlloc.DangerousGetHandle(), vBytes);
        }
        catch
        {
            DisposeDeviceVoiceprintTensors();
        }
    }

    private void TryInitializeCudaPinnedDynamicBuffers(InferenceSession session)
    {
        if (_wavShape.Length == 0 || _framesShape.Length == 0)
        {
            return;
        }

        (OrtAllocatorType AllocType, OrtMemType MemType)[] candidates =
        [
            (OrtAllocatorType.ArenaAllocator, OrtMemType.CpuOutput),
            (OrtAllocatorType.DeviceAllocator, OrtMemType.CpuOutput),
            (OrtAllocatorType.DeviceAllocator, OrtMemType.Default)
        ];

        for (int i = 0; i < candidates.Length; i++)
        {
            OrtMemoryInfo? memInfo = null;
            OrtAllocator? allocator = null;
            OrtValue? wav = null;
            OrtValue? frames = null;
            try
            {
                memInfo = new OrtMemoryInfo("CudaPinned", candidates[i].AllocType, 0, candidates[i].MemType);
                allocator = new OrtAllocator(session, memInfo);
                wav = OrtValue.CreateAllocatedTensorValue(allocator, TensorElementType.Float, _wavShape);
                frames = OrtValue.CreateAllocatedTensorValue(allocator, TensorElementType.Float, _framesShape);

                // Validate host span access before exposing on the audio hot path.
                wav.GetTensorMutableDataAsSpan<float>().Clear();
                frames.GetTensorMutableDataAsSpan<float>().Clear();
                _ = frames.GetTensorDataAsSpan<float>();

                _cudaPinnedMemInfo = memInfo;
                _cudaPinnedAllocator = allocator;
                _cudaPinnedWav = wav;
                _cudaPinnedFrames = frames;
                return;
            }
            catch
            {
                wav?.Dispose();
                frames?.Dispose();
                allocator?.Dispose();
                memInfo?.Dispose();
            }
        }
    }

    private static IntPtr ResolveCudaMemcpyFunction()
    {
        int state = Volatile.Read(ref _cudaMemcpyLookupState);
        if (state == 1)
        {
            return Volatile.Read(ref _cudaMemcpyFnPtr);
        }
        if (state == -1)
        {
            return IntPtr.Zero;
        }

        try
        {
            using Process proc = Process.GetCurrentProcess();
            foreach (ProcessModule module in proc.Modules)
            {
                string name = module.ModuleName ?? string.Empty;
                if (name.StartsWith("cudart64_", StringComparison.OrdinalIgnoreCase) &&
                    NativeLibrary.TryGetExport(module.BaseAddress, "cudaMemcpy", out IntPtr fnPtr))
                {
                    Volatile.Write(ref _cudaMemcpyFnPtr, fnPtr);
                    Volatile.Write(ref _cudaMemcpyLookupState, 1);
                    return fnPtr;
                }
            }
        }
        catch
        {
            // Ignore module enumeration failures; fallback safely to ORT host binding.
        }

        Volatile.Write(ref _cudaMemcpyLookupState, -1);
        return IntPtr.Zero;
    }

    private static unsafe bool TryCudaMemcpyHostToDevice(ReadOnlySpan<float> hostSrc, OrtMemoryAllocation? cudaAlloc)
    {
        if (cudaAlloc is null || cudaAlloc.IsInvalid || cudaAlloc.IsClosed || hostSrc.IsEmpty)
        {
            return false;
        }

        IntPtr fnPtr = ResolveCudaMemcpyFunction();
        if (fnPtr == IntPtr.Zero)
        {
            return false;
        }

        nuint byteCount = checked((nuint)hostSrc.Length * sizeof(float));
        if (cudaAlloc.Size < byteCount)
        {
            return false;
        }

        try
        {
            var cudaMemcpy = (delegate* unmanaged[Cdecl]<IntPtr, IntPtr, nuint, int, int>)fnPtr;
            fixed (float* pSrc = &MemoryMarshal.GetReference(hostSrc))
            {
                return cudaMemcpy(cudaAlloc.DangerousGetHandle(), (IntPtr)pSrc, byteCount, 1) == 0;
            }
        }
        catch
        {
            return false;
        }
    }

    private void RefreshCudaIoBindings()
    {
        if (_ioBindingReset is null || _ioBindingA is null || _ioBindingB is null ||
            _cudaHA is null || _cudaHB is null || _cudaCA is null || _cudaCB is null ||
            _ortWav is null || _ortEnrollN is null || _ortK is null || _ortV is null ||
            _ortHA is null || _ortCA is null || _ortFrames is null)
        {
            return;
        }

        _hasDeviceVoiceprintBuffers = _cudaEnrollN is not null && _cudaK is not null && _cudaV is not null
            && TryCudaMemcpyHostToDevice(_enrollNBuffer, _cudaEnrollNAlloc)
            && TryCudaMemcpyHostToDevice(_kBuffer, _cudaKAlloc)
            && TryCudaMemcpyHostToDevice(_vBuffer, _cudaVAlloc);

        ConfigureSingleIoBinding(_ioBindingReset, _ortHA, _ortCA, _cudaHB, _cudaCB);
        ConfigureSingleIoBinding(_ioBindingA, _cudaHA, _cudaCA, _cudaHB, _cudaCB);
        ConfigureSingleIoBinding(_ioBindingB, _cudaHB, _cudaCB, _cudaHA, _cudaCA);
    }

    private void ConfigureSingleIoBinding(
        OrtIoBinding binding,
        OrtValue hIn,
        OrtValue cIn,
        OrtValue hOut,
        OrtValue cOut)
    {
        binding.ClearBoundInputs();
        binding.ClearBoundOutputs();

        binding.BindInput(_wavInputName, _cudaPinnedWav ?? _ortWav!);
        binding.BindInput(_enrollNInputName, _hasDeviceVoiceprintBuffers ? _cudaEnrollN! : _ortEnrollN!);
        binding.BindInput(_kInputName, _hasDeviceVoiceprintBuffers ? _cudaK! : _ortK!);
        binding.BindInput(_vInputName, _hasDeviceVoiceprintBuffers ? _cudaV! : _ortV!);
        binding.BindInput(_hInputName, hIn);
        binding.BindInput(_cInputName, cIn);

        binding.BindOutput(_framesOutputName, _cudaPinnedFrames ?? _ortFrames!);
        binding.BindOutput(_hOutputName, hOut);
        binding.BindOutput(_cOutputName, cOut);
    }

    private void DisposeDeviceVoiceprintTensors()
    {
        _hasDeviceVoiceprintBuffers = false;

        _cudaEnrollN?.Dispose();
        _cudaK?.Dispose();
        _cudaV?.Dispose();
        _cudaEnrollN = null;
        _cudaK = null;
        _cudaV = null;

        _cudaEnrollNAlloc?.Dispose();
        _cudaKAlloc?.Dispose();
        _cudaVAlloc?.Dispose();
        _cudaEnrollNAlloc = null;
        _cudaKAlloc = null;
        _cudaVAlloc = null;
    }

    private void DisposeCudaIoBindings()
    {
        _useCudaIoBinding = false;
        _needsCudaStateInit = true;

        _ioBindingReset?.Dispose();
        _ioBindingA?.Dispose();
        _ioBindingB?.Dispose();
        _ioBindingReset = null;
        _ioBindingA = null;
        _ioBindingB = null;

        _cudaPinnedWav?.Dispose();
        _cudaPinnedFrames?.Dispose();
        _cudaPinnedWav = null;
        _cudaPinnedFrames = null;

        _cudaPinnedAllocator?.Dispose();
        _cudaPinnedAllocator = null;
        _cudaPinnedMemInfo?.Dispose();
        _cudaPinnedMemInfo = null;

        DisposeDeviceVoiceprintTensors();

        _cudaHA?.Dispose();
        _cudaHB?.Dispose();
        _cudaCA?.Dispose();
        _cudaCB?.Dispose();
        _cudaHA = null;
        _cudaHB = null;
        _cudaCA = null;
        _cudaCB = null;

        _cudaAllocator?.Dispose();
        _cudaAllocator = null;

        _cudaMemInfo?.Dispose();
        _cudaMemInfo = null;
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
        double sourceDurationSeconds,
        float[] enrollNData,
        float[] kData,
        float[] vData)
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

            WriteTensorPayload(writer, _enrollNShape, enrollNData);
            WriteTensorPayload(writer, _kShape, kData);
            WriteTensorPayload(writer, _vShape, vData);
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

            float[] tempEnrollN = new float[_enrollNBuffer.Length];
            float[] tempK = new float[_kBuffer.Length];
            float[] tempV = new float[_vBuffer.Length];
            if (!TryReadTensorPayload(reader, _enrollNShape, tempEnrollN) ||
                !TryReadTensorPayload(reader, _kShape, tempK) ||
                !TryReadTensorPayload(reader, _vShape, tempV))
            {
                return false;
            }

            lock (_inferenceLock)
            {
                tempEnrollN.AsSpan().CopyTo(_enrollNBuffer);
                tempK.AsSpan().CopyTo(_kBuffer);
                tempV.AsSpan().CopyTo(_vBuffer);
                RefreshCudaIoBindings();
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
        _lastOptions = null;
        _lastActiveProviderName = string.Empty;
        _lastProviderFallbackMessage = null;
        ReleaseSessionResourcesLocked(clearProfile: true, releaseVoiceprintArrays: true);
    }

    private void ReleaseSessionResourcesLocked(bool clearProfile, bool releaseVoiceprintArrays)
    {
        Volatile.Write(ref _isSessionLoaded, false);
        if (clearProfile)
        {
            Volatile.Write(ref _hasEnrolledVoiceprint, false);
            Volatile.Write(ref _currentVoiceprint, null);
        }

        DisposeCudaIoBindings();

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

        _inBuf = [];
        _olaBuffer = [];
        _wavBuffer = [];
        _framesBuffer = [];
        _hBufferA = [];
        _hBufferB = [];
        _cBufferA = [];
        _cBufferB = [];

        if (releaseVoiceprintArrays)
        {
            _enrollNBuffer = [];
            _kBuffer = [];
            _vBuffer = [];
        }

        _inputNames = [];
        _inputValues = [];
        _outputNames = [];
        _outputValues = [];
        _ortWorkerThreadIds.Clear();
        _lastBoundCoreMask = 0;
        _lastBoundProcessorGroup = 0;
    }
}
