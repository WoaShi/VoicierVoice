using System.Diagnostics;
using System.Reflection;
using System.Runtime.CompilerServices;
using NAudio.CoreAudioApi;
using NAudio.Wave;
using VoiceLockStudio.Models;
using VoiceLockStudio.Services.System;

namespace VoiceLockStudio.Services.Audio;

/// <summary>
/// Real-time WASAPI audio routing and neural DSP pipeline engine running on a dedicated,
/// CPU-affinity-pinned background thread with zero hot-path allocations.
/// </summary>
public sealed class AudioPipelineEngine : IDisposable
{
    public const int DspSampleRate = 16000;
    public const int HopSizeSamples = 256;          // 16 ms @ 16 kHz
    public const int VadFrameSizeSamples = 512;     // 32 ms @ 16 kHz (2 hops)
    private const int EventSyncCaptureBufferMs = 120;
    private const int PollingCaptureBufferMs = 30;
    private const int StandardRenderLatencyMs = 32;
    private const int StandardOutputPrebufferMs = 38;
    private const int StandardOutputMaxBacklogMs = 112;
    private const int LowLatencyRenderLatencyMs = 24;
    private const int LowLatencyOutputPrebufferMs = 26;
    private const int LowLatencyOutputMaxBacklogMs = 96;
    private const int VadHangoverFrames = 10;       // 10 * 32 ms = 320 ms
    private const int SilenceResetThresholdFrames = 5;
    private const double TelemetryIntervalMs = 80.0; // ~12.5 Hz

    private readonly AudioDeviceService _deviceService;
    private readonly CpuAffinityManager _affinityManager;
    private readonly object _lifecycleLock = new();

    // Pluggable DSP stages
    private IAudioFilterStage? _vadStage;
    private Func<float>? _vadProbabilityGetter;
    private IAudioFilterStage? _gtcrnStage;
    private IVoiceIsolatorBackend? _voxLockBackend;

    // Pre-allocated hot-path buffers & resamplers
    private readonly AudioRingBuffer _inputRing16k = new(16384);
    private readonly AudioRingBuffer _primaryOutRing = new(32768);
    private readonly AudioRingBuffer _monitorOutRing = new(32768);

    private readonly StreamingResampler _inputResampler = new(48000, DspSampleRate);
    private readonly StreamingResampler _primaryOutResampler = new(DspSampleRate, 48000);
    private readonly StreamingResampler _monitorOutResampler = new(DspSampleRate, 48000);

    private readonly float[] _captureMonoScratch = new float[16384];
    private readonly float[] _capture16kScratch = new float[16384];
    private readonly float[] _hopInput256 = new float[HopSizeSamples];
    private readonly float[] _gtcrnOut256 = new float[HopSizeSamples];
    private readonly float[] _voxStagingIn = new float[2048];
    private readonly float[] _voxOutScratch = new float[2048];
    private readonly float[] _hopOutput256 = new float[HopSizeSamples];
    private readonly float[] _vadAccum512 = new float[VadFrameSizeSamples];
    private readonly float[] _vadScratchOut512 = new float[VadFrameSizeSamples];
    private readonly float[] _primaryOutScratch = new float[8192];
    private readonly float[] _monitorOutScratch = new float[8192];
    private readonly float[] _zeroSilenceScratch = new float[8192];
    private readonly float[] _primeSilenceScratch = new float[4096];

    private int _silentOutputHops;
    private int _primarySilenceFrac;
    private int _monitorSilenceFrac;
    private float _lastEmittedSample16k;
    private int _crossfade16kSamplesRemaining;
    private int _crossfade16kTotalSamples = 64;
    private volatile int _adaptiveExtraCushionMs;

    private readonly AutoResetEvent _dspWorkSignal = new(false);
    private readonly ManualResetEventSlim _dspWarmupReady = new(false);

    // Live thread-safe parameters
    private volatile bool _vadEnabled = true;
    private volatile float _vadThreshold = 0.5f;
    private volatile bool _gtcrnEnabled = true;
    private volatile bool _voxLockEnabled = true;
    private volatile bool _monitorLoopbackEnabled;
    private volatile bool _autoAffinityEnabled = true;
    private long _selectedCoreMask;
    private int _selectedProcessorGroup;
    private volatile bool _affinityUpdatePending;
    private volatile ComputeBackend _activeComputeBackend = ComputeBackend.Cpu;
    private volatile string _activeBackendDescription = "CPU";

    // Runtime state
    private WasapiCapture? _capture;
    private WasapiOut? _primaryOut;
    private WasapiOut? _monitorOut;
    private RingBufferWaveProvider? _primaryWaveProvider;
    private RingBufferWaveProvider? _monitorWaveProvider;
    private MMDevice? _captureDevice;
    private MMDevice? _primaryRenderDevice;
    private MMDevice? _monitorRenderDevice;
    private string? _primaryRenderDeviceId;
    private bool _monitorSharesPrimaryDevice;
    private int _primaryRenderRate = 48000;
    private int _monitorRenderRate = 48000;

    private Thread? _dspThread;
    private CancellationTokenSource? _runCts;
    private IDisposable? _processAudioBoostScope;
    private volatile bool _isRunning;
    private bool _disposed;

    // Telemetry & counters
    private long _bufferOverruns;
    private long _bufferUnderruns;
    private TelemetrySnapshot _latestTelemetry = TelemetrySnapshot.Idle;
    private string _activeCoreDescription = "Unbound";
    private long _activeCoreMask;
    private int _activeProcessorGroup;

    public AudioPipelineEngine(
        AudioDeviceService? deviceService = null,
        CpuAffinityManager? affinityManager = null,
        IAudioFilterStage? vadStage = null,
        IAudioFilterStage? gtcrnStage = null,
        IVoiceIsolatorBackend? voxLockBackend = null)
    {
        _deviceService = deviceService ?? new AudioDeviceService();
        _affinityManager = affinityManager ?? new CpuAffinityManager();
        SetStages(vadStage, gtcrnStage, voxLockBackend);
    }

    public bool IsRunning => _isRunning;

    public TelemetrySnapshot LatestTelemetry => Volatile.Read(ref _latestTelemetry);

    public event EventHandler<TelemetrySnapshot>? TelemetryUpdated;
    public event EventHandler<Exception>? PipelineFaulted;

    /// <summary>
    /// Attaches or replaces the DSP stages while the pipeline is stopped.
    /// </summary>
    public void SetStages(
        IAudioFilterStage? vadStage,
        IAudioFilterStage? gtcrnStage,
        IVoiceIsolatorBackend? voxLockBackend)
    {
        lock (_lifecycleLock)
        {
            if (_isRunning)
            {
                throw new InvalidOperationException("Cannot replace DSP stages while the pipeline is running.");
            }

            _vadStage = vadStage;
            _vadProbabilityGetter = BindProbabilityGetter(vadStage);
            _gtcrnStage = gtcrnStage;
            _voxLockBackend = voxLockBackend;
        }
    }

    /// <summary>
    /// Applies live configuration changes safely (may be called while running).
    /// </summary>
    public void ApplySettings(AppSettings settings)
    {
        ArgumentNullException.ThrowIfNull(settings);

        _vadEnabled = settings.VadEnabled;
        _vadThreshold = Math.Clamp(settings.VadThreshold, 0.01f, 0.99f);
        _gtcrnEnabled = settings.GtcrnEnabled;
        _voxLockEnabled = settings.VoxLockEnabled;
        _activeComputeBackend = ResolveEffectiveComputeBackend(settings.ComputeBackend);
        _activeBackendDescription = _activeComputeBackend == ComputeBackend.Cuda ? "CUDA" : "CPU";

        UpdateCpuAffinity(settings.AutoAffinityEnabled, settings.SelectedCoreMask, settings.SelectedProcessorGroup);
        SetMonitorLoopback(settings.MonitorLoopbackEnabled);

        int prebufferMs = GetOutputPrebufferMs();
        int maxBacklogMs = GetOutputMaxBacklogMs();
        _primaryWaveProvider?.SetBaseCushionMs(prebufferMs, maxBacklogMs);
        _monitorWaveProvider?.SetBaseCushionMs(prebufferMs, maxBacklogMs);
    }

    public void UpdateCpuAffinity(bool autoAffinityEnabled, ulong selectedCoreMask, ushort processorGroup = 0)
    {
        _autoAffinityEnabled = autoAffinityEnabled;
        Interlocked.Exchange(ref _selectedCoreMask, unchecked((long)selectedCoreMask));
        Volatile.Write(ref _selectedProcessorGroup, processorGroup);
        _affinityUpdatePending = true;
        _dspWorkSignal.Set();
    }

    public void SetActiveBackend(ComputeBackend backend, string description)
    {
        _activeComputeBackend = backend;
        if (!string.IsNullOrWhiteSpace(description))
        {
            _activeBackendDescription = description;
        }
    }

    public void SetActiveBackendDescription(string description)
    {
        if (!string.IsNullOrWhiteSpace(description))
        {
            _activeBackendDescription = description;
        }
    }

    private ComputeBackend ResolveEffectiveComputeBackend(ComputeBackend requestedBackend)
    {
        if (_voxLockBackend is VoiceLockStudio.Services.Backends.VoxLockOnnxBackend { IsInitialized: true } voxBackend)
        {
            return string.Equals(voxBackend.ActiveProviderName, "CUDAExecutionProvider", StringComparison.OrdinalIgnoreCase)
                ? ComputeBackend.Cuda
                : ComputeBackend.Cpu;
        }

        return requestedBackend;
    }

    public void SetMonitorLoopback(bool enabled)
    {
        _monitorLoopbackEnabled = enabled;

        lock (_lifecycleLock)
        {
            if (!_isRunning)
            {
                return;
            }

            if (enabled && _monitorOut is null && !_monitorSharesPrimaryDevice)
            {
                TryStartMonitorOutputLocked();
            }
            else if (!enabled && _monitorOut is not null)
            {
                StopMonitorOutputLocked();
            }
        }
    }

    public void Start(AppSettings settings)
    {
        ArgumentNullException.ThrowIfNull(settings);
        ObjectDisposedException.ThrowIf(_disposed, this);

        lock (_lifecycleLock)
        {
            if (_isRunning)
            {
                return;
            }

            ApplySettings(settings);
            PublishTelemetry(TelemetrySnapshot.Idle with
            {
                TimestampUtc = DateTimeOffset.UtcNow,
                State = PipelineRunState.Starting,
                StatusText = "Starting WASAPI pipeline...",
                ActiveBackendDescription = _activeBackendDescription
            }, forceEvent: true);

            try
            {
                _processAudioBoostScope ??= CpuAffinityManager.BeginProcessAudioBoost();
                _inputRing16k.Reset();
                _primaryOutRing.Reset();
                _monitorOutRing.Reset();
                _silentOutputHops = 0;
                _primarySilenceFrac = 0;
                _monitorSilenceFrac = 0;
                _lastEmittedSample16k = 0f;
                _crossfade16kSamplesRemaining = 0;
                _crossfade16kTotalSamples = 64;
                _adaptiveExtraCushionMs = 0;
                Interlocked.Exchange(ref _bufferOverruns, 0);
                Interlocked.Exchange(ref _bufferUnderruns, 0);

                _vadStage?.ResetState();
                _gtcrnStage?.ResetState();
                _voxLockBackend?.ResetState();

                _captureDevice = _deviceService.ResolveDevice(settings.SelectedInputDeviceId, DataFlow.Capture)
                    ?? throw new InvalidOperationException("No active WASAPI capture device available.");

                _primaryRenderDevice = _deviceService.ResolveDevice(settings.SelectedOutputDeviceId, DataFlow.Render)
                    ?? throw new InvalidOperationException("No active WASAPI render device available.");
                _primaryRenderDeviceId = _primaryRenderDevice.ID;

                try
                {
                    _capture = new WasapiCapture(_captureDevice, useEventSync: true, audioBufferMillisecondsLength: EventSyncCaptureBufferMs);
                }
                catch
                {
                    _capture = new WasapiCapture(_captureDevice, useEventSync: false, audioBufferMillisecondsLength: PollingCaptureBufferMs);
                }

                int renderLatencyMs = GetRenderLatencyMs();
                int prebufferMs = GetOutputPrebufferMs();
                int maxBacklogMs = GetOutputMaxBacklogMs();
                _primaryOut = CreateAndInitWasapiOutWithFallback(
                    _primaryRenderDevice,
                    _primaryOutRing,
                    () => Interlocked.Increment(ref _bufferUnderruns),
                    renderLatencyMs,
                    prebufferMs,
                    maxBacklogMs,
                    out _primaryWaveProvider,
                    out bool primaryUsedEventSync);

                _primaryRenderRate = _primaryOut.OutputWaveFormat.SampleRate;
                _inputResampler.Configure(_capture.WaveFormat.SampleRate, DspSampleRate);
                _primaryOutResampler.Configure(DspSampleRate, _primaryRenderRate);

                string? defaultRenderId = _deviceService.GetDefaultDeviceId(DataFlow.Render, Role.Multimedia);
                _monitorSharesPrimaryDevice = string.IsNullOrEmpty(defaultRenderId)
                    || string.Equals(defaultRenderId, _primaryRenderDeviceId, StringComparison.OrdinalIgnoreCase);

                if (_monitorLoopbackEnabled && !_monitorSharesPrimaryDevice)
                {
                    TryStartMonitorOutputLocked();
                }

                _runCts = new CancellationTokenSource();
                _isRunning = true;
                _dspWarmupReady.Reset();

                _capture.DataAvailable += OnCaptureDataAvailable;
                _capture.RecordingStopped += OnCaptureStopped;

                _dspThread = new Thread(DspWorkerLoop)
                {
                    Name = "VoiceLock-DSP-Worker",
                    IsBackground = true
                };
                _dspThread.Start(_runCts.Token);

                // Wait for the affinity-pinned DSP thread to finish ORT warmup before opening capture flow.
                _dspWarmupReady.Wait(5000);

                PlayWasapiOutWithFallback(
                    ref _primaryOut,
                    ref _primaryWaveProvider,
                    primaryUsedEventSync,
                    _primaryRenderDevice,
                    _primaryOutRing,
                    () => Interlocked.Increment(ref _bufferUnderruns),
                    renderLatencyMs,
                    prebufferMs,
                    maxBacklogMs,
                    rate =>
                    {
                        _primaryRenderRate = rate;
                        _primaryOutResampler.Configure(DspSampleRate, _primaryRenderRate);
                    });
                try
                {
                    _capture.StartRecording();
                }
                catch
                {
                    _capture.DataAvailable -= OnCaptureDataAvailable;
                    _capture.RecordingStopped -= OnCaptureStopped;
                    _capture.Dispose();

                    _capture = new WasapiCapture(_captureDevice, useEventSync: false, audioBufferMillisecondsLength: PollingCaptureBufferMs);
                    _inputResampler.Configure(_capture.WaveFormat.SampleRate, DspSampleRate);
                    _capture.DataAvailable += OnCaptureDataAvailable;
                    _capture.RecordingStopped += OnCaptureStopped;
                    _capture.StartRecording();
                }
            }
            catch
            {
                StopInternalLocked();
                throw;
            }
        }
    }

    public void Stop()
    {
        lock (_lifecycleLock)
        {
            StopInternalLocked();
        }

        PublishTelemetry(TelemetrySnapshot.Idle with
        {
            TimestampUtc = DateTimeOffset.UtcNow,
            State = PipelineRunState.Stopped,
            StatusText = "Stopped",
            ActiveBackendDescription = _activeBackendDescription
        }, forceEvent: true);
    }

    private void StopInternalLocked()
    {
        _isRunning = false;
        _runCts?.Cancel();
        _dspWorkSignal.Set();

        if (_capture is not null)
        {
            _capture.DataAvailable -= OnCaptureDataAvailable;
            _capture.RecordingStopped -= OnCaptureStopped;
            try
            {
                _capture.StopRecording();
            }
            catch
            {
                // Ignore stop races.
            }
        }

        try
        {
            _primaryOut?.Stop();
        }
        catch
        {
            // Ignore stop races.
        }

        StopMonitorOutputLocked();

        if (_dspThread is not null && _dspThread.IsAlive && !ReferenceEquals(Thread.CurrentThread, _dspThread))
        {
            _dspThread.Join(1500);
        }

        _dspThread = null;
        _runCts?.Dispose();
        _runCts = null;

        _capture?.Dispose();
        _capture = null;
        _primaryOut?.Dispose();
        _primaryOut = null;
        _primaryWaveProvider = null;

        _captureDevice?.Dispose();
        _captureDevice = null;
        _primaryRenderDevice?.Dispose();
        _primaryRenderDevice = null;
        _primaryRenderDeviceId = null;

        Interlocked.Exchange(ref _processAudioBoostScope, null)?.Dispose();
    }

    private void TryStartMonitorOutputLocked()
    {
        try
        {
            MMDevice? defaultSpeakers = _deviceService.ResolveDevice(null, DataFlow.Render);
            if (defaultSpeakers is null ||
                string.Equals(defaultSpeakers.ID, _primaryRenderDeviceId, StringComparison.OrdinalIgnoreCase))
            {
                defaultSpeakers?.Dispose();
                _monitorSharesPrimaryDevice = true;
                return;
            }

            _monitorRenderDevice = defaultSpeakers;
            _monitorOutRing.Reset();
            _monitorSilenceFrac = 0;

            int renderLatencyMs = GetRenderLatencyMs();
            int prebufferMs = GetOutputPrebufferMs();
            int maxBacklogMs = GetOutputMaxBacklogMs();
            _monitorOut = CreateAndInitWasapiOutWithFallback(
                _monitorRenderDevice,
                _monitorOutRing,
                null,
                renderLatencyMs,
                prebufferMs,
                maxBacklogMs,
                out _monitorWaveProvider,
                out bool monitorUsedEventSync);
            _monitorWaveProvider?.SetAdaptiveCushionMs(_adaptiveExtraCushionMs);
            _monitorRenderRate = _monitorOut.OutputWaveFormat.SampleRate;
            _monitorOutResampler.Configure(DspSampleRate, _monitorRenderRate);
            PlayWasapiOutWithFallback(
                ref _monitorOut,
                ref _monitorWaveProvider,
                monitorUsedEventSync,
                _monitorRenderDevice,
                _monitorOutRing,
                null,
                renderLatencyMs,
                prebufferMs,
                maxBacklogMs,
                rate =>
                {
                    _monitorRenderRate = rate;
                    _monitorOutResampler.Configure(DspSampleRate, _monitorRenderRate);
                });
        }
        catch
        {
            StopMonitorOutputLocked();
        }
    }

    private void StopMonitorOutputLocked()
    {
        try
        {
            _monitorOut?.Stop();
        }
        catch
        {
            // Ignore stop races.
        }

        _monitorOut?.Dispose();
        _monitorOut = null;
        _monitorWaveProvider = null;
        _monitorRenderDevice?.Dispose();
        _monitorRenderDevice = null;
        _monitorOutRing.Reset();
    }

    private void OnCaptureDataAvailable(object? sender, WaveInEventArgs e)
    {
        if (!_isRunning || e.BytesRecorded <= 0 || _capture is null)
        {
            return;
        }

        WaveFormat fmt = _capture.WaveFormat;
        int blockAlign = Math.Max(1, fmt.BlockAlign);
        int totalFrames = e.BytesRecorded / blockAlign;
        int offset = 0;

        while (offset < totalFrames)
        {
            int chunkFrames = Math.Min(totalFrames - offset, _captureMonoScratch.Length);
            int byteOffset = offset * blockAlign;
            int byteLen = chunkFrames * blockAlign;

            int monoRead = WasapiPcmConverter.ReadMonoFloats(
                e.Buffer.AsSpan(byteOffset, byteLen),
                fmt,
                _captureMonoScratch);

            if (monoRead <= 0)
            {
                break;
            }

            int resampled = _inputResampler.Process(
                _captureMonoScratch.AsSpan(0, monoRead),
                _capture16kScratch);

            if (resampled > 0)
            {
                int written = _inputRing16k.Write(_capture16kScratch.AsSpan(0, resampled));
                if (written < resampled)
                {
                    Interlocked.Increment(ref _bufferOverruns);
                }
            }

            offset += chunkFrames;
        }

        if (_inputRing16k.AvailableRead >= HopSizeSamples)
        {
            _dspWorkSignal.Set();
        }
    }

    private void OnCaptureStopped(object? sender, StoppedEventArgs e)
    {
        if (e.Exception is not null && _isRunning)
        {
            _isRunning = false;
            _runCts?.Cancel();
            _dspWorkSignal.Set();

            PublishTelemetry(LatestTelemetry with
            {
                TimestampUtc = DateTimeOffset.UtcNow,
                State = PipelineRunState.Faulted,
                StatusText = $"Capture error: {e.Exception.Message}"
            }, forceEvent: true);

            PipelineFaulted?.Invoke(this, e.Exception);
        }
    }

    private void DspWorkerLoop(object? state)
    {
        var token = (CancellationToken)state!;
        using CpuAffinityManager.MmcssAudioScope? mmcssScope = CpuAffinityManager.TryRegisterCurrentThreadProAudio();
        try
        {
            ApplyAffinityOnDspThread(mmcssScope);
            _vadStage?.Warmup();
            _gtcrnStage?.Warmup();
            _voxLockBackend?.Warmup();
        }
        finally
        {
            _dspWarmupReady.Set();
        }

        int vadAccumOffset = 0;
        int voxStagingCount = 0;
        int hangoverRemaining = 0;
        int consecutiveSilenceFrames = 0;
        bool vadSpeechActive = true;
        bool hasHeldSilenceHop = false;
        float lastVadProb = 0f;
        bool downstreamStateClean = true;

        long lastTelemetryTick = Stopwatch.GetTimestamp();
        double ticksToMs = 1000.0 / Stopwatch.Frequency;

        double smoothedTotalInfMs = 0.0;
        double smoothedVadInfMs = 0.0;
        double smoothedGtcrnInfMs = 0.0;
        double smoothedVoxLockInfMs = 0.0;
        float lastInputRms = 0f;
        float lastInputPeak = 0f;
        float lastOutputRms = 0f;
        float lastOutputPeak = 0f;

        long lastObservedUnderruns = 0;
        int adaptiveExtraMs = 0;
        int stableCushionHops = 0;
        double hopAccumComputeMs = 0.0;
        int consecutiveHighLoadChunks = 0;
        bool escapedCoreCage = false;
        int lowLoadRecoveryHops = 0;
        int consecutiveOverbudgetChunks = 0;
        int overloadBypassHopsRemaining = 0;

        try
        {
            while (!token.IsCancellationRequested && _isRunning)
            {
                if (_affinityUpdatePending)
                {
                    _affinityUpdatePending = false;
                    escapedCoreCage = false;
                    consecutiveHighLoadChunks = 0;
                    lowLoadRecoveryHops = 0;
                    ApplyAffinityOnDspThread(mmcssScope);
                }

                if (_inputRing16k.AvailableRead < HopSizeSamples)
                {
                    _dspWorkSignal.WaitOne(16);
                    continue;
                }

                while (_inputRing16k.AvailableRead >= HopSizeSamples && !token.IsCancellationRequested && _isRunning)
                {
                    // Max-latency watermark ("Drop Oldest"): preserve BSRNN recurrent memory with soft buffer reset.
                    bool voxConfigured = _voxLockEnabled && _voxLockBackend is { IsInitialized: true, IsSessionLoaded: true, HasEnrolledVoiceprint: true };
                    int voxFrameSizeCheck = voxConfigured
                        ? Math.Clamp(_voxLockBackend!.FrameSizeSamples, HopSizeSamples, _voxOutScratch.Length)
                        : HopSizeSamples;
                    int maxInputBacklog = voxConfigured ? Math.Max(3200, voxFrameSizeCheck * 4) : 2048;
                    int keepInputSamples = voxConfigured ? Math.Max(HopSizeSamples * 2, voxFrameSizeCheck) : HopSizeSamples * 2;
                    int availIn = _inputRing16k.AvailableRead;
                    if (availIn > maxInputBacklog)
                    {
                        int dropped = _inputRing16k.Skip(availIn - keepInputSamples);
                        if (dropped > 0)
                        {
                            Interlocked.Increment(ref _bufferOverruns);
                            vadAccumOffset = 0;
                            voxStagingCount = 0;
                            hasHeldSilenceHop = false;
                            hopAccumComputeMs = 0.0;
                            _voxLockBackend?.SoftResetStreamingBuffers();
                            Request16kCrossfade(32);
                        }
                    }

                    int read = _inputRing16k.Read(_hopInput256);
                    if (read < HopSizeSamples)
                    {
                        break;
                    }

                    lastInputRms = WasapiPcmConverter.ComputeRmsAndPeak(_hopInput256, out lastInputPeak);

                    long frameStartTick = Stopwatch.GetTimestamp();
                    double vadInfMs = 0.0;
                    double gtcrnInfMs = 0.0;
                    double voxLockInfMs = 0.0;

                    // Accumulate 2x 256-sample hops into 512-sample Silero-VAD frames (32 ms)
                    _hopInput256.CopyTo(_vadAccum512.AsSpan(vadAccumOffset, HopSizeSamples));
                    vadAccumOffset += HopSizeSamples;

                    bool vadEnabled = _vadEnabled && _vadStage is { IsInitialized: true };
                    if (vadAccumOffset >= VadFrameSizeSamples)
                    {
                        vadAccumOffset = 0;

                        if (vadEnabled)
                        {
                            long t0 = Stopwatch.GetTimestamp();
                            _vadStage!.ProcessFrame(_vadAccum512, _vadScratchOut512);
                            vadInfMs = (Stopwatch.GetTimestamp() - t0) * ticksToMs;

                            lastVadProb = _vadProbabilityGetter is not null
                                ? _vadProbabilityGetter()
                                : Math.Clamp(_vadScratchOut512[0], 0f, 1f);

                            if (lastVadProb >= _vadThreshold)
                            {
                                hangoverRemaining = VadHangoverFrames;
                                consecutiveSilenceFrames = 0;
                                vadSpeechActive = true;
                            }
                            else if (hangoverRemaining > 0)
                            {
                                hangoverRemaining--;
                                vadSpeechActive = true;
                            }
                            else
                            {
                                vadSpeechActive = false;
                                consecutiveSilenceFrames++;

                                if (consecutiveSilenceFrames >= SilenceResetThresholdFrames && !downstreamStateClean)
                                {
                                    _gtcrnStage?.ResetState();
                                    _voxLockBackend?.ResetState();
                                    voxStagingCount = 0;
                                    downstreamStateClean = true;
                                }
                            }
                        }
                        else
                        {
                            vadSpeechActive = true;
                            lastVadProb = 1f;
                            consecutiveSilenceFrames = 0;
                        }
                    }
                    else if (!vadEnabled)
                    {
                        vadSpeechActive = true;
                        lastVadProb = 1f;
                    }

                    bool voxActive = _voxLockEnabled && _voxLockBackend is { IsInitialized: true, IsSessionLoaded: true, HasEnrolledVoiceprint: true };
                    int voxFrameSize = voxActive
                        ? Math.Clamp(_voxLockBackend!.FrameSizeSamples, HopSizeSamples, _voxOutScratch.Length)
                        : HopSizeSamples;
                    if (!voxActive)
                    {
                        overloadBypassHopsRemaining = 0;
                        consecutiveOverbudgetChunks = 0;
                        consecutiveHighLoadChunks = 0;
                    }

                    bool overloadBypassActive = voxActive && overloadBypassHopsRemaining > 0;
                    bool effectiveVoxActive = voxActive && !overloadBypassActive;
                    bool voxRanThisHop = false;

                    // Downstream cascade: Silero Gate -> GTCRN (256 hop) -> VoxLock (16 ms or 32 ms chunk)
                    if (vadEnabled && !vadSpeechActive)
                    {
                        overloadBypassHopsRemaining = 0;
                        consecutiveOverbudgetChunks = 0;
                        consecutiveHighLoadChunks = 0;
                        hopAccumComputeMs = 0.0;

                        if (escapedCoreCage && consecutiveSilenceFrames >= SilenceResetThresholdFrames)
                        {
                            escapedCoreCage = false;
                            lowLoadRecoveryHops = 0;
                            ApplyAffinityOnDspThread(mmcssScope);
                        }

                        if (voxStagingCount > 0)
                        {
                            // Drain any odd staged K=4 hop as silence so input/output sample counts remain 1:1.
                            int stagedHops = voxStagingCount / HopSizeSamples;
                            voxStagingCount = 0;
                            Array.Clear(_hopOutput256);
                            for (int s = 0; s < stagedHops; s++)
                            {
                                WriteOutputToRings(_hopOutput256, isDigitalSilence: true);
                            }
                        }

                        // Hold the first half-frame (256 samples) instead of emitting premature silence, preserving 1:1 sample conservation for VAD wake-up pre-roll.
                        if (vadAccumOffset == HopSizeSamples && !hasHeldSilenceHop)
                        {
                            hasHeldSilenceHop = true;
                        }
                        else
                        {
                            Array.Clear(_hopOutput256);
                            lastOutputRms = 0f;
                            lastOutputPeak = 0f;
                            if (hasHeldSilenceHop)
                            {
                                hasHeldSilenceHop = false;
                                WriteOutputToRings(_hopOutput256, isDigitalSilence: true);
                            }

                            WriteOutputToRings(_hopOutput256, isDigitalSilence: true);
                        }
                    }
                    else
                    {
                        // Flush any held silence hop cleanly without triggering a double-VoxLock compute spike on speech onset.
                        if (hasHeldSilenceHop)
                        {
                            hasHeldSilenceHop = false;
                            if (effectiveVoxActive && (voxFrameSize <= HopSizeSamples || smoothedTotalInfMs > 12.0))
                            {
                                ProcessGtcrnOnlyHop(
                                    _vadAccum512.AsSpan(0, HopSizeSamples),
                                    ticksToMs,
                                    ref downstreamStateClean,
                                    ref gtcrnInfMs,
                                    ref lastOutputRms,
                                    ref lastOutputPeak,
                                    applyFadeInFromZero: true);
                            }
                            else
                            {
                                ProcessSpeechHop(
                                    _vadAccum512.AsSpan(0, HopSizeSamples),
                                    ticksToMs,
                                    effectiveVoxActive,
                                    voxFrameSize,
                                    ref voxStagingCount,
                                    ref downstreamStateClean,
                                    ref gtcrnInfMs,
                                    ref voxLockInfMs,
                                    ref voxRanThisHop,
                                    ref lastOutputRms,
                                    ref lastOutputPeak);
                            }
                        }

                        ProcessSpeechHop(
                            _hopInput256,
                            ticksToMs,
                            effectiveVoxActive,
                            voxFrameSize,
                            ref voxStagingCount,
                            ref downstreamStateClean,
                            ref gtcrnInfMs,
                            ref voxLockInfMs,
                            ref voxRanThisHop,
                            ref lastOutputRms,
                            ref lastOutputPeak);
                    }

                    double totalInfMs = (Stopwatch.GetTimestamp() - frameStartTick) * ticksToMs;

                    if (!vadEnabled || vadSpeechActive)
                    {
                        hopAccumComputeMs += totalInfMs;

                        if (overloadBypassActive)
                        {
                            hopAccumComputeMs = 0.0;
                            overloadBypassHopsRemaining--;
                            if (overloadBypassHopsRemaining == 0)
                            {
                                Request16kCrossfade(64);
                            }
                        }
                        else if (!effectiveVoxActive || voxRanThisHop || _inputRing16k.AvailableRead >= 1600)
                        {
                            bool ranFullUnit = !effectiveVoxActive || voxRanThisHop;
                            double chunkWorkMs = hopAccumComputeMs;
                            if (ranFullUnit)
                            {
                                hopAccumComputeMs = 0.0;
                            }

                            double chunkBudgetMs = effectiveVoxActive
                                ? (voxFrameSize * 1000.0 / DspSampleRate)
                                : (HopSizeSamples * 1000.0 / DspSampleRate);

                            // Adaptive output jitter cushion (expands rapidly on underrun/high load, decays slowly when calm).
                            long currentUnderruns = Interlocked.Read(ref _bufferUnderruns);
                            if (currentUnderruns > lastObservedUnderruns || (ranFullUnit && chunkWorkMs > chunkBudgetMs * 0.80))
                            {
                                lastObservedUnderruns = currentUnderruns;
                                adaptiveExtraMs = Math.Min(48, adaptiveExtraMs + 16);
                                stableCushionHops = 0;
                                UpdateAdaptiveOutputCushion(adaptiveExtraMs);
                            }
                            else if (adaptiveExtraMs > 0 && ranFullUnit)
                            {
                                stableCushionHops += effectiveVoxActive && voxFrameSize > HopSizeSamples ? 2 : 1;
                                if (stableCushionHops >= 180)
                                {
                                    stableCushionHops = 0;
                                    adaptiveExtraMs = Math.Max(0, adaptiveExtraMs - 4);
                                    UpdateAdaptiveOutputCushion(adaptiveExtraMs);
                                }
                            }

                            // Stage A: Dynamic Core Escape when E-Core / tail-core affinity is saturated by external system load.
                            if (_autoAffinityEnabled && _activeComputeBackend == ComputeBackend.Cpu && effectiveVoxActive)
                            {
                                if (!escapedCoreCage)
                                {
                                    if ((ranFullUnit && chunkWorkMs > chunkBudgetMs * 0.88) || _inputRing16k.AvailableRead >= voxFrameSize * 2)
                                    {
                                        consecutiveHighLoadChunks++;
                                        if (consecutiveHighLoadChunks >= 3)
                                        {
                                            consecutiveHighLoadChunks = 0;
                                            ulong escMask = _affinityManager.GetOverloadEscapeMask(out ushort escGroup, out _);
                                            if (escMask != 0 &&
                                                (escMask != unchecked((ulong)Volatile.Read(ref _activeCoreMask)) ||
                                                 escGroup != Volatile.Read(ref _activeProcessorGroup)))
                                            {
                                                ApplyExplicitAffinityOnDspThread(escMask, escGroup, mmcssScope);
                                                escapedCoreCage = true;
                                                lowLoadRecoveryHops = 0;
                                            }
                                        }
                                    }
                                    else if (ranFullUnit)
                                    {
                                        consecutiveHighLoadChunks = 0;
                                    }
                                }
                                else if (ranFullUnit)
                                {
                                    if (chunkWorkMs < chunkBudgetMs * 0.42)
                                    {
                                        lowLoadRecoveryHops++;
                                        if (lowLoadRecoveryHops >= 150)
                                        {
                                            escapedCoreCage = false;
                                            lowLoadRecoveryHops = 0;
                                            ApplyAffinityOnDspThread(mmcssScope);
                                        }
                                    }
                                    else
                                    {
                                        lowLoadRecoveryHops = 0;
                                    }
                                }
                            }

                            // Stage B: Graceful Overload Shedding (smooth VoxLock -> GTCRN fallback to guarantee zero dropouts).
                            if (effectiveVoxActive)
                            {
                                bool queueBackedUp = _inputRing16k.AvailableRead >= 1600;
                                if ((ranFullUnit && chunkWorkMs > chunkBudgetMs * 1.02) || queueBackedUp)
                                {
                                    consecutiveOverbudgetChunks++;
                                    if (consecutiveOverbudgetChunks >= 3 || queueBackedUp)
                                    {
                                        consecutiveOverbudgetChunks = 0;
                                        overloadBypassHopsRemaining = 64; // ~1.02 s graceful GTCRN cushion
                                        hopAccumComputeMs = 0.0;

                                        if (voxStagingCount > 0)
                                        {
                                            int stagedHops = voxStagingCount / HopSizeSamples;
                                            for (int s = 0; s < stagedHops; s++)
                                            {
                                                Span<float> stagedSpan = _voxStagingIn.AsSpan(s * HopSizeSamples, HopSizeSamples);
                                                if (s == 0)
                                                {
                                                    Request16kCrossfade(64);
                                                }
                                                Apply16kTransitionCrossfade(stagedSpan);
                                                lastOutputRms = WasapiPcmConverter.ComputeRmsAndPeak(stagedSpan, out lastOutputPeak);
                                                WriteOutputToRings(stagedSpan);
                                            }
                                            voxStagingCount = 0;
                                        }
                                        else
                                        {
                                            Request16kCrossfade(64);
                                        }

                                        _voxLockBackend?.SoftResetStreamingBuffers();
                                    }
                                }
                                else if (ranFullUnit)
                                {
                                    consecutiveOverbudgetChunks = 0;
                                }
                            }
                        }
                    }

                    if (!effectiveVoxActive || voxRanThisHop || (vadEnabled && !vadSpeechActive))
                    {
                        smoothedTotalInfMs = smoothedTotalInfMs == 0.0 ? totalInfMs : smoothedTotalInfMs * 0.8 + totalInfMs * 0.2;
                    }
                    if (vadInfMs > 0.0)
                    {
                        smoothedVadInfMs = smoothedVadInfMs == 0.0 ? vadInfMs : smoothedVadInfMs * 0.7 + vadInfMs * 0.3;
                    }
                    smoothedGtcrnInfMs = _gtcrnEnabled ? (smoothedGtcrnInfMs == 0.0 ? gtcrnInfMs : smoothedGtcrnInfMs * 0.8 + gtcrnInfMs * 0.2) : 0.0;
                    if (effectiveVoxActive)
                    {
                        if (voxRanThisHop)
                        {
                            smoothedVoxLockInfMs = smoothedVoxLockInfMs == 0.0 ? voxLockInfMs : smoothedVoxLockInfMs * 0.8 + voxLockInfMs * 0.2;
                        }
                    }
                    else
                    {
                        smoothedVoxLockInfMs = 0.0;
                    }

                    long nowTick = Stopwatch.GetTimestamp();
                    if ((nowTick - lastTelemetryTick) * ticksToMs >= TelemetryIntervalMs)
                    {
                        lastTelemetryTick = nowTick;
                        double algorithmicLatencyMs = ComputeTotalLatencyMs(effectiveVoxActive);
                        string coreDesc = Volatile.Read(ref _activeCoreDescription);
                        ulong coreMask = unchecked((ulong)Volatile.Read(ref _activeCoreMask));

                        PublishTelemetry(new TelemetrySnapshot(
                            TimestampUtc: DateTimeOffset.UtcNow,
                            State: PipelineRunState.Running,
                            StatusText: vadSpeechActive ? "Running (Speech Active)" : "Running (VAD Gated)",
                            LatencyMs: algorithmicLatencyMs,
                            FrameInferenceMs: smoothedTotalInfMs,
                            VadInferenceMs: smoothedVadInfMs,
                            GtcrnInferenceMs: smoothedGtcrnInfMs,
                            VoxLockInferenceMs: smoothedVoxLockInfMs,
                            ActiveCoreDescription: coreDesc,
                            ActiveCoreMask: coreMask,
                            VadSpeechActive: vadSpeechActive,
                            VadProbability: lastVadProb,
                            InputRms: lastInputRms,
                            OutputRms: lastOutputRms,
                            InputPeak: lastInputPeak,
                            OutputPeak: lastOutputPeak,
                            BufferOverruns: Interlocked.Read(ref _bufferOverruns),
                            BufferUnderruns: Interlocked.Read(ref _bufferUnderruns),
                            ActiveBackendDescription: _activeBackendDescription),
                            forceEvent: false);
                    }
                }
            }
        }
        catch (Exception ex) when (!token.IsCancellationRequested)
        {
            _isRunning = false;
            Interlocked.Exchange(ref _processAudioBoostScope, null)?.Dispose();
            PublishTelemetry(LatestTelemetry with
            {
                TimestampUtc = DateTimeOffset.UtcNow,
                State = PipelineRunState.Faulted,
                StatusText = $"DSP error: {ex.Message}"
            }, forceEvent: true);

            PipelineFaulted?.Invoke(this, ex);
        }
    }

    private void ApplyAffinityOnDspThread(CpuAffinityManager.MmcssAudioScope? mmcssScope = null)
    {
        bool preferPCores = (_voxLockBackend?.FrameSizeSamples ?? 512) <= 256
            && _activeComputeBackend != ComputeBackend.Cuda;

        ulong mask;
        ushort group;
        if (_autoAffinityEnabled)
        {
            mask = _affinityManager.GetRecommendedAudioWorkerMask(out group, out _, preferPCores);
        }
        else
        {
            mask = unchecked((ulong)Interlocked.Read(ref _selectedCoreMask));
            group = (ushort)Math.Clamp(Volatile.Read(ref _selectedProcessorGroup), 0, ushort.MaxValue);
            if (mask == 0)
            {
                mask = _affinityManager.GetRecommendedAudioWorkerMask(out group, out _, preferPCores);
            }
        }

        ApplyExplicitAffinityOnDspThread(mask, group, mmcssScope);
    }

    private void ApplyExplicitAffinityOnDspThread(
        ulong mask,
        ushort group,
        CpuAffinityManager.MmcssAudioScope? mmcssScope = null)
    {
        AffinityBindResult result = _affinityManager.BindCurrentThread(mask, ThreadPriority.Highest, group);
        mmcssScope?.ReinforcePriority();
        _voxLockBackend?.UpdateCpuAffinity(result.AppliedMask, result.ProcessorGroup);
        Volatile.Write(ref _activeCoreDescription, result.Description);
        Volatile.Write(ref _activeCoreMask, unchecked((long)result.AppliedMask));
        Volatile.Write(ref _activeProcessorGroup, result.ProcessorGroup);
    }

    private int GetEffectiveDownstreamFrameSamples()
    {
        if (_voxLockEnabled && _voxLockBackend is { IsInitialized: true })
        {
            return _voxLockBackend.FrameSizeSamples;
        }

        return HopSizeSamples;
    }

    private int GetRenderLatencyMs() =>
        GetEffectiveDownstreamFrameSamples() <= HopSizeSamples ? LowLatencyRenderLatencyMs : StandardRenderLatencyMs;

    private int GetOutputPrebufferMs() =>
        GetEffectiveDownstreamFrameSamples() <= HopSizeSamples ? LowLatencyOutputPrebufferMs : StandardOutputPrebufferMs;

    private int GetOutputMaxBacklogMs() =>
        GetEffectiveDownstreamFrameSamples() <= HopSizeSamples ? LowLatencyOutputMaxBacklogMs : StandardOutputMaxBacklogMs;

    private void UpdateAdaptiveOutputCushion(int extraMs)
    {
        _adaptiveExtraCushionMs = extraMs;
        _primaryWaveProvider?.SetAdaptiveCushionMs(extraMs);
        _monitorWaveProvider?.SetAdaptiveCushionMs(extraMs);
    }

    private static WasapiOut CreateAndInitWasapiOutWithFallback(
        MMDevice device,
        AudioRingBuffer ring,
        Func<long>? onUnderrun,
        int renderLatencyMs,
        int prebufferMs,
        int maxBacklogMs,
        out RingBufferWaveProvider provider,
        out bool usedEventSync)
    {
        try
        {
            usedEventSync = true;
            return CreateAndInitWasapiOutCore(
                device,
                ring,
                onUnderrun,
                renderLatencyMs,
                prebufferMs,
                maxBacklogMs,
                useEventSync: true,
                out provider);
        }
        catch
        {
            usedEventSync = false;
            return CreateAndInitWasapiOutCore(
                device,
                ring,
                onUnderrun,
                renderLatencyMs,
                prebufferMs,
                maxBacklogMs,
                useEventSync: false,
                out provider);
        }
    }

    private static void PlayWasapiOutWithFallback(
        ref WasapiOut? output,
        ref RingBufferWaveProvider? waveProvider,
        bool usedEventSync,
        MMDevice device,
        AudioRingBuffer ring,
        Func<long>? onUnderrun,
        int renderLatencyMs,
        int prebufferMs,
        int maxBacklogMs,
        Action<int> onRenderRateResolved)
    {
        ArgumentNullException.ThrowIfNull(output);

        try
        {
            output.Play();
        }
        catch when (usedEventSync)
        {
            output.Dispose();
            output = null;
            waveProvider = null;
            WasapiOut fallback = CreateAndInitWasapiOutCore(
                device,
                ring,
                onUnderrun,
                renderLatencyMs,
                prebufferMs,
                maxBacklogMs,
                useEventSync: false,
                out RingBufferWaveProvider fallbackProvider);
            output = fallback;
            waveProvider = fallbackProvider;
            onRenderRateResolved(fallback.OutputWaveFormat.SampleRate);
            fallback.Play();
        }
    }

    private static WasapiOut CreateAndInitWasapiOutCore(
        MMDevice device,
        AudioRingBuffer ring,
        Func<long>? onUnderrun,
        int renderLatencyMs,
        int prebufferMs,
        int maxBacklogMs,
        bool useEventSync,
        out RingBufferWaveProvider provider)
    {
        var output = new WasapiOut(device, AudioClientShareMode.Shared, useEventSync: useEventSync, latency: renderLatencyMs);
        try
        {
            provider = new RingBufferWaveProvider(
                ring,
                output.OutputWaveFormat,
                onUnderrun,
                prebufferMs: prebufferMs,
                maxBacklogMs: maxBacklogMs);
            output.Init(provider);
            return output;
        }
        catch
        {
            output.Dispose();
            throw;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void ProcessGtcrnOnlyHop(
        ReadOnlySpan<float> hop16k,
        double ticksToMs,
        ref bool downstreamStateClean,
        ref double gtcrnInfMs,
        ref float lastOutputRms,
        ref float lastOutputPeak,
        bool applyFadeInFromZero)
    {
        ReadOnlySpan<float> current = hop16k;
        if (_gtcrnEnabled && _gtcrnStage is { IsInitialized: true })
        {
            long t0 = Stopwatch.GetTimestamp();
            _gtcrnStage.ProcessFrame(current, _gtcrnOut256);
            gtcrnInfMs += (Stopwatch.GetTimestamp() - t0) * ticksToMs;
            current = _gtcrnOut256;
            downstreamStateClean = false;
        }

        current.CopyTo(_hopOutput256);
        if (applyFadeInFromZero)
        {
            _lastEmittedSample16k = 0f;
            Request16kCrossfade(64);
        }
        Apply16kTransitionCrossfade(_hopOutput256);
        lastOutputRms = WasapiPcmConverter.ComputeRmsAndPeak(_hopOutput256, out lastOutputPeak);
        WriteOutputToRings(_hopOutput256);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void ProcessSpeechHop(
        ReadOnlySpan<float> hop16k,
        double ticksToMs,
        bool voxActive,
        int voxFrameSize,
        ref int voxStagingCount,
        ref bool downstreamStateClean,
        ref double gtcrnInfMs,
        ref double voxLockInfMs,
        ref bool voxRanThisHop,
        ref float lastOutputRms,
        ref float lastOutputPeak)
    {
        ReadOnlySpan<float> current = hop16k;

        if (_gtcrnEnabled && _gtcrnStage is { IsInitialized: true })
        {
            long t0 = Stopwatch.GetTimestamp();
            _gtcrnStage.ProcessFrame(current, _gtcrnOut256);
            gtcrnInfMs += (Stopwatch.GetTimestamp() - t0) * ticksToMs;
            current = _gtcrnOut256;
            downstreamStateClean = false;
        }

        if (voxActive)
        {
            if (voxStagingCount + HopSizeSamples > _voxStagingIn.Length)
            {
                voxStagingCount = 0;
            }

            current.CopyTo(_voxStagingIn.AsSpan(voxStagingCount, HopSizeSamples));
            voxStagingCount += HopSizeSamples;
            downstreamStateClean = false;

            while (voxStagingCount >= voxFrameSize)
            {
                Span<float> voxOutSpan = _voxOutScratch.AsSpan(0, voxFrameSize);
                long t0 = Stopwatch.GetTimestamp();
                _voxLockBackend!.ProcessFrame(_voxStagingIn.AsSpan(0, voxFrameSize), voxOutSpan);
                voxLockInfMs += (Stopwatch.GetTimestamp() - t0) * ticksToMs;
                voxRanThisHop = true;

                int remaining = voxStagingCount - voxFrameSize;
                if (remaining > 0)
                {
                    _voxStagingIn.AsSpan(voxFrameSize, remaining).CopyTo(_voxStagingIn.AsSpan(0, remaining));
                }
                voxStagingCount = remaining;

                Apply16kTransitionCrossfade(voxOutSpan);
                lastOutputRms = WasapiPcmConverter.ComputeRmsAndPeak(voxOutSpan, out lastOutputPeak);
                WriteOutputToRings(voxOutSpan);
            }
        }
        else
        {
            voxStagingCount = 0;
            current.CopyTo(_hopOutput256);
            Apply16kTransitionCrossfade(_hopOutput256);
            lastOutputRms = WasapiPcmConverter.ComputeRmsAndPeak(_hopOutput256, out lastOutputPeak);
            WriteOutputToRings(_hopOutput256);
        }
    }

    private void Request16kCrossfade(int samples)
    {
        _crossfade16kTotalSamples = Math.Max(1, samples);
        _crossfade16kSamplesRemaining = _crossfade16kTotalSamples;
    }

    private void Apply16kTransitionCrossfade(Span<float> samples)
    {
        if (samples.IsEmpty)
        {
            return;
        }

        int remaining = _crossfade16kSamplesRemaining;
        if (remaining > 0)
        {
            int totalRamp = Math.Max(remaining, _crossfade16kTotalSamples);
            int startStep = totalRamp - remaining;
            int count = Math.Min(samples.Length, remaining);
            float anchor = _lastEmittedSample16k;

            for (int i = 0; i < count; i++)
            {
                float alpha = (startStep + i + 1) / (float)totalRamp;
                samples[i] = anchor * (1f - alpha) + samples[i] * alpha;
            }

            _crossfade16kSamplesRemaining = remaining - count;
        }

        _lastEmittedSample16k = samples[^1];
    }

    private void WriteOutputToRings(ReadOnlySpan<float> samples16k, bool isDigitalSilence = false)
    {
        if (isDigitalSilence)
        {
            _lastEmittedSample16k = 0f;
            _silentOutputHops++;
            if (_silentOutputHops > 2)
            {
                if (_silentOutputHops == 3)
                {
                    _primaryOutResampler.Reset();
                    _monitorOutResampler.Reset();
                    _primarySilenceFrac = 0;
                    _monitorSilenceFrac = 0;
                }

                WriteTrimmedSilence(_primaryOutRing, samples16k.Length, _primaryRenderRate, ref _primarySilenceFrac);
                if (_monitorLoopbackEnabled && _monitorOut is not null && !_monitorSharesPrimaryDevice)
                {
                    WriteTrimmedSilence(_monitorOutRing, samples16k.Length, _monitorRenderRate, ref _monitorSilenceFrac);
                }
                return;
            }
        }
        else if (_silentOutputHops != 0)
        {
            _silentOutputHops = 0;
            _primarySilenceFrac = 0;
            _monitorSilenceFrac = 0;
        }

        int primarySamples = _primaryOutResampler.Process(samples16k, _primaryOutScratch);
        if (primarySamples > 0)
        {
            _primaryOutRing.Write(_primaryOutScratch.AsSpan(0, primarySamples));
        }

        if (_monitorLoopbackEnabled && _monitorOut is not null && !_monitorSharesPrimaryDevice)
        {
            if (_monitorRenderRate == _primaryRenderRate && primarySamples > 0)
            {
                _monitorOutRing.Write(_primaryOutScratch.AsSpan(0, primarySamples));
            }
            else
            {
                int monitorSamples = _monitorOutResampler.Process(samples16k, _monitorOutScratch);
                if (monitorSamples > 0)
                {
                    _monitorOutRing.Write(_monitorOutScratch.AsSpan(0, monitorSamples));
                }
            }
        }
    }

    private void WriteTrimmedSilence(AudioRingBuffer ring, int requested16kCount, int outputSampleRate, ref int fracRemainder)
    {
        int safeRate = Math.Max(1, outputSampleRate);
        int keepAliveSamples = (safeRate * (22 + _adaptiveExtraCushionMs / 2)) / 1000;
        int availOut = ring.AvailableRead;
        int step8Ms = Math.Max(1, (safeRate * 8) / 1000);

        int emit16k = requested16kCount;
        if (availOut > keepAliveSamples * 2)
        {
            emit16k = 0;
        }
        else if (availOut > keepAliveSamples + step8Ms)
        {
            emit16k = requested16kCount / 2;
        }

        if (emit16k > 0)
        {
            WriteSilenceSamples(ring, emit16k, safeRate, ref fracRemainder);
        }
    }

    private void WriteSilenceSamples(AudioRingBuffer ring, int samples16kCount, int outputSampleRate, ref int fracRemainder)
    {
        long num = (long)samples16kCount * Math.Max(1, outputSampleRate) + fracRemainder;
        int outCount = (int)(num / DspSampleRate);
        fracRemainder = (int)(num % DspSampleRate);

        while (outCount > 0)
        {
            int chunk = Math.Min(outCount, _zeroSilenceScratch.Length);
            ring.Write(_zeroSilenceScratch.AsSpan(0, chunk));
            outCount -= chunk;
        }
    }

    private double ComputeTotalLatencyMs(bool? voxOverrideActive = null)
    {
        double algoMs = _inputResampler.AlgorithmicDelayMs + _primaryOutResampler.AlgorithmicDelayMs;
        bool voxCurrentlyActive = voxOverrideActive
            ?? (_voxLockEnabled && _voxLockBackend is { IsInitialized: true, IsSessionLoaded: true, HasEnrolledVoiceprint: true });

        // Framing + STFT overlap latency:
        // - VoxLock active: 24 ms STFT OLA + chunk frame (32 ms at K=4), where 16 ms hops accumulate inside the frame.
        // - GTCRN only: 16 ms hop framing + 16 ms STFT lookahead.
        // - Bypass / VAD only: 16 ms hop framing.
        if (voxCurrentlyActive && _voxLockBackend is not null)
        {
            double frameMs = _voxLockBackend.FrameSizeSamples * 1000.0 / DspSampleRate;
            algoMs += 24.0 + Math.Max(16.0, frameMs);
        }
        else if (_gtcrnEnabled && _gtcrnStage is { IsInitialized: true })
        {
            algoMs += 16.0 + 16.0;
        }
        else
        {
            algoMs += 16.0;
        }

        double ringMs = (_inputRing16k.AvailableRead * 1000.0 / DspSampleRate)
                      + (_primaryOutRing.AvailableRead * 1000.0 / Math.Max(1, _primaryRenderRate));

        return algoMs + ringMs;
    }

    private void PublishTelemetry(TelemetrySnapshot snapshot, bool forceEvent)
    {
        Volatile.Write(ref _latestTelemetry, snapshot);
        if (forceEvent || _isRunning)
        {
            TelemetryUpdated?.Invoke(this, snapshot);
        }
    }

    private static Func<float>? BindProbabilityGetter(IAudioFilterStage? stage)
    {
        if (stage is null)
        {
            return null;
        }

        PropertyInfo? prop = stage.GetType().GetProperty("LastSpeechProbability", BindingFlags.Public | BindingFlags.Instance);
        MethodInfo? getter = prop?.GetGetMethod();
        if (getter is null || prop?.PropertyType != typeof(float))
        {
            return null;
        }

        return (Func<float>)Delegate.CreateDelegate(typeof(Func<float>), stage, getter);
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        Stop();
        _dspWorkSignal.Dispose();
        _dspWarmupReady.Dispose();
    }

    internal static void VerifyMicroUnderrunImmediateRecoveryForSelfTest()
    {
        var ring = new AudioRingBuffer(32768);
        long underruns = 0;
        WaveFormat fmt = WaveFormat.CreateIeeeFloatWaveFormat(48000, 1);
        var provider = new RingBufferWaveProvider(
            ring,
            fmt,
            () => Interlocked.Increment(ref underruns),
            prebufferMs: StandardOutputPrebufferMs,
            maxBacklogMs: StandardOutputMaxBacklogMs);

        // 1. Prebuffer 50 ms (2400 samples @ 48 kHz) and drain initial 10 ms (480 samples).
        float[] initial = new float[2400];
        for (int i = 0; i < initial.Length; i++)
        {
            initial[i] = 0.25f * MathF.Sin(2f * MathF.PI * 220f * i / 48000f);
        }
        ring.Write(initial);

        byte[] outBytes = new byte[480 * sizeof(float)];
        provider.Read(outBytes, 0, outBytes.Length);

        // 2. Drain remaining samples to trigger a transient micro-underrun.
        ring.Clear();
        provider.Read(outBytes, 0, outBytes.Length);
        if (underruns <= 0)
        {
            throw new InvalidOperationException("Expected micro-underrun counter to increment.");
        }

        // 3. Supply a single 16 ms burst (768 samples < 38 ms = 1824 prebuffer) and verify immediate non-zero output.
        float[] burst = new float[768];
        for (int i = 0; i < burst.Length; i++)
        {
            burst[i] = 0.35f;
        }
        ring.Write(burst);
        Array.Clear(outBytes);
        provider.Read(outBytes, 0, outBytes.Length);

        float maxAbs = 0f;
        ReadOnlySpan<float> emitted = global::System.Runtime.InteropServices.MemoryMarshal.Cast<byte, float>(outBytes);
        for (int i = 0; i < emitted.Length; i++)
        {
            maxAbs = MathF.Max(maxAbs, MathF.Abs(emitted[i]));
        }

        if (maxAbs < 0.05f)
        {
            throw new InvalidOperationException($"Micro-underrun recovery stalled into hard prebuffer silence (maxAbs={maxAbs:F4}).");
        }

        // 4. Verify adaptive cushion & base cushion setters remain stable.
        provider.SetAdaptiveCushionMs(32);
        provider.SetBaseCushionMs(LowLatencyOutputPrebufferMs, LowLatencyOutputMaxBacklogMs);
    }

    private sealed class RingBufferWaveProvider : IWaveProvider
    {
        private const int CrossfadeSamples = 64; // ~1.33 ms @ 48 kHz linear anti-click ramp
        private const int HardRebufferEmptyCallbacks = 6; // >= ~60 ms continuous starvation

        private readonly AudioRingBuffer _ring;
        private readonly Func<long>? _onUnderrun;
        private readonly int _sampleRate;
        private readonly float[] _monoScratch = new float[16384];

        private int _basePrebufferSamples;
        private int _baseMaxBacklogSamples;
        private int _dynamicTargetBacklogSamples;
        private int _dynamicMaxBacklogSamples;
        private int _adaptiveExtraMs;

        private bool _isInitialPrebuffering = true;
        private bool _hasStartedStreaming;
        private int _consecutiveEmptyReads;
        private float _lastSampleValue;
        private int _fadeInSamplesRemaining;

        public RingBufferWaveProvider(
            AudioRingBuffer ring,
            WaveFormat waveFormat,
            Func<long>? onUnderrun,
            int prebufferMs = StandardOutputPrebufferMs,
            int maxBacklogMs = StandardOutputMaxBacklogMs)
        {
            _ring = ring;
            WaveFormat = waveFormat;
            _onUnderrun = onUnderrun;
            _sampleRate = Math.Max(1, waveFormat.SampleRate);
            SetBaseCushionMs(prebufferMs, maxBacklogMs);
        }

        public WaveFormat WaveFormat { get; }

        public void SetBaseCushionMs(int prebufferMs, int maxBacklogMs)
        {
            int maxSamples = Math.Max(1024, (_sampleRate * maxBacklogMs) / 1000);
            int prebufferSamples = Math.Clamp((_sampleRate * prebufferMs) / 1000, 256, maxSamples);
            Volatile.Write(ref _baseMaxBacklogSamples, maxSamples);
            Volatile.Write(ref _basePrebufferSamples, prebufferSamples);
            RecomputeDynamicWatermarks();
        }

        public void SetAdaptiveCushionMs(int extraMs)
        {
            Volatile.Write(ref _adaptiveExtraMs, Math.Clamp(extraMs, 0, 48));
            RecomputeDynamicWatermarks();
        }

        private void RecomputeDynamicWatermarks()
        {
            int basePre = Volatile.Read(ref _basePrebufferSamples);
            int baseMax = Volatile.Read(ref _baseMaxBacklogSamples);
            int extraMs = Volatile.Read(ref _adaptiveExtraMs);

            int maxFloor = Math.Max(1024, baseMax + (_sampleRate * 48) / 1000);
            int target = basePre + (_sampleRate * extraMs) / 1000;
            int max = Math.Max(target + (_sampleRate * 24) / 1000, baseMax + (_sampleRate * extraMs) / 1000);

            int clampedTarget = Math.Clamp(target, 256, maxFloor);
            int clampedMax = Math.Clamp(max, clampedTarget, maxFloor);

            Volatile.Write(ref _dynamicTargetBacklogSamples, clampedTarget);
            Volatile.Write(ref _dynamicMaxBacklogSamples, clampedMax);
        }

        public int Read(byte[] buffer, int offset, int count)
        {
            int blockAlign = Math.Max(1, WaveFormat.BlockAlign);
            int totalFrames = count / blockAlign;
            if (totalFrames <= 0)
            {
                return 0;
            }

            if (_isInitialPrebuffering)
            {
                int prebufferSamples = Volatile.Read(ref _basePrebufferSamples);
                if (_ring.AvailableRead < prebufferSamples)
                {
                    EmitBufferingSilence(buffer, offset, totalFrames, blockAlign);
                    return totalFrames * blockAlign;
                }

                _isInitialPrebuffering = false;
                _consecutiveEmptyReads = 0;
                _fadeInSamplesRemaining = CrossfadeSamples;
            }

            // Consumer-side Drop-Oldest watermark so output buffer never snowballs latency.
            int targetBacklog = Volatile.Read(ref _dynamicTargetBacklogSamples);
            int maxBacklog = Math.Max(targetBacklog, Volatile.Read(ref _dynamicMaxBacklogSamples));
            int avail = _ring.AvailableRead;
            if (avail > maxBacklog)
            {
                _ring.Skip(avail - targetBacklog);
            }

            int framesRemaining = totalFrames;
            int currentByteOffset = offset;

            while (framesRemaining > 0)
            {
                int chunk = Math.Min(framesRemaining, _monoScratch.Length);
                Span<float> span = _monoScratch.AsSpan(0, chunk);
                int read = _ring.Read(span);

                if (read > 0)
                {
                    _hasStartedStreaming = true;
                    _consecutiveEmptyReads = 0;
                    if (_fadeInSamplesRemaining > 0)
                    {
                        ApplyFadeIn(span.Slice(0, read));
                    }
                }

                if (read < chunk)
                {
                    float tailStart = read > 0 ? span[read - 1] : _lastSampleValue;
                    ApplyFadeOutAndClear(span.Slice(read), tailStart);

                    _lastSampleValue = 0f;
                    _fadeInSamplesRemaining = CrossfadeSamples;

                    if (read == 0)
                    {
                        _consecutiveEmptyReads++;
                        if (_hasStartedStreaming && _consecutiveEmptyReads >= HardRebufferEmptyCallbacks)
                        {
                            _isInitialPrebuffering = true;
                        }
                    }

                    if (_hasStartedStreaming)
                    {
                        _onUnderrun?.Invoke();
                    }
                }
                else if (read > 0)
                {
                    _lastSampleValue = span[read - 1];
                }

                int chunkBytes = chunk * blockAlign;
                WasapiPcmConverter.WriteMonoFloats(span, WaveFormat, buffer.AsSpan(currentByteOffset, chunkBytes));

                currentByteOffset += chunkBytes;
                framesRemaining -= chunk;
            }

            return totalFrames * blockAlign;
        }

        private void EmitBufferingSilence(byte[] buffer, int offset, int totalFrames, int blockAlign)
        {
            int framesRemaining = totalFrames;
            int currentByteOffset = offset;

            while (framesRemaining > 0)
            {
                int chunk = Math.Min(framesRemaining, _monoScratch.Length);
                Span<float> span = _monoScratch.AsSpan(0, chunk);
                if (_lastSampleValue != 0f)
                {
                    ApplyFadeOutAndClear(span, _lastSampleValue);
                    _lastSampleValue = 0f;
                }
                else
                {
                    span.Clear();
                }

                int chunkBytes = chunk * blockAlign;
                WasapiPcmConverter.WriteMonoFloats(span, WaveFormat, buffer.AsSpan(currentByteOffset, chunkBytes));

                currentByteOffset += chunkBytes;
                framesRemaining -= chunk;
            }
        }

        private void ApplyFadeIn(Span<float> samples)
        {
            int count = Math.Min(samples.Length, _fadeInSamplesRemaining);
            int startStep = CrossfadeSamples - _fadeInSamplesRemaining;
            float anchor = _lastSampleValue;

            for (int i = 0; i < count; i++)
            {
                float alpha = (startStep + i + 1) / (float)CrossfadeSamples;
                samples[i] = anchor * (1f - alpha) + samples[i] * alpha;
            }

            _fadeInSamplesRemaining -= count;
        }

        private static void ApplyFadeOutAndClear(Span<float> gap, float startSample)
        {
            if (gap.IsEmpty)
            {
                return;
            }

            if (startSample == 0f)
            {
                gap.Clear();
                return;
            }

            int fadeCount = Math.Min(gap.Length, CrossfadeSamples);
            for (int i = 0; i < fadeCount; i++)
            {
                float gain = 1f - ((i + 1) / (float)fadeCount);
                gap[i] = startSample * gain;
            }

            if (fadeCount < gap.Length)
            {
                gap.Slice(fadeCount).Clear();
            }
        }
    }
}
