using System.Diagnostics;
using System.Reflection;
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
    private const int CaptureBufferMs = 25;
    private const int RenderLatencyMs = 45;
    private const int OutputPrebufferMs = 54;       // ~1.7x of 32 ms VoxLock chunk (50-60 ms jitter cushion)
    private const int OutputMaxBacklogMs = 150;
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
    private readonly float[] _primeSilenceScratch = new float[4096];

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
    private volatile string _activeBackendDescription = "CPU";

    // Runtime state
    private WasapiCapture? _capture;
    private WasapiOut? _primaryOut;
    private WasapiOut? _monitorOut;
    private MMDevice? _captureDevice;
    private MMDevice? _primaryRenderDevice;
    private MMDevice? _monitorRenderDevice;
    private string? _primaryRenderDeviceId;
    private bool _monitorSharesPrimaryDevice;
    private int _primaryRenderRate = 48000;
    private int _monitorRenderRate = 48000;

    private Thread? _dspThread;
    private CancellationTokenSource? _runCts;
    private volatile bool _isRunning;
    private bool _disposed;

    // Telemetry & counters
    private long _bufferOverruns;
    private long _bufferUnderruns;
    private TelemetrySnapshot _latestTelemetry = TelemetrySnapshot.Idle;
    private string _activeCoreDescription = "Unbound";
    private long _activeCoreMask;

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
        _activeBackendDescription = settings.ComputeBackend == ComputeBackend.Cuda ? "CUDA / CPU Fallback" : "CPU";

        UpdateCpuAffinity(settings.AutoAffinityEnabled, settings.SelectedCoreMask, settings.SelectedProcessorGroup);
        SetMonitorLoopback(settings.MonitorLoopbackEnabled);
    }

    public void UpdateCpuAffinity(bool autoAffinityEnabled, ulong selectedCoreMask, ushort processorGroup = 0)
    {
        _autoAffinityEnabled = autoAffinityEnabled;
        Interlocked.Exchange(ref _selectedCoreMask, unchecked((long)selectedCoreMask));
        Volatile.Write(ref _selectedProcessorGroup, processorGroup);
        _affinityUpdatePending = true;
        _dspWorkSignal.Set();
    }

    public void SetActiveBackendDescription(string description)
    {
        if (!string.IsNullOrWhiteSpace(description))
        {
            _activeBackendDescription = description;
        }
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
                _inputRing16k.Reset();
                _primaryOutRing.Reset();
                _monitorOutRing.Reset();
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

                _capture = new WasapiCapture(_captureDevice, useEventSync: false, audioBufferMillisecondsLength: CaptureBufferMs);
                _primaryOut = new WasapiOut(_primaryRenderDevice, AudioClientShareMode.Shared, useEventSync: false, latency: RenderLatencyMs);

                _primaryRenderRate = _primaryOut.OutputWaveFormat.SampleRate;
                _inputResampler.Configure(_capture.WaveFormat.SampleRate, DspSampleRate);
                _primaryOutResampler.Configure(DspSampleRate, _primaryRenderRate);

                var primaryProvider = new RingBufferWaveProvider(
                    _primaryOutRing,
                    _primaryOut.OutputWaveFormat,
                    () => Interlocked.Increment(ref _bufferUnderruns),
                    prebufferMs: OutputPrebufferMs,
                    maxBacklogMs: OutputMaxBacklogMs,
                    targetBacklogMs: OutputPrebufferMs);

                _primaryOut.Init(primaryProvider);

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

                _primaryOut.Play();
                _capture.StartRecording();
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

        _captureDevice?.Dispose();
        _captureDevice = null;
        _primaryRenderDevice?.Dispose();
        _primaryRenderDevice = null;
        _primaryRenderDeviceId = null;
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
            _monitorOut = new WasapiOut(_monitorRenderDevice, AudioClientShareMode.Shared, useEventSync: false, latency: RenderLatencyMs);
            _monitorRenderRate = _monitorOut.OutputWaveFormat.SampleRate;
            _monitorOutResampler.Configure(DspSampleRate, _monitorRenderRate);
            _monitorOutRing.Reset();

            var monitorProvider = new RingBufferWaveProvider(
                _monitorOutRing,
                _monitorOut.OutputWaveFormat,
                null,
                prebufferMs: OutputPrebufferMs,
                maxBacklogMs: OutputMaxBacklogMs,
                targetBacklogMs: OutputPrebufferMs);

            _monitorOut.Init(monitorProvider);
            _monitorOut.Play();
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
        try
        {
            ApplyAffinityOnDspThread();
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

        try
        {
            while (!token.IsCancellationRequested && _isRunning)
            {
                if (_affinityUpdatePending)
                {
                    _affinityUpdatePending = false;
                    ApplyAffinityOnDspThread();
                }

                if (_inputRing16k.AvailableRead < HopSizeSamples)
                {
                    _dspWorkSignal.WaitOne(16);
                    continue;
                }

                while (_inputRing16k.AvailableRead >= HopSizeSamples && !token.IsCancellationRequested && _isRunning)
                {
                    // Max-latency watermark ("Drop Oldest"): absorb 25 ms polling jitter without false drops.
                    bool voxConfigured = _voxLockEnabled && _voxLockBackend is { IsInitialized: true, HasEnrolledVoiceprint: true };
                    int voxFrameSizeCheck = voxConfigured
                        ? Math.Clamp(_voxLockBackend!.FrameSizeSamples, HopSizeSamples, _voxOutScratch.Length)
                        : HopSizeSamples;
                    int maxInputBacklog = voxConfigured ? Math.Max(1536, voxFrameSizeCheck * 3) : 1024;
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
                            _voxLockBackend?.ResetState();
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

                    bool voxActive = _voxLockEnabled && _voxLockBackend is { IsInitialized: true, HasEnrolledVoiceprint: true };
                    int voxFrameSize = voxActive
                        ? Math.Clamp(_voxLockBackend!.FrameSizeSamples, HopSizeSamples, _voxOutScratch.Length)
                        : HopSizeSamples;
                    bool voxRanThisHop = false;

                    // Downstream cascade: Silero Gate -> GTCRN (256 hop) -> VoxLock (512 / 32 ms chunk)
                    if (vadEnabled && !vadSpeechActive)
                    {
                        voxStagingCount = 0;
                        Array.Clear(_hopOutput256);
                        lastOutputRms = 0f;
                        lastOutputPeak = 0f;
                        WriteOutputToRings(_hopOutput256);
                    }
                    else
                    {
                        ReadOnlySpan<float> current = _hopInput256;

                        if (_gtcrnEnabled && _gtcrnStage is { IsInitialized: true })
                        {
                            long t0 = Stopwatch.GetTimestamp();
                            _gtcrnStage.ProcessFrame(current, _gtcrnOut256);
                            gtcrnInfMs = (Stopwatch.GetTimestamp() - t0) * ticksToMs;
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

                                lastOutputRms = WasapiPcmConverter.ComputeRmsAndPeak(voxOutSpan, out lastOutputPeak);
                                WriteOutputToRings(voxOutSpan);
                            }
                        }
                        else
                        {
                            voxStagingCount = 0;
                            current.CopyTo(_hopOutput256);
                            lastOutputRms = WasapiPcmConverter.ComputeRmsAndPeak(_hopOutput256, out lastOutputPeak);
                            WriteOutputToRings(_hopOutput256);
                        }
                    }

                    double totalInfMs = (Stopwatch.GetTimestamp() - frameStartTick) * ticksToMs;
                    if (!voxActive || voxRanThisHop || (vadEnabled && !vadSpeechActive))
                    {
                        smoothedTotalInfMs = smoothedTotalInfMs == 0.0 ? totalInfMs : smoothedTotalInfMs * 0.8 + totalInfMs * 0.2;
                    }
                    if (vadInfMs > 0.0)
                    {
                        smoothedVadInfMs = smoothedVadInfMs == 0.0 ? vadInfMs : smoothedVadInfMs * 0.7 + vadInfMs * 0.3;
                    }
                    smoothedGtcrnInfMs = _gtcrnEnabled ? (smoothedGtcrnInfMs == 0.0 ? gtcrnInfMs : smoothedGtcrnInfMs * 0.8 + gtcrnInfMs * 0.2) : 0.0;
                    if (voxActive)
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
                        double algorithmicLatencyMs = ComputeTotalLatencyMs();
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
            PublishTelemetry(LatestTelemetry with
            {
                TimestampUtc = DateTimeOffset.UtcNow,
                State = PipelineRunState.Faulted,
                StatusText = $"DSP error: {ex.Message}"
            }, forceEvent: true);

            PipelineFaulted?.Invoke(this, ex);
        }
    }

    private void ApplyAffinityOnDspThread()
    {
        ulong mask;
        ushort group;
        if (_autoAffinityEnabled)
        {
            mask = _affinityManager.GetRecommendedAudioWorkerMask(out group, out _);
        }
        else
        {
            mask = unchecked((ulong)Interlocked.Read(ref _selectedCoreMask));
            group = (ushort)Math.Clamp(Volatile.Read(ref _selectedProcessorGroup), 0, ushort.MaxValue);
            if (mask == 0)
            {
                mask = _affinityManager.GetRecommendedAudioWorkerMask(out group, out _);
            }
        }

        AffinityBindResult result = _affinityManager.BindCurrentThread(mask, ThreadPriority.AboveNormal, group);
        Volatile.Write(ref _activeCoreDescription, result.Description);
        Volatile.Write(ref _activeCoreMask, unchecked((long)result.AppliedMask));
    }

    private void WriteOutputToRings(ReadOnlySpan<float> samples16k)
    {
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

    private double ComputeTotalLatencyMs()
    {
        double algoMs = _inputResampler.AlgorithmicDelayMs + _primaryOutResampler.AlgorithmicDelayMs;

        // Framing + STFT overlap latency:
        // - VoxLock active: 24 ms STFT OLA + chunk frame (32 ms at K=4), where 16 ms hops accumulate inside the frame.
        // - GTCRN only: 16 ms hop framing + 16 ms STFT lookahead.
        // - Bypass / VAD only: 16 ms hop framing.
        if (_voxLockEnabled && _voxLockBackend is { IsInitialized: true, HasEnrolledVoiceprint: true })
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

    private sealed class RingBufferWaveProvider : IWaveProvider
    {
        private const int CrossfadeSamples = 64; // ~1.33 ms @ 48 kHz linear anti-click ramp

        private readonly AudioRingBuffer _ring;
        private readonly Func<long>? _onUnderrun;
        private readonly int _prebufferSamples;
        private readonly int _maxBacklogSamples;
        private readonly int _targetBacklogSamples;
        private readonly float[] _monoScratch = new float[16384];

        private bool _isOutputBuffering = true;
        private bool _hasStartedStreaming;
        private float _lastSampleValue;
        private int _fadeInSamplesRemaining;

        public RingBufferWaveProvider(
            AudioRingBuffer ring,
            WaveFormat waveFormat,
            Func<long>? onUnderrun,
            int prebufferMs = OutputPrebufferMs,
            int maxBacklogMs = OutputMaxBacklogMs,
            int targetBacklogMs = OutputPrebufferMs)
        {
            _ring = ring;
            WaveFormat = waveFormat;
            _onUnderrun = onUnderrun;
            _maxBacklogSamples = Math.Max(1024, (waveFormat.SampleRate * maxBacklogMs) / 1000);
            _prebufferSamples = Math.Clamp((waveFormat.SampleRate * prebufferMs) / 1000, 256, _maxBacklogSamples);
            _targetBacklogSamples = Math.Clamp((waveFormat.SampleRate * targetBacklogMs) / 1000, _prebufferSamples, _maxBacklogSamples);
        }

        public WaveFormat WaveFormat { get; }

        public int Read(byte[] buffer, int offset, int count)
        {
            int blockAlign = Math.Max(1, WaveFormat.BlockAlign);
            int totalFrames = count / blockAlign;
            if (totalFrames <= 0)
            {
                return 0;
            }

            if (_isOutputBuffering)
            {
                if (_ring.AvailableRead < _prebufferSamples)
                {
                    EmitBufferingSilence(buffer, offset, totalFrames, blockAlign);
                    return totalFrames * blockAlign;
                }

                _isOutputBuffering = false;
                _fadeInSamplesRemaining = CrossfadeSamples;
            }

            // Consumer-side Drop-Oldest watermark so output buffer never snowballs latency.
            int avail = _ring.AvailableRead;
            if (avail > _maxBacklogSamples)
            {
                _ring.Skip(avail - _targetBacklogSamples);
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
                    _isOutputBuffering = true;
                    _fadeInSamplesRemaining = CrossfadeSamples;

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
