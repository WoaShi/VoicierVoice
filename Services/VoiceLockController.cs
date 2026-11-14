using System.IO;
using VoiceLockStudio.Models;
using VoiceLockStudio.Services.Audio;
using VoiceLockStudio.Services.Backends;
using VoiceLockStudio.Services.Localization;
using VoiceLockStudio.Services.System;

namespace VoiceLockStudio.Services;

/// <summary>
/// Central application orchestrator owning the ONNX stages, WASAPI audio pipeline,
/// voiceprint recorder/migrator, CPU affinity manager, and persisted settings.
/// </summary>
public sealed class VoiceLockController : IDisposable
{
    private readonly string _modelsRootDirectory;
    private readonly SemaphoreSlim _operationGate = new(1, 1);

    private readonly SileroVadStage _vadStage = new();
    private readonly GtcrnDenoiserStage _gtcrnStage = new();
    private readonly VoxLockOnnxBackend _voxLockBackend = new();
    private readonly AudioDeviceService _deviceService = new();
    private readonly CpuAffinityManager _affinityManager = new();
    private readonly VoiceprintRecorderService _recorderService;
    private readonly AudioPipelineEngine _pipelineEngine;

    private ModelManifest? _manifest;
    private AppSettings _settings;
    private bool _isInitialized;
    private bool _disposed;

    public VoiceLockController(string? modelsRootDirectory = null)
    {
        _modelsRootDirectory = string.IsNullOrWhiteSpace(modelsRootDirectory)
            ? AppContext.BaseDirectory
            : modelsRootDirectory;

        _settings = AppSettings.Load();
        _recorderService = new VoiceprintRecorderService(_deviceService);
        _pipelineEngine = new AudioPipelineEngine(
            _deviceService,
            _affinityManager,
            _vadStage,
            _gtcrnStage,
            _voxLockBackend);

        _pipelineEngine.TelemetryUpdated += (_, snapshot) => TelemetryUpdated?.Invoke(this, snapshot);
        _pipelineEngine.PipelineFaulted += (_, ex) => PipelineFaulted?.Invoke(this, ex);
        _recorderService.RecordingProgress += (_, progress) => RecordingProgress?.Invoke(this, progress);
        _recorderService.PlaybackStopped += (_, _) => PlaybackStopped?.Invoke(this, EventArgs.Empty);
    }

    public AppSettings Settings => _settings;
    public ModelManifest? Manifest => _manifest;
    public CpuTopologyInfo CpuTopology => _affinityManager.GetTopology();
    public bool IsInitialized => _isInitialized;
    public bool IsPipelineRunning => _pipelineEngine.IsRunning;
    public bool IsRecording => _recorderService.IsRecording;
    public bool IsPlayingSample => _recorderService.IsPlaying;
    public TelemetrySnapshot LatestTelemetry => _pipelineEngine.LatestTelemetry;
    public VoiceprintProfileInfo? CurrentVoiceprint => _voxLockBackend.CurrentVoiceprint;
    public bool HasEnrolledVoiceprint => _voxLockBackend.HasEnrolledVoiceprint;

    public string ActiveProviderSummary { get; private set; } = AppStrings.Current.FormatBackendBanner(ComputeBackend.Cpu);
    public string? ProviderFallbackWarning { get; private set; }
    public bool IsCudaFallbackActive { get; private set; }

    public string VadActiveProvider => _vadStage.ActiveProviderName;
    public string GtcrnActiveProvider => _gtcrnStage.ActiveProviderName;
    public string VoxLockActiveProvider => _voxLockBackend.ActiveProviderName;

    public event EventHandler<TelemetrySnapshot>? TelemetryUpdated;
    public event EventHandler<Exception>? PipelineFaulted;
    public event EventHandler<VoiceprintRecordingProgress>? RecordingProgress;
    public event EventHandler? PlaybackStopped;
    public event EventHandler? VoiceprintChanged;
    public event EventHandler? BackendStateChanged;

    /// <summary>
    /// Loads the model manifest, initializes all DSP/ONNX stages, and silently auto-migrates
    /// any existing <c>user_raw_voice.wav</c> into the current model variant's binary cache.
    /// </summary>
    public async Task InitializeAsync(CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        await _operationGate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            if (_isInitialized)
            {
                return;
            }

            await Task.Run(() =>
            {
                cancellationToken.ThrowIfCancellationRequested();
                _manifest ??= ModelManifest.LoadDefault(_modelsRootDirectory);
                InitializeStagesLocked();
                _pipelineEngine.ApplySettings(_settings);
                _pipelineEngine.SetActiveBackendDescription(ActiveProviderSummary);
                _isInitialized = true;
            }, cancellationToken).ConfigureAwait(false);

            // Auto-migrate Golden Voiceprint silently on background thread if wav lives at custom or default path.
            await EnsureVoiceprintMigratedCoreAsync(forceRebuild: false, cancellationToken).ConfigureAwait(false);

            BackendStateChanged?.Invoke(this, EventArgs.Empty);
            VoiceprintChanged?.Invoke(this, EventArgs.Empty);
        }
        finally
        {
            _operationGate.Release();
        }
    }

    public IReadOnlyList<AudioDeviceInfo> GetCaptureDevices() => _deviceService.GetCaptureDevices();

    public IReadOnlyList<AudioDeviceInfo> GetRenderDevices() => _deviceService.GetRenderDevices();

    public async Task StartPipelineAsync(CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        if (!_isInitialized)
        {
            await InitializeAsync(cancellationToken).ConfigureAwait(false);
        }

        await _operationGate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            if (_pipelineEngine.IsRunning)
            {
                return;
            }

            await Task.Run(() =>
            {
                _pipelineEngine.Start(_settings);
                _pipelineEngine.SetActiveBackendDescription(ActiveProviderSummary);
            }, cancellationToken).ConfigureAwait(false);
        }
        finally
        {
            _operationGate.Release();
        }
    }

    public async Task StopPipelineAsync(CancellationToken cancellationToken = default)
    {
        if (_disposed)
        {
            return;
        }

        await _operationGate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            if (!_pipelineEngine.IsRunning)
            {
                return;
            }

            await Task.Run(() => _pipelineEngine.Stop(), cancellationToken).ConfigureAwait(false);
        }
        finally
        {
            _operationGate.Release();
        }
    }

    public async Task TogglePipelineAsync(CancellationToken cancellationToken = default)
    {
        if (_pipelineEngine.IsRunning)
        {
            await StopPipelineAsync(cancellationToken).ConfigureAwait(false);
        }
        else
        {
            await StartPipelineAsync(cancellationToken).ConfigureAwait(false);
        }
    }

    public async Task SetDevicesAsync(
        string? inputDeviceId,
        string? outputDeviceId,
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        await _operationGate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            bool changed = !string.Equals(_settings.SelectedInputDeviceId, inputDeviceId, StringComparison.OrdinalIgnoreCase)
                || !string.Equals(_settings.SelectedOutputDeviceId, outputDeviceId, StringComparison.OrdinalIgnoreCase);

            _settings.SelectedInputDeviceId = inputDeviceId;
            _settings.SelectedOutputDeviceId = outputDeviceId;
            _settings.Save();

            if (changed && _pipelineEngine.IsRunning)
            {
                await Task.Run(() =>
                {
                    _pipelineEngine.Stop();
                    _pipelineEngine.Start(_settings);
                    _pipelineEngine.SetActiveBackendDescription(ActiveProviderSummary);
                }, cancellationToken).ConfigureAwait(false);
            }
        }
        finally
        {
            _operationGate.Release();
        }
    }

    public void UpdateMonitorLoopback(bool enabled)
    {
        _settings.MonitorLoopbackEnabled = enabled;
        _settings.Save();
        _pipelineEngine.SetMonitorLoopback(enabled);
    }

    public void UpdateStageFlags(bool vadEnabled, float vadThreshold, bool gtcrnEnabled, bool voxLockEnabled)
    {
        _settings.VadEnabled = vadEnabled;
        _settings.VadThreshold = Math.Clamp(vadThreshold, 0.01f, 0.99f);
        _settings.GtcrnEnabled = gtcrnEnabled;
        _settings.VoxLockEnabled = voxLockEnabled;
        _settings.Save();

        _pipelineEngine.ApplySettings(_settings);
        _pipelineEngine.SetActiveBackendDescription(ActiveProviderSummary);
    }

    public void UpdateCpuAffinity(bool autoAffinityEnabled, ulong selectedCoreMask, ushort processorGroup = 0)
    {
        _settings.AutoAffinityEnabled = autoAffinityEnabled;
        _settings.SelectedCoreMask = selectedCoreMask;
        _settings.SelectedProcessorGroup = processorGroup;
        _settings.Save();

        _pipelineEngine.UpdateCpuAffinity(autoAffinityEnabled, selectedCoreMask, processorGroup);
    }

    public void UpdateAppPreferences(
        AppLanguage language,
        AppThemePreference themePreference,
        bool minimizeToTrayEnabled,
        bool? launchOnStartupEnabled = null)
    {
        _settings.Language = language;
        _settings.ThemePreference = themePreference;
        _settings.MinimizeToTrayEnabled = minimizeToTrayEnabled;
        if (launchOnStartupEnabled.HasValue)
        {
            _settings.LaunchOnStartupEnabled = launchOnStartupEnabled.Value;
        }
        _settings.Save();
    }

    /// <summary>
    /// Switches the ONNX compute backend (CPU/CUDA) and/or VoxLock model variant (INT8/INT8Full/FP16),
    /// cleanly pausing and resuming the audio pipeline if active and auto-migrating the voiceprint cache.
    /// </summary>
    public async Task SwitchComputeOrVariantAsync(
        ComputeBackend backend,
        VoxLockModelVariant variant,
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        await _operationGate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            if (_isInitialized &&
                _settings.ComputeBackend == backend &&
                _settings.VoxLockVariant == variant)
            {
                return;
            }

            _settings.ComputeBackend = backend;
            _settings.VoxLockVariant = variant;
            _settings.Save();

            bool wasRunning = _pipelineEngine.IsRunning;

            await Task.Run(() =>
            {
                if (wasRunning)
                {
                    _pipelineEngine.Stop();
                }

                _manifest ??= ModelManifest.LoadDefault(_modelsRootDirectory);
                InitializeStagesLocked();
                _pipelineEngine.SetStages(_vadStage, _gtcrnStage, _voxLockBackend);
                _pipelineEngine.ApplySettings(_settings);
                _pipelineEngine.SetActiveBackendDescription(ActiveProviderSummary);
                _isInitialized = true;
            }, cancellationToken).ConfigureAwait(false);

            // Auto-migrate Golden Voiceprint for the newly selected variant on background thread.
            await EnsureVoiceprintMigratedCoreAsync(forceRebuild: false, cancellationToken).ConfigureAwait(false);

            if (wasRunning)
            {
                await Task.Run(() =>
                {
                    _pipelineEngine.Start(_settings);
                    _pipelineEngine.SetActiveBackendDescription(ActiveProviderSummary);
                }, cancellationToken).ConfigureAwait(false);
            }

            BackendStateChanged?.Invoke(this, EventArgs.Empty);
            VoiceprintChanged?.Invoke(this, EventArgs.Empty);
        }
        finally
        {
            _operationGate.Release();
        }
    }

    public async Task<VoiceprintProfileInfo?> RebuildVoiceprintCacheAsync(
        bool forceRebuild = true,
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        if (!_isInitialized)
        {
            await InitializeAsync(cancellationToken).ConfigureAwait(false);
        }

        await _operationGate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            VoiceprintProfileInfo? profile = await EnsureVoiceprintMigratedCoreAsync(forceRebuild, cancellationToken)
                .ConfigureAwait(false);
            VoiceprintChanged?.Invoke(this, EventArgs.Empty);
            return profile;
        }
        finally
        {
            _operationGate.Release();
        }
    }

    public async Task<VoiceprintProfileInfo> RecordAndEnrollVoiceprintAsync(
        double durationSeconds = 5.0,
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        if (!_isInitialized)
        {
            await InitializeAsync(cancellationToken).ConfigureAwait(false);
        }

        string wavPath = await _recorderService
            .RecordAsync(_settings.SelectedInputDeviceId, durationSeconds, cancellationToken)
            .ConfigureAwait(false);

        await _operationGate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            VoiceprintProfileInfo profile = await Task.Run(() =>
            {
                VoiceprintProfileInfo enrolled = _voxLockBackend.EnrollVoiceprint(wavPath);
                _settings.VoiceprintWavPath = enrolled.SourceWavPath;
                _settings.Save();
                return enrolled;
            }, cancellationToken).ConfigureAwait(false);

            VoiceprintChanged?.Invoke(this, EventArgs.Empty);
            return profile;
        }
        finally
        {
            _operationGate.Release();
        }
    }

    public void StopVoiceprintRecording() => _recorderService.StopRecording();

    public void StartSamplePlayback()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        string? wavPath = ResolveExistingVoiceprintWavPath();
        if (wavPath is null)
        {
            throw new FileNotFoundException("尚未录制任何参考声纹文件。", VoiceprintStorage.DefaultUserRawVoiceWavPath);
        }

        _recorderService.StartPlayback(wavPath, _settings.SelectedOutputDeviceId);
    }

    public void StopSamplePlayback() => _recorderService.StopPlayback();

    public string? ResolveExistingVoiceprintWavPath()
    {
        if (File.Exists(VoiceprintStorage.DefaultUserRawVoiceWavPath))
        {
            return VoiceprintStorage.DefaultUserRawVoiceWavPath;
        }

        if (!string.IsNullOrWhiteSpace(_settings.VoiceprintWavPath) && File.Exists(_settings.VoiceprintWavPath))
        {
            return _settings.VoiceprintWavPath;
        }

        return null;
    }

    private void InitializeStagesLocked()
    {
        ModelManifest manifest = _manifest!;

        ulong affinityMask;
        ushort affinityGroup;
        if (_settings.AutoAffinityEnabled || _settings.SelectedCoreMask == 0)
        {
            affinityMask = _affinityManager.GetRecommendedAudioWorkerMask(out affinityGroup, out _);
        }
        else
        {
            affinityMask = _settings.SelectedCoreMask;
            affinityGroup = _settings.SelectedProcessorGroup;
        }

        // Benchmark #17 showed lightweight Silero-VAD + GTCRN run in <1.8 ms on CPU vs ~12.7 ms on CUDA
        // due to kernel launch overhead, whereas VoxLock accelerates on CUDA or 2-3 pinned E-Cores.
        // Keep VAD + GTCRN on CPU while routing VoxLock to the user-selected backend (CPU or CUDA).
        var vadGtcrnOptions = new AudioStageOptions(
            ModelsRootDirectory: _modelsRootDirectory,
            Manifest: manifest,
            Backend: ComputeBackend.Cpu,
            VoxLockVariant: _settings.VoxLockVariant,
            VadThreshold: _settings.VadThreshold,
            CpuAffinityMask: affinityMask,
            ProcessorGroup: affinityGroup);

        if (!_vadStage.IsInitialized)
        {
            _vadStage.Initialize(vadGtcrnOptions);
        }

        if (!_gtcrnStage.IsInitialized)
        {
            _gtcrnStage.Initialize(vadGtcrnOptions);
        }

        var voxLockOptions = new AudioStageOptions(
            ModelsRootDirectory: _modelsRootDirectory,
            Manifest: manifest,
            Backend: _settings.ComputeBackend,
            VoxLockVariant: _settings.VoxLockVariant,
            VadThreshold: _settings.VadThreshold,
            CpuAffinityMask: affinityMask,
            ProcessorGroup: affinityGroup);

        _voxLockBackend.Initialize(voxLockOptions);

        UpdateActiveProviderSummary();
    }

    private void UpdateActiveProviderSummary()
    {
        if (_settings.ComputeBackend == ComputeBackend.Cuda)
        {
            bool cudaActive = string.Equals(_voxLockBackend.ActiveProviderName, "CUDAExecutionProvider", StringComparison.OrdinalIgnoreCase);
            if (cudaActive && string.IsNullOrEmpty(_voxLockBackend.ProviderFallbackMessage))
            {
                IsCudaFallbackActive = false;
                ActiveProviderSummary = AppStrings.Current.FormatBackendBanner(ComputeBackend.Cuda, isFallback: false);
                ProviderFallbackWarning = null;
            }
            else
            {
                IsCudaFallbackActive = true;
                ActiveProviderSummary = AppStrings.Current.FormatBackendBanner(ComputeBackend.Cuda, isFallback: true);
                ProviderFallbackWarning = _voxLockBackend.ProviderFallbackMessage
                    ?? AppStrings.Current.FormatCudaFallbackDefault();
            }
        }
        else
        {
            IsCudaFallbackActive = false;
            ActiveProviderSummary = AppStrings.Current.FormatBackendBanner(ComputeBackend.Cpu, isFallback: false);
            ProviderFallbackWarning = null;
        }
    }

    private async Task<VoiceprintProfileInfo?> EnsureVoiceprintMigratedCoreAsync(
        bool forceRebuild,
        CancellationToken cancellationToken)
    {
        string? wavPath = ResolveExistingVoiceprintWavPath();
        if (wavPath is null)
        {
            return null;
        }

        if (!forceRebuild && _voxLockBackend.HasEnrolledVoiceprint && _voxLockBackend.CurrentVoiceprint is not null)
        {
            return _voxLockBackend.CurrentVoiceprint;
        }

        return await Task.Run(() =>
        {
            cancellationToken.ThrowIfCancellationRequested();
            return _voxLockBackend.LoadOrRebuildVoiceprintCache(wavPath, forceRebuild);
        }, cancellationToken).ConfigureAwait(false);
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        _pipelineEngine.Dispose();
        _recorderService.Dispose();
        _voxLockBackend.Dispose();
        _gtcrnStage.Dispose();
        _vadStage.Dispose();
        _operationGate.Dispose();
    }
}
