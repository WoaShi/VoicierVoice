using System.Collections.ObjectModel;
using System.Diagnostics;
using System.IO;
using System.Windows;
using System.Windows.Threading;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using VoiceLockStudio.Models;
using VoiceLockStudio.Services;
using VoiceLockStudio.Services.Audio;
using VoiceLockStudio.Services.Localization;
using VoiceLockStudio.Services.System;

namespace VoiceLockStudio.ViewModels;

public enum NavigationTab
{
    Pipeline,
    Voiceprint,
    Compute,
    OtherSettings
}

public enum GuidanceTarget
{
    None,
    AudioRouting,
    VoiceprintCapture,
    MasterSwitch
}

public enum PipelineStatusKind
{
    Idle,
    Running,
    Sleeping,
    Faulted
}

internal enum RecordingMessageKind
{
    Idle,
    Stopping,
    Completed,
    Canceled,
    Rebuilding,
    Rebuilt,
    RecordingFailed,
    PlaybackFailed,
    RebuildFailed
}

public sealed partial class CpuCoreItemViewModel : ObservableObject
{
    public CpuCoreItemViewModel(PhysicalCoreInfo core)
    {
        PhysicalIndex = core.CoreIndex;
        LogicalIndex = core.LogicalProcessorIndices.Count > 0 ? core.LogicalProcessorIndices[0] : core.CoreIndex;
        ProcessorGroup = core.ProcessorGroup;
        IsEfficiencyCore = core.IsEfficiencyCore;
        IsRecommended = core.IsRecommended;
        BitMask = core.LogicalProcessorMask;
        Label = $"C{core.CoreIndex} {(core.IsEfficiencyCore ? "E" : "P")}";
        TooltipText = core.DisplayName;
        _coreTypeLabel = ResolveRoleLabel(core.IsEfficiencyCore, core.IsRecommended);
    }

    public int LogicalIndex { get; }
    public int PhysicalIndex { get; }
    public ushort ProcessorGroup { get; }
    public string Label { get; }
    public bool IsEfficiencyCore { get; }
    public bool IsRecommended { get; }
    public ulong BitMask { get; }
    public string TooltipText { get; }

    [ObservableProperty]
    private string _coreTypeLabel;

    [ObservableProperty]
    private bool _isSelected;

    public void RefreshLocalization()
    {
        CoreTypeLabel = ResolveRoleLabel(IsEfficiencyCore, IsRecommended);
    }

    private static string ResolveRoleLabel(bool isEfficiency, bool isRecommended)
    {
        if (isEfficiency)
        {
            return AppStrings.Current.CoreRoleEfficiency;
        }

        return isRecommended ? AppStrings.Current.CoreRoleTail : AppStrings.Current.CoreRolePerformance;
    }
}

public sealed partial class MainViewModel : ObservableObject, IDisposable
{
    private readonly VoiceLockController _controller;
    private readonly Dispatcher _dispatcher;

    private bool _isHydrating;
    private bool _updatingPipelineFlagFromTelemetry;
    private bool _updatingBackendFlags;
    private bool _disposed;

    private TelemetrySnapshot? _latestSnapshot;
    private string? _latestActiveCoreRawDescription;
    private string? _lastStatusError;
    private RecordingMessageKind _recordingMessageKind = RecordingMessageKind.Idle;
    private double _lastCompletedDurationSeconds;
    private string? _lastRecordingError;

    // Navigation & Guidance
    private readonly Stack<NavigationTab> _navigationHistory = new();
    private NavigationTab _lastRecordedTab = NavigationTab.Pipeline;
    private bool _isNavigatingBack;

    [ObservableProperty]
    private NavigationTab _activeTab = NavigationTab.Pipeline;

    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(NavigateBackCommand))]
    private bool _canNavigateBack;

    [ObservableProperty]
    private bool _showQuickStartGuide = true;

    // Status & Telemetry
    [ObservableProperty]
    private bool _isPipelineRunning;

    [ObservableProperty]
    private bool _isBusy;

    [ObservableProperty]
    private string _statusBadgeText = string.Empty;

    [ObservableProperty]
    private PipelineStatusKind _statusKind = PipelineStatusKind.Idle;

    [ObservableProperty]
    private string _telemetryLatencyText = "--";

    [ObservableProperty]
    private string _telemetryInferenceText = "--";

    [ObservableProperty]
    private string _telemetryCoreText = string.Empty;

    [ObservableProperty]
    private double _inputLevelPercent;

    [ObservableProperty]
    private double _outputLevelPercent;

    [ObservableProperty]
    private string _backendStatusBanner = string.Empty;

    // Voiceprint Profile
    [ObservableProperty]
    private string _promptReadText = string.Empty;

    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(RebuildCacheCommand))]
    [NotifyCanExecuteChangedFor(nameof(PlaySampleCommand))]
    private bool _isRecording;

    [ObservableProperty]
    private double _recordingProgress;

    [ObservableProperty]
    private double _recordingLevel;

    [ObservableProperty]
    private string _recordingStatusText = string.Empty;

    [ObservableProperty]
    private bool _isPlayingSample;

    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(RebuildCacheCommand))]
    [NotifyCanExecuteChangedFor(nameof(PlaySampleCommand))]
    private bool _hasRawVoiceprint;

    [ObservableProperty]
    private bool _hasVoiceprintCache;

    [ObservableProperty]
    private string _voiceprintStatusText = string.Empty;

    [ObservableProperty]
    private string _voiceprintDetailText = string.Empty;

    // Audio Routing & Cascade DSP
    [ObservableProperty]
    private AudioDeviceInfo? _selectedCaptureDevice;

    [ObservableProperty]
    private AudioDeviceInfo? _selectedRenderDevice;

    [ObservableProperty]
    private bool _monitorLoopbackEnabled;

    [ObservableProperty]
    private bool _vadEnabled = true;

    [ObservableProperty]
    private double _vadThreshold = 0.5;

    [ObservableProperty]
    private string _vadThresholdText = "0.50";

    [ObservableProperty]
    private string _vadLiveProbText = "--";

    [ObservableProperty]
    private bool _gtcrnEnabled = true;

    [ObservableProperty]
    private bool _voxLockEnabled = true;

    [ObservableProperty]
    private VoxLockModelVariant _selectedVoxLockVariant = VoxLockModelVariant.Int8;

    // Compute & Core Affinity
    [ObservableProperty]
    private ComputeBackend _selectedComputeBackend = ComputeBackend.Cpu;

    [ObservableProperty]
    private bool _isCpuBackendSelected = true;

    [ObservableProperty]
    private bool _isCudaBackendSelected;

    [ObservableProperty]
    private string? _cudaFallbackNotice;

    [ObservableProperty]
    private bool _hasCudaFallbackNotice;

    [ObservableProperty]
    private string _cpuTopologySummary = string.Empty;

    [ObservableProperty]
    private bool _autoAffinityEnabled = true;

    [ObservableProperty]
    private bool _isCoreMatrixExpanded;

    // Preferences & About
    [ObservableProperty]
    private AppLanguage _selectedLanguage = AppLanguage.ZhCn;

    [ObservableProperty]
    private AppThemePreference _selectedThemePreference = AppThemePreference.System;

    [ObservableProperty]
    private bool _minimizeToTrayEnabled = true;

    [ObservableProperty]
    private bool _isDarkThemeActive;

    public MainViewModel()
        : this(new VoiceLockController())
    {
    }

    public MainViewModel(VoiceLockController controller)
    {
        _controller = controller ?? throw new ArgumentNullException(nameof(controller));
        _dispatcher = Application.Current?.Dispatcher ?? Dispatcher.CurrentDispatcher;

        _controller.TelemetryUpdated += OnControllerTelemetryUpdated;
        _controller.PipelineFaulted += OnControllerPipelineFaulted;
        _controller.RecordingProgress += OnControllerRecordingProgress;
        _controller.PlaybackStopped += OnControllerPlaybackStopped;
        _controller.VoiceprintChanged += OnControllerVoiceprintChanged;
        _controller.BackendStateChanged += OnControllerBackendStateChanged;

        AppStrings.Current.LanguageChanged += OnAppLanguageChanged;
        DwmBackdropHelper.ThemeChanged += OnDwmThemeChanged;

        HydrateFromSettingsAndTopology();
        RefreshDevices();
        RefreshLocalizedTexts();
    }

    public ObservableCollection<AudioDeviceInfo> CaptureDevices { get; } = [];
    public ObservableCollection<AudioDeviceInfo> RenderDevices { get; } = [];
    public ObservableCollection<CpuCoreItemViewModel> CoreMatrixItems { get; } = [];

    public IReadOnlyList<VoxLockModelVariant> VoxLockVariants { get; } =
    [
        VoxLockModelVariant.Int8,
        VoxLockModelVariant.Int8Full,
        VoxLockModelVariant.Fp16
    ];

    public string AppDataDirectoryPath => AppSettings.GetAppDataDirectory();

    public string ActiveTabTitle => ActiveTab switch
    {
        NavigationTab.Voiceprint => AppStrings.Current.NavVoiceprintTitle,
        NavigationTab.Compute => AppStrings.Current.NavComputeTitle,
        NavigationTab.OtherSettings => AppStrings.Current.NavOtherTitle,
        _ => AppStrings.Current.NavPipelineTitle
    };

    public string AboutCardTitle => AppStrings.Current.Language == AppLanguage.EnUs
        ? "About & Data"
        : "关于与本地数据";

    public string OpenFolderButtonText => AppStrings.Current.Language == AppLanguage.EnUs
        ? "Open Folder"
        : "打开目录";

    public string ResetQuickStartButtonText => AppStrings.Current.Language == AppLanguage.EnUs
        ? "Show Quick Start Guide"
        : "重新显示快速上手引导";

    public string TelemetryStripTitle => AppStrings.Current.Language == AppLanguage.EnUs
        ? "Audio Signal Levels"
        : "实时音频电平";

    public string VadLiveProbPrefix => AppStrings.Current.Language == AppLanguage.EnUs
        ? "Speech Prob"
        : "实时概率";

    public event EventHandler? ShowWindowRequested;
    public event EventHandler? ExitRequested;
    public event EventHandler<GuidanceTarget>? GuidanceRequested;

    public async Task InitializeAsync()
    {
        if (_disposed)
        {
            return;
        }

        IsBusy = true;
        try
        {
            await _controller.InitializeAsync();
            RefreshBackendBanner();
            RefreshVoiceprintStatus();
        }
        catch (Exception ex)
        {
            StatusKind = PipelineStatusKind.Faulted;
            _lastStatusError = ex.Message;
            StatusBadgeText = AppStrings.Current.FormatInitFailed(ex.Message);
        }
        finally
        {
            IsBusy = false;
        }
    }

    partial void OnActiveTabChanged(NavigationTab value)
    {
        if (!_isNavigatingBack && value != _lastRecordedTab)
        {
            _navigationHistory.Push(_lastRecordedTab);
            CanNavigateBack = _navigationHistory.Count > 0;
        }

        _lastRecordedTab = value;
        OnPropertyChanged(nameof(ActiveTabTitle));
    }

    partial void OnIsPipelineRunningChanged(bool value)
    {
        if (_isHydrating || _updatingPipelineFlagFromTelemetry)
        {
            return;
        }

        _ = SetPipelineRunningAsync(value);
    }

    partial void OnSelectedCaptureDeviceChanged(AudioDeviceInfo? value)
    {
        if (_isHydrating || value is null)
        {
            return;
        }

        _ = ApplySelectedDevicesAsync();
    }

    partial void OnSelectedRenderDeviceChanged(AudioDeviceInfo? value)
    {
        if (_isHydrating || value is null)
        {
            return;
        }

        _ = ApplySelectedDevicesAsync();
    }

    partial void OnMonitorLoopbackEnabledChanged(bool value)
    {
        if (_isHydrating)
        {
            return;
        }

        _controller.UpdateMonitorLoopback(value);
    }

    partial void OnVadEnabledChanged(bool value)
    {
        if (_isHydrating)
        {
            return;
        }

        PushStageFlagsToController();
    }

    partial void OnVadThresholdChanged(double value)
    {
        double clamped = Math.Clamp(value, 0.05, 0.95);
        VadThresholdText = $"{clamped:F2}";

        if (_isHydrating)
        {
            return;
        }

        PushStageFlagsToController();
    }

    partial void OnGtcrnEnabledChanged(bool value)
    {
        if (_isHydrating)
        {
            return;
        }

        PushStageFlagsToController();
    }

    partial void OnVoxLockEnabledChanged(bool value)
    {
        if (_isHydrating)
        {
            return;
        }

        PushStageFlagsToController();
    }

    partial void OnSelectedVoxLockVariantChanged(VoxLockModelVariant value)
    {
        if (_isHydrating)
        {
            return;
        }

        _ = SwitchComputeOrVariantAsync(SelectedComputeBackend, value);
    }

    partial void OnIsCpuBackendSelectedChanged(bool value)
    {
        if (_updatingBackendFlags || !value)
        {
            return;
        }

        _updatingBackendFlags = true;
        try
        {
            IsCudaBackendSelected = false;
            SelectedComputeBackend = ComputeBackend.Cpu;
        }
        finally
        {
            _updatingBackendFlags = false;
        }

        if (!_isHydrating)
        {
            _ = SwitchComputeOrVariantAsync(ComputeBackend.Cpu, SelectedVoxLockVariant);
        }
    }

    partial void OnIsCudaBackendSelectedChanged(bool value)
    {
        if (_updatingBackendFlags || !value)
        {
            return;
        }

        _updatingBackendFlags = true;
        try
        {
            IsCpuBackendSelected = false;
            SelectedComputeBackend = ComputeBackend.Cuda;
        }
        finally
        {
            _updatingBackendFlags = false;
        }

        if (!_isHydrating)
        {
            _ = SwitchComputeOrVariantAsync(ComputeBackend.Cuda, SelectedVoxLockVariant);
        }
    }

    partial void OnSelectedComputeBackendChanged(ComputeBackend value)
    {
        if (_updatingBackendFlags)
        {
            return;
        }

        _updatingBackendFlags = true;
        try
        {
            IsCpuBackendSelected = value == ComputeBackend.Cpu;
            IsCudaBackendSelected = value == ComputeBackend.Cuda;
        }
        finally
        {
            _updatingBackendFlags = false;
        }

        if (!_isHydrating)
        {
            _ = SwitchComputeOrVariantAsync(value, SelectedVoxLockVariant);
        }
    }

    partial void OnAutoAffinityEnabledChanged(bool value)
    {
        if (_isHydrating)
        {
            return;
        }

        if (value)
        {
            CpuTopologyInfo topology = _controller.CpuTopology;
            ApplyMaskSelectionToCoreItems(topology.RecommendedCoreMask, topology.RecommendedProcessorGroup);
            _controller.UpdateCpuAffinity(true, _controller.Settings.SelectedCoreMask, _controller.Settings.SelectedProcessorGroup);
        }
        else
        {
            (ulong mask, ushort group) = CollectSelectedCoreMask();
            _controller.UpdateCpuAffinity(false, mask, group);
        }
    }

    partial void OnSelectedLanguageChanged(AppLanguage value)
    {
        if (_isHydrating)
        {
            return;
        }

        AppStrings.Current.SetLanguage(value);
        _controller.UpdateAppPreferences(value, SelectedThemePreference, MinimizeToTrayEnabled);
    }

    partial void OnSelectedThemePreferenceChanged(AppThemePreference value)
    {
        if (_isHydrating)
        {
            return;
        }

        DwmBackdropHelper.ApplyThemePreference(value);
        _controller.UpdateAppPreferences(SelectedLanguage, value, MinimizeToTrayEnabled);
    }

    partial void OnMinimizeToTrayEnabledChanged(bool value)
    {
        if (_isHydrating)
        {
            return;
        }

        _controller.UpdateAppPreferences(SelectedLanguage, SelectedThemePreference, value);
    }

    [RelayCommand]
    private void NavigateToTab(NavigationTab tab)
    {
        ActiveTab = tab;
    }

    [RelayCommand]
    private void NavigateGuidance(GuidanceTarget target)
    {
        switch (target)
        {
            case GuidanceTarget.AudioRouting:
            case GuidanceTarget.MasterSwitch:
                ActiveTab = NavigationTab.Pipeline;
                break;

            case GuidanceTarget.VoiceprintCapture:
                ActiveTab = NavigationTab.Voiceprint;
                break;
        }

        _dispatcher.BeginInvoke(
            DispatcherPriority.Loaded,
            () => GuidanceRequested?.Invoke(this, target));
    }

    [RelayCommand]
    private void OpenVoiceprintTab()
    {
        NavigateGuidance(GuidanceTarget.VoiceprintCapture);
    }

    [RelayCommand]
    private void DismissQuickStart()
    {
        ShowQuickStartGuide = false;
    }

    [RelayCommand]
    private void ResetQuickStart()
    {
        ShowQuickStartGuide = true;
        ActiveTab = NavigationTab.Pipeline;
    }

    [RelayCommand(CanExecute = nameof(CanNavigateBack))]
    private void NavigateBack()
    {
        if (_navigationHistory.Count == 0)
        {
            return;
        }

        NavigationTab previous = _navigationHistory.Pop();
        CanNavigateBack = _navigationHistory.Count > 0;
        _isNavigatingBack = true;
        try
        {
            ActiveTab = previous;
        }
        finally
        {
            _isNavigatingBack = false;
        }
    }

    [RelayCommand]
    private void OpenVoiceprintFolder()
    {
        try
        {
            string? rawPath = _controller.ResolveExistingVoiceprintWavPath();
            if (!string.IsNullOrWhiteSpace(rawPath) && File.Exists(rawPath))
            {
                Process.Start(new ProcessStartInfo
                {
                    FileName = "explorer.exe",
                    Arguments = $"/select,\"{rawPath}\"",
                    UseShellExecute = true
                });
                return;
            }

            Directory.CreateDirectory(AppDataDirectoryPath);
            Process.Start(new ProcessStartInfo
            {
                FileName = AppDataDirectoryPath,
                UseShellExecute = true
            });
        }
        catch
        {
            // Ignore shell launch failures in restricted environments.
        }
    }

    [RelayCommand]
    private void OpenAppDataFolder()
    {
        try
        {
            Directory.CreateDirectory(AppDataDirectoryPath);
            Process.Start(new ProcessStartInfo
            {
                FileName = AppDataDirectoryPath,
                UseShellExecute = true
            });
        }
        catch
        {
            // Ignore shell launch failures in restricted environments.
        }
    }

    [RelayCommand]
    private void OpenExternalUrl(string? url)
    {
        if (string.IsNullOrWhiteSpace(url) ||
            !Uri.TryCreate(url, UriKind.Absolute, out Uri? uri) ||
            (uri.Scheme != Uri.UriSchemeHttp && uri.Scheme != Uri.UriSchemeHttps))
        {
            return;
        }

        try
        {
            Process.Start(new ProcessStartInfo
            {
                FileName = uri.AbsoluteUri,
                UseShellExecute = true
            });
        }
        catch
        {
            // Ignore shell launch failures in restricted environments.
        }
    }

    [RelayCommand]
    private async Task TogglePipelineAsync()
    {
        await SetPipelineRunningAsync(!_controller.IsPipelineRunning);
    }

    [RelayCommand]
    private void RefreshDevices()
    {
        bool prevHydrating = _isHydrating;
        _isHydrating = true;
        try
        {
            IReadOnlyList<AudioDeviceInfo> captures = _controller.GetCaptureDevices();
            IReadOnlyList<AudioDeviceInfo> renders = _controller.GetRenderDevices();

            CaptureDevices.Clear();
            foreach (AudioDeviceInfo dev in captures)
            {
                CaptureDevices.Add(dev);
            }

            RenderDevices.Clear();
            foreach (AudioDeviceInfo dev in renders)
            {
                RenderDevices.Add(dev);
            }

            string? savedInputId = _controller.Settings.SelectedInputDeviceId;
            string? savedOutputId = _controller.Settings.SelectedOutputDeviceId;

            SelectedCaptureDevice = captures.FirstOrDefault(d => string.Equals(d.DeviceId, savedInputId, StringComparison.OrdinalIgnoreCase))
                ?? captures.FirstOrDefault(d => d.IsDefault)
                ?? captures.FirstOrDefault();

            SelectedRenderDevice = renders.FirstOrDefault(d => string.Equals(d.DeviceId, savedOutputId, StringComparison.OrdinalIgnoreCase))
                ?? renders.FirstOrDefault(d => d.IsVirtualAudioDevice)
                ?? renders.FirstOrDefault(d => d.IsDefault)
                ?? renders.FirstOrDefault();
        }
        finally
        {
            _isHydrating = prevHydrating;
        }
    }

    [RelayCommand]
    private async Task RecordVoiceprintAsync()
    {
        if (IsRecording)
        {
            _controller.StopVoiceprintRecording();
            _recordingMessageKind = RecordingMessageKind.Stopping;
            RecordingStatusText = AppStrings.Current.RecordingStoppingStatus;
            return;
        }

        IsRecording = true;
        RecordingProgress = 0;
        RecordingLevel = 0;
        _recordingMessageKind = RecordingMessageKind.Idle;
        RecordingStatusText = AppStrings.Current.RecordingActiveDefaultStatus;

        try
        {
            VoiceprintProfileInfo profile = await _controller.RecordAndEnrollVoiceprintAsync(5.0);
            RecordingProgress = 100;
            RecordingLevel = 0;
            _lastCompletedDurationSeconds = profile.SourceDurationSeconds;
            _recordingMessageKind = RecordingMessageKind.Completed;
            RecordingStatusText = AppStrings.Current.FormatRecordingCompleted(profile.SourceDurationSeconds);
            RefreshVoiceprintStatus();
        }
        catch (OperationCanceledException)
        {
            _recordingMessageKind = RecordingMessageKind.Canceled;
            RecordingStatusText = AppStrings.Current.RecordingCanceledStatus;
            RecordingProgress = 0;
            RecordingLevel = 0;
        }
        catch (Exception ex)
        {
            _lastRecordingError = ex.Message;
            _recordingMessageKind = RecordingMessageKind.RecordingFailed;
            RecordingStatusText = AppStrings.Current.FormatRecordingFailed(ex.Message);
            RecordingProgress = 0;
            RecordingLevel = 0;
        }
        finally
        {
            IsRecording = false;
        }
    }

    private bool CanPlaySample() => HasRawVoiceprint && !IsRecording;

    [RelayCommand(CanExecute = nameof(CanPlaySample))]
    private void PlaySample()
    {
        if (IsPlayingSample)
        {
            _controller.StopSamplePlayback();
            IsPlayingSample = false;
            return;
        }

        try
        {
            _controller.StartSamplePlayback();
            IsPlayingSample = true;
        }
        catch (Exception ex)
        {
            IsPlayingSample = false;
            _lastRecordingError = ex.Message;
            _recordingMessageKind = RecordingMessageKind.PlaybackFailed;
            RecordingStatusText = AppStrings.Current.FormatPlaybackFailed(ex.Message);
        }
    }

    private bool CanRebuildCache() => HasRawVoiceprint && !IsRecording;

    [RelayCommand(CanExecute = nameof(CanRebuildCache))]
    private async Task RebuildCacheAsync()
    {
        IsBusy = true;
        _recordingMessageKind = RecordingMessageKind.Rebuilding;
        RecordingStatusText = AppStrings.Current.RebuildingCacheStatus;
        try
        {
            await _controller.RebuildVoiceprintCacheAsync(forceRebuild: true);
            RefreshVoiceprintStatus();
            _recordingMessageKind = RecordingMessageKind.Rebuilt;
            RecordingStatusText = AppStrings.Current.RebuildCacheDoneStatus;
        }
        catch (Exception ex)
        {
            _lastRecordingError = ex.Message;
            _recordingMessageKind = RecordingMessageKind.RebuildFailed;
            RecordingStatusText = AppStrings.Current.FormatRebuildFailed(ex.Message);
        }
        finally
        {
            IsBusy = false;
        }
    }

    [RelayCommand]
    private void ApplyCoreSelection()
    {
        (ulong mask, ushort group) = CollectSelectedCoreMask();
        if (mask == 0)
        {
            CpuTopologyInfo topology = _controller.CpuTopology;
            mask = topology.RecommendedCoreMask;
            group = topology.RecommendedProcessorGroup;
            ApplyMaskSelectionToCoreItems(mask, group);
        }

        bool prev = _isHydrating;
        _isHydrating = true;
        try
        {
            AutoAffinityEnabled = false;
        }
        finally
        {
            _isHydrating = prev;
        }

        _controller.UpdateCpuAffinity(false, mask, group);
    }

    [RelayCommand]
    private void ShowWindow()
    {
        ShowWindowRequested?.Invoke(this, EventArgs.Empty);
    }

    [RelayCommand]
    private void ExitApplication()
    {
        ExitRequested?.Invoke(this, EventArgs.Empty);
    }

    private async Task SetPipelineRunningAsync(bool shouldRun)
    {
        IsBusy = true;
        try
        {
            if (shouldRun)
            {
                StatusBadgeText = AppStrings.Current.StatusStarting;
                await _controller.StartPipelineAsync();
            }
            else
            {
                await _controller.StopPipelineAsync();
            }
        }
        catch (Exception ex)
        {
            SetPipelineRunningFlagFromTelemetry(false);
            StatusKind = PipelineStatusKind.Faulted;
            _lastStatusError = ex.Message;
            StatusBadgeText = AppStrings.Current.FormatStatusFaulted(ex.Message);
        }
        finally
        {
            SetPipelineRunningFlagFromTelemetry(_controller.IsPipelineRunning);
            IsBusy = false;
        }
    }

    private async Task ApplySelectedDevicesAsync()
    {
        try
        {
            await _controller.SetDevicesAsync(
                SelectedCaptureDevice?.DeviceId,
                SelectedRenderDevice?.DeviceId);
        }
        catch (Exception ex)
        {
            StatusKind = PipelineStatusKind.Faulted;
            _lastStatusError = ex.Message;
            StatusBadgeText = AppStrings.Current.FormatDeviceSwitchFailed(ex.Message);
        }
    }

    private async Task SwitchComputeOrVariantAsync(ComputeBackend backend, VoxLockModelVariant variant)
    {
        IsBusy = true;
        try
        {
            await _controller.SwitchComputeOrVariantAsync(backend, variant);
            RefreshBackendBanner();
            RefreshVoiceprintStatus();
        }
        catch (Exception ex)
        {
            CudaFallbackNotice = AppStrings.Current.FormatSwitchFailed(ex.Message);
            HasCudaFallbackNotice = true;
        }
        finally
        {
            IsBusy = false;
        }
    }

    private void PushStageFlagsToController()
    {
        _controller.UpdateStageFlags(
            VadEnabled,
            (float)Math.Clamp(VadThreshold, 0.05, 0.95),
            GtcrnEnabled,
            VoxLockEnabled);
    }

    private void HydrateFromSettingsAndTopology()
    {
        _isHydrating = true;
        _updatingBackendFlags = true;
        try
        {
            AppSettings settings = _controller.Settings;
            MonitorLoopbackEnabled = settings.MonitorLoopbackEnabled;
            VadEnabled = settings.VadEnabled;
            VadThreshold = Math.Clamp(settings.VadThreshold, 0.05, 0.95);
            VadThresholdText = $"{VadThreshold:F2}";
            GtcrnEnabled = settings.GtcrnEnabled;
            VoxLockEnabled = settings.VoxLockEnabled;
            SelectedVoxLockVariant = settings.VoxLockVariant;

            SelectedComputeBackend = settings.ComputeBackend;
            IsCpuBackendSelected = settings.ComputeBackend == ComputeBackend.Cpu;
            IsCudaBackendSelected = settings.ComputeBackend == ComputeBackend.Cuda;

            AutoAffinityEnabled = settings.AutoAffinityEnabled;

            SelectedLanguage = settings.Language;
            SelectedThemePreference = settings.ThemePreference;
            MinimizeToTrayEnabled = settings.MinimizeToTrayEnabled;
            IsDarkThemeActive = DwmBackdropHelper.IsDarkThemeActive();

            CpuTopologyInfo topology = _controller.CpuTopology;
            BuildCpuTopologyUi(topology, settings);
            _latestActiveCoreRawDescription = topology.RecommendedDescription;
        }
        finally
        {
            _updatingBackendFlags = false;
            _isHydrating = false;
        }
    }

    private void BuildCpuTopologyUi(CpuTopologyInfo topology, AppSettings settings)
    {
        CpuTopologySummary = AppStrings.Current.FormatCpuTopologySummary(topology);

        CoreMatrixItems.Clear();
        ulong activeMask = settings.AutoAffinityEnabled || settings.SelectedCoreMask == 0
            ? topology.RecommendedCoreMask
            : settings.SelectedCoreMask;
        ushort activeGroup = settings.AutoAffinityEnabled || settings.SelectedCoreMask == 0
            ? topology.RecommendedProcessorGroup
            : settings.SelectedProcessorGroup;

        foreach (PhysicalCoreInfo core in topology.PhysicalCores)
        {
            var item = new CpuCoreItemViewModel(core)
            {
                IsSelected = core.ProcessorGroup == activeGroup && (core.LogicalProcessorMask & activeMask) != 0
            };
            CoreMatrixItems.Add(item);
        }
    }

    private void ApplyMaskSelectionToCoreItems(ulong mask, ushort group)
    {
        foreach (CpuCoreItemViewModel item in CoreMatrixItems)
        {
            item.IsSelected = item.ProcessorGroup == group && (item.BitMask & mask) != 0;
        }
    }

    private (ulong Mask, ushort Group) CollectSelectedCoreMask()
    {
        ushort targetGroup = 0;
        bool foundFirst = false;
        ulong mask = 0;

        foreach (CpuCoreItemViewModel item in CoreMatrixItems)
        {
            if (!item.IsSelected)
            {
                continue;
            }

            if (!foundFirst)
            {
                targetGroup = item.ProcessorGroup;
                foundFirst = true;
            }

            if (item.ProcessorGroup == targetGroup)
            {
                mask |= item.BitMask;
            }
        }

        return (mask, targetGroup);
    }

    private void RefreshBackendBanner()
    {
        BackendStatusBanner = AppStrings.Current.FormatBackendBanner(
            _controller.Settings.ComputeBackend,
            _controller.IsCudaFallbackActive);

        if (_controller.IsCudaFallbackActive)
        {
            CudaFallbackNotice = string.IsNullOrWhiteSpace(_controller.ProviderFallbackWarning)
                ? AppStrings.Current.FormatCudaFallbackDefault()
                : _controller.ProviderFallbackWarning;
            HasCudaFallbackNotice = true;
        }
        else
        {
            CudaFallbackNotice = null;
            HasCudaFallbackNotice = false;
        }
    }

    private void RefreshVoiceprintStatus()
    {
        string? rawPath = _controller.ResolveExistingVoiceprintWavPath();
        HasRawVoiceprint = rawPath is not null;

        VoiceprintProfileInfo? profile = _controller.CurrentVoiceprint;
        HasVoiceprintCache = _controller.HasEnrolledVoiceprint && profile is not null;

        if (HasVoiceprintCache && profile is not null)
        {
            VoiceprintStatusText = profile.LoadedFromCache
                ? AppStrings.Current.VoiceprintReadyCached
                : AppStrings.Current.VoiceprintReadyFresh;
            string fileName = Path.GetFileName(profile.SourceWavPath);
            VoiceprintDetailText = AppStrings.Current.FormatVoiceprintDetail(
                fileName,
                profile.SourceDurationSeconds,
                SelectedVoxLockVariant);
        }
        else if (HasRawVoiceprint)
        {
            VoiceprintStatusText = AppStrings.Current.VoiceprintWaitingRebuild;
            VoiceprintDetailText = Path.GetFileName(rawPath)!;
        }
        else
        {
            VoiceprintStatusText = AppStrings.Current.VoiceprintNotEnrolled;
            VoiceprintDetailText = AppStrings.Current.VoiceprintPromptDetail;
        }
    }

    private void RefreshLocalizedTexts()
    {
        OnPropertyChanged(nameof(ActiveTabTitle));
        OnPropertyChanged(nameof(AboutCardTitle));
        OnPropertyChanged(nameof(OpenFolderButtonText));
        OnPropertyChanged(nameof(ResetQuickStartButtonText));
        OnPropertyChanged(nameof(TelemetryStripTitle));
        OnPropertyChanged(nameof(VadLiveProbPrefix));

        PromptReadText = AppStrings.Current.VoiceprintPromptText;

        CpuTopologyInfo topology = _controller.CpuTopology;
        CpuTopologySummary = AppStrings.Current.FormatCpuTopologySummary(topology);
        TelemetryCoreText = AppStrings.Current.FormatShortCoreDescription(
            _latestActiveCoreRawDescription ?? topology.RecommendedDescription,
            topology);

        foreach (CpuCoreItemViewModel core in CoreMatrixItems)
        {
            core.RefreshLocalization();
        }

        RefreshBackendBanner();
        RefreshVoiceprintStatus();
        RefreshStatusFromSnapshot();
        RefreshRecordingStatusText();
        NotifyVariantLabelsRefreshed();
    }

    private void RefreshStatusFromSnapshot()
    {
        if (_latestSnapshot is { } snapshot)
        {
            ApplyTelemetrySnapshot(snapshot);
            return;
        }

        if (StatusKind == PipelineStatusKind.Faulted && !string.IsNullOrWhiteSpace(_lastStatusError))
        {
            StatusBadgeText = AppStrings.Current.FormatStatusFaulted(_lastStatusError);
            return;
        }

        StatusKind = PipelineStatusKind.Idle;
        StatusBadgeText = AppStrings.Current.StatusStopped;
    }

    private void RefreshRecordingStatusText()
    {
        if (IsRecording)
        {
            RecordingStatusText = AppStrings.Current.RecordingActiveDefaultStatus;
            return;
        }

        RecordingStatusText = _recordingMessageKind switch
        {
            RecordingMessageKind.Stopping => AppStrings.Current.RecordingStoppingStatus,
            RecordingMessageKind.Completed => AppStrings.Current.FormatRecordingCompleted(_lastCompletedDurationSeconds),
            RecordingMessageKind.Canceled => AppStrings.Current.RecordingCanceledStatus,
            RecordingMessageKind.Rebuilding => AppStrings.Current.RebuildingCacheStatus,
            RecordingMessageKind.Rebuilt => AppStrings.Current.RebuildCacheDoneStatus,
            RecordingMessageKind.RecordingFailed when !string.IsNullOrEmpty(_lastRecordingError) =>
                AppStrings.Current.FormatRecordingFailed(_lastRecordingError),
            RecordingMessageKind.PlaybackFailed when !string.IsNullOrEmpty(_lastRecordingError) =>
                AppStrings.Current.FormatPlaybackFailed(_lastRecordingError),
            RecordingMessageKind.RebuildFailed when !string.IsNullOrEmpty(_lastRecordingError) =>
                AppStrings.Current.FormatRebuildFailed(_lastRecordingError),
            _ => AppStrings.Current.RecordingIdleStatus
        };
    }

    private void NotifyVariantLabelsRefreshed()
    {
        OnPropertyChanged(nameof(VoxLockVariants));
        OnPropertyChanged(nameof(SelectedVoxLockVariant));
    }

    private void SetPipelineRunningFlagFromTelemetry(bool running)
    {
        if (IsPipelineRunning == running)
        {
            return;
        }

        _updatingPipelineFlagFromTelemetry = true;
        try
        {
            IsPipelineRunning = running;
        }
        finally
        {
            _updatingPipelineFlagFromTelemetry = false;
        }
    }

    private void ApplyTelemetrySnapshot(TelemetrySnapshot snapshot)
    {
        switch (snapshot.State)
        {
            case PipelineRunState.Running:
                SetPipelineRunningFlagFromTelemetry(true);
                if (VadEnabled && !snapshot.VadSpeechActive)
                {
                    StatusKind = PipelineStatusKind.Sleeping;
                    StatusBadgeText = AppStrings.Current.StatusSleeping;
                }
                else
                {
                    StatusKind = PipelineStatusKind.Running;
                    StatusBadgeText = AppStrings.Current.StatusRunning;
                }

                TelemetryLatencyText = $"~{Math.Round(snapshot.LatencyMs):0} ms";
                TelemetryInferenceText = $"{snapshot.FrameInferenceMs:0.0} ms";
                _latestActiveCoreRawDescription = snapshot.ActiveCoreDescription;
                TelemetryCoreText = AppStrings.Current.FormatShortCoreDescription(
                    snapshot.ActiveCoreDescription,
                    _controller.CpuTopology);
                InputLevelPercent = RmsToMeterPercent(snapshot.InputRms);
                OutputLevelPercent = RmsToMeterPercent(snapshot.OutputRms);
                VadLiveProbText = AppStrings.Current.FormatVadLiveProb(
                    snapshot.VadProbability,
                    snapshot.VadSpeechActive,
                    VadEnabled);
                break;

            case PipelineRunState.Starting:
                StatusKind = PipelineStatusKind.Idle;
                StatusBadgeText = AppStrings.Current.StatusStarting;
                break;

            case PipelineRunState.Faulted:
                SetPipelineRunningFlagFromTelemetry(false);
                StatusKind = PipelineStatusKind.Faulted;
                _lastStatusError = snapshot.StatusText;
                StatusBadgeText = string.IsNullOrWhiteSpace(snapshot.StatusText)
                    ? AppStrings.Current.StatusFaultedPrefix
                    : snapshot.StatusText;
                InputLevelPercent = 0;
                OutputLevelPercent = 0;
                VadLiveProbText = "--";
                break;

            default:
                SetPipelineRunningFlagFromTelemetry(false);
                StatusKind = PipelineStatusKind.Idle;
                StatusBadgeText = AppStrings.Current.StatusStopped;
                TelemetryLatencyText = "--";
                TelemetryInferenceText = "--";
                InputLevelPercent = 0;
                OutputLevelPercent = 0;
                VadLiveProbText = "--";
                break;
        }
    }

    private void OnAppLanguageChanged(object? sender, EventArgs e)
    {
        BeginInvokeOnUi(() =>
        {
            if (_disposed)
            {
                return;
            }

            if (SelectedLanguage != AppStrings.Current.Language)
            {
                bool prev = _isHydrating;
                _isHydrating = true;
                try
                {
                    SelectedLanguage = AppStrings.Current.Language;
                }
                finally
                {
                    _isHydrating = prev;
                }
            }

            RefreshLocalizedTexts();
        });
    }

    private void OnDwmThemeChanged(object? sender, AppThemePreference preference)
    {
        BeginInvokeOnUi(() =>
        {
            if (_disposed)
            {
                return;
            }

            if (SelectedThemePreference != preference)
            {
                bool prev = _isHydrating;
                _isHydrating = true;
                try
                {
                    SelectedThemePreference = preference;
                }
                finally
                {
                    _isHydrating = prev;
                }
            }

            IsDarkThemeActive = DwmBackdropHelper.IsDarkThemeActive();
        });
    }

    private void OnControllerTelemetryUpdated(object? sender, TelemetrySnapshot snapshot)
    {
        BeginInvokeOnUi(() =>
        {
            if (_disposed)
            {
                return;
            }

            _latestSnapshot = snapshot;
            ApplyTelemetrySnapshot(snapshot);
        });
    }

    private void OnControllerPipelineFaulted(object? sender, Exception ex)
    {
        BeginInvokeOnUi(() =>
        {
            if (_disposed)
            {
                return;
            }

            SetPipelineRunningFlagFromTelemetry(false);
            StatusKind = PipelineStatusKind.Faulted;
            _lastStatusError = ex.Message;
            StatusBadgeText = AppStrings.Current.FormatStatusFaulted(ex.Message);
            InputLevelPercent = 0;
            OutputLevelPercent = 0;
        });
    }

    private void OnControllerRecordingProgress(object? sender, VoiceprintRecordingProgress progress)
    {
        BeginInvokeOnUi(() =>
        {
            if (_disposed)
            {
                return;
            }

            RecordingProgress = Math.Clamp(progress.Progress * 100.0, 0.0, 100.0);
            RecordingLevel = RmsToMeterPercent(progress.RmsLevel);
            if (progress.IsRecording)
            {
                RecordingStatusText = AppStrings.Current.FormatRecordingProgress(
                    progress.ElapsedSeconds,
                    progress.TotalSeconds);
            }
        });
    }

    private void OnControllerPlaybackStopped(object? sender, EventArgs e)
    {
        BeginInvokeOnUi(() =>
        {
            if (!_disposed)
            {
                IsPlayingSample = false;
            }
        });
    }

    private void OnControllerVoiceprintChanged(object? sender, EventArgs e)
    {
        BeginInvokeOnUi(() =>
        {
            if (!_disposed)
            {
                RefreshVoiceprintStatus();
            }
        });
    }

    private void OnControllerBackendStateChanged(object? sender, EventArgs e)
    {
        BeginInvokeOnUi(() =>
        {
            if (!_disposed)
            {
                RefreshBackendBanner();
            }
        });
    }

    private void BeginInvokeOnUi(Action action)
    {
        if (_dispatcher.CheckAccess())
        {
            action();
        }
        else
        {
            _dispatcher.BeginInvoke(DispatcherPriority.DataBind, action);
        }
    }

    private static double RmsToMeterPercent(float rms)
    {
        if (rms <= 1e-5f)
        {
            return 0.0;
        }

        double db = 20.0 * Math.Log10(rms);
        return Math.Clamp((db + 60.0) / 57.0 * 100.0, 0.0, 100.0);
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        AppStrings.Current.LanguageChanged -= OnAppLanguageChanged;
        DwmBackdropHelper.ThemeChanged -= OnDwmThemeChanged;

        _controller.TelemetryUpdated -= OnControllerTelemetryUpdated;
        _controller.PipelineFaulted -= OnControllerPipelineFaulted;
        _controller.RecordingProgress -= OnControllerRecordingProgress;
        _controller.PlaybackStopped -= OnControllerPlaybackStopped;
        _controller.VoiceprintChanged -= OnControllerVoiceprintChanged;
        _controller.BackendStateChanged -= OnControllerBackendStateChanged;
        _controller.Dispose();
    }
}
