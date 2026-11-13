using System.Collections.ObjectModel;
using System.IO;
using System.Windows;
using System.Windows.Threading;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using VoiceLockStudio.Models;
using VoiceLockStudio.Services;
using VoiceLockStudio.Services.Audio;
using VoiceLockStudio.Services.System;

namespace VoiceLockStudio.ViewModels;

public enum PipelineStatusKind
{
    Idle,
    Running,
    Sleeping,
    Faulted
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

        string shortRole = core.IsEfficiencyCore
            ? "E-Core"
            : (core.IsRecommended ? "Tail" : "P-Core");

        CoreTypeLabel = shortRole;
        Label = $"C{core.CoreIndex} · {(core.IsEfficiencyCore ? "E" : "P")}";
        TooltipText = core.DisplayName;
    }

    public int LogicalIndex { get; }
    public int PhysicalIndex { get; }
    public ushort ProcessorGroup { get; }
    public string Label { get; }
    public string CoreTypeLabel { get; }
    public bool IsEfficiencyCore { get; }
    public bool IsRecommended { get; }
    public ulong BitMask { get; }
    public string TooltipText { get; }

    [ObservableProperty]
    private bool _isSelected;
}

public sealed partial class MainViewModel : ObservableObject, IDisposable
{
    private const string DefaultPromptText =
        "今天天气真不错，我正在使用 VoiceLock Studio 录制个人声纹样本，用于在嘈杂环境中精准提取我的声音。";

    private readonly VoiceLockController _controller;
    private readonly Dispatcher _dispatcher;

    private bool _isHydrating;
    private bool _updatingPipelineFlagFromTelemetry;
    private bool _updatingBackendFlags;
    private bool _disposed;

    // Card 1: Status & Telemetry
    [ObservableProperty]
    private bool _isPipelineRunning;

    [ObservableProperty]
    private bool _isBusy;

    [ObservableProperty]
    private string _statusBadgeText = "已停止";

    [ObservableProperty]
    private PipelineStatusKind _statusKind = PipelineStatusKind.Idle;

    [ObservableProperty]
    private string _telemetryLatencyText = "--";

    [ObservableProperty]
    private string _telemetryInferenceText = "--";

    [ObservableProperty]
    private string _telemetryCoreText = "未绑定";

    [ObservableProperty]
    private double _inputLevelPercent;

    [ObservableProperty]
    private double _outputLevelPercent;

    [ObservableProperty]
    private string _backendStatusBanner = "CPU (Single-Thread Lock-Core)";

    // Card 2: Voiceprint Profile
    [ObservableProperty]
    private string _promptReadText = DefaultPromptText;

    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(RebuildCacheCommand))]
    [NotifyCanExecuteChangedFor(nameof(PlaySampleCommand))]
    private bool _isRecording;

    [ObservableProperty]
    private double _recordingProgress;

    [ObservableProperty]
    private double _recordingLevel;

    [ObservableProperty]
    private string _recordingStatusText = "点击按钮朗读上方文本 5 秒以录入声纹";

    [ObservableProperty]
    private bool _isPlayingSample;

    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(RebuildCacheCommand))]
    [NotifyCanExecuteChangedFor(nameof(PlaySampleCommand))]
    private bool _hasRawVoiceprint;

    [ObservableProperty]
    private bool _hasVoiceprintCache;

    [ObservableProperty]
    private string _voiceprintStatusText = "尚未录入声纹（VoxLock 将直通音频）";

    [ObservableProperty]
    private string _voiceprintDetailText = "请先录制 5 秒个人人声样本";

    // Card 3: Audio Routing & Cascade DSP
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

    // Card 4: Compute & Core Affinity
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
    private string _cpuTopologySummary = "正在检测 CPU 拓扑...";

    [ObservableProperty]
    private bool _autoAffinityEnabled = true;

    [ObservableProperty]
    private bool _isCoreMatrixExpanded;

    // Tray & Lifecycle
    [ObservableProperty]
    private bool _minimizeToTrayEnabled = true;

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

        HydrateFromSettingsAndTopology();
        RefreshDevices();
        RefreshVoiceprintStatus();
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

    public event EventHandler? ShowWindowRequested;
    public event EventHandler? ExitRequested;

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
            StatusBadgeText = $"初始化失败 · {ex.Message}";
        }
        finally
        {
            IsBusy = false;
        }
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
            RecordingStatusText = "正在停止录制...";
            return;
        }

        IsRecording = true;
        RecordingProgress = 0;
        RecordingLevel = 0;
        RecordingStatusText = "正在录制声纹样本 (5.0 秒)...";

        try
        {
            VoiceprintProfileInfo profile = await _controller.RecordAndEnrollVoiceprintAsync(5.0);
            RecordingProgress = 100;
            RecordingLevel = 0;
            RecordingStatusText = $"录制并建模完成 ({profile.SourceDurationSeconds:0.0} 秒)";
            RefreshVoiceprintStatus();
        }
        catch (OperationCanceledException)
        {
            RecordingStatusText = "声纹录制已取消";
            RecordingProgress = 0;
            RecordingLevel = 0;
        }
        catch (Exception ex)
        {
            RecordingStatusText = $"录制失败: {ex.Message}";
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
            RecordingStatusText = $"试听失败: {ex.Message}";
        }
    }

    private bool CanRebuildCache() => HasRawVoiceprint && !IsRecording;

    [RelayCommand(CanExecute = nameof(CanRebuildCache))]
    private async Task RebuildCacheAsync()
    {
        IsBusy = true;
        RecordingStatusText = "正在重建声纹模型缓存...";
        try
        {
            await _controller.RebuildVoiceprintCacheAsync(forceRebuild: true);
            RefreshVoiceprintStatus();
            RecordingStatusText = "声纹缓存重建完成";
        }
        catch (Exception ex)
        {
            RecordingStatusText = $"重建缓存失败: {ex.Message}";
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
                StatusBadgeText = "正在启动音频管线...";
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
            StatusBadgeText = $"启动失败 · {ex.Message}";
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
            StatusBadgeText = $"设备切换失败 · {ex.Message}";
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
            CudaFallbackNotice = $"模型/后端切换失败: {ex.Message}";
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

            CpuTopologyInfo topology = _controller.CpuTopology;
            BuildCpuTopologyUi(topology, settings);
            TelemetryCoreText = FormatShortCoreDescription(topology.RecommendedDescription, topology);
        }
        finally
        {
            _updatingBackendFlags = false;
            _isHydrating = false;
        }
    }

    private void BuildCpuTopologyUi(CpuTopologyInfo topology, AppSettings settings)
    {
        if (topology.IsHeterogeneous)
        {
            int pCores = topology.PhysicalCores.Count(c => !c.IsEfficiencyCore);
            int eCores = topology.PhysicalCores.Count(c => c.IsEfficiencyCore);
            CpuTopologySummary = $"{topology.TotalLogicalProcessors} 逻辑核心 ({pCores} P-Core + {eCores} E-Core) · 推荐自动锁定能效核";
        }
        else
        {
            CpuTopologySummary = $"{topology.TotalLogicalProcessors} 逻辑核心 ({topology.PhysicalCores.Count} 物理核心) · 推荐锁定尾核";
        }

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
        BackendStatusBanner = _controller.ActiveProviderSummary;
        CudaFallbackNotice = _controller.ProviderFallbackWarning;
        HasCudaFallbackNotice = !string.IsNullOrWhiteSpace(CudaFallbackNotice);
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
                ? "声纹已就绪 · 已加载本地特征缓存"
                : "声纹已就绪 · 已提取最新特征";
            string fileName = Path.GetFileName(profile.SourceWavPath);
            VoiceprintDetailText = $"{fileName} ({profile.SourceDurationSeconds:0.0}s) · 变体 {SelectedVoxLockVariant}";
        }
        else if (HasRawVoiceprint)
        {
            VoiceprintStatusText = "发现原始声纹样本，正在等待或可重建模型缓存";
            VoiceprintDetailText = Path.GetFileName(rawPath)!;
        }
        else
        {
            VoiceprintStatusText = "尚未录入声纹（VoxLock 将直通音频）";
            VoiceprintDetailText = "请朗读上方提示语并点击「录制我的声纹」";
        }
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

    private void OnControllerTelemetryUpdated(object? sender, TelemetrySnapshot snapshot)
    {
        BeginInvokeOnUi(() =>
        {
            if (_disposed)
            {
                return;
            }

            switch (snapshot.State)
            {
                case PipelineRunState.Running:
                    SetPipelineRunningFlagFromTelemetry(true);
                    if (VadEnabled && !snapshot.VadSpeechActive)
                    {
                        StatusKind = PipelineStatusKind.Sleeping;
                        StatusBadgeText = "静音休眠中 · 0% 算力占用";
                    }
                    else
                    {
                        StatusKind = PipelineStatusKind.Running;
                        StatusBadgeText = "运行中 · 正在过滤人声";
                    }

                    TelemetryLatencyText = $"~{Math.Round(snapshot.LatencyMs):0} ms";
                    TelemetryInferenceText = $"{snapshot.FrameInferenceMs:0.0} ms";
                    TelemetryCoreText = FormatShortCoreDescription(snapshot.ActiveCoreDescription, _controller.CpuTopology);
                    InputLevelPercent = RmsToMeterPercent(snapshot.InputRms);
                    OutputLevelPercent = RmsToMeterPercent(snapshot.OutputRms);
                    VadLiveProbText = VadEnabled
                        ? $"{snapshot.VadProbability:0.00} · {(snapshot.VadSpeechActive ? "人声" : "静音")}"
                        : "已旁路";
                    break;

                case PipelineRunState.Starting:
                    StatusKind = PipelineStatusKind.Idle;
                    StatusBadgeText = "正在启动音频管线...";
                    break;

                case PipelineRunState.Faulted:
                    SetPipelineRunningFlagFromTelemetry(false);
                    StatusKind = PipelineStatusKind.Faulted;
                    StatusBadgeText = snapshot.StatusText;
                    InputLevelPercent = 0;
                    OutputLevelPercent = 0;
                    VadLiveProbText = "--";
                    break;

                default:
                    SetPipelineRunningFlagFromTelemetry(false);
                    StatusKind = PipelineStatusKind.Idle;
                    StatusBadgeText = "已停止";
                    TelemetryLatencyText = "--";
                    TelemetryInferenceText = "--";
                    InputLevelPercent = 0;
                    OutputLevelPercent = 0;
                    VadLiveProbText = "--";
                    break;
            }

            if (!string.IsNullOrWhiteSpace(snapshot.ActiveBackendDescription))
            {
                BackendStatusBanner = snapshot.ActiveBackendDescription;
            }
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
            StatusBadgeText = $"管线异常 · {ex.Message}";
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
                RecordingStatusText = $"正在录制声纹... {progress.ElapsedSeconds:0.0} / {progress.TotalSeconds:0.0} 秒";
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

    private static string FormatShortCoreDescription(string rawDescription, CpuTopologyInfo topology)
    {
        if (string.IsNullOrWhiteSpace(rawDescription))
        {
            return "未绑定";
        }

        if (rawDescription.StartsWith("Auto E-Cores", StringComparison.OrdinalIgnoreCase))
        {
            int eCoreCount = topology.PhysicalCores.Count(c => c.IsEfficiencyCore);
            return $"自动锁定 E-Core ({eCoreCount}核)";
        }

        if (rawDescription.StartsWith("Auto Tail Cores", StringComparison.OrdinalIgnoreCase))
        {
            return "自动锁定尾核 (Tail)";
        }

        if (rawDescription.Length > 28)
        {
            int parenIdx = rawDescription.IndexOf('(');
            return parenIdx > 0 ? rawDescription[..parenIdx].Trim() : rawDescription[..28];
        }

        return rawDescription;
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        _controller.TelemetryUpdated -= OnControllerTelemetryUpdated;
        _controller.PipelineFaulted -= OnControllerPipelineFaulted;
        _controller.RecordingProgress -= OnControllerRecordingProgress;
        _controller.PlaybackStopped -= OnControllerPlaybackStopped;
        _controller.VoiceprintChanged -= OnControllerVoiceprintChanged;
        _controller.BackendStateChanged -= OnControllerBackendStateChanged;
        _controller.Dispose();
    }
}
