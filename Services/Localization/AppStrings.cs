using System.Windows;
using CommunityToolkit.Mvvm.ComponentModel;
using VoiceLockStudio.Models;
using VoiceLockStudio.Services.System;

namespace VoiceLockStudio.Services.Localization;

/// <summary>
/// Runtime-switchable bilingual localization source (Simplified Chinese &amp; English).
/// XAML bindings targeting <see cref="Current"/> update immediately when <see cref="SetLanguage"/> is called.
/// </summary>
public sealed class AppStrings : ObservableObject
{
    public static AppStrings Current { get; } = new();

    private AppStrings()
    {
    }

    public AppLanguage Language { get; private set; } = AppLanguage.ZhCn;

    public event EventHandler? LanguageChanged;

    public void SetLanguage(AppLanguage language)
    {
        if (Language == language)
        {
            return;
        }

        Language = language;
        OnPropertyChanged(string.Empty);
        LanguageChanged?.Invoke(this, EventArgs.Empty);
    }

    private string L(string zh, string en) => Language == AppLanguage.EnUs ? en : zh;

    // Typography Weights (Microsoft YaHei UI maps SemiBold to heavy 700 Bold; keep ZhCn crisp)
    public FontWeight HeadingFontWeight => Language == AppLanguage.ZhCn ? FontWeights.Normal : FontWeights.SemiBold;
    public FontWeight StrongFontWeight => Language == AppLanguage.ZhCn ? FontWeights.Normal : FontWeights.SemiBold;

    // App Metadata
    public string AppTitle => "VoicierVoice";
    public string WindowTitle => "VoicierVoice";
    public string AppSubtitle => L("实时声纹降噪与目标人声提取", "Real-Time Voiceprint Isolation & Denoising");

    // Sidebar Navigation Tabs
    public string NavPipelineTitle => L("主页", "Home");
    public string NavVoiceprintTitle => L("声纹", "Voiceprint");
    public string NavComputeTitle => L("占用偏好", "Usage Preference");
    public string NavOtherTitle => L("其他设置", "Other Settings");

    // Master Switch & Telemetry
    public string MasterSwitchTitle => L("降噪开关", "Noise Reduction");
    public string MasterSwitchTooltip => L(
        "开启或关闭实时级联降噪与声纹提取管线",
        "Start or stop the real-time cascade denoising and voiceprint isolation pipeline");

    public string TotalLatency => L("算法延迟", "Total Latency");
    public string FrameInference => L("单帧推理耗时", "Frame Inference");
    public string ActiveCore => L("活跃核心", "Active Core");
    public string InputLevelLabel => "IN";
    public string OutputLevelLabel => "OUT";
    public string CoreUnbound => L("未绑定", "Unbound");

    public string StatusStopped => L("已停止", "Stopped");
    public string StatusRunning => L("正在过滤人声", "Running, Isolating Voice");
    public string StatusSleeping => L("静音休眠中，零算力占用", "Silent Sleep, 0% Compute");
    public string StatusStarting => L("正在启动音频管线...", "Starting audio pipeline...");
    public string StatusFaultedPrefix => L("管线异常", "Pipeline Fault");
    public string StatusInitFailedPrefix => L("初始化失败", "Initialization Failed");
    public string StatusDeviceSwitchFailedPrefix => L("设备切换失败", "Device Switch Failed");

    // Overview Guidance Card
    public string QuickStartTitle => L("快速上手", "Quick Start");
    public string QuickStartSubtitle => L("三步开启精准目标人声提取", "Three steps to start isolating your target voice");
    public string QuickStartStep1Title => L("选择设备", "Select Devices");
    public string QuickStartStep1Desc => L("指定物理麦克风输入与虚拟声卡输出", "Choose your physical mic input and virtual cable output");
    public string QuickStartStep2Title => L("录制声纹", "Enroll Voiceprint");
    public string QuickStartStep2Desc => L("朗读 5 秒引导文本建立本地个人声纹", "Read the 5-second prompt to enroll your personal voiceprint");
    public string QuickStartStep3Title => L("开启降噪", "Start Isolation");
    public string QuickStartStep3Desc => L("打开主开关并在通话软件中选择输出设备", "Enable the master switch and pick the output device in your app");
    public string OpenVoiceprintButton => L("前往声纹档案", "Open Voiceprint");
    public string DismissQuickStartButton => L("收起引导", "Dismiss");

    // Overview: Audio Routing & Cascade DSP
    public string RoutingCardTitle => L("音频路由与级联降噪管线", "Audio Routing & Cascade Pipeline");
    public string RefreshDevicesButton => L("刷新设备", "Refresh");
    public string RefreshDevicesTooltip => L(
        "刷新 WASAPI 音频输入与输出设备列表",
        "Refresh WASAPI audio capture and render device lists");

    public string InputDeviceLabel => L("输入麦克风", "Input Microphone");
    public string InputDeviceHint => L(
        "通过 WASAPI Capture 采集原始麦克风音频流，建议选择您的主力物理麦克风。",
        "Captures raw microphone audio via WASAPI Capture. Choose your primary physical microphone.");

    public string OutputDeviceLabel => L("输出设备", "Output Device");
    public string OutputDeviceHint => L(
        "建议选择虚拟声卡的Input端",
        "Recommended: select the Input endpoint of a virtual audio device.");

    public string DeviceDefaultBadge => L("默认", "Default");
    public string DeviceVirtualBadge => L("虚拟声卡", "Virtual");

    public string MonitorLoopbackLabel => L("实时监听自己声音", "Monitor Own Voice");
    public string MonitorLoopbackHint => L(
        "调试回环监听开关。输出至 VB-Cable 或 VoiceMeeter 等虚拟声卡时建议保持关闭，以免在耳机或通话中产生回声。",
        "Debug loopback monitor. Keep off when routing to VB-Cable or VoiceMeeter virtual devices to prevent echo.");

    public string VadStageTitle => "Silero-VAD";
    public string VadStageSummary => L(
        "无人说话时自动进入静音休眠，降低下游算力占用",
        "Enters silent sleep during pauses to save downstream compute");
    public string VadStageHint => L(
        "基于 Silero-VAD 实时检测人声活动。当语音概率低于设定阈值时自动旁路下游推理，实现静音时段 0% 算力消耗。",
        "Uses Silero-VAD to detect speech activity in real time. Bypasses downstream inference below threshold for 0% idle compute.");

    public string VadThresholdLabel => L("检测灵敏度阈值", "Speech Sensitivity Threshold");
    public string VadThresholdHint => L(
        "数值越高越不容易被环境杂音误触发；数值越低越能保留轻声细语。",
        "Higher values reject faint background noise; lower values preserve soft speech.");
    public string VadLiveProbBypassed => L("已旁路", "Bypassed");

    public string GtcrnStageTitle => "GTCRN";
    public string GtcrnStageSummary => L(
        "超轻量语音增强模型，优先消除键盘声与风扇底噪",
        "Ultra-light speech enhancer suppressing steady keyboard and fan noise");
    public string GtcrnStageHint => L(
        "GTCRN 单帧推理耗时低于 1.8 ms，在声纹提取前先行清除稳态环境底噪与高频杂音，提升分离纯净度。",
        "GTCRN runs in under 1.8 ms per frame, cleaning steady background noise before voiceprint extraction.");

    public string VoxLockStageTitle => "VoxLock";
    public string VoxLockStageSummary => L(
        "依据个人声纹锁相提取目标人声，抑制旁人交谈干扰",
        "Locks onto your voiceprint to isolate your voice and suppress nearby speakers");
    public string VoxLockStageHint => L(
        "依据本地 5 秒参考声纹作为锁相条件，从混合人声中精准分离您的声音并抑制旁人讲话。",
        "Conditions target-speaker extraction on your 5-second voiceprint to suppress overlapping voices.");
    public string VoxLockVariantLabel => L("模型精度变体", "Model Variant");
    public string VoxLockVariantHint => L(
        "INT8 兼顾极低延迟与日常精度；INT8 Full 提供更完整权重量化；FP16 适合 CUDA GPU 高精度推理。",
        "INT8 balances ultra-low latency and quality; INT8 Full uses full quantized weights; FP16 targets CUDA GPU precision.");
    public string VoxLockLowLatencyLabel => L("16 ms 超低延迟分块", "16 ms Low-Latency Chunk");
    public string VoxLockLowLatencyHint => L(
        "将 VoxLock 分块从 32 ms 缩减至 16 ms（总延迟再降 ~24 ms）；CPU 模式自动调度至高性能 P 核，CUDA 模式同步启用低延迟缓冲。",
        "Halves VoxLock chunks from 32 ms to 16 ms (~24 ms lower total latency); CPU mode schedules on P-cores and CUDA mode uses low-latency buffering.");

    // Voiceprint Tab
    public string VoiceprintCardTitle => L("声纹", "Voiceprint");
    public string VoiceprintCardHint => L(
        "录制 5 秒个人人声样本并生成与模型精度匹配的本地特征缓存，全程离线保存。",
        "Records a 5-second voice sample and builds a hash-verified local embedding cache offline.");

    public string VoiceprintGuideTitle => L("录制指南", "Recording Guide");
    public string VoiceprintGuideBody => L(
        "请在安静环境中保持麦克风距离 20 至 30 厘米，使用日常自然说话音量清晰朗读 5 秒。",
        "Record in a quiet room with the microphone 20 to 30 cm away, speaking in your natural tone for 5 seconds.");

    public string VoiceprintPromptLabel => L("请清晰朗读下方引导文本", "Read the prompt below clearly");
    public string VoiceprintPromptText => L(
        "今天天气真不错，我正在使用 VoicierVoice 录制个人声纹样本，用于在嘈杂环境中精准提取我的声音。",
        "Today is a great day. I am recording my personal voiceprint sample with VoicierVoice to isolate my voice clearly in noisy environments.");

    public string NavigateBackTooltip => L("返回上一页", "Back");
    public string OpenVoiceprintFolderTooltip => L("打开本地录音存储位置", "Open local recording folder");
    public string RecordVoiceprintButton => L("录制我的声纹", "Record Voiceprint");
    public string StopRecordingButton => L("停止录制", "Stop Recording");
    public string PlaySampleButton => L("试听原始样本", "Play Sample");
    public string StopPlaybackButton => L("停止试听", "Stop Playback");
    public string RebuildCacheButton => L("重建声纹缓存", "Rebuild Cache");
    public string RebuildCacheTooltip => L(
        "从本地 user_raw_voice.wav 重新提取当前模型精度对应的声纹特征缓存",
        "Re-extract the voiceprint embedding cache from local user_raw_voice.wav for the active variant");

    public string RecordingProgressLabel => L("录制进度", "Recording Progress");
    public string RecordingMicLevelLabel => L("实时麦克风电平", "Live Mic Level");

    public string RecordingIdleStatus => L(
        "点击按钮朗读上方文本 5 秒以录入声纹",
        "Click Record and read the prompt above for 5 seconds");
    public string RecordingStoppingStatus => L("正在停止录制...", "Stopping recording...");
    public string RecordingActiveDefaultStatus => L("正在录制 5 秒声纹样本...", "Recording 5-second voiceprint sample...");
    public string RecordingCanceledStatus => L("声纹录制已取消", "Voiceprint recording canceled");
    public string RebuildingCacheStatus => L("正在重建声纹模型缓存...", "Rebuilding voiceprint cache...");
    public string RebuildCacheDoneStatus => L("声纹缓存重建完成", "Voiceprint cache rebuilt");

    public string VoiceprintReadyCached => L("声纹已就绪，已加载本地特征缓存", "Voiceprint Ready, Cached");
    public string VoiceprintReadyFresh => L("声纹已就绪，已提取最新特征", "Voiceprint Ready, Extracted");
    public string VoiceprintWaitingRebuild => L("发现原始声纹样本，等待重建模型缓存", "Raw sample found, ready to build cache");
    public string VoiceprintNotEnrolled => L("尚未录入声纹，将直通音频", "Not Enrolled, Passthrough");
    public string VoiceprintEmptyDetail => L("请先录制 5 秒个人人声样本", "Record a 5-second personal voice sample first");
    public string VoiceprintPromptDetail => L("请朗读上方引导文本并点击录制按钮", "Read the prompt above and click Record Voiceprint");

    // Compute & Cores Tab
    public string ComputeCardTitle => L("推理负载偏好", "Inference Load Preference");

    public string ComputeGuideTitle => L("建议", "Guidance");
    public string ComputeGuideBody => L(
        "运行大型 3D 游戏时，推荐使用 CPU 单线程锁核至能效核，可避免与游戏争抢显卡资源导致的帧率波动；追求极致分离精度或启用 FP16 模型时，可切换至 CUDA GPU 加速。",
        "While playing demanding 3D games, CPU Single-Thread E-Core locking avoids GPU contention and frame-time drops. Switch to CUDA GPU acceleration for maximum separation quality or FP16 inference.");

    public string CpuBackendTitle => L("CPU", "CPU");
    public string CpuBackendSubtitle => L("低延迟零拷贝，稳定控制实时语音抖动", "Low latency, zero-copy, stable frame pacing");
    public string CpuBackendBadge => L("推荐", "Recommended");
    public string CpuBackendHint => L(
        "将 VAD、GTCRN 与 VoxLock 全部绑定在独立物理核心单线程运行，不占用显卡算力，极适合游戏开黑。",
        "Runs VAD, GTCRN, and VoxLock single-threaded on pinned physical cores with zero GPU usage.");

    public string CudaBackendTitle => L("CUDA GPU", "CUDA GPU");
    public string CudaBackendSubtitle => L("使用 NVIDIA 显卡运行 VoxLock 推理", "Runs VoxLock ONNX inference on NVIDIA GPU");
    public string CudaBackendHint => L(
        "将 VoxLock 模型卸载至 NVIDIA CUDA 执行，轻量级 VAD 与 GTCRN 仍保留在 CPU 运行以规避小模型调度开销。",
        "Offloads VoxLock to NVIDIA CUDA while keeping lightweight VAD and GTCRN on CPU to avoid kernel launch overhead.");

    public string AutoAffinityTitle => L("自动绑定能效核并锁定单线程", "Auto-Lock Efficiency Cores");
    public string AutoAffinitySubtitle => L("自动识别大小核架构并避开高负载主核", "Detects hybrid topology and avoids primary cores");
    public string AutoAffinityHint => L(
        "在大小核 CPU 上自动锁定 E-Core 能效核，在均质 CPU 上锁定尾部物理核，避免音频线程被游戏主线程抢占。",
        "Pins audio workers to E-Cores on hybrid CPUs or tail cores on uniform CPUs to prevent audio dropouts.");

    public string CustomCoreMatrixTitle => L("自定义 CPU 核心亲和性矩阵", "Custom CPU Core Affinity Matrix");
    public string CustomCoreMatrixHint => L(
        "关闭上方自动绑定开关后，可手动勾选要绑定的物理核心。",
        "Turn off Auto-Lock above to manually choose which physical cores run the audio thread.");
    public string ManualCoreHintText => L(
        "关闭自动绑定开关后可手动勾选物理核",
        "Turn off Auto-Lock above to select cores manually");
    public string ApplyCoreSelectionButton => L("应用核心选择", "Apply Core Selection");

    public string CoreRoleEfficiency => "E-Core";
    public string CoreRoleTail => L("尾核", "Tail");
    public string CoreRolePerformance => "P-Core";

    // Other Settings Tab
    public string OtherSettingsTitle => L("外观、语言与系统行为", "Appearance, Language & System");
    public string LanguageLabel => L("界面语言", "Interface Language");
    public string LanguageHint => L(
        "即时切换简体中文或英文界面并自动保存偏好。",
        "Switch between Simplified Chinese and English immediately with automatic persistence.");
    public string LanguageOptionZhCn => "简体中文";
    public string LanguageOptionEnUs => "English";

    public string ThemeLabel => L("界面主题", "Color Theme");
    public string ThemeHint => L(
        "支持跟随 Windows 系统外观，或强制切换为浅色、深色 Fluent 2 主题并同步标题栏。",
        "Follow Windows personalization or force Light or Dark Fluent 2 theme with DWM titlebar sync.");
    public string ThemeOptionSystem => L("跟随系统", "System");
    public string ThemeOptionLight => L("浅色", "Light");
    public string ThemeOptionDark => L("深色", "Dark");

    public string MinimizeToTrayLabel => L("关闭窗口时最小化到系统托盘", "Minimize to System Tray on Close");
    public string MinimizeToTrayHint => L(
        "关闭主窗口后保持后台实时降噪运行，可通过右下角托盘图标随时打开窗口或彻底退出。",
        "Keeps background processing alive in the system tray when closed; reopen or exit from the tray icon.");

    public string LaunchOnStartupLabel => L("开机自动启动", "Launch on Windows Startup");
    public string LaunchOnStartupHint => L(
        "Windows 登录后自动在后台启动 VoicierVoice。",
        "Automatically start VoicierVoice minimized when signing in to Windows.");

    public string AppDataPathLabel => L("本地数据目录", "Local Data Directory");
    public string ExitApplicationButton => L("退出程序", "Exit VoicierVoice");

    // About & Acknowledgements
    public string AboutCreditsTitle => L("核心引擎与开源鸣谢", "Core Engine & Open-Source Credits");
    public string AboutVoxLockBadge => L("核心引擎", "Core Engine");
    public string AboutVoxLockDesc => L(
        "目标人声锁相提取与声纹嵌入模型核心",
        "Target-speaker extraction & voiceprint embedding core");
    public string AboutOpenSourceDesc => L(
        "致谢以下开源语音模型与基础依赖库：",
        "Built with the following open-source models and libraries:");
    public string AboutSileroDesc => L(
        "实时人声活动检测与静音休眠门控",
        "Real-time voice activity detection & idle sleep gating");
    public string AboutGtcrnDesc => L(
        "超轻量级流式语音增强前置降噪",
        "Ultra-lightweight streaming speech enhancement denoiser");
    public string AboutOrtDesc => L(
        "CPU 与 NVIDIA CUDA 神经网络推理引擎",
        "CPU & NVIDIA CUDA ONNX neural inference runtime");
    public string AboutNAudioDesc => L(
        "Windows WASAPI 低延迟音频采集与渲染",
        "Windows WASAPI low-latency audio capture & render");
    public string AboutMvvmDesc => L(
        ".NET 现代化 MVVM 数据绑定与命令架构",
        "Modern .NET MVVM bindings & command primitives");

    // Tray Icon & Menu
    public string TrayShowWindow => L("显示主窗口", "Show Window");
    public string TrayTogglePipeline => L("切换实时降噪", "Toggle Isolation");
    public string TrayExit => L("退出程序", "Exit VoicierVoice");
    public string TrayReadyTip => L("VoicierVoice 已就绪", "VoicierVoice Ready");

    // Dynamic Formatters
    public string FormatBackendBanner(ComputeBackend backend, bool isFallback = false)
    {
        if (backend == ComputeBackend.Cuda)
        {
            return isFallback
                ? L("已回退至 CPU 单线程模式", "CPU Fallback, CUDA Unavailable")
                : L("CUDA GPU 加速", "CUDA GPU Acceleration");
        }

        return L("CPU 单线程锁核", "CPU Single-Thread Lock");
    }

    public string FormatCudaFallbackDefault() => L(
        "CUDA 执行提供程序不可用，VoxLock 已自动回退至 CPU 单线程模式。",
        "CUDAExecutionProvider is unavailable; VoxLock automatically fell back to CPU single-thread mode.");

    public string FormatCpuTopologySummary(CpuTopologyInfo topology)
    {
        ArgumentNullException.ThrowIfNull(topology);

        if (topology.IsHeterogeneous)
        {
            int pCores = topology.PhysicalCores.Count(c => !c.IsEfficiencyCore);
            int eCores = topology.PhysicalCores.Count(c => c.IsEfficiencyCore);
            return L(
                $"{topology.TotalLogicalProcessors} 逻辑核心，{pCores} 个 P-Core 与 {eCores} 个 E-Core，推荐自动锁定能效核",
                $"{topology.TotalLogicalProcessors} Logical, {pCores} P-Core + {eCores} E-Core, Auto E-Core Recommended");
        }

        return L(
            $"{topology.TotalLogicalProcessors} 逻辑核心，{topology.PhysicalCores.Count} 物理核心，推荐锁定尾核",
            $"{topology.TotalLogicalProcessors} Logical, {topology.PhysicalCores.Count} Physical, Tail Core Recommended");
    }

    public string FormatShortCoreDescription(string? rawDescription, CpuTopologyInfo topology)
    {
        if (string.IsNullOrWhiteSpace(rawDescription))
        {
            return CoreUnbound;
        }

        if (rawDescription.StartsWith("Auto E-Cores", StringComparison.OrdinalIgnoreCase))
        {
            int eCoreCount = topology.PhysicalCores.Count(c => c.IsEfficiencyCore);
            return L($"自动锁定 E-Core {eCoreCount} 核", $"Auto E-Core, {eCoreCount} cores");
        }

        if (rawDescription.StartsWith("Auto P-Cores", StringComparison.OrdinalIgnoreCase))
        {
            int pCoreCount = topology.PhysicalCores.Count(c => !c.IsEfficiencyCore);
            return L($"自动锁定 P-Core {pCoreCount} 核", $"Auto P-Core, {pCoreCount} cores");
        }

        if (rawDescription.StartsWith("Auto Tail Cores", StringComparison.OrdinalIgnoreCase))
        {
            return L("自动锁定尾核", "Auto Tail Cores");
        }

        if (rawDescription.StartsWith("Auto All Cores", StringComparison.OrdinalIgnoreCase))
        {
            return L("自动全核释放", "Auto All Cores");
        }

        if (rawDescription.Length > 28)
        {
            int parenIdx = rawDescription.IndexOf('(');
            return parenIdx > 0 ? rawDescription[..parenIdx].Trim() : rawDescription[..28];
        }

        return rawDescription;
    }

    public string FormatRecordingProgress(double elapsedSeconds, double totalSeconds) => L(
        $"正在录制声纹... {elapsedSeconds:0.0} / {totalSeconds:0.0} 秒",
        $"Recording voiceprint... {elapsedSeconds:0.0} / {totalSeconds:0.0} s");

    public string FormatRecordingCompleted(double durationSeconds) => L(
        $"录制并建模完成，{durationSeconds:0.0} 秒",
        $"Enrollment complete, {durationSeconds:0.0} s");

    public string FormatRecordingFailed(string message) => L(
        $"录制失败：{message}",
        $"Recording failed: {message}");

    public string FormatPlaybackFailed(string message) => L(
        $"试听失败：{message}",
        $"Playback failed: {message}");

    public string FormatRebuildFailed(string message) => L(
        $"重建缓存失败：{message}",
        $"Cache rebuild failed: {message}");

    public string FormatSwitchFailed(string message) => L(
        $"模型或后端切换失败：{message}",
        $"Backend or variant switch failed: {message}");

    public string FormatVoiceprintDetail(string fileName, double durationSeconds, VoxLockModelVariant variant) => L(
        $"已保存本地样本 {fileName}，时长 {durationSeconds:0.0} 秒",
        $"Saved sample {fileName}, {durationSeconds:0.0} s");

    public string FormatStatusFaulted(string message) => L(
        $"{StatusFaultedPrefix}：{message}",
        $"{StatusFaultedPrefix}: {message}");

    public string FormatInitFailed(string message) => L(
        $"{StatusInitFailedPrefix}：{message}",
        $"{StatusInitFailedPrefix}: {message}");

    public string FormatDeviceSwitchFailed(string message) => L(
        $"{StatusDeviceSwitchFailedPrefix}：{message}",
        $"{StatusDeviceSwitchFailedPrefix}: {message}");

    public string FormatVadLiveProb(float probability, bool speechActive, bool vadEnabled = true)
    {
        if (!vadEnabled)
        {
            return VadLiveProbBypassed;
        }

        string state = speechActive ? L("人声", "Speech") : L("静音", "Silence");
        return $"{probability:0.00} {state}";
    }

    public string FormatVariantLabel(VoxLockModelVariant variant) => variant switch
    {
        VoxLockModelVariant.Int8 => L("INT8 快速", "INT8 Fast"),
        VoxLockModelVariant.Int8Full => "INT8 Full",
        VoxLockModelVariant.Fp16 => L("FP16 高精度", "FP16 Half Precision"),
        _ => variant.ToString()
    };

    public string FormatTrayTooltip(string? statusText, bool isRunning)
    {
        string cleanStatus = string.IsNullOrWhiteSpace(statusText)
            ? (isRunning ? StatusRunning : StatusStopped)
            : statusText.Trim();

        string tip = $"VoicierVoice - {cleanStatus}";
        return tip.Length > 127 ? tip[..127] : tip;
    }
}
