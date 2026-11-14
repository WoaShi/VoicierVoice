<div align="center">
  <img src="Assets/Icons/VoicierVoice.png" alt="VoicierVoice Logo" width="112" height="112" />
  <h1>VoicierVoice</h1>
  <p>
    <strong>面向 Windows 的实时声纹降噪与目标人声提取桌面应用</strong><br />
    简体中文 | <a href="README_EN.md">English</a>
  </p>
</div>

> **前提说明**：将处理后的音频接入 Discord、腾讯会议、OBS 或游戏内置语音等第三方软件时，本软件**依赖虚拟声卡**（如 [VB-Audio Virtual Cable](https://vb-audio.com/Cable/) 或 [VoiceMeeter](https://vb-audio.com/Voicemeeter/)）作为中间音频路由设备。请在使用前先完成虚拟声卡驱动的安装。

---

## 简介

**VoicierVoice** 是一款基于 **[VoxLock](https://github.com/arbazsiddiqui/voxlock)** 的 Windows 10 / 11 实时个人声纹提取与语音降噪桌面应用。软件通过串联 Silero VAD、GTCRN 与 VoxLock 处理管线，在本地使用 5 秒个人人声样本提取目标说话人声纹特征，在实时语音场景中保留目标人声并抑制背景噪声与旁人语音。

### 特性
- **处理管线**：按 16 kHz 单声道处理音频流，串联以下三个阶段：
  1. **Silero VAD**：实时语音活动检测（32 ms / 512 采样点），低于阈值时进入静音休眠并旁路下游推理。
  2. **GTCRN**：流式语音增强模型（16 ms / 256 采样点），用于抑制稳态背景噪声。
  3. **VoxLock**：依据本地 5 秒参考声纹进行目标人声提取，支持 `INT8`、`INT8 Full`、`FP16` 三种模型变体。
- **WASAPI 音频流**：采用定时器轮询模式 WASAPI Capture（25 ms）与 WASAPI Render（45 ms），配置常驻流式重采样器、54 ms 起播预缓冲水位线以及 64 采样点线性淡入淡出过渡，并自动识别 VB-Cable、VoiceMeeter 等虚拟声卡端点。
- **算力后端与核心调度**：支持 CPU 单线程锁核与 NVIDIA CUDA GPU 加速；在异构 CPU 上可自动识别 P-Core / E-Core 并绑定至能效核（均质 CPU 绑定尾部物理核），也支持手动配置核心亲和性掩码。
- **Fluent 2 界面**：支持跟随系统 / 浅色 / 深色主题（同步 Win11 DWM Mica 与标题栏外观）、简体中文 / English 实时切换、开机自动启动以及关闭窗口时最小化至系统托盘运行。

---

## 依赖与致谢

### 1. 音频驱动与核心模型
- **虚拟声卡驱动（必需，用于软件间音频路由）**：[VB-Audio Virtual Cable](https://vb-audio.com/Cable/) 或 [VoiceMeeter](https://vb-audio.com/Voicemeeter/)
- **[VoxLock](https://github.com/arbazsiddiqui/voxlock)**：目标人声提取与声纹嵌入模型核心（模型文件来源：[`arbazsiddiqui/voxlock`](https://huggingface.co/arbazsiddiqui/voxlock)，包含 `enroll.onnx`、`enroll_fp16.onnx`、`chunk_int8.onnx`、`chunk_int8_full.onnx`、`chunk_fp16.onnx`）。
- **[Silero VAD](https://github.com/snakers4/silero-vad)**：语音活动检测模型（`silero_vad.onnx`）。
- **[GTCRN](https://github.com/Xiaobin-Rong/gtcrn)** / **[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)**：流式语音增强模型（`gtcrn_simple.onnx`）。

### 2. 运行时与开源库
| 组件 / NuGet 包 | 版本 | 用途 | 仓库地址 |
| :--- | :--- | :--- | :--- |
| **.NET Desktop / WPF** | `.NET 10.0` (`net10.0-windows`) | Windows 桌面界面与系统集成 | [dotnet/wpf](https://github.com/dotnet/wpf) |
| **Microsoft.ML.OnnxRuntime.Gpu** | `1.17.1` | CPU 与 CUDA ONNX 推理运行时 | [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime) |
| **NAudio** | `2.3.0` | WASAPI 音频采集、重采样与渲染 | [naudio/NAudio](https://github.com/naudio/NAudio) |
| **CommunityToolkit.Mvvm** | `8.4.2` | MVVM 数据绑定与命令支持 | [CommunityToolkit/dotnet](https://github.com/CommunityToolkit/dotnet) |

### 3. 系统环境要求
- **操作系统**：Windows 10 / Windows 11（64 位）
- **自包含分发说明**：GitHub Releases 提供的压缩包已内含 .NET 10 运行时与依赖库，解压即可运行；仅在使用 CUDA GPU 加速时需要本机具备支持 CUDA 的 NVIDIA 显卡及 CUDA / cuDNN 运行环境（不可用时自动回退 CPU 模式）。

---

## 使用说明

### 第一步：安装并配置虚拟声卡与输入输出设备
1. 安装虚拟声卡驱动（推荐 **VB-Cable** 或 **VoiceMeeter**）。
2. 打开 VoicierVoice **主页**，向下滚动至「音频路由与级联降噪管线」：
   - **输入麦克风**：选择你的物理麦克风设备。
   - **输出设备**：选择虚拟声卡输入端（如 `CABLE Input (VB-Audio Virtual Cable)` 或 `VoiceMeeter Input`）。
3. 在你的通话或录制软件（如 Discord、腾讯会议、OBS）中，将麦克风设备设置为对应的虚拟声卡输出端（如 `CABLE Output` 或 `VoiceMeeter Out`）。

> 注：向虚拟声卡输出时，建议关闭「实时监听自己声音」以避免音频回声。

![选择物理输入麦克风与虚拟声卡输出设备](docs/images/zh-home-routing.png)

### 第二步：录制 5 秒个人声纹
1. 切换至左侧导航栏的 **声纹** 页面。
2. 在相对安静的环境中点击 **录制我的声纹**，按上方引导文本自然朗读 5 秒。
3. 录制完成后会自动生成本地声纹特征缓存（存储于 `%LocalAppData%\VoicierVoice`，可通过右上角文件夹图标打开）。

![录制 5 秒个人声纹样本](docs/images/zh-voiceprint.png)

### 第三步：开启实时声纹提取
点击右上角 **实时声纹提取** 开关即可开始处理。顶部状态颜文字会随管线运行状态动态变化：
- `( • vv • )`：已停止 / 待机
- `d( • vv • )b`：管线运行中，当前处于静音休眠状态
- `d( ^ ∀ ^ )b`：管线运行中，当前正在处理人声输入

![主页概览与实时声纹提取开关](docs/images/zh-home-hero.png)

### 第四步：算力调度与系统设置
- **CPU&GPU**：游戏等高 GPU 负载场景推荐使用 **CPU 单线程锁核** 并开启自动绑定能效核；需启用 `FP16` 或 GPU 推理时可切换至 **CUDA GPU 加速**。
- **其他设置**：支持切换 **跟随系统 / 浅色 / 深色** 主题、**简体中文 / English** 双语界面、**开机自动启动**（登录 Windows 后后台静默启动）以及 **关闭窗口时最小化到系统托盘**。

![其他设置与开机自启动选项](docs/images/zh-other-settings.png)

---

## 从源码构建与打包分发

### 1. 日常构建与测试

```powershell
# 1. 克隆仓库
git clone https://github.com/WoaShi/VoicierVoice.git
cd VoicierVoice

# 2. （可选）重新下载并校验 ONNX 模型与 model_manifest.json
python scripts/prepare_models.py

# 3. 编译 Release 版本
dotnet build VoicierVoice.csproj -c Release

# 4. 运行内置自检（模型清单、CPU 拓扑、推理、环形缓冲区与 UI 冒烟测试）
.\bin\Release\net10.0-windows\VoicierVoice.exe --self-test

# 5. 启动应用
dotnet run --project VoicierVoice.csproj -c Release
```

### 2. 自包含多架构 Release 打包

```powershell
# 在 dist/ 下生成自包含单文件 Release 压缩包（默认 win-x64 与 win-arm64）及 SHA256SUMS.txt
powershell -ExecutionPolicy Bypass -File scripts/build_releases.ps1 -Version 1.0.1
```

- `win-x64`：内置 .NET 10 桌面运行时与 `Microsoft.ML.OnnxRuntime.Gpu`，支持 CPU 与 NVIDIA CUDA GPU 加速。
- `win-arm64`：内置 .NET 10 桌面运行时与 `Microsoft.ML.OnnxRuntime`（CPU 执行后端），适用于 Windows on ARM64 设备。
- 注：已跳过 32 位 `win-x86` 作为正式分发目标，实测 32 位架构运行 VoxLock INT8 推理的 RTF 约为 1.40（无法维持低于 1.0 的实时率）。

---

## 许可证

本项目采用 [Apache License 2.0](LICENSE) 协议开源。详情请参阅根目录下的 [LICENSE](LICENSE) 文件。
