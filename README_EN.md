<div align="center">
  <img src="Assets/Icons/VoicierVoice.png" alt="VoicierVoice Logo" width="112" height="112" />
  <h1>VoicierVoice</h1>
  <p>
    <strong>Real-Time Voiceprint Isolation and Denoising Desktop Application for Windows</strong><br />
    <a href="README.md">简体中文</a> | English
  </p>
</div>

---

## Overview

**VoicierVoice** is a Windows 10 / 11 desktop application built on **[VoxLock](https://github.com/arbazsiddiqui/voxlock)** for real-time personal voiceprint isolation and speech denoising. By chaining Silero VAD, GTCRN, and VoxLock using a 5-second local voice sample, it isolates the enrolled speaker's voice while suppressing background noise and nearby speech.

### Features
- **Processing Pipeline**: Processes 16 kHz mono audio frames sequentially through three ONNX stages:
  1. **Silero VAD**: Real-time voice activity detection (32 ms / 512 samples). Enters idle sleep and bypasses downstream inference when speech probability drops below threshold.
  2. **GTCRN**: Streaming speech enhancement model (16 ms / 256 samples) for suppressing stationary background noise.
  3. **VoxLock**: Target-speaker extraction conditioned on a local 5-second voiceprint embedding, supporting `INT8`, `INT8 Full`, and `FP16` model variants.
- **WASAPI Audio Stream**: Uses timer-driven polling mode WASAPI Capture (25 ms) and WASAPI Render (45 ms), persistent streaming resamplers, a 54 ms output pre-buffer watermark, and a 64-sample linear crossfade on underrun/resume. Virtual audio endpoints such as VB-Cable and VoiceMeeter are automatically detected.
- **Compute Backend & Core Affinity**: Supports CPU single-thread affinity locking and NVIDIA CUDA GPU acceleration. Automatically detects hybrid P-Core / E-Core topology and binds audio worker threads to efficiency cores (or tail cores on homogeneous CPUs), with optional manual core selection.
- **Fluent 2 UI**: Supports System / Light / Dark themes synced with Windows 11 DWM Mica and caption bars, runtime Simplified Chinese / English switching, and minimize-to-tray background operation.

---

## Dependencies & Acknowledgements

### 1. Models & Core Engine
- **[VoxLock](https://github.com/arbazsiddiqui/voxlock)**: Core target-speaker extraction and voiceprint enrollment model (model weights from [`arbazsiddiqui/voxlock`](https://huggingface.co/arbazsiddiqui/voxlock): `enroll.onnx`, `enroll_fp16.onnx`, `chunk_int8.onnx`, `chunk_int8_full.onnx`, `chunk_fp16.onnx`).
- **[Silero VAD](https://github.com/snakers4/silero-vad)**: Voice activity detection model (`silero_vad.onnx`).
- **[GTCRN](https://github.com/Xiaobin-Rong/gtcrn)** / **[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)**: Streaming speech enhancement model (`gtcrn_simple.onnx`).

### 2. Runtime & Open-Source Libraries
| Component / NuGet Package | Version | Purpose | Repository |
| :--- | :--- | :--- | :--- |
| **.NET Desktop / WPF** | `.NET 10.0` (`net10.0-windows`) | Windows desktop UI and shell integration | [dotnet/wpf](https://github.com/dotnet/wpf) |
| **Microsoft.ML.OnnxRuntime.Gpu** | `1.17.1` | CPU & CUDA ONNX inference runtime | [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime) |
| **NAudio** | `2.3.0` | WASAPI audio capture, resampling, and render | [naudio/NAudio](https://github.com/naudio/NAudio) |
| **CommunityToolkit.Mvvm** | `8.4.2` | MVVM observable properties and commands | [CommunityToolkit/dotnet](https://github.com/CommunityToolkit/dotnet) |

### 3. Requirements
- **OS**: Windows 10 / Windows 11 (x64)
- **Virtual Audio Device** (required when routing processed audio into communication or recording apps): [VB-Audio Virtual Cable](https://vb-audio.com/Cable/) or [VoiceMeeter](https://vb-audio.com/Voicemeeter/)
- **GPU Acceleration (Optional)**: CUDA-capable NVIDIA GPU with compatible CUDA / cuDNN libraries; automatically falls back to CPU single-thread mode if unavailable

---

## Usage Guide

### 1. Configure a Virtual Audio Cable (Optional)
To route processed audio into third-party voice, meeting, or recording applications:
1. Install a virtual audio driver such as VB-Cable or VoiceMeeter.
2. In VoicierVoice (**Home** tab), select the virtual cable input as the output device (e.g., `CABLE Input` or `VoiceMeeter Input`).
3. In your communication or recording software, select the corresponding virtual output endpoint as the microphone (e.g., `CABLE Output` or `VoiceMeeter Out`).

> Note: Keep "Monitor Own Voice" disabled when outputting to a virtual audio device to avoid hearing loopback echo.

### 2. Basic Workflow
1. **Select Devices**: On the **Home** tab, choose your physical input microphone and target output device.
2. **Enroll Voiceprint**: Switch to the **Voiceprint** tab, click **Record Voiceprint** in a quiet environment, and read the 5-second prompt aloud to build a local embedding cache (stored under `%LocalAppData%\VoicierVoice`).
3. **Start Pipeline**: Toggle **Live Voice Isolation** in the top-right header. The status kaomoji reflects the current pipeline state:
   - `( • vv • )`: Stopped / Standby
   - `d( • vv • )b`: Running, currently in VAD silent sleep
   - `d( ^ ∀ ^ )b`: Running, actively isolating target speech

### 3. Compute & Preferences
- **CPU&GPU**: Switch between **CPU Single-Thread Lock** and **CUDA GPU Acceleration**, and configure automatic efficiency-core locking or custom CPU core affinity.
- **Other Settings**: Change theme preference (System / Light / Dark), interface language (Simplified Chinese / English), and minimize-to-tray behavior.

---

## Build and Release Packaging

### 1. Local Build & Self-Test

```powershell
# 1. Clone the repository
git clone https://github.com/WoaShi/VoicierVoice.git
cd VoicierVoice

# 2. (Optional) Re-download and verify ONNX models and model_manifest.json
python scripts/prepare_models.py

# 3. Build Release
dotnet build VoicierVoice.csproj -c Release

# 4. Run built-in headless self-test (manifest, topology, inference, ring buffer, UI smoke test)
.\bin\Release\net10.0-windows\VoicierVoice.exe --self-test

# 5. Launch the application
dotnet run --project VoicierVoice.csproj -c Release
```

### 2. Self-Contained Multi-Architecture Release Packages

```powershell
# Generate self-contained single-file release zips (win-x64 and win-arm64) and SHA256SUMS.txt under dist/
powershell -ExecutionPolicy Bypass -File scripts/build_releases.ps1 -Version 1.0.0
```

- `win-x64`: Bundles .NET 10 Desktop Runtime and `Microsoft.ML.OnnxRuntime.Gpu` (supports both CPU and NVIDIA CUDA GPU acceleration).
- `win-arm64`: Bundles .NET 10 Desktop Runtime and `Microsoft.ML.OnnxRuntime` (CPU execution provider for Windows on ARM64).
- Note: 32-bit `win-x86` is excluded from default release distribution because 32-bit x86 inference for VoxLock INT8 measured `RTF=1.400`, which exceeds real-time (`< 1.0`).

---

## License

This project is licensed under the [Apache License 2.0](LICENSE). See the [LICENSE](LICENSE) file in the repository root for details.
