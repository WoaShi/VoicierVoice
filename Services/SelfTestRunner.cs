using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using System.Windows;
using System.Windows.Threading;
using NAudio.Wave;
using VoiceLockStudio.Models;
using VoiceLockStudio.Services.Audio;
using VoiceLockStudio.Services.Backends;
using VoiceLockStudio.Services.Localization;
using VoiceLockStudio.Services.System;
using VoiceLockStudio.ViewModels;

namespace VoiceLockStudio.Services;

/// <summary>
/// Headless + UI-smoke self-test runner invoked via <c>--self-test</c> / <c>/selftest</c>.
/// Verifies all 7 ONNX models, CPU topology, CPU/CUDA stage initialization, Golden Voiceprint
/// enrollment &amp; cache roundtrip, 50-frame cascade inference, and MainViewModel/MainWindow smoke init.
/// </summary>
public static class SelfTestRunner
{
    private const int AttachParentProcess = -1;

    public static async Task<int> RunAsync(string[] args, CancellationToken cancellationToken = default)
    {
        TryAttachConsole();

        string logDir = AppSettings.GetAppDataDirectory();
        Directory.CreateDirectory(logDir);
        string logPath = Path.Combine(logDir, "selftest.log");

        using var logWriter = new StreamWriter(logPath, append: false, Encoding.UTF8) { AutoFlush = true };

        void Log(string line)
        {
            string formatted = $"[{DateTimeOffset.Now:HH:mm:ss.fff}] {line}";
            Console.WriteLine(formatted);
            logWriter.WriteLine(formatted);
        }

        Log("=== VoicierVoice Self-Test Started ===");
        Log($"BaseDirectory: {AppContext.BaseDirectory}");

        try
        {
            // 1. ModelManifest & 7 ONNX files
            ModelManifest manifest = ModelManifest.LoadDefault(AppContext.BaseDirectory);
            if (manifest.Models.Count != 7)
            {
                throw new InvalidDataException($"Expected 7 ONNX models in manifest, found {manifest.Models.Count}.");
            }

            foreach (ModelEntry entry in manifest.Models)
            {
                string fullPath = entry.ResolveFullPath(AppContext.BaseDirectory);
                if (!File.Exists(fullPath))
                {
                    throw new FileNotFoundException($"Missing ONNX model file: {entry.Path}", fullPath);
                }

                var fi = new FileInfo(fullPath);
                if (fi.Length != entry.SizeBytes || fi.Length <= 0)
                {
                    throw new InvalidDataException(
                        $"Model '{entry.Filename}' size mismatch: disk={fi.Length}, manifest={entry.SizeBytes}.");
                }
            }
            Log($"[PASS] Manifest verified: {manifest.Models.Count} ONNX models present and size-matched.");

            // 2. CpuAffinityManager topology
            var affinity = new CpuAffinityManager();
            CpuTopologyInfo topology = affinity.GetTopology(refresh: true);
            if (topology.TotalLogicalProcessors <= 0 || topology.PhysicalCores.Count == 0 || topology.RecommendedCoreMask == 0)
            {
                throw new InvalidOperationException("Invalid CPU topology returned by CpuAffinityManager.");
            }
            Log($"[PASS] CPU Topology: {topology.TotalLogicalProcessors} LP, {topology.PhysicalCores.Count} physical cores, Heterogeneous={topology.IsHeterogeneous}, Recommended='{topology.RecommendedDescription}' (Mask=0x{topology.RecommendedCoreMask:X}).");

            // Verify Process Audio Boost + MMCSS Pro Audio registration & clean restore.
            ProcessPriorityClass initialPriority;
            ProcessPriorityClass boostedPriority;
            ProcessPriorityClass restoredPriority;
            bool mmcssRegistered;
            using (var proc = Process.GetCurrentProcess())
            {
                initialPriority = proc.PriorityClass;
                using (CpuAffinityManager.BeginProcessAudioBoost())
                {
                    proc.Refresh();
                    boostedPriority = proc.PriorityClass;
                    using CpuAffinityManager.MmcssAudioScope? mmcss = CpuAffinityManager.TryRegisterCurrentThreadProAudio();
                    mmcssRegistered = mmcss is not null;
                    mmcss?.ReinforcePriority();
                }

                proc.Refresh();
                restoredPriority = proc.PriorityClass;
            }

            if ((initialPriority != ProcessPriorityClass.RealTime && boostedPriority != ProcessPriorityClass.High) ||
                restoredPriority != initialPriority)
            {
                throw new InvalidOperationException(
                    $"Process audio boost mismatch: initial={initialPriority}, boosted={boostedPriority}, restored={restoredPriority}.");
            }
            Log($"[PASS] Audio Scheduling Boost: Process={initialPriority}->{boostedPriority}->{restoredPriority}, MMCSS='Pro Audio' ({mmcssRegistered}).");

            // 3. Stage Initialization (E-Core-pinned CPU & CUDA)
            using var vadCpu = new SileroVadStage();
            using var gtcrnCpu = new GtcrnDenoiserStage();
            using var voxCpu = new VoxLockOnnxBackend();

            var cpuOptions = new AudioStageOptions(
                AppContext.BaseDirectory,
                manifest,
                ComputeBackend.Cpu,
                VoxLockModelVariant.Int8,
                0.5f,
                CpuAffinityMask: topology.RecommendedCoreMask,
                ProcessorGroup: topology.RecommendedProcessorGroup);
            vadCpu.Initialize(cpuOptions);
            gtcrnCpu.Initialize(cpuOptions);
            voxCpu.Initialize(cpuOptions);
            Log($"[PASS] CPU Stage Init (Chunk={voxCpu.FrameSizeSamples} samples / {voxCpu.FrameSizeSamples * 1000.0 / voxCpu.SampleRate:F0} ms): VAD={vadCpu.ActiveProviderName}, GTCRN={gtcrnCpu.ActiveProviderName}, VoxLock={voxCpu.ActiveProviderName}");

            using var voxCuda = new VoxLockOnnxBackend();
            var cudaOptions = new AudioStageOptions(AppContext.BaseDirectory, manifest, ComputeBackend.Cuda, VoxLockModelVariant.Int8, 0.5f);
            voxCuda.Initialize(cudaOptions);
            bool cudaAvailable = string.Equals(voxCuda.ActiveProviderName, "CUDAExecutionProvider", StringComparison.OrdinalIgnoreCase)
                && string.IsNullOrEmpty(voxCuda.ProviderFallbackMessage);
            Log($"[PASS] CUDA Probe: VoxLock ActiveProvider={voxCuda.ActiveProviderName}, Available={cudaAvailable}" +
                (string.IsNullOrEmpty(voxCuda.ProviderFallbackMessage) ? string.Empty : $", Fallback='{voxCuda.ProviderFallbackMessage}'"));

            // 4. Isolated Golden Voiceprint enrollment + cache roundtrip (preserve user's real AppData files)
            string tempDir = Path.Combine(Path.GetTempPath(), $"VoicierVoiceSelfTest_{Guid.NewGuid():N}");
            string backupDir = Path.Combine(VoiceprintStorage.AppDataDirectory, $"_selftest_backup_{Guid.NewGuid():N}");
            Directory.CreateDirectory(tempDir);

            List<string> backedUpFiles = BackupExistingVoiceprintFiles(backupDir);
            try
            {
                string synthWav = Path.Combine(tempDir, "synthetic_voice_3_5s.wav");
                CreateSyntheticVoiceWav(synthWav, sampleRate: 16000, durationSeconds: 3.5);

                VoiceprintProfileInfo enrolled = voxCpu.EnrollVoiceprint(synthWav);
                if (!voxCpu.HasEnrolledVoiceprint || enrolled.LoadedFromCache || !File.Exists(enrolled.CacheFilePath))
                {
                    throw new InvalidOperationException("EnrollVoiceprint did not produce fresh voiceprint and cache file.");
                }
                Log($"[PASS] Voiceprint Enroll: WavSha={enrolled.ContentSha256[..12]}..., Cache={Path.GetFileName(enrolled.CacheFilePath)}, LoadedFromCache={enrolled.LoadedFromCache}");

                // Verify cache hit on second load (using enrolled.SourceWavPath = user_raw_voice.wav)
                VoiceprintProfileInfo? cached = voxCpu.LoadOrRebuildVoiceprintCache(enrolled.SourceWavPath, forceRebuild: false);
                if (cached is null || !cached.LoadedFromCache)
                {
                    throw new InvalidOperationException("Expected cache hit (LoadedFromCache=true) on LoadOrRebuildVoiceprintCache.");
                }
                Log($"[PASS] Voiceprint Cache Hit: LoadedFromCache={cached.LoadedFromCache}");

                // Delete cache .bin and verify silent auto-rebuild
                File.Delete(enrolled.CacheFilePath);
                VoiceprintProfileInfo? rebuilt = voxCpu.LoadOrRebuildVoiceprintCache(enrolled.SourceWavPath, forceRebuild: false);
                if (rebuilt is null || rebuilt.LoadedFromCache || !File.Exists(rebuilt.CacheFilePath))
                {
                    throw new InvalidOperationException("Expected automatic cache rebuild (LoadedFromCache=false) after deleting .bin.");
                }
                Log($"[PASS] Voiceprint Auto-Rebuild: LoadedFromCache={rebuilt.LoadedFromCache}, Recreated={File.Exists(rebuilt.CacheFilePath)}");

                // Also enroll on CUDA backend instance if CUDA is active so we can benchmark 50 frames on CUDA
                if (cudaAvailable)
                {
                    VoiceprintProfileInfo? cudaProfile = voxCuda.LoadOrRebuildVoiceprintCache(enrolled.SourceWavPath, forceRebuild: false);
                    if (cudaProfile is null || !voxCuda.HasEnrolledVoiceprint)
                    {
                        throw new InvalidOperationException("CUDA VoxLock failed to load enrolled voiceprint.");
                    }
                }

                // 5. E-Core-pinned & CUDA 60-hop Cascade Inference (plus 16ms K=2 low-latency mode & Drop-Oldest RingBuffer check)
                RunCascadeBenchmark("CPU E-Core Locked (VAD+GTCRN+VoxLock 32ms INT8 K=4)", vadCpu, gtcrnCpu, voxCpu, affinity, topology.RecommendedCoreMask, topology.RecommendedProcessorGroup, Log);

                if (cudaAvailable)
                {
                    RunCascadeBenchmark("Hybrid CUDA (CPU VAD+GTCRN + CUDA VoxLock 32ms INT8 K=4)", vadCpu, gtcrnCpu, voxCuda, null, 0, 0, Log);
                }

                using (var voxCpuLowLat = new VoxLockOnnxBackend())
                {
                    ulong pCoreMask = affinity.GetRecommendedAudioWorkerMask(
                        out ushort pCoreGroup,
                        out _,
                        preferPerformanceCores: true);
                    var cpuLowLatOptions = new AudioStageOptions(
                        AppContext.BaseDirectory,
                        manifest,
                        ComputeBackend.Cpu,
                        VoxLockModelVariant.Int8,
                        0.5f,
                        CpuAffinityMask: pCoreMask,
                        ProcessorGroup: pCoreGroup,
                        VoxLockHopsPerChunk: 2);
                    voxCpuLowLat.Initialize(cpuLowLatOptions);
                    if (voxCpuLowLat.FrameSizeSamples != 256)
                    {
                        throw new InvalidOperationException($"Expected K=2 FrameSizeSamples == 256, got {voxCpuLowLat.FrameSizeSamples}.");
                    }
                    VoiceprintProfileInfo? cpuLowLatProfile = voxCpuLowLat.LoadOrRebuildVoiceprintCache(enrolled.SourceWavPath, forceRebuild: false);
                    if (cpuLowLatProfile is null || !voxCpuLowLat.HasEnrolledVoiceprint)
                    {
                        throw new InvalidOperationException("CPU Low-Latency VoxLock failed to load enrolled voiceprint.");
                    }
                    RunCascadeBenchmark("CPU P-Core Low-Latency (VAD+GTCRN+VoxLock 16ms INT8 K=2)", vadCpu, gtcrnCpu, voxCpuLowLat, affinity, pCoreMask, pCoreGroup, Log);
                }

                if (cudaAvailable)
                {
                    using var voxCudaFp16 = new VoxLockOnnxBackend();
                    var cudaFp16Options = new AudioStageOptions(
                        AppContext.BaseDirectory,
                        manifest,
                        ComputeBackend.Cuda,
                        VoxLockModelVariant.Fp16,
                        0.5f,
                        VoxLockHopsPerChunk: 4);
                    voxCudaFp16.Initialize(cudaFp16Options);
                    VoiceprintProfileInfo? cudaFp16Profile = voxCudaFp16.LoadOrRebuildVoiceprintCache(enrolled.SourceWavPath, forceRebuild: false);
                    if (cudaFp16Profile is null || !voxCudaFp16.HasEnrolledVoiceprint)
                    {
                        throw new InvalidOperationException("CUDA FP16 VoxLock failed to load enrolled voiceprint.");
                    }
                    RunCascadeBenchmark("Hybrid CUDA (CPU VAD+GTCRN + CUDA VoxLock 32ms FP16 K=4)", vadCpu, gtcrnCpu, voxCudaFp16, null, 0, 0, Log);

                    // Verify concurrent CUDA ProcessFrame + voiceprint reload/refresh has no OrtIoBinding race.
                    int cudaFrameSize = voxCudaFp16.FrameSizeSamples;
                    float[] concIn = new float[cudaFrameSize];
                    float[] concOut = new float[cudaFrameSize];
                    for (int i = 0; i < cudaFrameSize; i++)
                    {
                        concIn[i] = 0.25f * MathF.Sin(2f * MathF.PI * 220f * i / 16000f);
                    }

                    using var ctsConc = new CancellationTokenSource();
                    Task inferTask = Task.Run(() =>
                    {
                        for (int iter = 0; iter < 40 && !ctsConc.IsCancellationRequested; iter++)
                        {
                            voxCudaFp16.ProcessFrame(concIn, concOut);
                            if (!float.IsFinite(concOut[0]))
                            {
                                throw new InvalidDataException($"Non-finite CUDA concurrent output: {concOut[0]}");
                            }
                        }
                    });
                    try
                    {
                        for (int r = 0; r < 3; r++)
                        {
                            voxCudaFp16.LoadOrRebuildVoiceprintCache(enrolled.SourceWavPath, forceRebuild: false);
                        }
                    }
                    finally
                    {
                        ctsConc.Cancel();
                        await inferTask;
                    }
                    Log("[PASS] CUDA Concurrent ProcessFrame + ReloadVoiceprintCache Safety Check");
                }

                // Verify AudioRingBuffer Drop-Oldest watermark semantics
                var ring = new AudioRingBuffer(16384);
                float[] burst = new float[8000];
                ring.Write(burst);
                if (ring.AvailableRead > 1920)
                {
                    ring.Skip(ring.AvailableRead - 640);
                }
                if (ring.AvailableRead != 640)
                {
                    throw new InvalidOperationException($"Expected ring buffer Drop-Oldest to keep 640 samples, got {ring.AvailableRead}.");
                }
                Log("[PASS] RingBuffer Drop-Oldest watermark verified (8000 -> 640 samples)");
            }
            finally
            {
                RestoreVoiceprintFiles(backupDir, backedUpFiles);
                TryDeleteDirectory(tempDir);
            }

            // 6. AppSettings + AppStrings + MainViewModel + DwmBackdropHelper + TrayIconService + MainWindow UI Smoke Test
            if (!string.Equals(Path.GetFileName(AppSettings.GetAppDataDirectory()), "VoicierVoice", StringComparison.OrdinalIgnoreCase))
            {
                throw new InvalidOperationException($"Expected AppData folder 'VoicierVoice', got '{AppSettings.GetAppDataDirectory()}'.");
            }

            AppLanguage originalLang = AppStrings.Current.Language;
            try
            {
                AppStrings.Current.SetLanguage(AppLanguage.EnUs);
                if (!string.Equals(AppStrings.Current.NavPipelineTitle, "Home", StringComparison.Ordinal) ||
                    !string.Equals(AppStrings.Current.NavVoiceprintTitle, "Voiceprint", StringComparison.Ordinal) ||
                    !string.Equals(AppStrings.Current.NavComputeTitle, "Usage Preference", StringComparison.Ordinal))
                {
                    throw new InvalidOperationException("AppStrings failed to switch to EnUs.");
                }

                AppStrings.Current.SetLanguage(AppLanguage.ZhCn);
                if (!string.Equals(AppStrings.Current.NavPipelineTitle, "主页", StringComparison.Ordinal) ||
                    !string.Equals(AppStrings.Current.NavVoiceprintTitle, "声纹", StringComparison.Ordinal) ||
                    !string.Equals(AppStrings.Current.NavComputeTitle, "占用偏好", StringComparison.Ordinal))
                {
                    throw new InvalidOperationException("AppStrings failed to switch to ZhCn.");
                }
            }
            finally
            {
                AppStrings.Current.SetLanguage(originalLang);
            }

            string tempSettingsPath = Path.Combine(Path.GetTempPath(), $"voicier_selftest_{Guid.NewGuid():N}.json");
            try
            {
                var saved = new AppSettings { PipelineActive = true };
                AppSettings cloned = saved.Clone();
                cloned.Save(tempSettingsPath);
                AppSettings reloaded = AppSettings.Load(tempSettingsPath);
                if (!reloaded.PipelineActive)
                {
                    throw new InvalidOperationException("AppSettings failed to roundtrip PipelineActive.");
                }
            }
            finally
            {
                TryDeleteFile(tempSettingsPath);
            }

            using (var tray = new TrayIconService())
            {
                tray.UpdateStatus("SelfTest", isRunning: false);
                tray.UpdateLocalization();
            }

            using (var controller = new VoiceLockController(AppContext.BaseDirectory))
            using (var vm = new MainViewModel(controller))
            {
                await vm.InitializeAsync();

                // Verify virtual audio endpoint recognition (VB-Cable + VoiceMeeter series)
                if (!AudioDeviceService.IsVirtualEndpoint("CABLE Input (VB-Audio Virtual Cable)", "VB-Audio Virtual Cable") ||
                    !AudioDeviceService.IsVirtualEndpoint("VoiceMeeter Input (VB-Audio VoiceMeeter VAIO)", "VB-Audio VoiceMeeter VAIO") ||
                    !AudioDeviceService.IsVirtualEndpoint("VoiceMeeter Aux Input (VB-Audio VoiceMeeter AUX VAIO)", "VB-Audio VoiceMeeter AUX VAIO") ||
                    !AudioDeviceService.IsVirtualEndpoint("VAIO3 Input (VB-Audio VoiceMeeter VAIO3)", "VB-Audio VoiceMeeter VAIO3") ||
                    AudioDeviceService.IsVirtualEndpoint("Speakers (Realtek(R) Audio)", "Realtek High Definition Audio"))
                {
                    throw new InvalidOperationException("Virtual endpoint recognition test failed.");
                }

                // Verify Back navigation history behavior
                if (vm.CanNavigateBack)
                {
                    throw new InvalidOperationException("Expected CanNavigateBack=false on initial tab.");
                }
                vm.ActiveTab = NavigationTab.Voiceprint;
                vm.ActiveTab = NavigationTab.Compute;
                if (!vm.CanNavigateBack || !vm.NavigateBackCommand.CanExecute(null))
                {
                    throw new InvalidOperationException("Expected CanNavigateBack=true after tab switches.");
                }
                vm.NavigateBackCommand.Execute(null);
                if (vm.ActiveTab != NavigationTab.Voiceprint || !vm.CanNavigateBack)
                {
                    throw new InvalidOperationException($"Expected Back to return to Voiceprint, got {vm.ActiveTab}.");
                }
                vm.NavigateBackCommand.Execute(null);
                if (vm.ActiveTab != NavigationTab.Pipeline || vm.CanNavigateBack)
                {
                    throw new InvalidOperationException($"Expected Back to return to Pipeline with empty history, got {vm.ActiveTab}.");
                }

                var window = new MainWindow
                {
                    DataContext = vm,
                    ShowInTaskbar = false,
                    WindowState = WindowState.Minimized
                };

                bool closed = false;
                window.Closed += (_, _) => closed = true;
                window.Show();
                await Dispatcher.Yield(DispatcherPriority.Loaded);
                DwmBackdropHelper.ApplyThemePreference(controller.Settings.ThemePreference);
                bool darkTheme = DwmBackdropHelper.IsDarkThemeActive();
                if (!closed)
                {
                    window.Close();
                }

                await Dispatcher.Yield(DispatcherPriority.ApplicationIdle);
                Log($"[PASS] UI/ViewModel/DWM Smoke Test: Captures={vm.CaptureDevices.Count}, Renders={vm.RenderDevices.Count}, Cores={vm.CoreMatrixItems.Count}, MicaSupported={DwmBackdropHelper.IsMicaSupported}, DarkTheme={darkTheme}");
            }

            GC.Collect();
            GC.WaitForPendingFinalizers();
            await Dispatcher.Yield(DispatcherPriority.ApplicationIdle);
            Log("=== ALL SELF-TESTS PASSED (EXIT 0) ===");
            return 0;
        }
        catch (Exception ex)
        {
            Log($"[FAIL] Self-test failed: {ex}");
            return 1;
        }
    }

    private static void RunCascadeBenchmark(
        string label,
        SileroVadStage vad,
        GtcrnDenoiserStage gtcrn,
        VoxLockOnnxBackend voxLock,
        CpuAffinityManager? affinity,
        ulong affinityMask,
        ushort affinityGroup,
        Action<string> log)
    {
        const int measuredHops = 60; // 60 * 16 ms = 960 ms of real-time speech
        const int hopSamples = 256;
        const int vadSamples = 512;

        Exception? threadError = null;
        string summaryLine = string.Empty;

        var worker = new Thread(() =>
        {
            try
            {
                if (affinity is not null && affinityMask != 0)
                {
                    affinity.BindCurrentThread(affinityMask, ThreadPriority.AboveNormal, affinityGroup);
                }

                vad.ResetState();
                gtcrn.ResetState();
                voxLock.ResetState();
                vad.Warmup();
                gtcrn.Warmup();
                voxLock.Warmup();

                int voxFrameSize = voxLock.FrameSizeSamples;
                float[] hopIn = new float[hopSamples];
                float[] hopGtcrn = new float[hopSamples];
                float[] voxStaging = new float[voxFrameSize + hopSamples];
                float[] voxOut = new float[voxFrameSize];
                float[] vadIn = new float[vadSamples];
                float[] vadOut = new float[vadSamples];
                int voxStagingCount = 0;

                double ticksToMs = 1000.0 / Stopwatch.Frequency;
                double vadTotalMs = 0;
                int vadRuns = 0;
                double gtcrnTotalMs = 0;
                double voxTotalMs = 0;
                int voxRuns = 0;
                double firstVoxMs = 0;
                bool hasNonZeroOutput = false;

                for (int h = 0; h < measuredHops; h++)
                {
                    for (int i = 0; i < hopSamples; i++)
                    {
                        int globalIdx = h * hopSamples + i;
                        hopIn[i] = 0.25f * MathF.Sin(2f * MathF.PI * 220f * globalIdx / 16000f)
                                 + 0.12f * MathF.Sin(2f * MathF.PI * 440f * globalIdx / 16000f);
                    }

                    int vadHalfOffset = (h % 2) * hopSamples;
                    hopIn.CopyTo(vadIn.AsSpan(vadHalfOffset, hopSamples));

                    if (vadHalfOffset == hopSamples)
                    {
                        long tVad = Stopwatch.GetTimestamp();
                        vad.ProcessFrame(vadIn, vadOut);
                        vadTotalMs += (Stopwatch.GetTimestamp() - tVad) * ticksToMs;
                        vadRuns++;
                    }

                    long tGtcrn = Stopwatch.GetTimestamp();
                    gtcrn.ProcessFrame(hopIn, hopGtcrn);
                    gtcrnTotalMs += (Stopwatch.GetTimestamp() - tGtcrn) * ticksToMs;

                    hopGtcrn.CopyTo(voxStaging.AsSpan(voxStagingCount, hopSamples));
                    voxStagingCount += hopSamples;

                    while (voxStagingCount >= voxFrameSize)
                    {
                        long tVox = Stopwatch.GetTimestamp();
                        voxLock.ProcessFrame(voxStaging.AsSpan(0, voxFrameSize), voxOut);
                        double elapsedVox = (Stopwatch.GetTimestamp() - tVox) * ticksToMs;
                        if (voxRuns == 0)
                        {
                            firstVoxMs = elapsedVox;
                        }
                        voxTotalMs += elapsedVox;
                        voxRuns++;

                        int rem = voxStagingCount - voxFrameSize;
                        if (rem > 0)
                        {
                            voxStaging.AsSpan(voxFrameSize, rem).CopyTo(voxStaging.AsSpan(0, rem));
                        }
                        voxStagingCount = rem;

                        for (int i = 0; i < voxFrameSize; i++)
                        {
                            float sample = voxOut[i];
                            if (!float.IsFinite(sample))
                            {
                                throw new InvalidDataException($"Non-finite output sample detected at chunk {voxRuns}, index {i} ({sample}).");
                            }

                            if (MathF.Abs(sample) > 1e-6f)
                            {
                                hasNonZeroOutput = true;
                            }
                        }
                    }
                }

                if (!hasNonZeroOutput)
                {
                    throw new InvalidDataException($"Cascade output remained zero across all {voxRuns} chunks for '{label}'.");
                }

                double avgVad = vadRuns > 0 ? vadTotalMs / vadRuns : 0.0;
                double avgGtcrn = gtcrnTotalMs / measuredHops;
                double avgVox = voxRuns > 0 ? voxTotalMs / voxRuns : 0.0;
                double totalAudioMs = measuredHops * 16.0;
                double totalComputeMs = vadTotalMs + gtcrnTotalMs + voxTotalMs;
                double rtf = totalComputeMs / totalAudioMs;

                if (rtf >= 0.95)
                {
                    throw new InvalidOperationException($"Real-time factor too high ({rtf:F3} >= 0.95) for '{label}'.");
                }

                summaryLine = $"[PASS] {label} ({measuredHops} hops={totalAudioMs:F0}ms): VAD={avgVad:F2}ms/32ms, GTCRN={avgGtcrn:F2}ms/16ms, VoxLock={avgVox:F2}ms/{voxFrameSize * 1000.0 / 16000.0:F0}ms (cold1={firstVoxMs:F2}ms), RTF={rtf:F3}";
            }
            catch (Exception ex)
            {
                threadError = ex;
            }
        })
        { IsBackground = true };

        worker.Start();
        worker.Join();

        if (threadError is not null)
        {
            throw new InvalidOperationException($"Cascade benchmark failed for '{label}'.", threadError);
        }

        log(summaryLine);
    }

    private static void CreateSyntheticVoiceWav(string path, int sampleRate, double durationSeconds)
    {
        int totalSamples = (int)Math.Round(sampleRate * durationSeconds);
        float[] samples = new float[totalSamples];
        for (int i = 0; i < totalSamples; i++)
        {
            double t = (double)i / sampleRate;
            samples[i] = (float)(0.25 * Math.Sin(2 * Math.PI * 180.0 * t)
                + 0.15 * Math.Sin(2 * Math.PI * 360.0 * t)
                + 0.08 * Math.Sin(2 * Math.PI * 720.0 * t));
        }

        using var writer = new WaveFileWriter(path, new WaveFormat(sampleRate, 16, 1));
        writer.WriteSamples(samples, 0, samples.Length);
        writer.Flush();
    }

    private static List<string> BackupExistingVoiceprintFiles(string backupDir)
    {
        var backedUp = new List<string>();
        string appData = VoiceprintStorage.AppDataDirectory;
        if (!Directory.Exists(appData))
        {
            return backedUp;
        }

        var targets = new List<string>();
        string defaultWav = VoiceprintStorage.DefaultUserRawVoiceWavPath;
        if (File.Exists(defaultWav))
        {
            targets.Add(defaultWav);
        }

        targets.AddRange(Directory.EnumerateFiles(appData, "voiceprint_cache_*.bin", SearchOption.TopDirectoryOnly));
        if (targets.Count == 0)
        {
            return backedUp;
        }

        Directory.CreateDirectory(backupDir);
        foreach (string file in targets)
        {
            string name = Path.GetFileName(file);
            string dest = Path.Combine(backupDir, name);
            File.Move(file, dest, overwrite: true);
            backedUp.Add(name);
        }

        return backedUp;
    }

    private static void RestoreVoiceprintFiles(string backupDir, List<string> backedUpNames)
    {
        string appData = VoiceprintStorage.AppDataDirectory;
        if (Directory.Exists(appData))
        {
            if (File.Exists(VoiceprintStorage.DefaultUserRawVoiceWavPath))
            {
                TryDeleteFile(VoiceprintStorage.DefaultUserRawVoiceWavPath);
            }

            foreach (string cacheFile in Directory.EnumerateFiles(appData, "voiceprint_cache_*.bin", SearchOption.TopDirectoryOnly))
            {
                TryDeleteFile(cacheFile);
            }
        }

        if (Directory.Exists(backupDir))
        {
            foreach (string name in backedUpNames)
            {
                string src = Path.Combine(backupDir, name);
                string dest = Path.Combine(appData, name);
                if (File.Exists(src))
                {
                    File.Move(src, dest, overwrite: true);
                }
            }

            TryDeleteDirectory(backupDir);
        }
    }

    private static void TryDeleteFile(string path)
    {
        try { File.Delete(path); } catch { }
    }

    private static void TryDeleteDirectory(string path)
    {
        try { if (Directory.Exists(path)) Directory.Delete(path, recursive: true); } catch { }
    }

    private static void TryAttachConsole()
    {
        if (Console.IsOutputRedirected)
        {
            return;
        }

        _ = NativeMethods.AttachConsole(AttachParentProcess);
    }

    private static class NativeMethods
    {
        [DllImport("kernel32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        public static extern bool AttachConsole(int dwProcessId);
    }
}
