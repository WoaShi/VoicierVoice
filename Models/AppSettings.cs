using System.IO;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace VoiceLockStudio.Models;

[JsonConverter(typeof(JsonStringEnumConverter))]
public enum VoxLockModelVariant
{
    Int8,
    Int8Full,
    Fp16
}

[JsonConverter(typeof(JsonStringEnumConverter))]
public enum ComputeBackend
{
    Cpu,
    Cuda
}

[JsonConverter(typeof(JsonStringEnumConverter))]
public enum AppLanguage
{
    ZhCn,
    EnUs
}

[JsonConverter(typeof(JsonStringEnumConverter))]
public enum AppThemePreference
{
    System,
    Light,
    Dark
}

/// <summary>
/// Persisted application configuration for audio routing, DSP stages, ONNX backend, CPU affinity, and UI preferences.
/// </summary>
public sealed class AppSettings
{
    private const string LegacyAppDataFolderName = "VoiceLockStudio";
    private const string AppDataFolderName = "VoicierVoice";

    private static readonly JsonSerializerOptions SerializerOptions = new()
    {
        WriteIndented = true,
        PropertyNameCaseInsensitive = true
    };

    public string? SelectedInputDeviceId { get; set; }
    public string? SelectedOutputDeviceId { get; set; }
    public bool MonitorLoopbackEnabled { get; set; }

    public bool VadEnabled { get; set; } = true;
    public float VadThreshold { get; set; } = 0.5f;

    public bool GtcrnEnabled { get; set; } = true;

    public bool VoxLockEnabled { get; set; } = true;
    public VoxLockModelVariant VoxLockVariant { get; set; } = VoxLockModelVariant.Int8;
    public bool VoxLockLowLatencyMode { get; set; }

    public ComputeBackend ComputeBackend { get; set; } = ComputeBackend.Cpu;

    public bool AutoAffinityEnabled { get; set; } = true;
    public ulong SelectedCoreMask { get; set; }
    public ushort SelectedProcessorGroup { get; set; }

    public string VoiceprintWavPath { get; set; } = GetDefaultVoiceprintWavPath();

    public AppLanguage Language { get; set; } = AppLanguage.ZhCn;
    public AppThemePreference ThemePreference { get; set; } = AppThemePreference.System;
    public bool MinimizeToTrayEnabled { get; set; } = true;
    public bool LaunchOnStartupEnabled { get; set; }
    public bool PipelineActive { get; set; }

    public static string GetAppDataDirectory() =>
        Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            AppDataFolderName);

    public static string GetLegacyAppDataDirectory() =>
        Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            LegacyAppDataFolderName);

    public static string GetDefaultSettingsPath() =>
        Path.Combine(GetAppDataDirectory(), "settings.json");

    public static string GetDefaultVoiceprintWavPath() =>
        Path.Combine(GetAppDataDirectory(), "user_raw_voice.wav");

    public AppSettings Clone() => new()
    {
        SelectedInputDeviceId = SelectedInputDeviceId,
        SelectedOutputDeviceId = SelectedOutputDeviceId,
        MonitorLoopbackEnabled = MonitorLoopbackEnabled,
        VadEnabled = VadEnabled,
        VadThreshold = VadThreshold,
        GtcrnEnabled = GtcrnEnabled,
        VoxLockEnabled = VoxLockEnabled,
        VoxLockVariant = VoxLockVariant,
        VoxLockLowLatencyMode = VoxLockLowLatencyMode,
        ComputeBackend = ComputeBackend,
        AutoAffinityEnabled = AutoAffinityEnabled,
        SelectedCoreMask = SelectedCoreMask,
        SelectedProcessorGroup = SelectedProcessorGroup,
        VoiceprintWavPath = VoiceprintWavPath,
        Language = Language,
        ThemePreference = ThemePreference,
        MinimizeToTrayEnabled = MinimizeToTrayEnabled,
        LaunchOnStartupEnabled = LaunchOnStartupEnabled,
        PipelineActive = PipelineActive
    };

    public static AppSettings Load(string? path = null)
    {
        bool isDefaultPath = string.IsNullOrWhiteSpace(path);
        if (isDefaultPath)
        {
            MigrateLegacyAppDataIfNeeded();
        }

        string targetPath = isDefaultPath ? GetDefaultSettingsPath() : path!;
        if (!File.Exists(targetPath))
        {
            return new AppSettings();
        }

        try
        {
            string json = File.ReadAllText(targetPath);
            AppSettings? loaded = JsonSerializer.Deserialize<AppSettings>(json, SerializerOptions);
            if (loaded is null)
            {
                return new AppSettings();
            }

            if (isDefaultPath && RewriteLegacyVoiceprintPath(loaded))
            {
                try
                {
                    loaded.Save(targetPath);
                }
                catch
                {
                    // Best-effort persistence of migrated path.
                }
            }

            return loaded;
        }
        catch
        {
            return new AppSettings();
        }
    }

    /// <summary>
    /// Non-destructively migrates persisted settings, raw voiceprint WAV, and hash-keyed caches
    /// from the legacy <c>VoiceLockStudio</c> LocalAppData directory to <c>VoicierVoice</c>.
    /// </summary>
    public static void MigrateLegacyAppDataIfNeeded()
    {
        try
        {
            string legacyDir = GetLegacyAppDataDirectory();
            if (!Directory.Exists(legacyDir))
            {
                return;
            }

            string newDir = GetAppDataDirectory();
            string newSettingsPath = Path.Combine(newDir, "settings.json");
            string newWavPath = Path.Combine(newDir, "user_raw_voice.wav");

            bool needSettings = !File.Exists(newSettingsPath);
            bool needWav = !File.Exists(newWavPath);
            if (!needSettings && !needWav)
            {
                return;
            }

            Directory.CreateDirectory(newDir);

            if (needSettings)
            {
                string legacySettingsPath = Path.Combine(legacyDir, "settings.json");
                if (File.Exists(legacySettingsPath))
                {
                    File.Copy(legacySettingsPath, newSettingsPath, overwrite: false);
                }
            }

            if (needWav)
            {
                string legacyWavPath = Path.Combine(legacyDir, "user_raw_voice.wav");
                if (File.Exists(legacyWavPath))
                {
                    File.Copy(legacyWavPath, newWavPath, overwrite: false);
                }
            }

            foreach (string legacyCache in Directory.EnumerateFiles(legacyDir, "voiceprint_cache_*.bin", SearchOption.TopDirectoryOnly))
            {
                string destCache = Path.Combine(newDir, Path.GetFileName(legacyCache));
                if (!File.Exists(destCache))
                {
                    File.Copy(legacyCache, destCache, overwrite: false);
                }
            }
        }
        catch
        {
            // Never block application startup if legacy migration hits an I/O restriction.
        }
    }

    private static bool RewriteLegacyVoiceprintPath(AppSettings settings)
    {
        if (string.IsNullOrWhiteSpace(settings.VoiceprintWavPath))
        {
            settings.VoiceprintWavPath = GetDefaultVoiceprintWavPath();
            return true;
        }

        try
        {
            string legacyDir = Path.GetFullPath(GetLegacyAppDataDirectory())
                .TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
            string newDir = Path.GetFullPath(GetAppDataDirectory())
                .TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
            string currentFullPath = Path.GetFullPath(settings.VoiceprintWavPath);

            if (currentFullPath.Equals(legacyDir, StringComparison.OrdinalIgnoreCase) ||
                currentFullPath.StartsWith(legacyDir + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase))
            {
                string relative = Path.GetRelativePath(legacyDir, currentFullPath);
                settings.VoiceprintWavPath = relative == "."
                    ? GetDefaultVoiceprintWavPath()
                    : Path.Combine(newDir, relative);
                return true;
            }
        }
        catch
        {
            // Leave custom/unparseable paths untouched.
        }

        return false;
    }

    public void Save(string? path = null)
    {
        string targetPath = string.IsNullOrWhiteSpace(path) ? GetDefaultSettingsPath() : path;
        string? directory = Path.GetDirectoryName(targetPath);
        if (!string.IsNullOrEmpty(directory))
        {
            Directory.CreateDirectory(directory);
        }

        string json = JsonSerializer.Serialize(this, SerializerOptions);
        File.WriteAllText(targetPath, json);
    }
}
