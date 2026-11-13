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

/// <summary>
/// Persisted application configuration for audio routing, DSP stages, ONNX backend, and CPU affinity.
/// </summary>
public sealed class AppSettings
{
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

    public ComputeBackend ComputeBackend { get; set; } = ComputeBackend.Cpu;

    public bool AutoAffinityEnabled { get; set; } = true;
    public ulong SelectedCoreMask { get; set; }
    public ushort SelectedProcessorGroup { get; set; }

    public string VoiceprintWavPath { get; set; } = GetDefaultVoiceprintWavPath();

    public static string GetAppDataDirectory() =>
        Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "VoiceLockStudio");

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
        ComputeBackend = ComputeBackend,
        AutoAffinityEnabled = AutoAffinityEnabled,
        SelectedCoreMask = SelectedCoreMask,
        SelectedProcessorGroup = SelectedProcessorGroup,
        VoiceprintWavPath = VoiceprintWavPath
    };

    public static AppSettings Load(string? path = null)
    {
        string targetPath = string.IsNullOrWhiteSpace(path) ? GetDefaultSettingsPath() : path;
        if (!File.Exists(targetPath))
        {
            return new AppSettings();
        }

        try
        {
            string json = File.ReadAllText(targetPath);
            AppSettings? loaded = JsonSerializer.Deserialize<AppSettings>(json, SerializerOptions);
            return loaded ?? new AppSettings();
        }
        catch
        {
            return new AppSettings();
        }
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
