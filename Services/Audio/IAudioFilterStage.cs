using System.IO;
using System.Security.Cryptography;
using VoiceLockStudio.Models;

namespace VoiceLockStudio.Services.Audio;

/// <summary>
/// Configuration passed when initializing an <see cref="IAudioFilterStage"/>.
/// </summary>
public sealed record AudioStageOptions(
    string ModelsRootDirectory,
    ModelManifest? Manifest = null,
    ComputeBackend Backend = ComputeBackend.Cpu,
    VoxLockModelVariant VoxLockVariant = VoxLockModelVariant.Int8,
    float VadThreshold = 0.5f,
    int OrtIntraOpThreads = 1,
    int OrtInterOpThreads = 1,
    ulong CpuAffinityMask = 0,
    ushort ProcessorGroup = 0,
    int VoxLockHopsPerChunk = 4)
{
    public static AudioStageOptions CreateDefault(string? baseDirectory = null)
    {
        string root = string.IsNullOrWhiteSpace(baseDirectory) ? AppContext.BaseDirectory : baseDirectory;
        return new AudioStageOptions(ModelsRootDirectory: root);
    }
}

/// <summary>
/// Streaming single-channel audio filter or neural DSP stage operating on fixed-size float PCM frames.
/// </summary>
public interface IAudioFilterStage : IDisposable
{
    string Name { get; }
    int FrameSizeSamples { get; }
    int SampleRate { get; }
    bool IsInitialized { get; }
    bool IsSessionLoaded => IsInitialized;

    /// <summary>
    /// Initializes model sessions, pre-allocated scratch buffers, and stage parameters.
    /// </summary>
    void Initialize(AudioStageOptions options);

    /// <summary>
    /// Processes exactly <see cref="FrameSizeSamples"/> mono float samples from <paramref name="input"/>
    /// into <paramref name="output"/> without heap allocations on the audio thread.
    /// </summary>
    void ProcessFrame(ReadOnlySpan<float> input, Span<float> output);

    /// <summary>
    /// Clears streaming recurrent/STFT state between utterances or pipeline restarts.
    /// </summary>
    void ResetState();

    /// <summary>
    /// Clears only short-term STFT/OLA staging buffers after an overrun skip or transient bypass
    /// without erasing recurrent speaker-tracking memory.
    /// </summary>
    void SoftResetStreamingBuffers() => ResetState();

    /// <summary>
    /// Executes dummy warmup inference(s) so ONNX Runtime allocates arenas/kernels
    /// before live microphone capture begins, eliminating cold-start latency spikes.
    /// </summary>
    void Warmup() { }

    /// <summary>
    /// Updates CPU affinity for internal stage worker threads (e.g. ONNX intra-op pool threads) at runtime.
    /// </summary>
    void UpdateCpuAffinity(ulong coreMask, ushort processorGroup = 0) { }
}

/// <summary>
/// Metadata describing an enrolled speaker voiceprint and its binary cache file.
/// </summary>
public sealed record VoiceprintProfileInfo(
    string SourceWavPath,
    string CacheFilePath,
    string ContentSha256,
    DateTimeOffset EnrolledAtUtc,
    double SourceDurationSeconds,
    bool LoadedFromCache);

/// <summary>
/// Target-speaker voice isolation backend supporting voiceprint enrollment and disk caching.
/// </summary>
public interface IVoiceIsolatorBackend : IAudioFilterStage
{
    bool HasEnrolledVoiceprint { get; }
    VoiceprintProfileInfo? CurrentVoiceprint { get; }

    /// <summary>
    /// Runs speaker enrollment on <paramref name="wavFilePath"/>, copies/normalizes the reference
    /// source to <c>AppData/Local/VoiceLockStudio/user_raw_voice.wav</c> when needed, and writes
    /// the hash-keyed <c>voiceprint_cache_{hash}.bin</c> file.
    /// </summary>
    VoiceprintProfileInfo EnrollVoiceprint(string wavFilePath);

    /// <summary>
    /// Loads the cached voiceprint tensors matching <paramref name="wavFilePath"/> (defaulting to
    /// <c>AppData/Local/VoiceLockStudio/user_raw_voice.wav</c>), or runs <see cref="EnrollVoiceprint"/>
    /// to rebuild the cache when missing or stale.
    /// </summary>
    VoiceprintProfileInfo? LoadOrRebuildVoiceprintCache(string? wavFilePath = null, bool forceRebuild = false);
}

/// <summary>
/// Standard LocalAppData storage paths and content-hashing helpers for enrolled voiceprints.
/// </summary>
public static class VoiceprintStorage
{
    public static string AppDataDirectory => AppSettings.GetAppDataDirectory();

    public static string DefaultUserRawVoiceWavPath => AppSettings.GetDefaultVoiceprintWavPath();

    public static string GetCacheFilePath(string contentHashHex)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(contentHashHex);
        return Path.Combine(AppDataDirectory, $"voiceprint_cache_{contentHashHex.ToLowerInvariant()}.bin");
    }

    public static string ComputeFileSha256Hex(string filePath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(filePath);
        using FileStream stream = File.OpenRead(filePath);
        byte[] hash = SHA256.HashData(stream);
        return Convert.ToHexStringLower(hash);
    }
}
