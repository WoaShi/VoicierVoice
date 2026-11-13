namespace VoiceLockStudio.Models;

public enum PipelineRunState
{
    Stopped,
    Starting,
    Running,
    Faulted
}

/// <summary>
/// Immutable live telemetry snapshot published by the audio processing pipeline for the UI.
/// </summary>
public sealed record TelemetrySnapshot(
    DateTimeOffset TimestampUtc,
    PipelineRunState State,
    string StatusText,
    double LatencyMs,
    double FrameInferenceMs,
    double VadInferenceMs,
    double GtcrnInferenceMs,
    double VoxLockInferenceMs,
    string ActiveCoreDescription,
    ulong ActiveCoreMask,
    bool VadSpeechActive,
    float VadProbability,
    float InputRms,
    float OutputRms,
    float InputPeak,
    float OutputPeak,
    long BufferOverruns,
    long BufferUnderruns,
    string ActiveBackendDescription)
{
    public static TelemetrySnapshot Idle { get; } = new(
        TimestampUtc: DateTimeOffset.MinValue,
        State: PipelineRunState.Stopped,
        StatusText: "Idle",
        LatencyMs: 0.0,
        FrameInferenceMs: 0.0,
        VadInferenceMs: 0.0,
        GtcrnInferenceMs: 0.0,
        VoxLockInferenceMs: 0.0,
        ActiveCoreDescription: "Unbound",
        ActiveCoreMask: 0UL,
        VadSpeechActive: false,
        VadProbability: 0f,
        InputRms: 0f,
        OutputRms: 0f,
        InputPeak: 0f,
        OutputPeak: 0f,
        BufferOverruns: 0L,
        BufferUnderruns: 0L,
        ActiveBackendDescription: "CPU");
}
