using System.IO;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace VoiceLockStudio.Models;

/// <summary>
/// Strongly-typed representation of <c>Assets/Models/model_manifest.json</c>.
/// </summary>
public sealed class ModelManifest
{
    private static readonly JsonSerializerOptions SerializerOptions = new()
    {
        PropertyNameCaseInsensitive = true,
        ReadCommentHandling = JsonCommentHandling.Skip,
        AllowTrailingCommas = true
    };

    [JsonPropertyName("generated_at_utc")]
    public string GeneratedAtUtc { get; init; } = string.Empty;

    [JsonPropertyName("python_executable")]
    public string PythonExecutable { get; init; } = string.Empty;

    [JsonPropertyName("python_version")]
    public string PythonVersion { get; init; } = string.Empty;

    [JsonPropertyName("onnxruntime_version")]
    public string OnnxRuntimeVersion { get; init; } = string.Empty;

    [JsonPropertyName("available_providers")]
    public IReadOnlyList<string> AvailableProviders { get; init; } = [];

    [JsonPropertyName("models")]
    public IReadOnlyList<ModelEntry> Models { get; init; } = [];

    [JsonPropertyName("secondary_cuda_onnx_env")]
    public JsonElement? SecondaryCudaOnnxEnv { get; init; }

    /// <summary>
    /// Loads and deserializes a <see cref="ModelManifest"/> from disk.
    /// </summary>
    public static ModelManifest LoadFromFile(string manifestPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(manifestPath);

        using FileStream stream = File.OpenRead(manifestPath);
        ModelManifest? manifest = JsonSerializer.Deserialize<ModelManifest>(stream, SerializerOptions);
        return manifest ?? throw new InvalidDataException($"Failed to deserialize model manifest from '{manifestPath}'.");
    }

    /// <summary>
    /// Loads <c>Assets/Models/model_manifest.json</c> relative to <paramref name="baseDirectory"/>
    /// (defaulting to <see cref="AppContext.BaseDirectory"/>).
    /// </summary>
    public static ModelManifest LoadDefault(string? baseDirectory = null)
    {
        string root = string.IsNullOrWhiteSpace(baseDirectory) ? AppContext.BaseDirectory : baseDirectory;
        string manifestPath = Path.Combine(root, "Assets", "Models", "model_manifest.json");
        return LoadFromFile(manifestPath);
    }

    /// <summary>
    /// Finds a model entry by exact filename (e.g. <c>silero_vad.onnx</c> or <c>chunk_int8.onnx</c>).
    /// </summary>
    public ModelEntry? FindByFilename(string filename)
    {
        if (string.IsNullOrWhiteSpace(filename))
        {
            return null;
        }

        return Models.FirstOrDefault(m =>
            string.Equals(m.Filename, filename, StringComparison.OrdinalIgnoreCase));
    }

    /// <summary>
    /// Finds a model entry by manifest relative path (slash-insensitive).
    /// </summary>
    public ModelEntry? FindByPath(string relativePath)
    {
        if (string.IsNullOrWhiteSpace(relativePath))
        {
            return null;
        }

        string normalized = relativePath.Replace('\\', '/');
        return Models.FirstOrDefault(m =>
            string.Equals(m.Path.Replace('\\', '/'), normalized, StringComparison.OrdinalIgnoreCase));
    }
}

public sealed class ModelEntry
{
    [JsonPropertyName("path")]
    public string Path { get; init; } = string.Empty;

    [JsonPropertyName("filename")]
    public string Filename { get; init; } = string.Empty;

    [JsonPropertyName("size_bytes")]
    public long SizeBytes { get; init; }

    [JsonPropertyName("sha256")]
    public string Sha256 { get; init; } = string.Empty;

    [JsonPropertyName("source")]
    public ModelSourceInfo? Source { get; init; }

    [JsonPropertyName("onnx_protobuf")]
    public OnnxProtobufInfo? OnnxProtobuf { get; init; }

    [JsonPropertyName("audio_params")]
    public ModelAudioParams? AudioParams { get; init; }

    [JsonPropertyName("ort_cpu")]
    public OrtInspectionInfo? OrtCpu { get; init; }

    [JsonPropertyName("ort_cuda")]
    public OrtInspectionInfo? OrtCuda { get; init; }

    [JsonPropertyName("ort_secondary_cpu")]
    public OrtInspectionInfo? OrtSecondaryCpu { get; init; }

    /// <summary>
    /// Resolves the absolute model path on disk given the application base directory.
    /// </summary>
    public string ResolveFullPath(string? baseDirectory = null)
    {
        string root = string.IsNullOrWhiteSpace(baseDirectory) ? AppContext.BaseDirectory : baseDirectory;
        string osRelative = Path.Replace('/', System.IO.Path.DirectorySeparatorChar)
                                .Replace('\\', System.IO.Path.DirectorySeparatorChar);
        return System.IO.Path.GetFullPath(System.IO.Path.Combine(root, osRelative));
    }

    /// <summary>
    /// Returns the inspected input tensor descriptors (preferring ORT CPU metadata, falling back to protobuf).
    /// </summary>
    public IReadOnlyList<TensorDescriptor> GetEffectiveInputs() =>
        OrtCpu?.Inputs is { Count: > 0 } ? OrtCpu.Inputs :
        OnnxProtobuf?.Inputs ?? [];

    /// <summary>
    /// Returns the inspected output tensor descriptors (preferring ORT CPU metadata, falling back to protobuf).
    /// </summary>
    public IReadOnlyList<TensorDescriptor> GetEffectiveOutputs() =>
        OrtCpu?.Outputs is { Count: > 0 } ? OrtCpu.Outputs :
        OnnxProtobuf?.Outputs ?? [];
}

public sealed class ModelSourceInfo
{
    [JsonPropertyName("source_repo")]
    public string? SourceRepo { get; init; }

    [JsonPropertyName("download_url")]
    public string? DownloadUrl { get; init; }

    [JsonPropertyName("remote_path")]
    public string? RemotePath { get; init; }
}

public sealed class OnnxProtobufInfo
{
    [JsonPropertyName("available")]
    public bool Available { get; init; }

    [JsonPropertyName("onnx_version")]
    public string? OnnxVersion { get; init; }

    [JsonPropertyName("ir_version")]
    public int IrVersion { get; init; }

    [JsonPropertyName("producer_name")]
    public string? ProducerName { get; init; }

    [JsonPropertyName("producer_version")]
    public string? ProducerVersion { get; init; }

    [JsonPropertyName("model_version")]
    public long ModelVersion { get; init; }

    [JsonPropertyName("opsets")]
    public IReadOnlyList<OpsetInfo> Opsets { get; init; } = [];

    [JsonPropertyName("metadata_props")]
    public IReadOnlyDictionary<string, string> MetadataProps { get; init; } =
        new Dictionary<string, string>();

    [JsonPropertyName("inputs")]
    public IReadOnlyList<TensorDescriptor> Inputs { get; init; } = [];

    [JsonPropertyName("outputs")]
    public IReadOnlyList<TensorDescriptor> Outputs { get; init; } = [];

    [JsonPropertyName("node_count")]
    public int NodeCount { get; init; }
}

public sealed class OpsetInfo
{
    [JsonPropertyName("domain")]
    public string Domain { get; init; } = string.Empty;

    [JsonPropertyName("version")]
    public int Version { get; init; }
}

public sealed class TensorDescriptor
{
    [JsonPropertyName("name")]
    public string Name { get; init; } = string.Empty;

    [JsonPropertyName("elem_type")]
    public string? ElemType { get; init; }

    [JsonPropertyName("type")]
    public string? Type { get; init; }

    [JsonPropertyName("shape")]
    public IReadOnlyList<TensorDimension> Shape { get; init; } = [];

    [JsonIgnore]
    public string EffectiveType => Type ?? ElemType ?? string.Empty;
}

/// <summary>
/// Represents a tensor dimension that may be static (<see cref="int"/>/<see cref="long"/>),
/// symbolic (<see cref="string"/> such as <c>"S"</c> or <c>"K"</c>), or dynamic (<see langword="null"/>).
/// </summary>
[JsonConverter(typeof(TensorDimensionJsonConverter))]
public readonly record struct TensorDimension
{
    public bool IsStatic { get; }
    public long? StaticValue { get; }
    public string? SymbolicName { get; }

    [JsonIgnore]
    public bool IsDynamic => !IsStatic;

    private TensorDimension(bool isStatic, long? staticValue, string? symbolicName)
    {
        IsStatic = isStatic;
        StaticValue = staticValue;
        SymbolicName = symbolicName;
    }

    public static TensorDimension FromStatic(long value) => new(true, value, null);
    public static TensorDimension FromSymbolic(string name) => new(false, null, name);
    public static TensorDimension DynamicUnknown() => new(false, null, null);

    public override string ToString() =>
        IsStatic ? StaticValue!.Value.ToString() :
        !string.IsNullOrEmpty(SymbolicName) ? SymbolicName :
        "?";
}

public sealed class TensorDimensionJsonConverter : JsonConverter<TensorDimension>
{
    public override TensorDimension Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
    {
        return reader.TokenType switch
        {
            JsonTokenType.Number => TensorDimension.FromStatic(reader.GetInt64()),
            JsonTokenType.String => TensorDimension.FromSymbolic(reader.GetString() ?? string.Empty),
            JsonTokenType.Null => TensorDimension.DynamicUnknown(),
            _ => throw new JsonException($"Unexpected token {reader.TokenType} for tensor dimension.")
        };
    }

    public override void Write(Utf8JsonWriter writer, TensorDimension value, JsonSerializerOptions options)
    {
        if (value.IsStatic && value.StaticValue.HasValue)
        {
            writer.WriteNumberValue(value.StaticValue.Value);
        }
        else if (!string.IsNullOrEmpty(value.SymbolicName))
        {
            writer.WriteStringValue(value.SymbolicName);
        }
        else
        {
            writer.WriteNullValue();
        }
    }
}

public sealed class ModelAudioParams
{
    [JsonPropertyName("sample_rate")]
    public int SampleRate { get; init; }

    [JsonPropertyName("supported_sample_rates")]
    public IReadOnlyList<int>? SupportedSampleRates { get; init; }

    [JsonPropertyName("n_fft")]
    public int? NFft { get; init; }

    [JsonPropertyName("hop_length")]
    public int? HopLength { get; init; }

    [JsonPropertyName("window_length")]
    public int? WindowLength { get; init; }

    [JsonPropertyName("window_type")]
    public string? WindowType { get; init; }

    [JsonPropertyName("freq_bins")]
    public int? FreqBins { get; init; }

    [JsonPropertyName("domain")]
    public string? Domain { get; init; }

    [JsonPropertyName("frame_duration_ms")]
    public double? FrameDurationMs { get; init; }

    [JsonPropertyName("window_duration_ms")]
    public double? WindowDurationMs { get; init; }

    [JsonPropertyName("window_size_samples_16k")]
    public int? WindowSizeSamples16k { get; init; }

    [JsonPropertyName("context_samples_16k")]
    public int? ContextSamples16k { get; init; }

    [JsonPropertyName("effective_model_input_samples_16k")]
    public int? EffectiveModelInputSamples16k { get; init; }

    [JsonPropertyName("window_size_samples_8k")]
    public int? WindowSizeSamples8k { get; init; }

    [JsonPropertyName("reflect_pad_samples")]
    public int? ReflectPadSamples { get; init; }

    [JsonPropertyName("chunk_formula_samples")]
    public string? ChunkFormulaSamples { get; init; }

    [JsonPropertyName("overlap_samples")]
    public int? OverlapSamples { get; init; }

    [JsonPropertyName("algorithmic_latency_ms")]
    public double? AlgorithmicLatencyMs { get; init; }

    [JsonPropertyName("output_gain_multiplier")]
    public double? OutputGainMultiplier { get; init; }

    [JsonPropertyName("min_enroll_samples")]
    public int? MinEnrollSamples { get; init; }

    [JsonPropertyName("min_enroll_seconds")]
    public double? MinEnrollSeconds { get; init; }

    [JsonPropertyName("fbank_win_length_samples")]
    public int? FbankWinLengthSamples { get; init; }

    [JsonPropertyName("fbank_hop_length_samples")]
    public int? FbankHopLengthSamples { get; init; }

    [JsonPropertyName("fbank_num_mel_bins")]
    public int? FbankNumMelBins { get; init; }

    [JsonPropertyName("stft_n_fft")]
    public int? StftNFft { get; init; }

    [JsonPropertyName("stft_hop_length")]
    public int? StftHopLength { get; init; }
}

public sealed class OrtInspectionInfo
{
    [JsonPropertyName("ort_version")]
    public string? OrtVersion { get; init; }

    [JsonPropertyName("requested_providers")]
    public IReadOnlyList<string> RequestedProviders { get; init; } = [];

    [JsonPropertyName("session_ok")]
    public bool SessionOk { get; init; }

    [JsonPropertyName("active_providers")]
    public IReadOnlyList<string> ActiveProviders { get; init; } = [];

    [JsonPropertyName("model_meta")]
    public OrtModelMetadata? ModelMeta { get; init; }

    [JsonPropertyName("inputs")]
    public IReadOnlyList<TensorDescriptor> Inputs { get; init; } = [];

    [JsonPropertyName("outputs")]
    public IReadOnlyList<TensorDescriptor> Outputs { get; init; } = [];

    [JsonPropertyName("dummy_input_shapes")]
    public IReadOnlyDictionary<string, DummyTensorShape>? DummyInputShapes { get; init; }

    [JsonPropertyName("inference_ok")]
    public bool InferenceOk { get; init; }

    [JsonPropertyName("dummy_output_shapes")]
    public IReadOnlyDictionary<string, DummyTensorShape>? DummyOutputShapes { get; init; }
}

public sealed class OrtModelMetadata
{
    [JsonPropertyName("producer_name")]
    public string? ProducerName { get; init; }

    [JsonPropertyName("graph_name")]
    public string? GraphName { get; init; }

    [JsonPropertyName("domain")]
    public string? Domain { get; init; }

    [JsonPropertyName("description")]
    public string? Description { get; init; }

    [JsonPropertyName("graph_description")]
    public string? GraphDescription { get; init; }

    [JsonPropertyName("version")]
    public long Version { get; init; }

    [JsonPropertyName("custom_metadata_map")]
    public IReadOnlyDictionary<string, string> CustomMetadataMap { get; init; } =
        new Dictionary<string, string>();
}

public sealed class DummyTensorShape
{
    [JsonPropertyName("shape")]
    public IReadOnlyList<int> Shape { get; init; } = [];

    [JsonPropertyName("dtype")]
    public string Dtype { get; init; } = string.Empty;
}
