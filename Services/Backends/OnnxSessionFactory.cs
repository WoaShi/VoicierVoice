using System.IO;
using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime;
using VoiceLockStudio.Models;
using VoiceLockStudio.Services.Audio;

namespace VoiceLockStudio.Services.Backends;

/// <summary>
/// Holds an initialized ONNX Runtime <see cref="InferenceSession"/> alongside its resolved
/// <see cref="ModelEntry"/> metadata, active execution provider, and any CUDA-to-CPU fallback telemetry.
/// </summary>
public sealed class OnnxSessionResult : IDisposable
{
    public OnnxSessionResult(
        InferenceSession session,
        ModelEntry modelEntry,
        string resolvedModelPath,
        ComputeBackend requestedBackend,
        ComputeBackend activeBackend,
        string activeProviderName,
        string? fallbackMessage)
    {
        Session = session;
        ModelEntry = modelEntry;
        ResolvedModelPath = resolvedModelPath;
        RequestedBackend = requestedBackend;
        ActiveBackend = activeBackend;
        ActiveProviderName = activeProviderName;
        FallbackMessage = fallbackMessage;
    }

    public InferenceSession Session { get; }
    public ModelEntry ModelEntry { get; }
    public string ResolvedModelPath { get; }
    public ComputeBackend RequestedBackend { get; }
    public ComputeBackend ActiveBackend { get; }
    public string ActiveProviderName { get; }
    public string? FallbackMessage { get; }
    public bool FellBackToCpu => RequestedBackend == ComputeBackend.Cuda && ActiveBackend == ComputeBackend.Cpu;

    public void Dispose()
    {
        Session.Dispose();
    }
}

/// <summary>
/// Central factory for creating single-threaded, fully graph-optimized ONNX Runtime sessions
/// driven strictly by <see cref="ModelManifest"/> entries with safe CUDA 11.8 DLL path setup and CPU fallback.
/// </summary>
public static class OnnxSessionFactory
{
    private const string DefaultCuda118BinDirectory = @"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin";
    private const string CpuProviderName = "CPUExecutionProvider";
    private const string CudaProviderName = "CUDAExecutionProvider";

    private static readonly object CudaPathSync = new();
    private static bool _cudaPathInitialized;

    /// <summary>
    /// Returns <see cref="AudioStageOptions.Manifest"/> when provided, or loads the default
    /// <c>Assets/Models/model_manifest.json</c> relative to <see cref="AudioStageOptions.ModelsRootDirectory"/>.
    /// </summary>
    public static ModelManifest ResolveManifest(AudioStageOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        if (options.Manifest is not null)
        {
            return options.Manifest;
        }

        string root = string.IsNullOrWhiteSpace(options.ModelsRootDirectory)
            ? AppContext.BaseDirectory
            : options.ModelsRootDirectory;

        return ModelManifest.LoadDefault(root);
    }

    /// <summary>
    /// Looks up a <see cref="ModelEntry"/> in <paramref name="manifest"/> by filename or manifest relative path.
    /// </summary>
    public static ModelEntry ResolveModelEntry(ModelManifest manifest, string filenameOrRelativePath)
    {
        ArgumentNullException.ThrowIfNull(manifest);
        ArgumentException.ThrowIfNullOrWhiteSpace(filenameOrRelativePath);

        ModelEntry? entry = manifest.FindByFilename(filenameOrRelativePath)
            ?? manifest.FindByPath(filenameOrRelativePath);

        return entry ?? throw new FileNotFoundException(
            $"Model '{filenameOrRelativePath}' was not found in model_manifest.json.",
            filenameOrRelativePath);
    }

    /// <summary>
    /// Creates an <see cref="OnnxSessionResult"/> for <paramref name="modelFilenameOrPath"/> using
    /// <c>IntraOpNumThreads = 1</c>, <c>InterOpNumThreads = 1</c>, and <c>ORT_ENABLE_ALL</c>.
    /// Attempts CUDA when requested and falls back cleanly to CPU on any provider/library load error.
    /// </summary>
    public static OnnxSessionResult CreateSession(
        ModelManifest manifest,
        string modelFilenameOrPath,
        string? modelsRootDirectory,
        ComputeBackend requestedBackend,
        int intraOpNumThreads = 1)
    {
        int safeThreads = Math.Clamp(intraOpNumThreads, 1, 8);
        ModelEntry entry = ResolveModelEntry(manifest, modelFilenameOrPath);
        string root = string.IsNullOrWhiteSpace(modelsRootDirectory)
            ? AppContext.BaseDirectory
            : modelsRootDirectory;

        string fullPath = entry.ResolveFullPath(root);
        if (!File.Exists(fullPath))
        {
            throw new FileNotFoundException(
                $"ONNX model file '{entry.Path}' does not exist at '{fullPath}'.",
                fullPath);
        }

        if (requestedBackend == ComputeBackend.Cuda)
        {
            string? cudaFallbackReason = null;
            try
            {
                EnsureCudaDllSearchPath();
                using SessionOptions cudaOptions = CreateBaseSessionOptions(safeThreads);
                using var cudaProviderOptions = new OrtCUDAProviderOptions();
                cudaProviderOptions.UpdateOptions(new Dictionary<string, string>(StringComparer.Ordinal)
                {
                    ["device_id"] = "0",
                    ["cudnn_conv_algo_search"] = "HEURISTIC",
                    ["do_copy_in_default_stream"] = "1"
                });
                cudaOptions.AppendExecutionProvider_CUDA(cudaProviderOptions);

                var cudaSession = new InferenceSession(fullPath, cudaOptions);
                return new OnnxSessionResult(
                    session: cudaSession,
                    modelEntry: entry,
                    resolvedModelPath: fullPath,
                    requestedBackend: requestedBackend,
                    activeBackend: ComputeBackend.Cuda,
                    activeProviderName: CudaProviderName,
                    fallbackMessage: null);
            }
            catch (Exception ex)
            {
                cudaFallbackReason = $"CUDA EP init failed for '{entry.Filename}' ({ex.Message}); fell back to CPU.";
            }

            using SessionOptions fallbackCpuOptions = CreateBaseSessionOptions(safeThreads);
            var fallbackSession = new InferenceSession(fullPath, fallbackCpuOptions);
            return new OnnxSessionResult(
                session: fallbackSession,
                modelEntry: entry,
                resolvedModelPath: fullPath,
                requestedBackend: requestedBackend,
                activeBackend: ComputeBackend.Cpu,
                activeProviderName: CpuProviderName,
                fallbackMessage: cudaFallbackReason);
        }

        using SessionOptions cpuOptions = CreateBaseSessionOptions(safeThreads);
        var cpuSession = new InferenceSession(fullPath, cpuOptions);
        return new OnnxSessionResult(
            session: cpuSession,
            modelEntry: entry,
            resolvedModelPath: fullPath,
            requestedBackend: requestedBackend,
            activeBackend: ComputeBackend.Cpu,
            activeProviderName: CpuProviderName,
            fallbackMessage: null);
    }

    /// <summary>
    /// Converts a manifest <see cref="TensorDescriptor"/> shape into concrete <see cref="long"/> dimensions,
    /// applying optional symbolic dimension overrides (e.g., <c>"S"</c> or <c>"K"</c>) or a dynamic fallback.
    /// </summary>
    public static long[] ResolveTensorShape(
        TensorDescriptor descriptor,
        IReadOnlyDictionary<string, long>? symbolicOverrides = null,
        long dynamicFallback = 1)
    {
        ArgumentNullException.ThrowIfNull(descriptor);

        if (descriptor.Shape.Count == 0)
        {
            return [];
        }

        var dims = new long[descriptor.Shape.Count];
        for (int i = 0; i < descriptor.Shape.Count; i++)
        {
            TensorDimension dim = descriptor.Shape[i];
            if (dim.IsStatic && dim.StaticValue.HasValue)
            {
                dims[i] = dim.StaticValue.Value;
            }
            else if (!string.IsNullOrEmpty(dim.SymbolicName)
                && symbolicOverrides is not null
                && symbolicOverrides.TryGetValue(dim.SymbolicName, out long resolved))
            {
                dims[i] = resolved;
            }
            else
            {
                dims[i] = dynamicFallback;
            }
        }

        return dims;
    }

    /// <summary>
    /// Computes the total number of elements for a tensor shape (returns <c>1</c> for rank-0 scalars).
    /// </summary>
    public static int ComputeElementCount(ReadOnlySpan<long> shape)
    {
        if (shape.IsEmpty)
        {
            return 1;
        }

        long total = 1;
        for (int i = 0; i < shape.Length; i++)
        {
            total = checked(total * shape[i]);
        }

        return checked((int)total);
    }

    private static SessionOptions CreateBaseSessionOptions(int intraOpNumThreads)
    {
        var opts = new SessionOptions
        {
            IntraOpNumThreads = intraOpNumThreads,
            InterOpNumThreads = 1,
            ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
            EnableCpuMemArena = true,
            EnableMemoryPattern = true,
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
            LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR
        };
        opts.AddSessionConfigEntry("session.set_denormal_as_zero", "1");
        if (intraOpNumThreads <= 1)
        {
            opts.AddSessionConfigEntry("session.intra_op.allow_spinning", "0");
        }
        return opts;
    }

    private static void EnsureCudaDllSearchPath()
    {
        lock (CudaPathSync)
        {
            if (_cudaPathInitialized)
            {
                return;
            }

            if (Directory.Exists(DefaultCuda118BinDirectory))
            {
                // Prepend CUDA 11.8 bin before newer CUDA toolkits (e.g. 13.x) so ORT 1.17.1 resolves
                // cudart64_110 / cublas64_11 / cufft64_10 / cudnn64_8 reliably.
                string currentPath = Environment.GetEnvironmentVariable("PATH") ?? string.Empty;
                if (!currentPath.StartsWith(DefaultCuda118BinDirectory, StringComparison.OrdinalIgnoreCase))
                {
                    string updatedPath = string.IsNullOrEmpty(currentPath)
                        ? DefaultCuda118BinDirectory
                        : $"{DefaultCuda118BinDirectory};{currentPath}";
                    Environment.SetEnvironmentVariable("PATH", updatedPath);
                }

                NativeMethods.SetDllDirectory(DefaultCuda118BinDirectory);
            }

            _cudaPathInitialized = true;
        }
    }

    private static class NativeMethods
    {
        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        public static extern bool SetDllDirectory(string lpPathName);
    }
}
