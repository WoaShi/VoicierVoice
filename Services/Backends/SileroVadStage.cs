using System.IO;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime;
using VoiceLockStudio.Models;
using VoiceLockStudio.Services.Audio;

namespace VoiceLockStudio.Services.Backends;

/// <summary>
/// Streaming Silero-VAD v5 voice activity detection stage operating on 512-sample (32 ms)
/// 16 kHz mono frames with a 64-sample context prefix and persistent recurrent state.
/// </summary>
public sealed class SileroVadStage : IAudioFilterStage
{
    private const string DefaultModelKey = "silero_vad.onnx";

    private readonly object _lifecycleLock = new();

    private OnnxSessionResult? _sessionResult;
    private RunOptions? _runOptions;

    private int _sampleRate = 16000;
    private int _frameSizeSamples = 512;
    private int _contextSamples = 64;
    private int _effectiveInputSamples = 576;

    private float[] _inputBuffer = [];
    private float[] _stateBufferA = [];
    private float[] _stateBufferB = [];
    private long[] _sampleRateBuffer = [];
    private float[] _outputProbBuffer = [];

    private OrtValue? _ortInput;
    private OrtValue? _ortStateA;
    private OrtValue? _ortStateB;
    private OrtValue? _ortSampleRate;
    private OrtValue? _ortOutputProb;

    private string[] _inputNames = [];
    private OrtValue[] _inputValues = [];
    private string[] _outputNames = [];
    private OrtValue[] _outputValues = [];
    private int _stateInputSlot;
    private int _stateOutputSlot;
    private bool _stateIsA = true;

    private float _lastSpeechProbability;
    private bool _isInitialized;
    private bool _disposed;

    public string Name => "Silero-VAD";
    public int FrameSizeSamples => _frameSizeSamples;
    public int SampleRate => _sampleRate;
    public bool IsInitialized => Volatile.Read(ref _isInitialized);

    /// <summary>
    /// Speech probability in [0, 1] from the most recent <see cref="ProcessFrame"/> invocation.
    /// </summary>
    public float LastSpeechProbability => Volatile.Read(ref _lastSpeechProbability);

    public string ActiveProviderName => _sessionResult?.ActiveProviderName ?? string.Empty;
    public string? ProviderFallbackMessage => _sessionResult?.FallbackMessage;

    public void Initialize(AudioStageOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        ObjectDisposedException.ThrowIf(_disposed, this);

        lock (_lifecycleLock)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            ReleaseResources();

            ModelManifest manifest = OnnxSessionFactory.ResolveManifest(options);
            ModelEntry entry = ResolveVadModelEntry(manifest);

            ModelAudioParams audio = entry.AudioParams ?? throw new InvalidDataException(
                $"Model '{entry.Filename}' is missing audio_params in model_manifest.json.");

            _sampleRate = audio.SampleRate > 0
                ? audio.SampleRate
                : throw new InvalidDataException($"Model '{entry.Filename}' has invalid sample_rate.");

            _frameSizeSamples = audio.WindowSizeSamples16k
                ?? throw new InvalidDataException($"Model '{entry.Filename}' is missing window_size_samples_16k.");

            _contextSamples = audio.ContextSamples16k
                ?? throw new InvalidDataException($"Model '{entry.Filename}' is missing context_samples_16k.");

            _effectiveInputSamples = audio.EffectiveModelInputSamples16k
                ?? (_frameSizeSamples + _contextSamples);

            if (_effectiveInputSamples != _frameSizeSamples + _contextSamples)
            {
                throw new InvalidDataException(
                    $"Model '{entry.Filename}' effective input ({_effectiveInputSamples}) != frame ({_frameSizeSamples}) + context ({_contextSamples}).");
            }

            IReadOnlyList<TensorDescriptor> inputs = entry.GetEffectiveInputs();
            IReadOnlyList<TensorDescriptor> outputs = entry.GetEffectiveOutputs();
            if (inputs.Count < 3 || outputs.Count < 2)
            {
                throw new InvalidDataException(
                    $"Model '{entry.Filename}' has unexpected I/O count ({inputs.Count} inputs, {outputs.Count} outputs).");
            }

            _sessionResult = OnnxSessionFactory.CreateSession(
                manifest,
                entry.Filename,
                options.ModelsRootDirectory,
                options.Backend);

            TensorDescriptor audioInDesc = inputs.First(d =>
                d.Shape.Count == 2 && d.EffectiveType.Contains("float", StringComparison.OrdinalIgnoreCase));
            TensorDescriptor stateInDesc = inputs.First(d =>
                d.Shape.Count == 3 && d.EffectiveType.Contains("float", StringComparison.OrdinalIgnoreCase));
            TensorDescriptor srInDesc = inputs.First(d =>
                d.Shape.Count == 0 && d.EffectiveType.Contains("int64", StringComparison.OrdinalIgnoreCase));

            TensorDescriptor probOutDesc = outputs.First(d =>
                d.Shape.Count == 2 && d.EffectiveType.Contains("float", StringComparison.OrdinalIgnoreCase));
            TensorDescriptor stateOutDesc = outputs.First(d =>
                d.Shape.Count == 3 && d.EffectiveType.Contains("float", StringComparison.OrdinalIgnoreCase));

            long[] inputShape = [1L, _effectiveInputSamples];
            long[] stateShape = OnnxSessionFactory.ResolveTensorShape(stateInDesc, dynamicFallback: 1);
            long[] srShape = OnnxSessionFactory.ResolveTensorShape(srInDesc);
            long[] probShape = [1L, 1L];

            int stateElemCount = OnnxSessionFactory.ComputeElementCount(stateShape);
            int probElemCount = OnnxSessionFactory.ComputeElementCount(probShape);

            _inputBuffer = GC.AllocateArray<float>(_effectiveInputSamples, pinned: true);
            _stateBufferA = GC.AllocateArray<float>(stateElemCount, pinned: true);
            _stateBufferB = GC.AllocateArray<float>(stateElemCount, pinned: true);
            _sampleRateBuffer = GC.AllocateArray<long>(1, pinned: true);
            _sampleRateBuffer[0] = _sampleRate;
            _outputProbBuffer = GC.AllocateArray<float>(probElemCount, pinned: true);

            _ortInput = OrtValue.CreateTensorValueFromMemory(_inputBuffer, inputShape);
            _ortStateA = OrtValue.CreateTensorValueFromMemory(_stateBufferA, stateShape);
            _ortStateB = OrtValue.CreateTensorValueFromMemory(_stateBufferB, stateShape);
            _ortSampleRate = OrtValue.CreateTensorValueFromMemory(_sampleRateBuffer, srShape);
            _ortOutputProb = OrtValue.CreateTensorValueFromMemory(_outputProbBuffer, probShape);

            _runOptions = new RunOptions();
            _inputNames = [audioInDesc.Name, stateInDesc.Name, srInDesc.Name];
            _outputNames = [probOutDesc.Name, stateOutDesc.Name];
            _stateInputSlot = 1;
            _stateOutputSlot = 1;
            _stateIsA = true;

            _inputValues = [_ortInput, _ortStateA, _ortSampleRate];
            _outputValues = [_ortOutputProb, _ortStateB];

            Volatile.Write(ref _lastSpeechProbability, 0f);
            Volatile.Write(ref _isInitialized, true);

            Warmup();
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public void ProcessFrame(ReadOnlySpan<float> input, Span<float> output)
    {
        if (!Volatile.Read(ref _isInitialized) || _sessionResult is null || _runOptions is null)
        {
            throw new InvalidOperationException($"{Name} must be initialized before calling ProcessFrame.");
        }

        if (input.Length != _frameSizeSamples)
        {
            throw new ArgumentException($"Expected {_frameSizeSamples} input samples, got {input.Length}.", nameof(input));
        }

        if (output.Length != _frameSizeSamples)
        {
            throw new ArgumentException($"Expected {_frameSizeSamples} output samples, got {output.Length}.", nameof(output));
        }

        // Prepend previous 64-sample context (already at indices 0..63) with current 512-sample frame.
        input.CopyTo(_inputBuffer.AsSpan(_contextSamples, _frameSizeSamples));

        _sessionResult.Session.Run(_runOptions, _inputNames, _inputValues, _outputNames, _outputValues);

        // Save trailing 64 samples of the current frame as context for the next hop.
        _inputBuffer.AsSpan(_frameSizeSamples, _contextSamples).CopyTo(_inputBuffer.AsSpan(0, _contextSamples));

        // Ping-pong recurrent state buffers without allocations.
        _stateIsA = !_stateIsA;
        if (_stateIsA)
        {
            _inputValues[_stateInputSlot] = _ortStateA!;
            _outputValues[_stateOutputSlot] = _ortStateB!;
        }
        else
        {
            _inputValues[_stateInputSlot] = _ortStateB!;
            _outputValues[_stateOutputSlot] = _ortStateA!;
        }

        Volatile.Write(ref _lastSpeechProbability, _outputProbBuffer[0]);

        if (!Unsafe.AreSame(
            ref MemoryMarshal.GetReference(input),
            ref MemoryMarshal.GetReference(output)))
        {
            input.CopyTo(output);
        }
    }

    public void ResetState()
    {
        if (!Volatile.Read(ref _isInitialized))
        {
            return;
        }

        Array.Clear(_inputBuffer);
        Array.Clear(_stateBufferA);
        Array.Clear(_stateBufferB);
        Array.Clear(_outputProbBuffer);

        _stateIsA = true;
        if (_ortStateA is not null && _ortStateB is not null)
        {
            _inputValues[_stateInputSlot] = _ortStateA;
            _outputValues[_stateOutputSlot] = _ortStateB;
        }

        Volatile.Write(ref _lastSpeechProbability, 0f);
    }

    public void Warmup()
    {
        if (!Volatile.Read(ref _isInitialized) || _sessionResult is null || _runOptions is null)
        {
            return;
        }

        Array.Clear(_inputBuffer);
        for (int i = 0; i < 2; i++)
        {
            _sessionResult.Session.Run(_runOptions, _inputNames, _inputValues, _outputNames, _outputValues);
        }
        ResetState();
    }

    public void Dispose()
    {
        lock (_lifecycleLock)
        {
            if (_disposed)
            {
                return;
            }

            _disposed = true;
            ReleaseResources();
        }
    }

    private static ModelEntry ResolveVadModelEntry(ModelManifest manifest)
    {
        return manifest.FindByFilename(DefaultModelKey)
            ?? manifest.Models.FirstOrDefault(m =>
                m.Filename.Contains("silero", StringComparison.OrdinalIgnoreCase))
            ?? throw new FileNotFoundException("Silero-VAD model entry not found in model_manifest.json.", DefaultModelKey);
    }

    private void ReleaseResources()
    {
        Volatile.Write(ref _isInitialized, false);
        Volatile.Write(ref _lastSpeechProbability, 0f);

        _ortInput?.Dispose();
        _ortStateA?.Dispose();
        _ortStateB?.Dispose();
        _ortSampleRate?.Dispose();
        _ortOutputProb?.Dispose();
        _runOptions?.Dispose();
        _sessionResult?.Dispose();

        _ortInput = null;
        _ortStateA = null;
        _ortStateB = null;
        _ortSampleRate = null;
        _ortOutputProb = null;
        _runOptions = null;
        _sessionResult = null;

        _inputNames = [];
        _inputValues = [];
        _outputNames = [];
        _outputValues = [];
    }
}
