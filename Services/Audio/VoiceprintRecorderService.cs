using System.IO;
using NAudio.CoreAudioApi;
using NAudio.Wave;

namespace VoiceLockStudio.Services.Audio;

public sealed record VoiceprintRecordingProgress(
    double Progress,
    float RmsLevel,
    double ElapsedSeconds,
    double TotalSeconds,
    bool IsRecording);

/// <summary>
/// Records a 3–5 second reference voiceprint from a WASAPI capture device, resamples to 16 kHz mono,
/// writes uncompressed PCM WAV to <see cref="VoiceprintStorage.DefaultUserRawVoiceWavPath"/>,
/// and supports audition playback via <see cref="WasapiOut"/>.
/// </summary>
public sealed class VoiceprintRecorderService : IDisposable
{
    private const int TargetSampleRate = 16000;

    private readonly AudioDeviceService _deviceService;
    private readonly object _sync = new();

    private WasapiCapture? _capture;
    private CancellationTokenSource? _recordCts;

    private WasapiOut? _playbackOut;
    private WaveFileReader? _playbackReader;

    private volatile bool _isRecording;
    private volatile bool _isPlaying;
    private bool _disposed;

    public VoiceprintRecorderService(AudioDeviceService? deviceService = null)
    {
        _deviceService = deviceService ?? new AudioDeviceService();
    }

    public bool IsRecording => _isRecording;
    public bool IsPlaying => _isPlaying;

    public event EventHandler<VoiceprintRecordingProgress>? RecordingProgress;
    public event EventHandler? PlaybackStopped;

    public async Task<string> RecordAsync(
        string? captureDeviceId,
        double durationSeconds = 5.0,
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        durationSeconds = Math.Clamp(durationSeconds, 3.0, 10.0);
        StopPlayback();

        lock (_sync)
        {
            if (_isRecording)
            {
                throw new InvalidOperationException("Voiceprint recording is already in progress.");
            }
            _isRecording = true;
            _recordCts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        }

        string outputPath = VoiceprintStorage.DefaultUserRawVoiceWavPath;
        string? dir = Path.GetDirectoryName(outputPath);
        if (!string.IsNullOrEmpty(dir))
        {
            Directory.CreateDirectory(dir);
        }

        int targetSamples = (int)Math.Round(TargetSampleRate * durationSeconds);
        float[] recorded16k = new float[targetSamples];
        int recordedCount = 0;

        MMDevice? captureDevice = _deviceService.ResolveDevice(captureDeviceId, DataFlow.Capture)
            ?? throw new InvalidOperationException("No active WASAPI capture device was found.");

        var completionTcs = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        WasapiCapture? capture = null;

        try
        {
            capture = new WasapiCapture(captureDevice, useEventSync: true, audioBufferMillisecondsLength: 20);
            var resampler = new StreamingResampler(capture.WaveFormat.SampleRate, TargetSampleRate);
            float[] monoScratch = new float[Math.Max(4096, capture.WaveFormat.SampleRate / 2)];
            float[] resampledScratch = new float[resampler.GetMaxOutputSamples(monoScratch.Length)];

            lock (_sync)
            {
                _capture = capture;
            }

            capture.DataAvailable += (_, e) =>
            {
                if (e.BytesRecorded <= 0 || recordedCount >= targetSamples)
                {
                    return;
                }

                int monoFrames = WasapiPcmConverter.ReadMonoFloats(
                    e.Buffer.AsSpan(0, e.BytesRecorded),
                    capture.WaveFormat,
                    monoScratch);

                if (monoFrames <= 0)
                {
                    return;
                }

                float rms = WasapiPcmConverter.ComputeRmsAndPeak(monoScratch.AsSpan(0, monoFrames), out float _);
                int outSamples = resampler.Process(monoScratch.AsSpan(0, monoFrames), resampledScratch);
                if (outSamples > 0)
                {
                    int copyCount = Math.Min(outSamples, targetSamples - recordedCount);
                    resampledScratch.AsSpan(0, copyCount).CopyTo(recorded16k.AsSpan(recordedCount, copyCount));
                    recordedCount += copyCount;
                }

                double progress = Math.Clamp((double)recordedCount / targetSamples, 0.0, 1.0);
                double elapsed = (double)recordedCount / TargetSampleRate;
                RecordingProgress?.Invoke(this, new VoiceprintRecordingProgress(
                    Progress: progress,
                    RmsLevel: rms,
                    ElapsedSeconds: elapsed,
                    TotalSeconds: durationSeconds,
                    IsRecording: recordedCount < targetSamples));

                if (recordedCount >= targetSamples)
                {
                    try
                    {
                        capture.StopRecording();
                    }
                    catch
                    {
                        // Ignore stop races from callback thread.
                    }
                }
            };

            capture.RecordingStopped += (_, args) =>
            {
                if (args.Exception is not null)
                {
                    completionTcs.TrySetException(args.Exception);
                }
                else
                {
                    completionTcs.TrySetResult();
                }
            };

            CancellationToken linkedToken = _recordCts!.Token;
            using CancellationTokenRegistration reg = linkedToken.Register(() =>
            {
                try
                {
                    capture.StopRecording();
                }
                catch
                {
                    completionTcs.TrySetCanceled(linkedToken);
                }
            });

            capture.StartRecording();
            await completionTcs.Task.ConfigureAwait(false);
            linkedToken.ThrowIfCancellationRequested();

            int validSamples = Math.Max(0, Math.Min(recordedCount, recorded16k.Length));
            if (validSamples == 0)
            {
                throw new InvalidOperationException("No audio samples were captured from the selected microphone.");
            }

            string tempPath = outputPath + ".tmp";
            using (var writer = new WaveFileWriter(tempPath, new WaveFormat(TargetSampleRate, 16, 1)))
            {
                writer.WriteSamples(recorded16k, 0, validSamples);
                writer.Flush();
            }

            File.Move(tempPath, outputPath, overwrite: true);
            RecordingProgress?.Invoke(this, new VoiceprintRecordingProgress(
                Progress: 1.0,
                RmsLevel: 0f,
                ElapsedSeconds: (double)validSamples / TargetSampleRate,
                TotalSeconds: durationSeconds,
                IsRecording: false));

            return outputPath;
        }
        finally
        {
            lock (_sync)
            {
                if (ReferenceEquals(_capture, capture))
                {
                    _capture = null;
                }
                _recordCts?.Dispose();
                _recordCts = null;
                _isRecording = false;
            }

            capture?.Dispose();
            captureDevice.Dispose();
        }
    }

    public void StopRecording()
    {
        lock (_sync)
        {
            _recordCts?.Cancel();
            try
            {
                _capture?.StopRecording();
            }
            catch
            {
                // Ignore if already stopped.
            }
        }
    }

    public void StartPlayback(string? wavFilePath = null, string? renderDeviceId = null)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        string targetPath = string.IsNullOrWhiteSpace(wavFilePath)
            ? VoiceprintStorage.DefaultUserRawVoiceWavPath
            : wavFilePath;

        if (!File.Exists(targetPath))
        {
            throw new FileNotFoundException("Reference voiceprint WAV file not found.", targetPath);
        }

        StopPlayback();

        MMDevice? renderDevice = _deviceService.ResolveDevice(renderDeviceId, DataFlow.Render)
            ?? throw new InvalidOperationException("No active WASAPI render device was found.");

        try
        {
            var reader = new WaveFileReader(targetPath);
            var wasapiOut = new WasapiOut(renderDevice, AudioClientShareMode.Shared, useEventSync: true, latency: 20);
            var provider = new WavFileResamplingProvider(reader, wasapiOut.OutputWaveFormat);

            lock (_sync)
            {
                _playbackReader = reader;
                _playbackOut = wasapiOut;
                _isPlaying = true;
            }

            wasapiOut.PlaybackStopped += (_, _) =>
            {
                StopPlayback();
                PlaybackStopped?.Invoke(this, EventArgs.Empty);
            };

            wasapiOut.Init(provider);
            wasapiOut.Play();
        }
        catch
        {
            StopPlayback();
            throw;
        }
        finally
        {
            renderDevice.Dispose();
        }
    }

    public void StopPlayback()
    {
        WasapiOut? wasapiOut;
        WaveFileReader? reader;
        lock (_sync)
        {
            wasapiOut = _playbackOut;
            reader = _playbackReader;
            _playbackOut = null;
            _playbackReader = null;
            _isPlaying = false;
        }

        try
        {
            wasapiOut?.Stop();
        }
        catch
        {
            // Ignore stop races.
        }

        wasapiOut?.Dispose();
        reader?.Dispose();
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        StopRecording();
        StopPlayback();
    }

    private sealed class WavFileResamplingProvider : IWaveProvider
    {
        private readonly WaveFileReader _reader;
        private readonly StreamingResampler _resampler;
        private readonly byte[] _readBytes;
        private readonly float[] _monoIn;
        private readonly float[] _monoOut;
        private readonly AudioRingBuffer _outRing;

        public WavFileResamplingProvider(WaveFileReader reader, WaveFormat targetFormat)
        {
            _reader = reader;
            WaveFormat = targetFormat;
            _resampler = new StreamingResampler(reader.WaveFormat.SampleRate, targetFormat.SampleRate);
            _readBytes = new byte[Math.Max(4096, reader.WaveFormat.AverageBytesPerSecond / 10)];
            _monoIn = new float[_readBytes.Length / Math.Max(1, reader.WaveFormat.BlockAlign) + 16];
            _monoOut = new float[_resampler.GetMaxOutputSamples(_monoIn.Length) + 64];
            _outRing = new AudioRingBuffer(Math.Max(8192, targetFormat.SampleRate));
        }

        public WaveFormat WaveFormat { get; }

        public int Read(byte[] buffer, int offset, int count)
        {
            int framesNeeded = count / WaveFormat.BlockAlign;
            if (framesNeeded <= 0)
            {
                return 0;
            }

            Span<float> outSpan = _monoOut.Length >= framesNeeded
                ? _monoOut.AsSpan(0, framesNeeded)
                : new float[framesNeeded];

            while (_outRing.AvailableRead < framesNeeded)
            {
                int bytesRead = _reader.Read(_readBytes, 0, _readBytes.Length);
                if (bytesRead <= 0)
                {
                    break;
                }

                int monoCount = WasapiPcmConverter.ReadMonoFloats(
                    _readBytes.AsSpan(0, bytesRead),
                    _reader.WaveFormat,
                    _monoIn);

                if (monoCount <= 0)
                {
                    break;
                }

                int produced = _resampler.Process(_monoIn.AsSpan(0, monoCount), _monoOut);
                if (produced > 0)
                {
                    _outRing.Write(_monoOut.AsSpan(0, produced));
                }
            }

            int readCount = _outRing.Read(outSpan);
            if (readCount <= 0)
            {
                return 0;
            }

            int byteCount = readCount * WaveFormat.BlockAlign;
            WasapiPcmConverter.WriteMonoFloats(
                outSpan.Slice(0, readCount),
                WaveFormat,
                buffer.AsSpan(offset, byteCount));

            return byteCount;
        }
    }
}
