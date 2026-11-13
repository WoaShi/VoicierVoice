using System.Buffers.Binary;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using NAudio.Wave;

namespace VoiceLockStudio.Services.Audio;

/// <summary>
/// Zero-allocation PCM format conversion and metering helpers for WASAPI capture/render buffers.
/// </summary>
internal static class WasapiPcmConverter
{
    private static readonly Guid SubtypeIeeeFloat = new("00000003-0000-0010-8000-00aa00389b71");

    public static bool IsIeeeFloat(WaveFormat format)
    {
        if (format.Encoding == WaveFormatEncoding.IeeeFloat)
        {
            return true;
        }

        return format is WaveFormatExtensible ext && ext.SubFormat == SubtypeIeeeFloat;
    }

    public static int ReadMonoFloats(ReadOnlySpan<byte> sourceBytes, WaveFormat format, Span<float> destinationMono)
    {
        int bytesPerSample = format.BitsPerSample / 8;
        int channels = Math.Max(1, format.Channels);
        int blockAlign = Math.Max(bytesPerSample * channels, format.BlockAlign);
        if (blockAlign <= 0 || sourceBytes.Length < blockAlign)
        {
            return 0;
        }

        int frameCount = Math.Min(sourceBytes.Length / blockAlign, destinationMono.Length);
        bool isFloat = IsIeeeFloat(format);
        float invChannels = 1f / channels;

        if (isFloat && format.BitsPerSample == 32)
        {
            ReadOnlySpan<float> floatSrc = MemoryMarshal.Cast<byte, float>(sourceBytes);
            if (channels == 1)
            {
                floatSrc.Slice(0, frameCount).CopyTo(destinationMono);
                return frameCount;
            }

            for (int f = 0; f < frameCount; f++)
            {
                int baseIdx = f * channels;
                float sum = 0f;
                for (int c = 0; c < channels; c++)
                {
                    sum += floatSrc[baseIdx + c];
                }
                destinationMono[f] = sum * invChannels;
            }

            return frameCount;
        }

        if (format.BitsPerSample == 16)
        {
            const float scale16 = 1f / 32768f;
            for (int f = 0; f < frameCount; f++)
            {
                int frameOffset = f * blockAlign;
                float sum = 0f;
                for (int c = 0; c < channels; c++)
                {
                    short s = BinaryPrimitives.ReadInt16LittleEndian(sourceBytes.Slice(frameOffset + c * 2, 2));
                    sum += s * scale16;
                }
                destinationMono[f] = sum * invChannels;
            }

            return frameCount;
        }

        if (format.BitsPerSample == 24)
        {
            const float scale24 = 1f / 8388608f;
            for (int f = 0; f < frameCount; f++)
            {
                int frameOffset = f * blockAlign;
                float sum = 0f;
                for (int c = 0; c < channels; c++)
                {
                    int off = frameOffset + c * 3;
                    int val = sourceBytes[off] | (sourceBytes[off + 1] << 8) | ((sbyte)sourceBytes[off + 2] << 16);
                    sum += val * scale24;
                }
                destinationMono[f] = sum * invChannels;
            }

            return frameCount;
        }

        if (format.BitsPerSample == 32)
        {
            const float scale32 = 1f / 2147483648f;
            for (int f = 0; f < frameCount; f++)
            {
                int frameOffset = f * blockAlign;
                float sum = 0f;
                for (int c = 0; c < channels; c++)
                {
                    int s = BinaryPrimitives.ReadInt32LittleEndian(sourceBytes.Slice(frameOffset + c * 4, 4));
                    sum += s * scale32;
                }
                destinationMono[f] = sum * invChannels;
            }

            return frameCount;
        }

        return 0;
    }

    public static void WriteMonoFloats(ReadOnlySpan<float> sourceMono, WaveFormat targetFormat, Span<byte> destinationBytes)
    {
        int channels = Math.Max(1, targetFormat.Channels);
        int bytesPerSample = targetFormat.BitsPerSample / 8;
        int blockAlign = Math.Max(bytesPerSample * channels, targetFormat.BlockAlign);
        int frameCount = Math.Min(sourceMono.Length, destinationBytes.Length / blockAlign);

        bool isFloat = IsIeeeFloat(targetFormat);
        if (isFloat && targetFormat.BitsPerSample == 32)
        {
            Span<float> dstFloat = MemoryMarshal.Cast<byte, float>(destinationBytes);
            for (int f = 0; f < frameCount; f++)
            {
                float sample = Math.Clamp(sourceMono[f], -1f, 1f);
                int baseIdx = f * channels;
                for (int c = 0; c < channels; c++)
                {
                    dstFloat[baseIdx + c] = sample;
                }
            }
            return;
        }

        if (targetFormat.BitsPerSample == 16)
        {
            for (int f = 0; f < frameCount; f++)
            {
                float clamped = Math.Clamp(sourceMono[f], -1f, 1f);
                short pcm = (short)MathF.Round(clamped * 32767f);
                int frameOffset = f * blockAlign;
                for (int c = 0; c < channels; c++)
                {
                    BinaryPrimitives.WriteInt16LittleEndian(destinationBytes.Slice(frameOffset + c * 2, 2), pcm);
                }
            }
            return;
        }

        if (targetFormat.BitsPerSample == 32)
        {
            for (int f = 0; f < frameCount; f++)
            {
                float clamped = Math.Clamp(sourceMono[f], -1f, 1f);
                int pcm = (int)Math.Round(clamped * 2147483647.0);
                int frameOffset = f * blockAlign;
                for (int c = 0; c < channels; c++)
                {
                    BinaryPrimitives.WriteInt32LittleEndian(destinationBytes.Slice(frameOffset + c * 4, 4), pcm);
                }
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float ComputeRmsAndPeak(ReadOnlySpan<float> samples, out float peak)
    {
        if (samples.IsEmpty)
        {
            peak = 0f;
            return 0f;
        }

        double sumSq = 0.0;
        float maxAbs = 0f;
        for (int i = 0; i < samples.Length; i++)
        {
            float s = samples[i];
            float abs = MathF.Abs(s);
            if (abs > maxAbs)
            {
                maxAbs = abs;
            }
            sumSq += s * s;
        }

        peak = maxAbs;
        return (float)Math.Sqrt(sumSq / samples.Length);
    }
}
