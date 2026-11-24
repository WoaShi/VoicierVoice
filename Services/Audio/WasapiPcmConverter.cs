using System.Buffers.Binary;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;
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

            if (channels == 2)
            {
                int f = 0;
                if (Vector128.IsHardwareAccelerated && frameCount >= 4)
                {
                    ref float srcRef = ref MemoryMarshal.GetReference(floatSrc);
                    ref float dstRef = ref MemoryMarshal.GetReference(destinationMono);
                    Vector128<float> half = Vector128.Create(0.5f);
                    int limit = frameCount - 4;

                    for (; f <= limit; f += 4)
                    {
                        nuint srcOff = (nuint)(f * 2);
                        Vector128<float> v0 = Vector128.LoadUnsafe(ref srcRef, srcOff);
                        Vector128<float> v1 = Vector128.LoadUnsafe(ref srcRef, srcOff + 4);

                        Vector128<float> left;
                        Vector128<float> right;
                        if (Sse.IsSupported)
                        {
                            left = Sse.Shuffle(v0, v1, 0x88);
                            right = Sse.Shuffle(v0, v1, 0xDD);
                        }
                        else if (AdvSimd.Arm64.IsSupported)
                        {
                            left = AdvSimd.Arm64.UnzipEven(v0, v1);
                            right = AdvSimd.Arm64.UnzipOdd(v0, v1);
                        }
                        else
                        {
                            left = Vector128.Create(v0[0], v0[2], v1[0], v1[2]);
                            right = Vector128.Create(v0[1], v0[3], v1[1], v1[3]);
                        }

                        ((left + right) * half).StoreUnsafe(ref dstRef, (nuint)f);
                    }
                }

                for (; f < frameCount; f++)
                {
                    int baseIdx = f * 2;
                    destinationMono[f] = (floatSrc[baseIdx] + floatSrc[baseIdx + 1]) * 0.5f;
                }

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

        if (!isFloat && format.BitsPerSample == 16)
        {
            const float scale16 = 1f / 32768f;
            if (channels == 1 && blockAlign == 2 && BitConverter.IsLittleEndian)
            {
                int f = 0;
                if (Vector128.IsHardwareAccelerated && frameCount >= 8)
                {
                    ReadOnlySpan<short> shortSrc = MemoryMarshal.Cast<byte, short>(sourceBytes);
                    ref short srcRef = ref MemoryMarshal.GetReference(shortSrc);
                    ref float dstRef = ref MemoryMarshal.GetReference(destinationMono);
                    Vector128<float> vScale = Vector128.Create(scale16);
                    int limit = frameCount - 8;

                    for (; f <= limit; f += 8)
                    {
                        Vector128<short> vShort = Vector128.LoadUnsafe(ref srcRef, (nuint)f);
                        (Vector128<int> lo, Vector128<int> hi) = Vector128.Widen(vShort);
                        (Vector128.ConvertToSingle(lo) * vScale).StoreUnsafe(ref dstRef, (nuint)f);
                        (Vector128.ConvertToSingle(hi) * vScale).StoreUnsafe(ref dstRef, (nuint)(f + 4));
                    }
                }

                for (; f < frameCount; f++)
                {
                    short s = BinaryPrimitives.ReadInt16LittleEndian(sourceBytes.Slice(f * 2, 2));
                    destinationMono[f] = s * scale16;
                }

                return frameCount;
            }

            if (channels == 2 && blockAlign == 4 && BitConverter.IsLittleEndian)
            {
                int f = 0;
                if (Vector128.IsHardwareAccelerated && frameCount >= 4)
                {
                    ReadOnlySpan<short> shortSrc = MemoryMarshal.Cast<byte, short>(sourceBytes);
                    ref short srcRef = ref MemoryMarshal.GetReference(shortSrc);
                    ref float dstRef = ref MemoryMarshal.GetReference(destinationMono);
                    Vector128<float> vHalfScale = Vector128.Create(0.5f * scale16);
                    int limit = frameCount - 4;

                    for (; f <= limit; f += 4)
                    {
                        Vector128<short> vShort = Vector128.LoadUnsafe(ref srcRef, (nuint)(f * 2));
                        (Vector128<int> lo, Vector128<int> hi) = Vector128.Widen(vShort);
                        Vector128<float> v0 = Vector128.ConvertToSingle(lo);
                        Vector128<float> v1 = Vector128.ConvertToSingle(hi);

                        Vector128<float> left;
                        Vector128<float> right;
                        if (Sse.IsSupported)
                        {
                            left = Sse.Shuffle(v0, v1, 0x88);
                            right = Sse.Shuffle(v0, v1, 0xDD);
                        }
                        else if (AdvSimd.Arm64.IsSupported)
                        {
                            left = AdvSimd.Arm64.UnzipEven(v0, v1);
                            right = AdvSimd.Arm64.UnzipOdd(v0, v1);
                        }
                        else
                        {
                            left = Vector128.Create(v0[0], v0[2], v1[0], v1[2]);
                            right = Vector128.Create(v0[1], v0[3], v1[1], v1[3]);
                        }

                        ((left + right) * vHalfScale).StoreUnsafe(ref dstRef, (nuint)f);
                    }
                }

                for (; f < frameCount; f++)
                {
                    int frameOffset = f * 4;
                    short s0 = BinaryPrimitives.ReadInt16LittleEndian(sourceBytes.Slice(frameOffset, 2));
                    short s1 = BinaryPrimitives.ReadInt16LittleEndian(sourceBytes.Slice(frameOffset + 2, 2));
                    destinationMono[f] = (s0 + s1) * (0.5f * scale16);
                }

                return frameCount;
            }

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
            if (channels == 1)
            {
                int f = 0;
                if (Vector<float>.IsSupported && frameCount >= Vector<float>.Count)
                {
                    ref float srcRef = ref MemoryMarshal.GetReference(sourceMono);
                    ref float dstRef = ref MemoryMarshal.GetReference(dstFloat);
                    int vecCount = Vector<float>.Count;
                    int limit = frameCount - vecCount;
                    var vNegOne = new Vector<float>(-1f);
                    var vPosOne = new Vector<float>(1f);

                    for (; f <= limit; f += vecCount)
                    {
                        nuint off = (nuint)f;
                        Vector<float> v = Vector.LoadUnsafe(ref srcRef, off);
                        Vector<float> clamped = Vector.Max(vNegOne, Vector.Min(vPosOne, v));
                        clamped.StoreUnsafe(ref dstRef, off);
                    }
                }

                for (; f < frameCount; f++)
                {
                    dstFloat[f] = Math.Clamp(sourceMono[f], -1f, 1f);
                }
                return;
            }

            if (channels == 2)
            {
                int f = 0;
                if (Vector128.IsHardwareAccelerated && frameCount >= 4)
                {
                    ref float srcRef = ref MemoryMarshal.GetReference(sourceMono);
                    ref float dstRef = ref MemoryMarshal.GetReference(dstFloat);
                    Vector128<float> vNegOne = Vector128.Create(-1f);
                    Vector128<float> vPosOne = Vector128.Create(1f);
                    int limit = frameCount - 4;

                    for (; f <= limit; f += 4)
                    {
                        Vector128<float> v = Vector128.LoadUnsafe(ref srcRef, (nuint)f);
                        Vector128<float> clamped = Vector128.Max(vNegOne, Vector128.Min(vPosOne, v));

                        Vector128<float> lo;
                        Vector128<float> hi;
                        if (Sse.IsSupported)
                        {
                            lo = Sse.UnpackLow(clamped, clamped);
                            hi = Sse.UnpackHigh(clamped, clamped);
                        }
                        else if (AdvSimd.Arm64.IsSupported)
                        {
                            lo = AdvSimd.Arm64.ZipLow(clamped, clamped);
                            hi = AdvSimd.Arm64.ZipHigh(clamped, clamped);
                        }
                        else
                        {
                            lo = Vector128.Create(clamped[0], clamped[0], clamped[1], clamped[1]);
                            hi = Vector128.Create(clamped[2], clamped[2], clamped[3], clamped[3]);
                        }

                        nuint dstOff = (nuint)(f * 2);
                        lo.StoreUnsafe(ref dstRef, dstOff);
                        hi.StoreUnsafe(ref dstRef, dstOff + 4);
                    }
                }

                for (; f < frameCount; f++)
                {
                    float sample = Math.Clamp(sourceMono[f], -1f, 1f);
                    int baseIdx = f * 2;
                    dstFloat[baseIdx] = sample;
                    dstFloat[baseIdx + 1] = sample;
                }
                return;
            }

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

    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static float ComputeRmsAndPeak(ReadOnlySpan<float> samples, out float peak)
    {
        int len = samples.Length;
        if (len == 0)
        {
            peak = 0f;
            return 0f;
        }

        double sumSq = 0.0;
        float maxAbs = 0f;
        int i = 0;

        if (Vector<float>.IsSupported && len >= Vector<float>.Count)
        {
            ref float srcRef = ref MemoryMarshal.GetReference(samples);
            int vecCount = Vector<float>.Count;
            int limit = len - vecCount;

            var accSq = Vector<float>.Zero;
            var accMax = Vector<float>.Zero;

            for (; i <= limit; i += vecCount)
            {
                var v = Vector.LoadUnsafe(ref srcRef, (nuint)i);
                accSq += v * v;
                accMax = Vector.Max(accMax, Vector.Abs(v));
            }

            sumSq = Vector.Sum(accSq);
            for (int lane = 0; lane < vecCount; lane++)
            {
                float laneVal = accMax[lane];
                if (laneVal > maxAbs)
                {
                    maxAbs = laneVal;
                }
            }
        }

        for (; i < len; i++)
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
        return MathF.Sqrt((float)(sumSq / len));
    }

    /// <summary>
    /// Computes a fixed 24-tap dot product using AVX2/NEON vectorization when available.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static float Dot24(ReadOnlySpan<float> samples, ReadOnlySpan<float> coeffs)
    {
        if (Vector256.IsHardwareAccelerated && samples.Length >= 24 && coeffs.Length >= 24)
        {
            ref float xRef = ref MemoryMarshal.GetReference(samples);
            ref float cRef = ref MemoryMarshal.GetReference(coeffs);

            var acc = Vector256.LoadUnsafe(ref xRef, 0) * Vector256.LoadUnsafe(ref cRef, 0);
            acc += Vector256.LoadUnsafe(ref xRef, 8) * Vector256.LoadUnsafe(ref cRef, 8);
            acc += Vector256.LoadUnsafe(ref xRef, 16) * Vector256.LoadUnsafe(ref cRef, 16);
            return Vector256.Sum(acc);
        }

        if (Vector128.IsHardwareAccelerated && samples.Length >= 24 && coeffs.Length >= 24)
        {
            ref float xRef = ref MemoryMarshal.GetReference(samples);
            ref float cRef = ref MemoryMarshal.GetReference(coeffs);

            var acc0 = Vector128.LoadUnsafe(ref xRef, 0) * Vector128.LoadUnsafe(ref cRef, 0);
            var acc1 = Vector128.LoadUnsafe(ref xRef, 4) * Vector128.LoadUnsafe(ref cRef, 4);
            acc0 += Vector128.LoadUnsafe(ref xRef, 8) * Vector128.LoadUnsafe(ref cRef, 8);
            acc1 += Vector128.LoadUnsafe(ref xRef, 12) * Vector128.LoadUnsafe(ref cRef, 12);
            acc0 += Vector128.LoadUnsafe(ref xRef, 16) * Vector128.LoadUnsafe(ref cRef, 16);
            acc1 += Vector128.LoadUnsafe(ref xRef, 20) * Vector128.LoadUnsafe(ref cRef, 20);
            return Vector128.Sum(acc0 + acc1);
        }

        float sum = 0f;
        int len = Math.Min(24, Math.Min(samples.Length, coeffs.Length));
        for (int k = 0; k < len; k++)
        {
            sum += samples[k] * coeffs[k];
        }
        return sum;
    }

    /// <summary>
    /// Computes a linearly interpolated 24-tap polyphase dot product: sum(samples[k] * (c0[k] + alpha * (c1[k] - c0[k]))).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static float Dot24Interpolated(
        ReadOnlySpan<float> samples,
        ReadOnlySpan<float> coeffsRow0,
        ReadOnlySpan<float> coeffsRow1,
        float alpha)
    {
        if (Vector256.IsHardwareAccelerated &&
            samples.Length >= 24 &&
            coeffsRow0.Length >= 24 &&
            coeffsRow1.Length >= 24)
        {
            ref float xRef = ref MemoryMarshal.GetReference(samples);
            ref float c0Ref = ref MemoryMarshal.GetReference(coeffsRow0);
            ref float c1Ref = ref MemoryMarshal.GetReference(coeffsRow1);

            var vAlpha = Vector256.Create(alpha);

            var c0_0 = Vector256.LoadUnsafe(ref c0Ref, 0);
            var c_0 = c0_0 + vAlpha * (Vector256.LoadUnsafe(ref c1Ref, 0) - c0_0);
            var acc = Vector256.LoadUnsafe(ref xRef, 0) * c_0;

            var c0_1 = Vector256.LoadUnsafe(ref c0Ref, 8);
            var c_1 = c0_1 + vAlpha * (Vector256.LoadUnsafe(ref c1Ref, 8) - c0_1);
            acc += Vector256.LoadUnsafe(ref xRef, 8) * c_1;

            var c0_2 = Vector256.LoadUnsafe(ref c0Ref, 16);
            var c_2 = c0_2 + vAlpha * (Vector256.LoadUnsafe(ref c1Ref, 16) - c0_2);
            acc += Vector256.LoadUnsafe(ref xRef, 16) * c_2;

            return Vector256.Sum(acc);
        }

        if (Vector128.IsHardwareAccelerated &&
            samples.Length >= 24 &&
            coeffsRow0.Length >= 24 &&
            coeffsRow1.Length >= 24)
        {
            ref float xRef = ref MemoryMarshal.GetReference(samples);
            ref float c0Ref = ref MemoryMarshal.GetReference(coeffsRow0);
            ref float c1Ref = ref MemoryMarshal.GetReference(coeffsRow1);

            var vAlpha = Vector128.Create(alpha);
            var acc = Vector128<float>.Zero;

            for (nuint offset = 0; offset < 24; offset += 4)
            {
                var c0 = Vector128.LoadUnsafe(ref c0Ref, offset);
                var c = c0 + vAlpha * (Vector128.LoadUnsafe(ref c1Ref, offset) - c0);
                acc += Vector128.LoadUnsafe(ref xRef, offset) * c;
            }

            return Vector128.Sum(acc);
        }

        float sum = 0f;
        int len = Math.Min(24, Math.Min(samples.Length, Math.Min(coeffsRow0.Length, coeffsRow1.Length)));
        for (int k = 0; k < len; k++)
        {
            float c0 = coeffsRow0[k];
            float c = c0 + alpha * (coeffsRow1[k] - c0);
            sum += samples[k] * c;
        }
        return sum;
    }

    /// <summary>
    /// Element-wise multiplication: dst[i] = a[i] * b[i].
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static void Multiply(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> dst)
    {
        int len = Math.Min(dst.Length, Math.Min(a.Length, b.Length));
        int i = 0;

        if (Vector<float>.IsSupported && len >= Vector<float>.Count)
        {
            ref float aRef = ref MemoryMarshal.GetReference(a);
            ref float bRef = ref MemoryMarshal.GetReference(b);
            ref float dRef = ref MemoryMarshal.GetReference(dst);
            int vecCount = Vector<float>.Count;
            int limit = len - vecCount;

            for (; i <= limit; i += vecCount)
            {
                nuint off = (nuint)i;
                var prod = Vector.LoadUnsafe(ref aRef, off) * Vector.LoadUnsafe(ref bRef, off);
                prod.StoreUnsafe(ref dRef, off);
            }
        }

        for (; i < len; i++)
        {
            dst[i] = a[i] * b[i];
        }
    }

    /// <summary>
    /// Element-wise multiply-add in place: dst[i] += a[i] * b[i].
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static void MultiplyAddInPlace(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> dst)
    {
        int len = Math.Min(dst.Length, Math.Min(a.Length, b.Length));
        int i = 0;

        if (Vector<float>.IsSupported && len >= Vector<float>.Count)
        {
            ref float aRef = ref MemoryMarshal.GetReference(a);
            ref float bRef = ref MemoryMarshal.GetReference(b);
            ref float dRef = ref MemoryMarshal.GetReference(dst);
            int vecCount = Vector<float>.Count;
            int limit = len - vecCount;

            for (; i <= limit; i += vecCount)
            {
                nuint off = (nuint)i;
                var res = Vector.LoadUnsafe(ref dRef, off) + (Vector.LoadUnsafe(ref aRef, off) * Vector.LoadUnsafe(ref bRef, off));
                res.StoreUnsafe(ref dRef, off);
            }
        }

        for (; i < len; i++)
        {
            dst[i] += a[i] * b[i];
        }
    }

    /// <summary>
    /// Element-wise addition in place: dst[i] += src[i].
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static void AddInPlace(ReadOnlySpan<float> src, Span<float> dst)
    {
        int len = Math.Min(src.Length, dst.Length);
        int i = 0;

        if (Vector<float>.IsSupported && len >= Vector<float>.Count)
        {
            ref float sRef = ref MemoryMarshal.GetReference(src);
            ref float dRef = ref MemoryMarshal.GetReference(dst);
            int vecCount = Vector<float>.Count;
            int limit = len - vecCount;

            for (; i <= limit; i += vecCount)
            {
                nuint off = (nuint)i;
                var res = Vector.LoadUnsafe(ref dRef, off) + Vector.LoadUnsafe(ref sRef, off);
                res.StoreUnsafe(ref dRef, off);
            }
        }

        for (; i < len; i++)
        {
            dst[i] += src[i];
        }
    }

    /// <summary>
    /// Element-wise scalar multiplication: dst[i] = src[i] * scalar.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static void MultiplyScalar(ReadOnlySpan<float> src, float scalar, Span<float> dst)
    {
        int len = Math.Min(src.Length, dst.Length);
        int i = 0;

        if (Vector<float>.IsSupported && len >= Vector<float>.Count)
        {
            ref float sRef = ref MemoryMarshal.GetReference(src);
            ref float dRef = ref MemoryMarshal.GetReference(dst);
            int vecCount = Vector<float>.Count;
            int limit = len - vecCount;
            var vScalar = new Vector<float>(scalar);

            for (; i <= limit; i += vecCount)
            {
                nuint off = (nuint)i;
                var res = Vector.LoadUnsafe(ref sRef, off) * vScalar;
                res.StoreUnsafe(ref dRef, off);
            }
        }

        for (; i < len; i++)
        {
            dst[i] = src[i] * scalar;
        }
    }
}
