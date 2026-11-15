using System.Runtime.CompilerServices;

namespace VoiceLockStudio.Services.Audio;

/// <summary>
/// Zero-allocation streaming audio resampler using a Kaiser-windowed sinc polyphase FIR filter
/// with automatic anti-aliasing for arbitrary WASAPI sample rates (e.g. 44.1 kHz / 48 kHz &lt;-&gt; 16 kHz).
/// </summary>
public sealed class StreamingResampler
{
    private const int HalfWidth = 12;
    private const int Taps = HalfWidth * 2; // 24 taps
    private const int HistoryLength = Taps;
    private const int NumPhases = 512;
    private const int MaxWorkChunk = 4096;
    private const double KaiserBeta = 7.0;

    private readonly float[] _workBuffer;
    private readonly float[] _coeffTable;
    private readonly float[] _fastUp3Coeffs;

    private int _inputSampleRate;
    private int _outputSampleRate;
    private double _step;
    private double _phase;
    private int _bufferedSamples;
    private bool _isFastDecimate3;
    private bool _isFastInterpolate3;
    private int _up3SubPhase;

    public StreamingResampler(int inputSampleRate = 48000, int outputSampleRate = 16000)
    {
        _workBuffer = new float[HistoryLength + MaxWorkChunk];
        _coeffTable = new float[(NumPhases + 1) * Taps];
        _fastUp3Coeffs = new float[3 * Taps];
        Configure(inputSampleRate, outputSampleRate);
    }

    public int InputSampleRate => _inputSampleRate;
    public int OutputSampleRate => _outputSampleRate;
    public bool IsIdentity => _inputSampleRate == _outputSampleRate;

    /// <summary>
    /// Algorithmic filter group delay in input samples.
    /// </summary>
    public double AlgorithmicDelaySamples => IsIdentity ? 0.0 : HalfWidth - 1;

    /// <summary>
    /// Algorithmic filter group delay in milliseconds.
    /// </summary>
    public double AlgorithmicDelayMs => IsIdentity ? 0.0 : AlgorithmicDelaySamples * 1000.0 / _inputSampleRate;

    /// <summary>
    /// Reconfigures sample rates and precomputes the anti-aliased windowed-sinc phase table.
    /// </summary>
    public void Configure(int inputSampleRate, int outputSampleRate)
    {
        if (inputSampleRate <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(inputSampleRate), "Sample rate must be positive.");
        }

        if (outputSampleRate <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(outputSampleRate), "Sample rate must be positive.");
        }

        if (_inputSampleRate == inputSampleRate && _outputSampleRate == outputSampleRate)
        {
            Reset();
            return;
        }

        _inputSampleRate = inputSampleRate;
        _outputSampleRate = outputSampleRate;
        _step = (double)inputSampleRate / outputSampleRate;
        _isFastDecimate3 = inputSampleRate == outputSampleRate * 3;
        _isFastInterpolate3 = outputSampleRate == inputSampleRate * 3;

        if (!IsIdentity)
        {
            BuildKaiserSincTable(inputSampleRate, outputSampleRate);
            if (_isFastInterpolate3)
            {
                BuildFastUp3Table(inputSampleRate, outputSampleRate);
            }
        }

        Reset();
    }

    /// <summary>
    /// Resets internal delay-line state and fractional phase accumulator without heap allocations.
    /// </summary>
    public void Reset()
    {
        Array.Clear(_workBuffer);
        _phase = 0.0;
        _bufferedSamples = 0;
        _up3SubPhase = 0;
    }

    /// <summary>
    /// Upper bound on the number of output samples that may be produced for <paramref name="inputSampleCount"/>
    /// new input samples (including any samples currently buffered internally).
    /// </summary>
    public int GetMaxOutputSamples(int inputSampleCount)
    {
        if (inputSampleCount < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(inputSampleCount));
        }

        if (IsIdentity)
        {
            return inputSampleCount;
        }

        return (int)Math.Ceiling((_bufferedSamples + inputSampleCount) / _step) + 2;
    }

    /// <summary>
    /// Resamples all samples from <paramref name="input"/> into <paramref name="output"/> without allocation.
    /// Throws <see cref="ArgumentException"/> if <paramref name="output"/> is smaller than required.
    /// </summary>
    public int Process(ReadOnlySpan<float> input, Span<float> output)
    {
        Process(input, output, out int consumed, out int written);
        if (consumed < input.Length)
        {
            throw new ArgumentException(
                $"Output buffer too small: consumed {consumed}/{input.Length} input samples.",
                nameof(output));
        }

        return written;
    }

    /// <summary>
    /// Streaming zero-allocation resample step. Consumes as much of <paramref name="input"/> as can be
    /// processed into <paramref name="output"/> without dropping samples.
    /// </summary>
    public void Process(
        ReadOnlySpan<float> input,
        Span<float> output,
        out int inputConsumed,
        out int outputWritten)
    {
        inputConsumed = 0;
        outputWritten = 0;

        if (IsIdentity)
        {
            int copyLen = Math.Min(input.Length, output.Length);
            if (copyLen > 0)
            {
                input.Slice(0, copyLen).CopyTo(output);
                inputConsumed = copyLen;
                outputWritten = copyLen;
            }
            return;
        }

        while (true)
        {
            int inputRemaining = input.Length - inputConsumed;
            int freeInWork = MaxWorkChunk - _bufferedSamples;
            int take = Math.Min(inputRemaining, freeInWork);

            if (take > 0)
            {
                input.Slice(inputConsumed, take)
                    .CopyTo(_workBuffer.AsSpan(HistoryLength + _bufferedSamples, take));
                _bufferedSamples += take;
                inputConsumed += take;
            }

            if (_bufferedSamples == 0 || outputWritten >= output.Length)
            {
                break;
            }

            int emittedBefore = outputWritten;
            ProcessWorkBuffer(output, ref outputWritten);

            if (outputWritten == emittedBefore && take == 0)
            {
                break;
            }

            if (inputConsumed >= input.Length && _phase >= _bufferedSamples)
            {
                break;
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void ProcessWorkBuffer(Span<float> output, ref int outputWritten)
    {
        if (_isFastDecimate3)
        {
            ProcessDecimate3(output, ref outputWritten);
            return;
        }

        if (_isFastInterpolate3)
        {
            ProcessInterpolate3(output, ref outputWritten);
            return;
        }

        double phase = _phase;
        double step = _step;
        int buffered = _bufferedSamples;
        int outLen = output.Length;

        float[] work = _workBuffer;
        float[] coeffs = _coeffTable;

        while (phase < buffered && outputWritten < outLen)
        {
            int intPos = (int)phase;
            double frac = phase - intPos;

            double phasePos = frac * NumPhases;
            int row0 = (int)phasePos;
            float alpha = (float)(phasePos - row0);
            int row0Offset = row0 * Taps;
            int row1Offset = row0Offset + Taps;

            int windowStart = HistoryLength - HalfWidth + intPos - (HalfWidth - 1);

            output[outputWritten++] = WasapiPcmConverter.Dot24Interpolated(
                work.AsSpan(windowStart, Taps),
                coeffs.AsSpan(row0Offset, Taps),
                coeffs.AsSpan(row1Offset, Taps),
                alpha);

            phase += step;
        }

        int consumed = Math.Min(buffered, (int)Math.Floor(phase));
        if (consumed > 0)
        {
            int remainingTotal = HistoryLength + buffered - consumed;
            work.AsSpan(consumed, remainingTotal).CopyTo(work.AsSpan(0, remainingTotal));
            buffered -= consumed;
            phase -= consumed;
        }

        _bufferedSamples = buffered;
        _phase = phase;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void ProcessDecimate3(Span<float> output, ref int outputWritten)
    {
        int intPos = (int)_phase;
        int buffered = _bufferedSamples;
        int outLen = output.Length;

        float[] work = _workBuffer;
        ReadOnlySpan<float> coeffsRow0 = _coeffTable.AsSpan(0, Taps);
        const int baseOffset = HistoryLength - HalfWidth - (HalfWidth - 1);

        while (intPos < buffered && outputWritten < outLen)
        {
            output[outputWritten++] = WasapiPcmConverter.Dot24(
                work.AsSpan(baseOffset + intPos, Taps),
                coeffsRow0);
            intPos += 3;
        }

        int consumed = Math.Min(buffered, intPos);
        if (consumed > 0)
        {
            int remainingTotal = HistoryLength + buffered - consumed;
            work.AsSpan(consumed, remainingTotal).CopyTo(work.AsSpan(0, remainingTotal));
            buffered -= consumed;
            intPos -= consumed;
        }

        _bufferedSamples = buffered;
        _phase = intPos;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void ProcessInterpolate3(Span<float> output, ref int outputWritten)
    {
        int buffered = _bufferedSamples;
        int outLen = output.Length;
        int intPos = 0;
        int subPhase = _up3SubPhase;

        float[] work = _workBuffer;
        float[] up3 = _fastUp3Coeffs;
        ReadOnlySpan<float> c0 = up3.AsSpan(0, Taps);
        ReadOnlySpan<float> c1 = up3.AsSpan(Taps, Taps);
        ReadOnlySpan<float> c2 = up3.AsSpan(2 * Taps, Taps);
        const int baseOffset = HistoryLength - HalfWidth - (HalfWidth - 1);

        // Align subPhase to 0 if resuming partway through a 1:3 triplet.
        while (subPhase != 0 && intPos < buffered && outputWritten < outLen)
        {
            ReadOnlySpan<float> win = work.AsSpan(baseOffset + intPos, Taps);
            ReadOnlySpan<float> coeff = subPhase == 1 ? c1 : c2;
            output[outputWritten++] = WasapiPcmConverter.Dot24(win, coeff);
            if (++subPhase == 3)
            {
                subPhase = 0;
                intPos++;
            }
        }

        // Fast triplet emission: one 24-sample window reused across all 3 polyphase filters.
        while (intPos < buffered && outputWritten + 3 <= outLen)
        {
            ReadOnlySpan<float> win = work.AsSpan(baseOffset + intPos, Taps);
            output[outputWritten] = WasapiPcmConverter.Dot24(win, c0);
            output[outputWritten + 1] = WasapiPcmConverter.Dot24(win, c1);
            output[outputWritten + 2] = WasapiPcmConverter.Dot24(win, c2);
            outputWritten += 3;
            intPos++;
        }

        // Remaining tail (1-2 output samples when output buffer fills mid-triplet).
        while (intPos < buffered && outputWritten < outLen)
        {
            ReadOnlySpan<float> win = work.AsSpan(baseOffset + intPos, Taps);
            ReadOnlySpan<float> coeff = subPhase switch
            {
                0 => c0,
                1 => c1,
                _ => c2
            };
            output[outputWritten++] = WasapiPcmConverter.Dot24(win, coeff);
            if (++subPhase == 3)
            {
                subPhase = 0;
                intPos++;
            }
        }

        int consumed = Math.Min(buffered, intPos);
        if (consumed > 0)
        {
            int remainingTotal = HistoryLength + buffered - consumed;
            work.AsSpan(consumed, remainingTotal).CopyTo(work.AsSpan(0, remainingTotal));
            buffered -= consumed;
            intPos -= consumed;
        }

        _bufferedSamples = buffered;
        _up3SubPhase = subPhase;
        _phase = intPos + (subPhase / 3.0);
    }

    private void BuildKaiserSincTable(int inputSampleRate, int outputSampleRate)
    {
        double fc = ResolveNormalizedCutoff(inputSampleRate, outputSampleRate);
        double invI0Beta = 1.0 / BesselI0(KaiserBeta);

        for (int p = 0; p <= NumPhases; p++)
        {
            double frac = (double)p / NumPhases;
            BuildKaiserSincRow(frac, fc, invI0Beta, _coeffTable.AsSpan(p * Taps, Taps));
        }
    }

    private void BuildFastUp3Table(int inputSampleRate, int outputSampleRate)
    {
        double fc = ResolveNormalizedCutoff(inputSampleRate, outputSampleRate);
        double invI0Beta = 1.0 / BesselI0(KaiserBeta);

        for (int p = 0; p < 3; p++)
        {
            double frac = p / 3.0;
            BuildKaiserSincRow(frac, fc, invI0Beta, _fastUp3Coeffs.AsSpan(p * Taps, Taps));
        }
    }

    private static double ResolveNormalizedCutoff(int inputSampleRate, int outputSampleRate)
    {
        // Normalized cutoff relative to input sample rate (0.5 = Nyquist).
        return inputSampleRate > outputSampleRate
            ? 0.45 * outputSampleRate / inputSampleRate
            : 0.45;
    }

    private static void BuildKaiserSincRow(double frac, double fc, double invI0Beta, Span<float> dst)
    {
        double sum = 0.0;

        for (int k = 0; k < Taps; k++)
        {
            double t = (k - (HalfWidth - 1)) - frac;
            double norm = t / HalfWidth;

            double window;
            if (Math.Abs(norm) >= 1.0)
            {
                window = 0.0;
            }
            else
            {
                double arg = KaiserBeta * Math.Sqrt(1.0 - norm * norm);
                window = BesselI0(arg) * invI0Beta;
            }

            double x = 2.0 * fc * t;
            double sinc = Math.Abs(x) < 1e-9
                ? 1.0
                : Math.Sin(Math.PI * x) / (Math.PI * x);

            double h = 2.0 * fc * sinc * window;
            dst[k] = (float)h;
            sum += h;
        }

        if (sum > 0.0)
        {
            float invSum = (float)(1.0 / sum);
            for (int k = 0; k < Taps; k++)
            {
                dst[k] *= invSum;
            }
        }
    }

    private static double BesselI0(double x)
    {
        double sum = 1.0;
        double term = 1.0;
        double halfX = 0.5 * x;
        double halfXSq = halfX * halfX;

        for (int k = 1; k <= 20; k++)
        {
            term *= halfXSq / (k * k);
            sum += term;
            if (term < 1e-12 * sum)
            {
                break;
            }
        }

        return sum;
    }
}
