using System.Numerics;
using System.Runtime.CompilerServices;

namespace VoiceLockStudio.Services.Audio;

/// <summary>
/// Lock-free, single-producer single-consumer (SPSC) ring buffer for 32-bit float PCM samples
/// with zero heap allocations on read, write, and clear paths.
/// </summary>
public sealed class AudioRingBuffer
{
    private readonly float[] _buffer;
    private readonly int _mask;

    private long _writeCount;
    private long _readCount;

    public AudioRingBuffer(int minCapacitySamples)
    {
        if (minCapacitySamples <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(minCapacitySamples), "Capacity must be positive.");
        }

        uint pow2 = BitOperations.RoundUpToPowerOf2((uint)Math.Max(2, minCapacitySamples));
        if (pow2 > int.MaxValue)
        {
            throw new ArgumentOutOfRangeException(nameof(minCapacitySamples), "Capacity exceeds Int32 range.");
        }

        Capacity = (int)pow2;
        _mask = Capacity - 1;
        _buffer = GC.AllocateUninitializedArray<float>(Capacity);
        Array.Clear(_buffer);
    }

    /// <summary>
    /// Total usable sample capacity (rounded up to a power of two).
    /// </summary>
    public int Capacity { get; }

    /// <summary>
    /// Number of samples currently available to read by the consumer.
    /// </summary>
    public int AvailableRead
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get
        {
            long write = Volatile.Read(ref _writeCount);
            long read = Volatile.Read(ref _readCount);
            long diff = write - read;
            if (diff <= 0)
            {
                return 0;
            }

            return diff > Capacity ? Capacity : (int)diff;
        }
    }

    /// <summary>
    /// Number of free sample slots currently available to write by the producer.
    /// </summary>
    public int AvailableWrite
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get
        {
            long write = Volatile.Read(ref _writeCount);
            long read = Volatile.Read(ref _readCount);
            long used = write - read;
            if (used <= 0)
            {
                return Capacity;
            }

            return used >= Capacity ? 0 : Capacity - (int)used;
        }
    }

    /// <summary>
    /// Writes up to <paramref name="source"/>.Length samples from the producer thread.
    /// Returns the actual number of samples written (may be less than requested if full).
    /// </summary>
    public int Write(ReadOnlySpan<float> source)
    {
        if (source.IsEmpty)
        {
            return 0;
        }

        long write = _writeCount;
        long read = Volatile.Read(ref _readCount);
        int free = Capacity - (int)(write - read);
        if (free <= 0)
        {
            return 0;
        }

        int toWrite = Math.Min(source.Length, free);
        int startIndex = (int)(write & _mask);
        int firstPart = Math.Min(toWrite, Capacity - startIndex);

        source.Slice(0, firstPart).CopyTo(_buffer.AsSpan(startIndex, firstPart));

        int secondPart = toWrite - firstPart;
        if (secondPart > 0)
        {
            source.Slice(firstPart, secondPart).CopyTo(_buffer.AsSpan(0, secondPart));
        }

        // Release semantics: ensure sample writes are visible before advancing _writeCount.
        Volatile.Write(ref _writeCount, write + toWrite);
        return toWrite;
    }

    /// <summary>
    /// Reads up to <paramref name="destination"/>.Length samples from the consumer thread.
    /// Returns the actual number of samples read.
    /// </summary>
    public int Read(Span<float> destination)
    {
        if (destination.IsEmpty)
        {
            return 0;
        }

        long read = _readCount;
        long write = Volatile.Read(ref _writeCount);
        int available = (int)(write - read);
        if (available <= 0)
        {
            return 0;
        }

        int toRead = Math.Min(destination.Length, available);
        CopyOut(read, toRead, destination);

        // Release semantics: ensure sample reads complete before freeing slots via _readCount.
        Volatile.Write(ref _readCount, read + toRead);
        return toRead;
    }

    /// <summary>
    /// Copies up to <paramref name="destination"/>.Length available samples without advancing the read cursor.
    /// </summary>
    public int Peek(Span<float> destination)
    {
        if (destination.IsEmpty)
        {
            return 0;
        }

        long read = _readCount;
        long write = Volatile.Read(ref _writeCount);
        int available = (int)(write - read);
        if (available <= 0)
        {
            return 0;
        }

        int toPeek = Math.Min(destination.Length, available);
        CopyOut(read, toPeek, destination);
        return toPeek;
    }

    /// <summary>
    /// Discards up to <paramref name="sampleCount"/> available samples from the consumer side.
    /// </summary>
    public int Skip(int sampleCount)
    {
        if (sampleCount <= 0)
        {
            return 0;
        }

        long read = _readCount;
        long write = Volatile.Read(ref _writeCount);
        int available = (int)(write - read);
        if (available <= 0)
        {
            return 0;
        }

        int toSkip = Math.Min(sampleCount, available);
        Volatile.Write(ref _readCount, read + toSkip);
        return toSkip;
    }

    /// <summary>
    /// Consumer-safe clear: advances the read cursor to match the current write cursor without allocation.
    /// </summary>
    public void Clear()
    {
        long write = Volatile.Read(ref _writeCount);
        Volatile.Write(ref _readCount, write);
    }

    /// <summary>
    /// Resets both cursors and zeroes backing memory. Call only when producer and consumer are idle.
    /// </summary>
    public void Reset()
    {
        Volatile.Write(ref _readCount, 0);
        Volatile.Write(ref _writeCount, 0);
        Array.Clear(_buffer);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void CopyOut(long readCursor, int count, Span<float> destination)
    {
        int startIndex = (int)(readCursor & _mask);
        int firstPart = Math.Min(count, Capacity - startIndex);

        _buffer.AsSpan(startIndex, firstPart).CopyTo(destination.Slice(0, firstPart));

        int secondPart = count - firstPart;
        if (secondPart > 0)
        {
            _buffer.AsSpan(0, secondPart).CopyTo(destination.Slice(firstPart, secondPart));
        }
    }
}
