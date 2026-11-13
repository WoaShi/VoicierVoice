using System.ComponentModel;
using System.Diagnostics;
using System.Numerics;
using System.Runtime.InteropServices;

namespace VoiceLockStudio.Services.System;

/// <summary>
/// Describes a physical CPU core returned by <c>GetLogicalProcessorInformationEx(RelationProcessorCore)</c>.
/// </summary>
public sealed record PhysicalCoreInfo(
    int CoreIndex,
    byte EfficiencyClass,
    bool IsEfficiencyCore,
    bool IsRecommended,
    bool IsHyperThreaded,
    ushort ProcessorGroup,
    ulong LogicalProcessorMask,
    IReadOnlyList<int> LogicalProcessorIndices,
    string DisplayName);

/// <summary>
/// Full physical core topology and recommended audio worker affinity mask.
/// </summary>
public sealed record CpuTopologyInfo(
    int TotalLogicalProcessors,
    int ProcessorGroupCount,
    bool IsHeterogeneous,
    IReadOnlyList<PhysicalCoreInfo> PhysicalCores,
    ushort RecommendedProcessorGroup,
    ulong RecommendedCoreMask,
    string RecommendedDescription);

/// <summary>
/// Result of binding the current thread's CPU affinity and priority.
/// </summary>
public sealed record AffinityBindResult(
    bool Success,
    ulong AppliedMask,
    ushort ProcessorGroup,
    ThreadPriority AppliedPriority,
    int Win32Error,
    string Description);

/// <summary>
/// Queries Windows physical core topology (<c>EfficiencyClass</c>, processor groups, logical masks)
/// and pins audio worker threads to E-Cores (heterogeneous) or tail logical cores (homogeneous/AMD).
/// </summary>
public sealed class CpuAffinityManager
{
    private const int ErrorInsufficientBuffer = 122;
    private const byte LtpPcSmt = 0x01;

    private CpuTopologyInfo? _cachedTopology;

    public CpuTopologyInfo GetTopology(bool refresh = false)
    {
        if (!refresh && _cachedTopology is not null)
        {
            return _cachedTopology;
        }

        _cachedTopology = QueryTopologyCore();
        return _cachedTopology;
    }

    /// <summary>
    /// Computes the recommended affinity mask: all E-Cores when heterogeneous, or the last 2 logical cores
    /// when all cores share the same <c>EfficiencyClass</c> (e.g. AMD Ryzen).
    /// </summary>
    public ulong GetRecommendedAudioWorkerMask(out ushort processorGroup, out string description)
    {
        CpuTopologyInfo topology = GetTopology();
        processorGroup = topology.RecommendedProcessorGroup;
        description = topology.RecommendedDescription;
        return topology.RecommendedCoreMask;
    }

    /// <summary>
    /// Binds the calling managed thread to <paramref name="mask"/> (within <paramref name="processorGroup"/>)
    /// and sets <paramref name="priority"/>. Uses <c>SetThreadAffinityMask</c> for single-group group-0
    /// masks and <c>SetThreadGroupAffinity</c> when multiple processor groups (&gt;64 CPUs) are present.
    /// </summary>
    public AffinityBindResult BindCurrentThread(
        ulong mask,
        ThreadPriority priority = ThreadPriority.AboveNormal,
        ushort processorGroup = 0)
    {
        if (mask == 0)
        {
            return new AffinityBindResult(
                Success: false,
                AppliedMask: 0UL,
                ProcessorGroup: processorGroup,
                AppliedPriority: Thread.CurrentThread.Priority,
                Win32Error: 0,
                Description: "Unbound (empty mask)");
        }

        Thread.CurrentThread.Priority = priority;
        IntPtr currentThread = NativeMethods.GetCurrentThread();

        CpuTopologyInfo topology = GetTopology();
        bool useGroupAffinity = processorGroup > 0 || topology.ProcessorGroupCount > 1;

        if (!useGroupAffinity)
        {
            UIntPtr prev = NativeMethods.SetThreadAffinityMask(currentThread, ToNativeAffinityMask(mask));
            if (prev != UIntPtr.Zero)
            {
                return new AffinityBindResult(
                    Success: true,
                    AppliedMask: mask,
                    ProcessorGroup: 0,
                    AppliedPriority: Thread.CurrentThread.Priority,
                    Win32Error: 0,
                    Description: FormatMaskDescription(topology, 0, mask));
            }

            int err = Marshal.GetLastWin32Error();
            return new AffinityBindResult(
                Success: false,
                AppliedMask: 0UL,
                ProcessorGroup: 0,
                AppliedPriority: Thread.CurrentThread.Priority,
                Win32Error: err,
                Description: $"SetThreadAffinityMask failed ({new Win32Exception(err).Message})");
        }

        var groupAffinity = new NativeMethods.GroupAffinity
        {
            Mask = ToNativeAffinityMask(mask),
            Group = processorGroup
        };

        if (NativeMethods.SetThreadGroupAffinity(currentThread, ref groupAffinity, out _))
        {
            return new AffinityBindResult(
                Success: true,
                AppliedMask: mask,
                ProcessorGroup: processorGroup,
                AppliedPriority: Thread.CurrentThread.Priority,
                Win32Error: 0,
                Description: FormatMaskDescription(topology, processorGroup, mask));
        }

        int groupErr = Marshal.GetLastWin32Error();
        return new AffinityBindResult(
            Success: false,
            AppliedMask: 0UL,
            ProcessorGroup: processorGroup,
            AppliedPriority: Thread.CurrentThread.Priority,
            Win32Error: groupErr,
            Description: $"SetThreadGroupAffinity failed ({new Win32Exception(groupErr).Message})");
    }

    public static HashSet<int> CaptureCurrentOsThreadIds()
    {
        var ids = new HashSet<int>();
        using Process proc = Process.GetCurrentProcess();
        foreach (ProcessThread t in proc.Threads)
        {
            ids.Add(t.Id);
        }
        return ids;
    }

    public int BindNewProcessThreads(HashSet<int> baselineThreadIds, ulong mask, ushort processorGroup = 0)
    {
        if (mask == 0)
        {
            return 0;
        }

        int boundCount = 0;
        CpuTopologyInfo topology = GetTopology();
        bool useGroupAffinity = processorGroup > 0 || topology.ProcessorGroupCount > 1;

        using Process proc = Process.GetCurrentProcess();
        foreach (ProcessThread t in proc.Threads)
        {
            if (baselineThreadIds.Contains(t.Id))
            {
                continue;
            }

            IntPtr hThread = NativeMethods.OpenThread(
                NativeMethods.ThreadSetInformation | NativeMethods.ThreadQueryInformation,
                false,
                (uint)t.Id);
            if (hThread == IntPtr.Zero)
            {
                continue;
            }

            try
            {
                if (!useGroupAffinity)
                {
                    if (NativeMethods.SetThreadAffinityMask(hThread, ToNativeAffinityMask(mask)) != UIntPtr.Zero)
                    {
                        boundCount++;
                    }
                }
                else
                {
                    var ga = new NativeMethods.GroupAffinity
                    {
                        Mask = ToNativeAffinityMask(mask),
                        Group = processorGroup
                    };
                    if (NativeMethods.SetThreadGroupAffinity(hThread, ref ga, out _))
                    {
                        boundCount++;
                    }
                }
            }
            finally
            {
                NativeMethods.CloseHandle(hThread);
            }
        }

        return boundCount;
    }

    private static UIntPtr ToNativeAffinityMask(ulong mask)
    {
        return Environment.Is64BitProcess
            ? new UIntPtr(mask)
            : new UIntPtr((uint)(mask & uint.MaxValue));
    }

    private static CpuTopologyInfo QueryTopologyCore()
    {
        uint requiredBytes = 0;
        NativeMethods.GetLogicalProcessorInformationEx(
            NativeMethods.LogicalProcessorRelationship.RelationProcessorCore,
            IntPtr.Zero,
            ref requiredBytes);

        int firstError = Marshal.GetLastWin32Error();
        if (requiredBytes == 0 || firstError != ErrorInsufficientBuffer)
        {
            return BuildFallbackTopology();
        }

        IntPtr buffer = Marshal.AllocHGlobal((int)requiredBytes);
        try
        {
            if (!NativeMethods.GetLogicalProcessorInformationEx(
                NativeMethods.LogicalProcessorRelationship.RelationProcessorCore,
                buffer,
                ref requiredBytes))
            {
                return BuildFallbackTopology();
            }

            return ParseTopologyBuffer(buffer, (int)requiredBytes);
        }
        finally
        {
            Marshal.FreeHGlobal(buffer);
        }
    }

    private static unsafe CpuTopologyInfo ParseTopologyBuffer(IntPtr buffer, int byteLength)
    {
        var rawCores = new List<(byte Flags, byte EfficiencyClass, ushort Group, ulong Mask)>();
        var distinctGroups = new HashSet<ushort>();
        int totalLogical = 0;

        byte* cursor = (byte*)buffer.ToPointer();
        byte* end = cursor + byteLength;
        int groupAffinitySize = Marshal.SizeOf<NativeMethods.GroupAffinity>();

        while (cursor + 32 <= end)
        {
            uint relationship = *(uint*)cursor;
            uint entrySize = *(uint*)(cursor + 4);
            if (entrySize == 0 || cursor + entrySize > end)
            {
                break;
            }

            if (relationship == (uint)NativeMethods.LogicalProcessorRelationship.RelationProcessorCore)
            {
                byte flags = *(cursor + 8);
                byte efficiencyClass = *(cursor + 9);
                ushort groupCount = *(ushort*)(cursor + 30);

                byte* groupPtr = cursor + 32;
                for (int g = 0; g < groupCount; g++)
                {
                    if (groupPtr + groupAffinitySize > cursor + entrySize)
                    {
                        break;
                    }

                    var ga = Marshal.PtrToStructure<NativeMethods.GroupAffinity>((IntPtr)groupPtr);
                    ulong mask = ga.Mask.ToUInt64();
                    if (mask != 0)
                    {
                        rawCores.Add((flags, efficiencyClass, ga.Group, mask));
                        distinctGroups.Add(ga.Group);
                        totalLogical += BitOperations.PopCount(mask);
                    }

                    groupPtr += groupAffinitySize;
                }
            }

            cursor += entrySize;
        }

        if (rawCores.Count == 0)
        {
            return BuildFallbackTopology();
        }

        byte minEff = rawCores.Min(c => c.EfficiencyClass);
        byte maxEff = rawCores.Max(c => c.EfficiencyClass);
        bool isHeterogeneous = maxEff > minEff;

        ushort recommendedGroup;
        ulong recommendedMask = 0;
        var tailLogicalSet = new HashSet<(ushort Group, int LogicalIndex)>();

        if (isHeterogeneous)
        {
            // Prefer the group containing E-Cores (usually Group 0).
            recommendedGroup = rawCores.First(c => c.EfficiencyClass == minEff).Group;
            foreach (var core in rawCores)
            {
                if (core.EfficiencyClass == minEff && core.Group == recommendedGroup)
                {
                    recommendedMask |= core.Mask;
                }
            }
        }
        else
        {
            // Homogeneous CPU (e.g. AMD): pick the last 2 logical processors in the highest group.
            recommendedGroup = rawCores.Max(c => c.Group);
            var logicalsInGroup = new List<int>();
            foreach (var core in rawCores.Where(c => c.Group == recommendedGroup))
            {
                ExtractBitIndices(core.Mask, logicalsInGroup);
            }

            logicalsInGroup.Sort();
            int takeCount = Math.Min(2, logicalsInGroup.Count);
            for (int i = logicalsInGroup.Count - takeCount; i < logicalsInGroup.Count; i++)
            {
                int lp = logicalsInGroup[i];
                tailLogicalSet.Add((recommendedGroup, lp));
                recommendedMask |= 1UL << lp;
            }
        }

        var physicalCores = new List<PhysicalCoreInfo>(rawCores.Count);
        for (int i = 0; i < rawCores.Count; i++)
        {
            var (flags, eff, group, mask) = rawCores[i];
            var indices = new List<int>(4);
            ExtractBitIndices(mask, indices);

            bool isECore = isHeterogeneous && eff == minEff;
            bool isRecommended = isHeterogeneous
                ? (isECore && group == recommendedGroup)
                : indices.Any(lp => tailLogicalSet.Contains((group, lp)));
            bool isSmt = (flags & LtpPcSmt) != 0 || indices.Count > 1;

            string lpLabel = indices.Count == 1
                ? $"LP {indices[0]}"
                : $"LP {string.Join(',', indices)}";

            string roleTag = isHeterogeneous
                ? (isECore ? "E-Core" : "P-Core")
                : (isRecommended ? "Tail" : "Core");

            string groupPrefix = distinctGroups.Count > 1 ? $"G{group} " : string.Empty;
            string displayName = $"{groupPrefix}Core {i} ({roleTag}, {lpLabel})";

            physicalCores.Add(new PhysicalCoreInfo(
                CoreIndex: i,
                EfficiencyClass: eff,
                IsEfficiencyCore: isECore,
                IsRecommended: isRecommended,
                IsHyperThreaded: isSmt,
                ProcessorGroup: group,
                LogicalProcessorMask: mask,
                LogicalProcessorIndices: indices,
                DisplayName: displayName));
        }

        string recDesc = BuildRecommendedDescription(isHeterogeneous, physicalCores, recommendedGroup, recommendedMask);

        return new CpuTopologyInfo(
            TotalLogicalProcessors: totalLogical,
            ProcessorGroupCount: distinctGroups.Count,
            IsHeterogeneous: isHeterogeneous,
            PhysicalCores: physicalCores,
            RecommendedProcessorGroup: recommendedGroup,
            RecommendedCoreMask: recommendedMask,
            RecommendedDescription: recDesc);
    }

    private static CpuTopologyInfo BuildFallbackTopology()
    {
        int count = Math.Clamp(Environment.ProcessorCount, 1, 64);
        var cores = new List<PhysicalCoreInfo>(count);
        ulong recMask = 0;

        int tailStart = Math.Max(0, count - 2);
        for (int i = 0; i < count; i++)
        {
            ulong bit = 1UL << i;
            bool isRec = i >= tailStart;
            if (isRec)
            {
                recMask |= bit;
            }

            cores.Add(new PhysicalCoreInfo(
                CoreIndex: i,
                EfficiencyClass: 0,
                IsEfficiencyCore: false,
                IsRecommended: isRec,
                IsHyperThreaded: false,
                ProcessorGroup: 0,
                LogicalProcessorMask: bit,
                LogicalProcessorIndices: [i],
                DisplayName: $"Core {i} ({(isRec ? "Tail" : "Core")}, LP {i})"));
        }

        return new CpuTopologyInfo(
            TotalLogicalProcessors: count,
            ProcessorGroupCount: 1,
            IsHeterogeneous: false,
            PhysicalCores: cores,
            RecommendedProcessorGroup: 0,
            RecommendedCoreMask: recMask,
            RecommendedDescription: BuildRecommendedDescription(false, cores, 0, recMask));
    }

    private static void ExtractBitIndices(ulong mask, List<int> destination)
    {
        while (mask != 0)
        {
            int idx = BitOperations.TrailingZeroCount(mask);
            destination.Add(idx);
            mask &= mask - 1;
        }
    }

    private static string BuildRecommendedDescription(
        bool isHeterogeneous,
        IReadOnlyList<PhysicalCoreInfo> cores,
        ushort group,
        ulong mask)
    {
        var lps = new List<int>();
        ExtractBitIndices(mask, lps);
        string lpList = string.Join(", ", lps);

        if (isHeterogeneous)
        {
            int eCoreCount = cores.Count(c => c.IsEfficiencyCore && c.ProcessorGroup == group);
            return $"Auto E-Cores ({eCoreCount} cores, LP {lpList})";
        }

        return $"Auto Tail Cores (LP {lpList})";
    }

    private static string FormatMaskDescription(CpuTopologyInfo topology, ushort group, ulong mask)
    {
        var lps = new List<int>();
        ExtractBitIndices(mask, lps);
        string lpText = string.Join(", ", lps);

        if (mask == topology.RecommendedCoreMask && group == topology.RecommendedProcessorGroup)
        {
            return topology.RecommendedDescription;
        }

        string prefix = topology.ProcessorGroupCount > 1 ? $"G{group} " : string.Empty;
        return $"{prefix}LP {lpText} (0x{mask:X})";
    }

    private static class NativeMethods
    {
        public enum LogicalProcessorRelationship : uint
        {
            RelationProcessorCore = 0
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct GroupAffinity
        {
            public UIntPtr Mask;
            public ushort Group;
            public ushort Reserved0;
            public ushort Reserved1;
            public ushort Reserved2;
        }

        [DllImport("kernel32.dll", SetLastError = true)]
        public static extern bool GetLogicalProcessorInformationEx(
            LogicalProcessorRelationship relationshipType,
            IntPtr buffer,
            ref uint returnedLength);

        [DllImport("kernel32.dll")]
        public static extern IntPtr GetCurrentThread();

        [DllImport("kernel32.dll", SetLastError = true)]
        public static extern UIntPtr SetThreadAffinityMask(IntPtr hThread, UIntPtr dwThreadAffinityMask);

        [DllImport("kernel32.dll", SetLastError = true)]
        public static extern bool SetThreadGroupAffinity(
            IntPtr hThread,
            ref GroupAffinity groupAffinity,
            out GroupAffinity previousGroupAffinity);

        public const uint ThreadSetInformation = 0x0020;
        public const uint ThreadQueryInformation = 0x0040;

        [DllImport("kernel32.dll", SetLastError = true)]
        public static extern IntPtr OpenThread(uint dwDesiredAccess, bool bInheritHandle, uint dwThreadId);

        [DllImport("kernel32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        public static extern bool CloseHandle(IntPtr hObject);
    }
}
