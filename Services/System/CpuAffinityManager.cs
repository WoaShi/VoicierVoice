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
    private static readonly object ProcessBoostSync = new();
    private static int _processBoostRefCount;
    private static ProcessPriorityClass? _savedProcessPriorityClass;
    private static bool _powerThrottlingDisabled;
    private static bool _timerResolutionActive;

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
    public ulong GetRecommendedAudioWorkerMask(
        out ushort processorGroup,
        out string description,
        bool preferPerformanceCores = false)
    {
        CpuTopologyInfo topology = GetTopology();
        if (preferPerformanceCores &&
            TryGetPerformanceCoresMask(topology, out ulong pCoreMask, out ushort pCoreGroup, out int pCoreCount))
        {
            processorGroup = pCoreGroup;
            description = $"Auto P-Cores ({pCoreCount} cores)";
            return pCoreMask;
        }

        processorGroup = topology.RecommendedProcessorGroup;
        description = topology.RecommendedDescription;
        return topology.RecommendedCoreMask;
    }

    /// <summary>
    /// Returns an expanded CPU affinity mask used when the default E-Core or tail-core subset
    /// experiences thermal/scheduling saturation under extreme system load.
    /// </summary>
    public ulong GetOverloadEscapeMask(out ushort processorGroup, out string description)
    {
        CpuTopologyInfo topology = GetTopology();
        if (topology.IsHeterogeneous &&
            TryGetPerformanceCoresMask(topology, out ulong pCoreMask, out ushort pCoreGroup, out int pCoreCount))
        {
            processorGroup = pCoreGroup;
            description = $"Auto P-Cores ({pCoreCount} cores)";
            return pCoreMask;
        }

        ushort targetGroup = topology.RecommendedProcessorGroup;
        ulong allCoresMask = 0;
        int coreCount = 0;
        foreach (PhysicalCoreInfo core in topology.PhysicalCores)
        {
            if (core.ProcessorGroup == targetGroup)
            {
                allCoresMask |= core.LogicalProcessorMask;
                coreCount++;
            }
        }

        if (allCoresMask != 0)
        {
            processorGroup = targetGroup;
            description = $"Auto All Cores ({coreCount} cores)";
            return allCoresMask;
        }

        processorGroup = topology.RecommendedProcessorGroup;
        description = topology.RecommendedDescription;
        return topology.RecommendedCoreMask;
    }

    private static bool TryGetPerformanceCoresMask(
        CpuTopologyInfo topology,
        out ulong pCoreMask,
        out ushort pCoreGroup,
        out int pCoreCount)
    {
        pCoreMask = 0;
        pCoreGroup = topology.RecommendedProcessorGroup;
        pCoreCount = 0;

        if (!topology.IsHeterogeneous)
        {
            return false;
        }

        PhysicalCoreInfo? firstPCore = topology.PhysicalCores.FirstOrDefault(c => !c.IsEfficiencyCore);
        if (firstPCore is null)
        {
            return false;
        }

        pCoreGroup = firstPCore.ProcessorGroup;
        foreach (PhysicalCoreInfo core in topology.PhysicalCores)
        {
            if (!core.IsEfficiencyCore && core.ProcessorGroup == pCoreGroup)
            {
                pCoreMask |= core.LogicalProcessorMask;
                pCoreCount++;
            }
        }

        return pCoreMask != 0;
    }

    /// <summary>
    /// Binds the calling managed thread to <paramref name="mask"/> (within <paramref name="processorGroup"/>)
    /// and sets <paramref name="priority"/>. Uses <c>SetThreadAffinityMask</c> for single-group group-0
    /// masks and <c>SetThreadGroupAffinity</c> when multiple processor groups (&gt;64 CPUs) are present.
    /// </summary>
    public AffinityBindResult BindCurrentThread(
        ulong mask,
        ThreadPriority priority = ThreadPriority.Highest,
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
        TryDisableThreadPowerThrottling(currentThread);

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

    public static List<int> GetNewProcessThreadIds(HashSet<int> baselineThreadIds, int maxExpectedThreads = 0)
    {
        using Process proc = Process.GetCurrentProcess();
        var modules = SnapshotProcessModules(proc);
        var candidates = new List<(int ThreadId, int Tier)>();

        foreach (ProcessThread t in proc.Threads)
        {
            if (baselineThreadIds.Contains(t.Id))
            {
                continue;
            }

            string? moduleName = TryResolveThreadStartModule(t, modules);
            if (IsExcludedRuntimeOrAudioModule(moduleName))
            {
                continue;
            }

            int tier = ClassifyThreadModuleTier(moduleName);
            candidates.Add((t.Id, tier));
        }

        if (candidates.Count <= 1)
        {
            return candidates.ConvertAll(c => c.ThreadId);
        }

        candidates.Sort(static (a, b) =>
        {
            int tierCmp = a.Tier.CompareTo(b.Tier);
            return tierCmp != 0 ? tierCmp : a.ThreadId.CompareTo(b.ThreadId);
        });

        int limit = maxExpectedThreads > 0 ? Math.Min(candidates.Count, maxExpectedThreads) : candidates.Count;
        var result = new List<int>(limit);
        for (int i = 0; i < limit; i++)
        {
            result.Add(candidates[i].ThreadId);
        }
        return result;
    }

    private static List<(ulong Start, ulong End, string Name)> SnapshotProcessModules(Process proc)
    {
        var list = new List<(ulong, ulong, string)>();
        try
        {
            foreach (ProcessModule module in proc.Modules)
            {
                ulong start = unchecked((ulong)module.BaseAddress.ToInt64());
                ulong end = start + (uint)module.ModuleMemorySize;
                if (start != 0 && end > start && !string.IsNullOrEmpty(module.ModuleName))
                {
                    list.Add((start, end, module.ModuleName));
                }
            }
        }
        catch
        {
            // Ignore module enumeration races; thread filtering will fall back safely.
        }
        return list;
    }

    private static string? TryResolveThreadStartModule(
        ProcessThread thread,
        List<(ulong Start, ulong End, string Name)> modules)
    {
        if (modules.Count == 0)
        {
            return null;
        }

        try
        {
            ulong addr = unchecked((ulong)thread.StartAddress.ToInt64());
            if (addr == 0)
            {
                return null;
            }

            for (int i = 0; i < modules.Count; i++)
            {
                if (addr >= modules[i].Start && addr < modules[i].End)
                {
                    return modules[i].Name;
                }
            }
        }
        catch
        {
            // Thread may exit or deny StartAddress query during enumeration.
        }

        return null;
    }

    private static bool IsExcludedRuntimeOrAudioModule(string? moduleName)
    {
        if (string.IsNullOrEmpty(moduleName))
        {
            return false;
        }

        return moduleName.Contains("coreclr", StringComparison.OrdinalIgnoreCase)
            || moduleName.Equals("clr.dll", StringComparison.OrdinalIgnoreCase)
            || moduleName.Contains("clrjit", StringComparison.OrdinalIgnoreCase)
            || moduleName.Contains("hostpolicy", StringComparison.OrdinalIgnoreCase)
            || moduleName.Contains("hostfxr", StringComparison.OrdinalIgnoreCase)
            || moduleName.Equals("mmdevapi.dll", StringComparison.OrdinalIgnoreCase)
            || moduleName.Equals("audioses.dll", StringComparison.OrdinalIgnoreCase)
            || moduleName.Equals("wdmaud.drv", StringComparison.OrdinalIgnoreCase)
            || moduleName.Equals("avrt.dll", StringComparison.OrdinalIgnoreCase)
            || moduleName.StartsWith("wpfgfx", StringComparison.OrdinalIgnoreCase)
            || moduleName.StartsWith("presentationnative", StringComparison.OrdinalIgnoreCase)
            || moduleName.Equals("dwmapi.dll", StringComparison.OrdinalIgnoreCase);
    }

    private static int ClassifyThreadModuleTier(string? moduleName)
    {
        if (string.IsNullOrEmpty(moduleName))
        {
            return 2;
        }

        if (moduleName.Contains("onnxruntime", StringComparison.OrdinalIgnoreCase))
        {
            return 0;
        }

        if (moduleName.StartsWith("ucrtbase", StringComparison.OrdinalIgnoreCase)
            || moduleName.StartsWith("msvcrt", StringComparison.OrdinalIgnoreCase)
            || moduleName.StartsWith("msvcp", StringComparison.OrdinalIgnoreCase)
            || moduleName.StartsWith("vcruntime", StringComparison.OrdinalIgnoreCase))
        {
            return 1;
        }

        return 2;
    }

    public int BindSpecificThreads(
        IReadOnlyList<int> threadIds,
        ulong mask,
        ushort processorGroup = 0,
        ThreadPriority priority = ThreadPriority.Highest)
    {
        if (threadIds.Count == 0)
        {
            return 0;
        }

        int boundCount = 0;
        CpuTopologyInfo topology = GetTopology();
        bool useGroupAffinity = processorGroup > 0 || topology.ProcessorGroupCount > 1;
        int win32Priority = ToWin32ThreadPriority(priority);

        for (int i = 0; i < threadIds.Count; i++)
        {
            IntPtr hThread = NativeMethods.OpenThread(
                NativeMethods.ThreadSetInformation | NativeMethods.ThreadQueryInformation,
                false,
                (uint)threadIds[i]);
            if (hThread == IntPtr.Zero)
            {
                continue;
            }

            try
            {
                bool prioritySet = NativeMethods.SetThreadPriority(hThread, win32Priority);
                TryDisableThreadPowerThrottling(hThread);

                if (mask == 0)
                {
                    if (prioritySet)
                    {
                        boundCount++;
                    }
                    continue;
                }

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

    /// <summary>
    /// Ref-counted scope that temporarily elevates the process to <see cref="ProcessPriorityClass.High"/>
    /// and disables Windows EcoQoS power throttling while real-time audio processing is active.
    /// </summary>
    public static IDisposable BeginProcessAudioBoost()
    {
        lock (ProcessBoostSync)
        {
            if (_processBoostRefCount == 0)
            {
                try
                {
                    using Process proc = Process.GetCurrentProcess();
                    ProcessPriorityClass current = proc.PriorityClass;
                    if (current != ProcessPriorityClass.High && current != ProcessPriorityClass.RealTime)
                    {
                        _savedProcessPriorityClass = current;
                        proc.PriorityClass = ProcessPriorityClass.High;
                    }
                    else
                    {
                        _savedProcessPriorityClass = null;
                    }

                    _powerThrottlingDisabled = TrySetProcessPowerThrottling(proc.Handle, disableThrottling: true);
                }
                catch
                {
                    // Best-effort boost; continue running if restricted by OS policy.
                }

                try
                {
                    if (NativeMethods.timeBeginPeriod(1) == 0)
                    {
                        _timerResolutionActive = true;
                    }
                }
                catch
                {
                    // Best-effort 1ms multimedia timer resolution.
                }
            }

            _processBoostRefCount++;
        }

        return new ProcessAudioBoostScope();
    }

    private static void EndProcessAudioBoost()
    {
        lock (ProcessBoostSync)
        {
            if (_processBoostRefCount <= 0)
            {
                _processBoostRefCount = 0;
                return;
            }

            _processBoostRefCount--;
            if (_processBoostRefCount > 0)
            {
                return;
            }

            try
            {
                using Process proc = Process.GetCurrentProcess();
                if (_savedProcessPriorityClass is { } original && proc.PriorityClass == ProcessPriorityClass.High)
                {
                    proc.PriorityClass = original;
                }

                if (_powerThrottlingDisabled)
                {
                    TryResetProcessPowerThrottling(proc.Handle);
                }
            }
            catch
            {
                // Ignore teardown/permission errors when restoring priority.
            }
            finally
            {
                if (_timerResolutionActive)
                {
                    try
                    {
                        NativeMethods.timeEndPeriod(1);
                    }
                    catch
                    {
                        // Ignore timer teardown errors.
                    }
                    _timerResolutionActive = false;
                }

                _savedProcessPriorityClass = null;
                _powerThrottlingDisabled = false;
            }
        }
    }

    /// <summary>
    /// Registers the calling thread with the Windows MMCSS "Pro Audio" scheduler task and raises
    /// its MMCSS priority to <c>AVRT_PRIORITY_CRITICAL</c> (with fallback to <c>AVRT_PRIORITY_HIGH</c>).
    /// Must be disposed on the same thread.
    /// </summary>
    public static MmcssAudioScope? TryRegisterCurrentThreadProAudio()
    {
        try
        {
            uint taskIndex = 0;
            IntPtr handle = NativeMethods.AvSetMmThreadCharacteristicsW("Pro Audio", ref taskIndex);
            if (handle == IntPtr.Zero)
            {
                return null;
            }

            if (!NativeMethods.AvSetMmThreadPriority(handle, NativeMethods.AvrtPriorityCritical))
            {
                NativeMethods.AvSetMmThreadPriority(handle, NativeMethods.AvrtPriorityHigh);
            }
            return new MmcssAudioScope(handle);
        }
        catch
        {
            return null;
        }
    }

    public sealed class MmcssAudioScope : IDisposable
    {
        private IntPtr _handle;

        internal MmcssAudioScope(IntPtr handle)
        {
            _handle = handle;
        }

        public void ReinforcePriority()
        {
            IntPtr handle = _handle;
            if (handle != IntPtr.Zero)
            {
                try
                {
                    if (!NativeMethods.AvSetMmThreadPriority(handle, NativeMethods.AvrtPriorityCritical))
                    {
                        NativeMethods.AvSetMmThreadPriority(handle, NativeMethods.AvrtPriorityHigh);
                    }
                }
                catch
                {
                    // Ignore best-effort MMCSS priority refresh failures.
                }
            }
        }

        public void Dispose()
        {
            IntPtr handle = Interlocked.Exchange(ref _handle, IntPtr.Zero);
            if (handle != IntPtr.Zero)
            {
                try
                {
                    NativeMethods.AvRevertMmThreadCharacteristics(handle);
                }
                catch
                {
                    // Ignore MMCSS teardown failures.
                }
            }
        }
    }

    private sealed class ProcessAudioBoostScope : IDisposable
    {
        private int _disposed;

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) == 0)
            {
                EndProcessAudioBoost();
            }
        }
    }

    private static int ToWin32ThreadPriority(ThreadPriority priority) => priority switch
    {
        ThreadPriority.Lowest => NativeMethods.ThreadPriorityLowest,
        ThreadPriority.BelowNormal => NativeMethods.ThreadPriorityBelowNormal,
        ThreadPriority.AboveNormal => NativeMethods.ThreadPriorityAboveNormal,
        ThreadPriority.Highest => NativeMethods.ThreadPriorityHighest,
        _ => NativeMethods.ThreadPriorityNormal
    };

    private static bool TrySetProcessPowerThrottling(IntPtr processHandle, bool disableThrottling)
    {
        var state = new NativeMethods.ProcessPowerThrottlingState
        {
            Version = NativeMethods.ProcessPowerThrottlingCurrentVersion,
            ControlMask = NativeMethods.ProcessPowerThrottlingExecutionSpeed,
            StateMask = disableThrottling ? 0u : NativeMethods.ProcessPowerThrottlingExecutionSpeed
        };

        return NativeMethods.SetProcessInformation(
            processHandle,
            NativeMethods.ProcessPowerThrottling,
            ref state,
            (uint)Marshal.SizeOf<NativeMethods.ProcessPowerThrottlingState>());
    }

    private static void TryDisableThreadPowerThrottling(IntPtr threadHandle)
    {
        if (threadHandle == IntPtr.Zero)
        {
            return;
        }

        try
        {
            var state = new NativeMethods.ProcessPowerThrottlingState
            {
                Version = NativeMethods.ThreadPowerThrottlingCurrentVersion,
                ControlMask = NativeMethods.ThreadPowerThrottlingExecutionSpeed,
                StateMask = 0u
            };

            NativeMethods.SetThreadInformation(
                threadHandle,
                NativeMethods.ThreadPowerThrottling,
                ref state,
                (uint)Marshal.SizeOf<NativeMethods.ProcessPowerThrottlingState>());
        }
        catch
        {
            // Ignore on older Windows builds lacking ThreadPowerThrottling.
        }
    }

    private static void TryResetProcessPowerThrottling(IntPtr processHandle)
    {
        var state = new NativeMethods.ProcessPowerThrottlingState
        {
            Version = NativeMethods.ProcessPowerThrottlingCurrentVersion,
            ControlMask = 0u,
            StateMask = 0u
        };

        NativeMethods.SetProcessInformation(
            processHandle,
            NativeMethods.ProcessPowerThrottling,
            ref state,
            (uint)Marshal.SizeOf<NativeMethods.ProcessPowerThrottlingState>());
    }

    public int BindNewProcessThreads(HashSet<int> baselineThreadIds, ulong mask, ushort processorGroup = 0)
    {
        if (mask == 0)
        {
            return 0;
        }

        List<int> newIds = GetNewProcessThreadIds(baselineThreadIds);
        return BindSpecificThreads(newIds, mask, processorGroup);
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
            // Homogeneous CPU (e.g. AMD): pick the last 2-3 physical cores (1 LP per core when >=3 cores)
            // so VoxLock intra-op threads never contend on a single SMT core under system load.
            recommendedGroup = rawCores.Max(c => c.Group);
            var groupCoreMasks = new List<ulong>();
            foreach (var core in rawCores)
            {
                if (core.Group == recommendedGroup)
                {
                    groupCoreMasks.Add(core.Mask);
                }
            }

            int takeCores = groupCoreMasks.Count >= 4 ? 3 : Math.Min(2, groupCoreMasks.Count);
            bool singleLpPerCore = groupCoreMasks.Count >= 3;
            int startIdx = Math.Max(0, groupCoreMasks.Count - takeCores);

            for (int i = startIdx; i < groupCoreMasks.Count; i++)
            {
                ulong coreMask = groupCoreMasks[i];
                ulong selectedMask = singleLpPerCore ? (coreMask & (0UL - coreMask)) : coreMask;
                recommendedMask |= selectedMask;

                if (singleLpPerCore)
                {
                    int firstLp = BitOperations.TrailingZeroCount(selectedMask);
                    tailLogicalSet.Add((recommendedGroup, firstLp));
                }
                else
                {
                    var lps = new List<int>(4);
                    ExtractBitIndices(selectedMask, lps);
                    foreach (int lp in lps)
                    {
                        tailLogicalSet.Add((recommendedGroup, lp));
                    }
                }
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

        int tailStart = count >= 6 ? count - 3 : Math.Max(0, count - 2);
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

        if (TryGetPerformanceCoresMask(topology, out ulong pCoreMask, out ushort pCoreGroup, out int pCoreCount) &&
            mask == pCoreMask &&
            group == pCoreGroup)
        {
            return $"Auto P-Cores ({pCoreCount} cores)";
        }

        ulong allGroupMask = 0;
        int groupCoreCount = 0;
        foreach (PhysicalCoreInfo core in topology.PhysicalCores)
        {
            if (core.ProcessorGroup == group)
            {
                allGroupMask |= core.LogicalProcessorMask;
                groupCoreCount++;
            }
        }

        if (allGroupMask != 0 && mask == allGroupMask)
        {
            return $"Auto All Cores ({groupCoreCount} cores)";
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

        public const int ThreadPriorityLowest = -2;
        public const int ThreadPriorityBelowNormal = -1;
        public const int ThreadPriorityNormal = 0;
        public const int ThreadPriorityAboveNormal = 1;
        public const int ThreadPriorityHighest = 2;

        public const int ProcessPowerThrottling = 4;
        public const uint ProcessPowerThrottlingCurrentVersion = 1;
        public const uint ProcessPowerThrottlingExecutionSpeed = 0x1;

        public const int ThreadPowerThrottling = 3;
        public const uint ThreadPowerThrottlingCurrentVersion = 1;
        public const uint ThreadPowerThrottlingExecutionSpeed = 0x1;

        public const int AvrtPriorityHigh = 1;
        public const int AvrtPriorityCritical = 2;

        [StructLayout(LayoutKind.Sequential)]
        public struct ProcessPowerThrottlingState
        {
            public uint Version;
            public uint ControlMask;
            public uint StateMask;
        }

        [DllImport("kernel32.dll", SetLastError = true)]
        public static extern IntPtr OpenThread(uint dwDesiredAccess, bool bInheritHandle, uint dwThreadId);

        [DllImport("kernel32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        public static extern bool SetThreadPriority(IntPtr hThread, int nPriority);

        [DllImport("kernel32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        public static extern bool SetThreadInformation(
            IntPtr hThread,
            int threadInformationClass,
            ref ProcessPowerThrottlingState threadInformation,
            uint threadInformationSize);

        [DllImport("kernel32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        public static extern bool SetProcessInformation(
            IntPtr hProcess,
            int processInformationClass,
            ref ProcessPowerThrottlingState processInformation,
            uint processInformationSize);

        [DllImport("winmm.dll", SetLastError = false)]
        public static extern uint timeBeginPeriod(uint uPeriod);

        [DllImport("winmm.dll", SetLastError = false)]
        public static extern uint timeEndPeriod(uint uPeriod);

        [DllImport("avrt.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        public static extern IntPtr AvSetMmThreadCharacteristicsW(string taskName, ref uint taskIndex);

        [DllImport("avrt.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        public static extern bool AvSetMmThreadPriority(IntPtr avrtHandle, int priority);

        [DllImport("avrt.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        public static extern bool AvRevertMmThreadCharacteristics(IntPtr avrtHandle);

        [DllImport("kernel32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        public static extern bool CloseHandle(IntPtr hObject);
    }
}
