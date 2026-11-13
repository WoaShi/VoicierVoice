using NAudio.CoreAudioApi;

namespace VoiceLockStudio.Services.Audio;

/// <summary>
/// Describes an active WASAPI capture or render endpoint.
/// </summary>
public sealed record AudioDeviceInfo(
    string DeviceId,
    string FriendlyName,
    string DeviceDescription,
    DataFlow DataFlow,
    bool IsDefault,
    bool IsVirtualAudioDevice,
    int SampleRate,
    int Channels)
{
    public string DisplayName
    {
        get
        {
            if (IsVirtualAudioDevice)
            {
                return $"[Virtual] {FriendlyName}";
            }

            return IsDefault ? $"{FriendlyName} (Default)" : FriendlyName;
        }
    }
}

/// <summary>
/// Enumerates Windows WASAPI capture (microphones) and render (virtual cables / playback) endpoints.
/// </summary>
public sealed class AudioDeviceService
{
    public IReadOnlyList<AudioDeviceInfo> GetCaptureDevices() => EnumerateDevices(DataFlow.Capture);

    public IReadOnlyList<AudioDeviceInfo> GetRenderDevices() => EnumerateDevices(DataFlow.Render);

    public string? GetDefaultDeviceId(DataFlow dataFlow, Role role = Role.Communications)
    {
        using var enumerator = new MMDeviceEnumerator();
        try
        {
            using MMDevice device = enumerator.GetDefaultAudioEndpoint(dataFlow, role);
            return device.ID;
        }
        catch
        {
            try
            {
                using MMDevice fallback = enumerator.GetDefaultAudioEndpoint(dataFlow, Role.Multimedia);
                return fallback.ID;
            }
            catch
            {
                return null;
            }
        }
    }

    public MMDevice? ResolveDevice(string? deviceId, DataFlow dataFlow)
    {
        using var enumerator = new MMDeviceEnumerator();
        if (!string.IsNullOrWhiteSpace(deviceId))
        {
            try
            {
                MMDevice specific = enumerator.GetDevice(deviceId);
                if (specific.DataFlow == dataFlow && specific.State == DeviceState.Active)
                {
                    return specific;
                }

                specific.Dispose();
            }
            catch
            {
                // Fall back to default endpoint below.
            }
        }

        try
        {
            return enumerator.GetDefaultAudioEndpoint(dataFlow, Role.Multimedia);
        }
        catch
        {
            try
            {
                return enumerator.GetDefaultAudioEndpoint(dataFlow, Role.Communications);
            }
            catch
            {
                return null;
            }
        }
    }

    private static IReadOnlyList<AudioDeviceInfo> EnumerateDevices(DataFlow dataFlow)
    {
        using var enumerator = new MMDeviceEnumerator();
        string? defaultId = null;
        try
        {
            using MMDevice defaultDevice = enumerator.GetDefaultAudioEndpoint(dataFlow, Role.Multimedia);
            defaultId = defaultDevice.ID;
        }
        catch
        {
            // No default endpoint configured.
        }

        var list = new List<AudioDeviceInfo>();
        MMDeviceCollection collection = enumerator.EnumerateAudioEndPoints(dataFlow, DeviceState.Active);
        foreach (MMDevice device in collection)
        {
            using (device)
            {
                string id = device.ID;
                string friendlyName = SafeGet(() => device.FriendlyName) ?? id;
                string deviceDesc = SafeGet(() => device.DeviceFriendlyName) ?? friendlyName;

                int sampleRate = 48000;
                int channels = 2;
                try
                {
                    var mixFormat = device.AudioClient.MixFormat;
                    sampleRate = mixFormat.SampleRate;
                    channels = mixFormat.Channels;
                }
                catch
                {
                    // Keep standard fallback format metadata if endpoint query is temporarily unavailable.
                }

                bool isVirtual = dataFlow == DataFlow.Render && IsVirtualEndpoint(friendlyName, deviceDesc);
                bool isDefault = string.Equals(id, defaultId, StringComparison.OrdinalIgnoreCase);

                list.Add(new AudioDeviceInfo(
                    DeviceId: id,
                    FriendlyName: friendlyName,
                    DeviceDescription: deviceDesc,
                    DataFlow: dataFlow,
                    IsDefault: isDefault,
                    IsVirtualAudioDevice: isVirtual,
                    SampleRate: sampleRate,
                    Channels: channels));
            }
        }

        if (dataFlow == DataFlow.Render)
        {
            list.Sort((a, b) =>
            {
                int virtualCmp = b.IsVirtualAudioDevice.CompareTo(a.IsVirtualAudioDevice);
                if (virtualCmp != 0) return virtualCmp;

                int defaultCmp = b.IsDefault.CompareTo(a.IsDefault);
                if (defaultCmp != 0) return defaultCmp;

                return string.Compare(a.FriendlyName, b.FriendlyName, StringComparison.OrdinalIgnoreCase);
            });
        }
        else
        {
            list.Sort((a, b) =>
            {
                int defaultCmp = b.IsDefault.CompareTo(a.IsDefault);
                if (defaultCmp != 0) return defaultCmp;

                return string.Compare(a.FriendlyName, b.FriendlyName, StringComparison.OrdinalIgnoreCase);
            });
        }

        return list;
    }

    private static bool IsVirtualEndpoint(string friendlyName, string deviceDescription)
    {
        return friendlyName.Contains("CABLE Input", StringComparison.OrdinalIgnoreCase)
            || friendlyName.Contains("Virtual", StringComparison.OrdinalIgnoreCase)
            || deviceDescription.Contains("CABLE Input", StringComparison.OrdinalIgnoreCase)
            || deviceDescription.Contains("Virtual", StringComparison.OrdinalIgnoreCase);
    }

    private static string? SafeGet(Func<string> getter)
    {
        try
        {
            return getter();
        }
        catch
        {
            return null;
        }
    }
}
