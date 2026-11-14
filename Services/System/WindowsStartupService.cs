using System.IO;
using Microsoft.Win32;

namespace VoiceLockStudio.Services.System;

/// <summary>
/// Manages per-user Windows logon startup registration under
/// <c>HKCU\Software\Microsoft\Windows\CurrentVersion\Run</c>.
/// </summary>
public static class WindowsStartupService
{
    private const string RunSubKeyPath = @"Software\Microsoft\Windows\CurrentVersion\Run";
    private const string AppValueName = "VoicierVoice";

    /// <summary>
    /// Returns <see langword="true"/> when a startup entry for VoicierVoice exists in the current user's Run key.
    /// </summary>
    public static bool IsStartupEnabled()
    {
        try
        {
            using RegistryKey? key = Registry.CurrentUser.OpenSubKey(RunSubKeyPath, writable: false);
            string? value = key?.GetValue(AppValueName) as string;
            return !string.IsNullOrWhiteSpace(value);
        }
        catch
        {
            return false;
        }
    }

    /// <summary>
    /// Enables or disables automatic launch of VoicierVoice when the current Windows user signs in.
    /// </summary>
    public static bool SetStartupEnabled(bool enabled)
    {
        try
        {
            using RegistryKey? key = Registry.CurrentUser.OpenSubKey(RunSubKeyPath, writable: true)
                ?? (enabled ? Registry.CurrentUser.CreateSubKey(RunSubKeyPath, writable: true) : null);

            if (key is null)
            {
                return !enabled;
            }

            if (enabled)
            {
                string? exePath = ResolveExecutablePath();
                if (string.IsNullOrWhiteSpace(exePath))
                {
                    return false;
                }

                key.SetValue(AppValueName, $"\"{exePath}\" --minimized", RegistryValueKind.String);
                return true;
            }

            key.DeleteValue(AppValueName, throwOnMissingValue: false);
            return true;
        }
        catch
        {
            return false;
        }
    }

    private static string? ResolveExecutablePath()
    {
        string? processPath = Environment.ProcessPath;
        if (string.IsNullOrWhiteSpace(processPath) || !File.Exists(processPath))
        {
            return null;
        }

        // Avoid registering dotnet.exe if launched via `dotnet run`.
        string fileName = Path.GetFileName(processPath);
        if (string.Equals(fileName, "dotnet.exe", StringComparison.OrdinalIgnoreCase))
        {
            string baseDirExe = Path.Combine(AppContext.BaseDirectory, "VoicierVoice.exe");
            return File.Exists(baseDirExe) ? baseDirExe : null;
        }

        return processPath;
    }
}
