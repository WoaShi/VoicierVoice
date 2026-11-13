using System.Runtime.InteropServices;
using System.Windows;
using System.Windows.Interop;
using System.Windows.Media;
using Microsoft.Win32;
using VoiceLockStudio.Models;

namespace VoiceLockStudio.Services.System;

/// <summary>
/// Applies Windows 11 DWM Mica backdrop, rounded window corners, and dark/light titlebar
/// synchronization based on <see cref="Application.ThemeMode"/> / Windows theme settings,
/// with graceful fallback on Windows 10.
/// </summary>
public static class DwmBackdropHelper
{
    private const int DwmwaUseImmersiveDarkMode = 20;
    private const int DwmwaWindowCornerPreference = 33;
    private const int DwmwaSystemBackdropType = 38;

    private const int DwmwcpRound = 2;
    private const int DwmsbtMainWindow = 2;

    public static bool IsWindows11OrGreater => OperatingSystem.IsWindowsVersionAtLeast(10, 0, 22000);
    public static bool IsMicaSupported => OperatingSystem.IsWindowsVersionAtLeast(10, 0, 22621);

    public static AppThemePreference CurrentThemePreference { get; private set; } = AppThemePreference.System;

    public static event EventHandler<AppThemePreference>? ThemeChanged;

    /// <summary>
    /// Applies the requested <see cref="AppThemePreference"/> to <see cref="Application.ThemeMode"/>
    /// at runtime and refreshes DWM titlebar theming on all currently open windows.
    /// </summary>
    public static void ApplyThemePreference(AppThemePreference preference)
    {
        CurrentThemePreference = preference;

        if (Application.Current is { } app)
        {
#pragma warning disable WPF0001
            app.ThemeMode = preference switch
            {
                AppThemePreference.Light => ThemeMode.Light,
                AppThemePreference.Dark => ThemeMode.Dark,
                _ => ThemeMode.System
            };
#pragma warning restore WPF0001

            foreach (Window window in app.Windows)
            {
                UpdateTitleBarTheme(window);
            }
        }

        ThemeChanged?.Invoke(null, preference);
    }

    /// <summary>
    /// Applies Mica (Win11 22H2+), rounded corners (Win11+), and immersive dark mode titlebar
    /// to <paramref name="window"/>. Should be called from <see cref="Window.OnSourceInitialized"/>
    /// or after the window handle is created.
    /// </summary>
    /// <returns><see langword="true"/> if DWM Mica backdrop was applied; otherwise <see langword="false"/>.</returns>
    public static bool ApplyBackdrop(Window window, bool attachThemeListener = true)
    {
        ArgumentNullException.ThrowIfNull(window);

        IntPtr hwnd = new WindowInteropHelper(window).Handle;
        if (hwnd == IntPtr.Zero)
        {
            return false;
        }

        if (attachThemeListener)
        {
            AttachThemeSync(window);
        }

        UpdateTitleBarTheme(window);

        if (IsWindows11OrGreater)
        {
            int cornerPreference = DwmwcpRound;
            _ = NativeMethods.DwmSetWindowAttribute(
                hwnd,
                DwmwaWindowCornerPreference,
                ref cornerPreference,
                sizeof(int));
        }

        if (!IsMicaSupported)
        {
            return false;
        }

        if (HwndSource.FromHwnd(hwnd)?.CompositionTarget is { } compositionTarget)
        {
            compositionTarget.BackgroundColor = Colors.Transparent;
        }

        var margins = new NativeMethods.Margins(-1);
        _ = NativeMethods.DwmExtendFrameIntoClientArea(hwnd, ref margins);

        int backdropType = DwmsbtMainWindow;
        int hr = NativeMethods.DwmSetWindowAttribute(
            hwnd,
            DwmwaSystemBackdropType,
            ref backdropType,
            sizeof(int));

        return hr == 0;
    }

    /// <summary>
    /// Synchronizes DWM titlebar dark mode (<c>DWMWA_USE_IMMERSIVE_DARK_MODE</c>) with the current
    /// application <see cref="ThemeMode"/> or Windows system personalization setting.
    /// </summary>
    public static void UpdateTitleBarTheme(Window window)
    {
        ArgumentNullException.ThrowIfNull(window);

        IntPtr hwnd = new WindowInteropHelper(window).Handle;
        if (hwnd == IntPtr.Zero)
        {
            return;
        }

        int useDark = IsDarkThemeActive() ? 1 : 0;
        _ = NativeMethods.DwmSetWindowAttribute(
            hwnd,
            DwmwaUseImmersiveDarkMode,
            ref useDark,
            sizeof(int));
    }

    /// <summary>
    /// Determines whether dark mode is currently active according to <see cref="Application.ThemeMode"/>,
    /// <see cref="CurrentThemePreference"/>, or the Windows <c>AppsUseLightTheme</c> registry preference.
    /// </summary>
    public static bool IsDarkThemeActive()
    {
        if (Application.Current is { } app)
        {
#pragma warning disable WPF0001
            ThemeMode mode = app.ThemeMode;
            if (mode == ThemeMode.Dark)
            {
                return true;
            }

            if (mode == ThemeMode.Light)
            {
                return false;
            }
#pragma warning restore WPF0001
        }

        if (CurrentThemePreference == AppThemePreference.Dark)
        {
            return true;
        }

        if (CurrentThemePreference == AppThemePreference.Light)
        {
            return false;
        }

        try
        {
            object? value = Registry.GetValue(
                @"HKEY_CURRENT_USER\Software\Microsoft\Windows\CurrentVersion\Themes\Personalize",
                "AppsUseLightTheme",
                1);

            if (value is int intVal)
            {
                return intVal == 0;
            }
        }
        catch
        {
            // Default to light if registry query is blocked.
        }

        return false;
    }

    private static void AttachThemeSync(Window window)
    {
        if (ThemeSyncTracker.IsAttached(window))
        {
            return;
        }

        ThemeSyncTracker.Attach(window);
    }

    private sealed class ThemeSyncTracker
    {
        private static readonly DependencyProperty TrackerProperty = DependencyProperty.RegisterAttached(
            "Tracker",
            typeof(ThemeSyncTracker),
            typeof(ThemeSyncTracker),
            new PropertyMetadata(null));

        private readonly WeakReference<Window> _windowRef;

        private ThemeSyncTracker(Window window)
        {
            _windowRef = new WeakReference<Window>(window);
            SystemEvents.UserPreferenceChanged += OnUserPreferenceChanged;
            window.Closed += OnWindowClosed;
        }

        public static bool IsAttached(Window window) => window.GetValue(TrackerProperty) is ThemeSyncTracker;

        public static void Attach(Window window)
        {
            var tracker = new ThemeSyncTracker(window);
            window.SetValue(TrackerProperty, tracker);
        }

        private void OnUserPreferenceChanged(object? sender, UserPreferenceChangedEventArgs e)
        {
            if (e.Category != UserPreferenceCategory.General &&
                e.Category != UserPreferenceCategory.Color)
            {
                return;
            }

            if (_windowRef.TryGetTarget(out Window? window))
            {
                window.Dispatcher.BeginInvoke(() => UpdateTitleBarTheme(window));
            }
        }

        private void OnWindowClosed(object? sender, EventArgs e)
        {
            SystemEvents.UserPreferenceChanged -= OnUserPreferenceChanged;
            if (sender is Window window)
            {
                window.Closed -= OnWindowClosed;
                window.ClearValue(TrackerProperty);
            }
        }
    }

    private static class NativeMethods
    {
        [StructLayout(LayoutKind.Sequential)]
        public readonly struct Margins
        {
            public readonly int LeftWidth;
            public readonly int RightWidth;
            public readonly int TopHeight;
            public readonly int BottomHeight;

            public Margins(int uniform)
            {
                LeftWidth = uniform;
                RightWidth = uniform;
                TopHeight = uniform;
                BottomHeight = uniform;
            }
        }

        [DllImport("dwmapi.dll", PreserveSig = true)]
        public static extern int DwmSetWindowAttribute(
            IntPtr hwnd,
            int dwAttribute,
            ref int pvAttribute,
            int cbAttribute);

        [DllImport("dwmapi.dll", PreserveSig = true)]
        public static extern int DwmExtendFrameIntoClientArea(
            IntPtr hwnd,
            ref Margins pMarInset);
    }
}
