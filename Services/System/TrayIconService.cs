using System.IO;
using System.Runtime.InteropServices;
using System.Windows.Interop;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using VoiceLockStudio.Services.Localization;

namespace VoiceLockStudio.Services.System;

/// <summary>
/// Zero-dependency Win32 system tray icon and context menu service built on
/// <c>Shell_NotifyIconW</c>, <see cref="HwndSource"/>, and <c>TrackPopupMenuEx</c>.
/// </summary>
public sealed class TrayIconService : IDisposable
{
    private const uint TrayIconId = 1;
    private const uint WmTrayCallback = 0x8001; // WM_APP + 1
    private const uint WmNull = 0x0000;
    private const uint WmLButtonDblClk = 0x0203;
    private const uint WmRButtonUp = 0x0205;
    private const uint WmContextMenu = 0x007B;

    private const uint NimAdd = 0x00000000;
    private const uint NimModify = 0x00000001;
    private const uint NimDelete = 0x00000002;

    private const uint NifMessage = 0x00000001;
    private const uint NifIcon = 0x00000002;
    private const uint NifTip = 0x00000004;
    private const uint NifShowTip = 0x00000080;

    private const uint MfString = 0x00000000;
    private const uint MfSeparator = 0x00000800;
    private const uint MfChecked = 0x00000008;
    private const uint MfUnchecked = 0x00000000;

    private const uint TpmRightButton = 0x0002;
    private const uint TpmReturnCmd = 0x0100;
    private const uint TpmBottomAlign = 0x0020;

    private const uint CmdShowWindow = 1001;
    private const uint CmdTogglePipeline = 1002;
    private const uint CmdExit = 1003;

    private readonly HwndSource _messageWindow;
    private readonly uint _taskbarCreatedMsg;
    private readonly IntPtr _idleIconHandle;
    private readonly IntPtr _activeIconHandle;

    private string _tooltipText = AppStrings.Current.TrayReadyTip;
    private string? _lastStatusText;
    private bool _isRunning;
    private bool _iconAdded;
    private bool _disposed;

    public TrayIconService()
    {
        _idleIconHandle = TryCreateBrandIcon(active: false);
        if (_idleIconHandle == IntPtr.Zero)
        {
            _idleIconHandle = CreateMicrophoneBadgeIcon(active: false);
        }

        _activeIconHandle = TryCreateBrandIcon(active: true);
        if (_activeIconHandle == IntPtr.Zero)
        {
            _activeIconHandle = CreateMicrophoneBadgeIcon(active: true);
        }
        _taskbarCreatedMsg = NativeMethods.RegisterWindowMessageW("TaskbarCreated");

        var parameters = new HwndSourceParameters("VoicierVoiceTrayMessageWindow")
        {
            Width = 0,
            Height = 0,
            WindowStyle = 0,
            ExtendedWindowStyle = 0,
            ParentWindow = IntPtr.Zero
        };

        _messageWindow = new HwndSource(parameters);
        _messageWindow.AddHook(WndProc);

        AddTrayIcon();
    }

    public event EventHandler? ShowWindowRequested;
    public event EventHandler? TogglePipelineRequested;
    public event EventHandler? ExitRequested;

    public void UpdateStatus(string statusText, bool isRunning)
    {
        if (_disposed)
        {
            return;
        }

        _isRunning = isRunning;
        _lastStatusText = statusText;
        _tooltipText = AppStrings.Current.FormatTrayTooltip(statusText, isRunning);

        ApplyIconAndTooltip();
    }

    public void UpdateLocalization()
    {
        if (_disposed)
        {
            return;
        }

        _tooltipText = string.IsNullOrWhiteSpace(_lastStatusText)
            ? AppStrings.Current.TrayReadyTip
            : AppStrings.Current.FormatTrayTooltip(_lastStatusText, _isRunning);

        ApplyIconAndTooltip();
    }

    private void ApplyIconAndTooltip()
    {
        if (!_iconAdded)
        {
            AddTrayIcon();
            return;
        }

        var data = CreateNotifyIconData(NifIcon | NifTip | NifShowTip);
        NativeMethods.Shell_NotifyIconW(NimModify, ref data);
    }

    private void AddTrayIcon()
    {
        if (_disposed || _messageWindow.Handle == IntPtr.Zero)
        {
            return;
        }

        var data = CreateNotifyIconData(NifMessage | NifIcon | NifTip | NifShowTip);
        _iconAdded = NativeMethods.Shell_NotifyIconW(NimAdd, ref data);
    }

    private NativeMethods.NotifyIconData CreateNotifyIconData(uint flags)
    {
        return new NativeMethods.NotifyIconData
        {
            cbSize = Marshal.SizeOf<NativeMethods.NotifyIconData>(),
            hWnd = _messageWindow.Handle,
            uID = TrayIconId,
            uFlags = flags,
            uCallbackMessage = WmTrayCallback,
            hIcon = _isRunning ? _activeIconHandle : _idleIconHandle,
            szTip = _tooltipText
        };
    }

    private IntPtr WndProc(IntPtr hwnd, int msg, IntPtr wParam, IntPtr lParam, ref bool handled)
    {
        if (_disposed)
        {
            return IntPtr.Zero;
        }

        uint uMsg = unchecked((uint)msg);
        if (_taskbarCreatedMsg != 0 && uMsg == _taskbarCreatedMsg)
        {
            AddTrayIcon();
            handled = true;
            return IntPtr.Zero;
        }

        if (uMsg == WmTrayCallback)
        {
            uint mouseMsg = unchecked((uint)(lParam.ToInt64() & 0xFFFF));
            switch (mouseMsg)
            {
                case WmLButtonDblClk:
                    ShowWindowRequested?.Invoke(this, EventArgs.Empty);
                    handled = true;
                    break;

                case WmRButtonUp:
                case WmContextMenu:
                    ShowNativeContextMenu();
                    handled = true;
                    break;
            }
        }

        return IntPtr.Zero;
    }

    private void ShowNativeContextMenu()
    {
        IntPtr menu = NativeMethods.CreatePopupMenu();
        if (menu == IntPtr.Zero)
        {
            return;
        }

        try
        {
            NativeMethods.AppendMenuW(menu, MfString, CmdShowWindow, AppStrings.Current.TrayShowWindow);
            uint toggleFlags = MfString | (_isRunning ? MfChecked : MfUnchecked);
            NativeMethods.AppendMenuW(menu, toggleFlags, CmdTogglePipeline, AppStrings.Current.TrayTogglePipeline);
            NativeMethods.AppendMenuW(menu, MfSeparator, 0, string.Empty);
            NativeMethods.AppendMenuW(menu, MfString, CmdExit, AppStrings.Current.TrayExit);

            NativeMethods.GetCursorPos(out NativeMethods.Point pt);
            NativeMethods.SetForegroundWindow(_messageWindow.Handle);

            uint selected = NativeMethods.TrackPopupMenuEx(
                menu,
                TpmReturnCmd | TpmRightButton | TpmBottomAlign,
                pt.X,
                pt.Y,
                _messageWindow.Handle,
                IntPtr.Zero);

            NativeMethods.PostMessageW(_messageWindow.Handle, WmNull, IntPtr.Zero, IntPtr.Zero);

            switch (selected)
            {
                case CmdShowWindow:
                    ShowWindowRequested?.Invoke(this, EventArgs.Empty);
                    break;

                case CmdTogglePipeline:
                    TogglePipelineRequested?.Invoke(this, EventArgs.Empty);
                    break;

                case CmdExit:
                    ExitRequested?.Invoke(this, EventArgs.Empty);
                    break;
            }
        }
        finally
        {
            NativeMethods.DestroyMenu(menu);
        }
    }

    private static unsafe IntPtr TryCreateBrandIcon(bool active)
    {
        const int size = 32;
        try
        {
            var uri = new Uri("pack://application:,,,/VoicierVoice;component/Assets/Icons/VoicierVoice.png", UriKind.Absolute);
            var streamInfo = global::System.Windows.Application.GetResourceStream(uri);
            if (streamInfo?.Stream is null)
            {
                return IntPtr.Zero;
            }

            byte[] buffer = new byte[size * size * 4];
            using (Stream stream = streamInfo.Stream)
            {
                var bitmap = new BitmapImage();
                bitmap.BeginInit();
                bitmap.StreamSource = stream;
                bitmap.CacheOption = BitmapCacheOption.OnLoad;
                bitmap.DecodePixelWidth = size;
                bitmap.DecodePixelHeight = size;
                bitmap.EndInit();
                bitmap.Freeze();

                BitmapSource pbgra = bitmap.Format == PixelFormats.Pbgra32
                    ? bitmap
                    : new FormatConvertedBitmap(bitmap, PixelFormats.Pbgra32, null, 0);
                pbgra.CopyPixels(buffer, size * 4, 0);
            }

            if (active)
            {
                // Draw a crisp emerald status dot at bottom-right so active vs idle stays legible in tray.
                const float dotCx = 25.0f;
                const float dotCy = 25.0f;
                const float dotOuterR = 5.4f;
                const float dotInnerR = 3.8f;

                for (int y = 0; y < size; y++)
                {
                    for (int x = 0; x < size; x++)
                    {
                        float dx = x - dotCx;
                        float dy = y - dotCy;
                        float dist = MathF.Sqrt(dx * dx + dy * dy);
                        if (dist > dotOuterR + 0.5f)
                        {
                            continue;
                        }

                        int idx = (y * size + x) * 4;
                        if (dist <= dotInnerR)
                        {
                            // Emerald dot (#10B981)
                            buffer[idx + 0] = 0x81;
                            buffer[idx + 1] = 0xB9;
                            buffer[idx + 2] = 0x10;
                            buffer[idx + 3] = 0xFF;
                        }
                        else
                        {
                            float rimAlpha = Math.Clamp(dotOuterR + 0.5f - dist, 0f, 1f);
                            byte a = (byte)Math.Round(rimAlpha * 255f);
                            if (a > buffer[idx + 3])
                            {
                                buffer[idx + 0] = a;
                                buffer[idx + 1] = a;
                                buffer[idx + 2] = a;
                                buffer[idx + 3] = a;
                            }
                        }
                    }
                }
            }

            var bmi = new NativeMethods.BitmapInfo
            {
                bmiHeader = new NativeMethods.BitmapInfoHeader
                {
                    biSize = Marshal.SizeOf<NativeMethods.BitmapInfoHeader>(),
                    biWidth = size,
                    biHeight = -size,
                    biPlanes = 1,
                    biBitCount = 32,
                    biCompression = 0
                }
            };

            IntPtr hdcScreen = NativeMethods.GetDC(IntPtr.Zero);
            IntPtr colorBitmap = NativeMethods.CreateDIBSection(hdcScreen, ref bmi, 0, out IntPtr bitsPtr, IntPtr.Zero, 0);
            NativeMethods.ReleaseDC(IntPtr.Zero, hdcScreen);

            if (colorBitmap == IntPtr.Zero || bitsPtr == IntPtr.Zero)
            {
                return IntPtr.Zero;
            }

            IntPtr maskBitmap = NativeMethods.CreateBitmap(size, size, 1, 1, IntPtr.Zero);
            try
            {
                Marshal.Copy(buffer, 0, bitsPtr, buffer.Length);
                var iconInfo = new NativeMethods.IconInfo
                {
                    fIcon = true,
                    xHotspot = 0,
                    yHotspot = 0,
                    hbmMask = maskBitmap,
                    hbmColor = colorBitmap
                };
                return NativeMethods.CreateIconIndirect(ref iconInfo);
            }
            finally
            {
                NativeMethods.DeleteObject(colorBitmap);
                if (maskBitmap != IntPtr.Zero)
                {
                    NativeMethods.DeleteObject(maskBitmap);
                }
            }
        }
        catch
        {
            return IntPtr.Zero;
        }
    }

    /// <summary>
    /// Renders a clean 32x32 ARGB badge icon using pure GDI DIB sections so the tray service
    /// requires zero WinForms or System.Drawing dependencies.
    /// </summary>
    private static unsafe IntPtr CreateMicrophoneBadgeIcon(bool active)
    {
        const int size = 32;

        var bmi = new NativeMethods.BitmapInfo
        {
            bmiHeader = new NativeMethods.BitmapInfoHeader
            {
                biSize = Marshal.SizeOf<NativeMethods.BitmapInfoHeader>(),
                biWidth = size,
                biHeight = -size, // Top-down DIB
                biPlanes = 1,
                biBitCount = 32,
                biCompression = 0 // BI_RGB
            }
        };

        IntPtr hdcScreen = NativeMethods.GetDC(IntPtr.Zero);
        IntPtr colorBitmap = NativeMethods.CreateDIBSection(hdcScreen, ref bmi, 0, out IntPtr bitsPtr, IntPtr.Zero, 0);
        NativeMethods.ReleaseDC(IntPtr.Zero, hdcScreen);

        if (colorBitmap == IntPtr.Zero || bitsPtr == IntPtr.Zero)
        {
            return IntPtr.Zero;
        }

        IntPtr maskBitmap = NativeMethods.CreateBitmap(size, size, 1, 1, IntPtr.Zero);
        try
        {
            uint* pixels = (uint*)bitsPtr.ToPointer();

            // Fluent accent blue (#0078D4) when idle, or emerald (#107C10) when active
            byte bgR = active ? (byte)0x10 : (byte)0x00;
            byte bgG = active ? (byte)0x7C : (byte)0x78;
            byte bgB = active ? (byte)0x10 : (byte)0xD4;

            const float cx = 15.5f;
            const float cy = 15.5f;
            const float radius = 14.0f;

            for (int y = 0; y < size; y++)
            {
                for (int x = 0; x < size; x++)
                {
                    float dx = x - cx;
                    float dy = y - cy;
                    float dist = MathF.Sqrt(dx * dx + dy * dy);

                    // Soft anti-aliased circular badge
                    float circleAlpha = Math.Clamp(radius + 0.5f - dist, 0f, 1f);
                    if (circleAlpha <= 0f)
                    {
                        pixels[y * size + x] = 0;
                        continue;
                    }

                    bool isWhiteGlyph = IsInsideMicrophoneGlyph(x, y);
                    byte r = isWhiteGlyph ? (byte)255 : bgR;
                    byte g = isWhiteGlyph ? (byte)255 : bgG;
                    byte b = isWhiteGlyph ? (byte)255 : bgB;
                    byte a = (byte)Math.Round(circleAlpha * 255f);

                    // Premultiplied BGRA
                    byte pr = (byte)(r * a / 255);
                    byte pg = (byte)(g * a / 255);
                    byte pb = (byte)(b * a / 255);
                    pixels[y * size + x] = ((uint)a << 24) | ((uint)pr << 16) | ((uint)pg << 8) | pb;
                }
            }

            var iconInfo = new NativeMethods.IconInfo
            {
                fIcon = true,
                xHotspot = 0,
                yHotspot = 0,
                hbmMask = maskBitmap,
                hbmColor = colorBitmap
            };

            return NativeMethods.CreateIconIndirect(ref iconInfo);
        }
        finally
        {
            NativeMethods.DeleteObject(colorBitmap);
            if (maskBitmap != IntPtr.Zero)
            {
                NativeMethods.DeleteObject(maskBitmap);
            }
        }
    }

    private static bool IsInsideMicrophoneGlyph(int x, int y)
    {
        // Capsule head: x in [13..18], y in [7..16] with rounded top/bottom
        if (x >= 13 && x <= 18 && y >= 7 && y <= 16)
        {
            float dx = x - 15.5f;
            if (y < 10)
            {
                float dy = y - 10f;
                return dx * dx + dy * dy <= 9.0f;
            }
            if (y > 13)
            {
                float dy = y - 13f;
                return dx * dx + dy * dy <= 9.0f;
            }
            return true;
        }

        // Outer U-arc cradle around capsule
        if (y >= 13 && y <= 20)
        {
            float dx = x - 15.5f;
            float dy = y - 14.5f;
            float r = MathF.Sqrt(dx * dx + dy * dy);
            if (r >= 4.4f && r <= 6.2f && y >= 14)
            {
                return true;
            }
        }

        // Stem and base
        if ((x == 15 || x == 16) && y >= 20 && y <= 23)
        {
            return true;
        }

        if (y == 24 && x >= 12 && x <= 19)
        {
            return true;
        }

        return false;
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;

        if (_iconAdded && _messageWindow.Handle != IntPtr.Zero)
        {
            var data = CreateNotifyIconData(0);
            NativeMethods.Shell_NotifyIconW(NimDelete, ref data);
            _iconAdded = false;
        }

        _messageWindow.RemoveHook(WndProc);
        _messageWindow.Dispose();

        if (_idleIconHandle != IntPtr.Zero)
        {
            NativeMethods.DestroyIcon(_idleIconHandle);
        }

        if (_activeIconHandle != IntPtr.Zero)
        {
            NativeMethods.DestroyIcon(_activeIconHandle);
        }
    }

    private static class NativeMethods
    {
        [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
        public struct NotifyIconData
        {
            public int cbSize;
            public IntPtr hWnd;
            public uint uID;
            public uint uFlags;
            public uint uCallbackMessage;
            public IntPtr hIcon;
            [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 128)]
            public string szTip;
            public uint dwState;
            public uint dwStateMask;
            [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 256)]
            public string szInfo;
            public uint uTimeoutOrVersion;
            [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 64)]
            public string szInfoTitle;
            public uint dwInfoFlags;
            public Guid guidItem;
            public IntPtr hBalloonIcon;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct Point
        {
            public int X;
            public int Y;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct BitmapInfoHeader
        {
            public int biSize;
            public int biWidth;
            public int biHeight;
            public short biPlanes;
            public short biBitCount;
            public int biCompression;
            public int biSizeImage;
            public int biXPelsPerMeter;
            public int biYPelsPerMeter;
            public int biClrUsed;
            public int biClrImportant;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct BitmapInfo
        {
            public BitmapInfoHeader bmiHeader;
            public uint bmiColors;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct IconInfo
        {
            [MarshalAs(UnmanagedType.Bool)]
            public bool fIcon;
            public int xHotspot;
            public int yHotspot;
            public IntPtr hbmMask;
            public IntPtr hbmColor;
        }

        [DllImport("shell32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        public static extern bool Shell_NotifyIconW(uint dwMessage, ref NotifyIconData lpData);

        [DllImport("user32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        public static extern uint RegisterWindowMessageW(string lpString);

        [DllImport("user32.dll", SetLastError = true)]
        public static extern IntPtr CreatePopupMenu();

        [DllImport("user32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        public static extern bool AppendMenuW(IntPtr hMenu, uint uFlags, uint uIDNewItem, string lpNewItem);

        [DllImport("user32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        public static extern bool DestroyMenu(IntPtr hMenu);

        [DllImport("user32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        public static extern bool GetCursorPos(out Point lpPoint);

        [DllImport("user32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        public static extern bool SetForegroundWindow(IntPtr hWnd);

        [DllImport("user32.dll", SetLastError = true)]
        public static extern uint TrackPopupMenuEx(
            IntPtr hMenu,
            uint uFlags,
            int x,
            int y,
            IntPtr hwnd,
            IntPtr lptpm);

        [DllImport("user32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        public static extern bool PostMessageW(IntPtr hWnd, uint msg, IntPtr wParam, IntPtr lParam);

        [DllImport("user32.dll")]
        public static extern IntPtr GetDC(IntPtr hWnd);

        [DllImport("user32.dll")]
        public static extern int ReleaseDC(IntPtr hWnd, IntPtr hDC);

        [DllImport("gdi32.dll", SetLastError = true)]
        public static extern IntPtr CreateDIBSection(
            IntPtr hdc,
            ref BitmapInfo pbmi,
            uint usage,
            out IntPtr ppvBits,
            IntPtr hSection,
            uint offset);

        [DllImport("gdi32.dll", SetLastError = true)]
        public static extern IntPtr CreateBitmap(int nWidth, int nHeight, uint nPlanes, uint nBitCount, IntPtr lpBits);

        [DllImport("gdi32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        public static extern bool DeleteObject(IntPtr ho);

        [DllImport("user32.dll", SetLastError = true)]
        public static extern IntPtr CreateIconIndirect(ref IconInfo piconinfo);

        [DllImport("user32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        public static extern bool DestroyIcon(IntPtr hIcon);
    }
}
