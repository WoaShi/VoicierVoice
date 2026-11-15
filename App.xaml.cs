using System.ComponentModel;
using System.Runtime.InteropServices;
using System.Windows;
using Microsoft.ML.OnnxRuntime;
using VoiceLockStudio.Services;
using VoiceLockStudio.Services.Localization;
using VoiceLockStudio.Services.System;
using VoiceLockStudio.ViewModels;

namespace VoiceLockStudio;

public partial class App : Application
{
    private VoiceLockController? _controller;
    private MainViewModel? _viewModel;
    private TrayIconService? _trayIconService;
    private MainWindow? _mainWindow;
    private bool _isExiting;

    protected override async void OnStartup(StartupEventArgs e)
    {
        base.OnStartup(e);
        ShutdownMode = ShutdownMode.OnExplicitShutdown;

        if (IsSelfTestRequested(e.Args))
        {
            int exitCode = await SelfTestRunner.RunAsync(e.Args);
            Shutdown(exitCode);
            return;
        }

        _controller = new VoiceLockController();
        AppStrings.Current.SetLanguage(_controller.Settings.Language);
        DwmBackdropHelper.ApplyThemePreference(_controller.Settings.ThemePreference);
        AppStrings.Current.LanguageChanged += OnLanguageChanged;

        _viewModel = new MainViewModel(_controller);
        _trayIconService = new TrayIconService();

        _trayIconService.ShowWindowRequested += (_, _) => Dispatcher.BeginInvoke(ShowMainWindow);
        _trayIconService.TogglePipelineRequested += (_, _) => Dispatcher.BeginInvoke(() =>
        {
            if (_viewModel is not null && _viewModel.TogglePipelineCommand.CanExecute(null))
            {
                _viewModel.TogglePipelineCommand.Execute(null);
            }
        });
        _trayIconService.ExitRequested += (_, _) => Dispatcher.BeginInvoke(ExitApplication);

        _viewModel.ShowWindowRequested += (_, _) => ShowMainWindow();
        _viewModel.ExitRequested += (_, _) => ExitApplication();
        _viewModel.PropertyChanged += OnViewModelPropertyChanged;

        _mainWindow = new MainWindow
        {
            DataContext = _viewModel
        };
        _mainWindow.Closing += OnMainWindowClosing;

        MainWindow = _mainWindow;
        bool startMinimized = HasArgument(e.Args, "--minimized");
        if (startMinimized && _viewModel.MinimizeToTrayEnabled)
        {
            // Keep hidden in tray on Windows logon autostart when minimize-to-tray is enabled.
        }
        else
        {
            if (startMinimized)
            {
                _mainWindow.WindowState = WindowState.Minimized;
            }
            _mainWindow.Show();
        }

        _trayIconService.UpdateStatus(_viewModel.StatusBadgeText, _viewModel.IsPipelineRunning);

        await _viewModel.InitializeAsync();
    }

    private void OnViewModelPropertyChanged(object? sender, PropertyChangedEventArgs e)
    {
        if (_trayIconService is null || _viewModel is null)
        {
            return;
        }

        if (e.PropertyName is nameof(MainViewModel.StatusBadgeText) or nameof(MainViewModel.IsPipelineRunning))
        {
            _trayIconService.UpdateStatus(_viewModel.StatusBadgeText, _viewModel.IsPipelineRunning);
        }
    }

    private void OnLanguageChanged(object? sender, EventArgs e)
    {
        if (_trayIconService is null)
        {
            return;
        }

        Dispatcher.BeginInvoke(() =>
        {
            if (_trayIconService is null)
            {
                return;
            }

            if (_viewModel is not null)
            {
                _trayIconService.UpdateStatus(_viewModel.StatusBadgeText, _viewModel.IsPipelineRunning);
            }
            else
            {
                _trayIconService.UpdateLocalization();
            }
        });
    }

    private void OnMainWindowClosing(object? sender, CancelEventArgs e)
    {
        if (!_isExiting && _viewModel is { MinimizeToTrayEnabled: true } && _mainWindow is not null)
        {
            e.Cancel = true;
            _mainWindow.Hide();
            return;
        }

        if (!_isExiting)
        {
            ExitApplication();
        }
    }

    private void ShowMainWindow()
    {
        if (_mainWindow is null)
        {
            return;
        }

        if (!_mainWindow.IsVisible)
        {
            _mainWindow.Show();
        }

        if (_mainWindow.WindowState == WindowState.Minimized)
        {
            _mainWindow.WindowState = WindowState.Normal;
        }

        _mainWindow.Activate();
        _mainWindow.Focus();
    }

    private void ExitApplication()
    {
        if (_isExiting)
        {
            return;
        }

        _isExiting = true;

        if (_mainWindow is not null)
        {
            _mainWindow.Closing -= OnMainWindowClosing;
            _mainWindow.Close();
        }

        Shutdown();
    }

    protected override void OnExit(ExitEventArgs e)
    {
        AppStrings.Current.LanguageChanged -= OnLanguageChanged;

        if (_viewModel is not null)
        {
            _viewModel.PropertyChanged -= OnViewModelPropertyChanged;
            _viewModel.Dispose();
            _viewModel = null;
        }

        _trayIconService?.Dispose();
        _trayIconService = null;

        _controller?.Dispose();
        _controller = null;

        GC.Collect();
        GC.WaitForPendingFinalizers();

        if (NativeMethods.GetModuleHandleW("onnxruntime.dll") != IntPtr.Zero)
        {
            try
            {
                OrtEnv.Instance().Dispose();
            }
            catch
            {
                // Best-effort native OrtEnv teardown before process exit.
            }
        }

        base.OnExit(e);
    }

    private static class NativeMethods
    {
        [DllImport("kernel32.dll", CharSet = CharSet.Unicode)]
        public static extern IntPtr GetModuleHandleW(string lpModuleName);
    }

    private static bool IsSelfTestRequested(string[] args)
    {
        return HasArgument(args, "--self-test") ||
               HasArgument(args, "/selftest") ||
               HasArgument(args, "--self-test-ui") ||
               HasArgument(args, "/selftest-ui");
    }

    private static bool HasArgument(string[] args, string flag)
    {
        foreach (string arg in args)
        {
            if (string.Equals(arg, flag, StringComparison.OrdinalIgnoreCase))
            {
                return true;
            }
        }

        return false;
    }
}
