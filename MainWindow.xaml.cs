using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Animation;
using VoiceLockStudio.Controls;
using VoiceLockStudio.Services.System;
using VoiceLockStudio.ViewModels;

namespace VoiceLockStudio;

public partial class MainWindow : Window
{
    private MainViewModel? _subscribedViewModel;
    private bool _splashStarted;

    public MainWindow()
    {
        InitializeComponent();
        HeaderKaomoji.SetFlightTargetsHidden(true);
        DataContextChanged += OnMainWindowDataContextChanged;
        ContentRendered += OnMainWindowContentRendered;
        StateChanged += OnWindowStateChanged;
        IsVisibleChanged += OnWindowIsVisibleChanged;
        Closed += OnMainWindowClosed;
    }

    protected override void OnSourceInitialized(EventArgs e)
    {
        base.OnSourceInitialized(e);

        bool micaApplied = DwmBackdropHelper.ApplyBackdrop(this, attachThemeListener: true);
        if (micaApplied)
        {
            Background = Brushes.Transparent;
        }
        else
        {
            SetResourceReference(BackgroundProperty, "SolidBackgroundFillColorBaseBrush");
        }

        UpdateMaximizedLayoutState();
    }

    private void OnMainWindowContentRendered(object? sender, EventArgs e)
    {
        if (_splashStarted)
        {
            return;
        }

        _splashStarted = true;
        StartLaunchSplashAnimation();
    }

    private void StartLaunchSplashAnimation()
    {
        if (WindowState == WindowState.Minimized || !IsVisible)
        {
            CompleteSplashImmediately();
            return;
        }

        UpdateLayout();

        if (!HeaderKaomoji.TryGetMouthVCenters(RootLayoutGrid, out Point v1Target, out Point v2Target) ||
            SplashV1.ActualWidth <= 0 ||
            SplashV2.ActualWidth <= 0)
        {
            CompleteSplashImmediately();
            return;
        }

        Point v1Start = SplashV1.TransformToAncestor(RootLayoutGrid)
            .Transform(new Point(SplashV1.ActualWidth / 2.0, SplashV1.ActualHeight / 2.0));
        Point v2Start = SplashV2.TransformToAncestor(RootLayoutGrid)
            .Transform(new Point(SplashV2.ActualWidth / 2.0, SplashV2.ActualHeight / 2.0));

        double dx1 = v1Target.X - v1Start.X;
        double dy1 = v1Target.Y - v1Start.Y;
        double dx2 = v2Target.X - v2Start.X;
        double dy2 = v2Target.Y - v2Start.Y;

        double targetScale = KaomojiStatusView.KaomojiFontSize / SplashV1.FontSize;

        Color splashInk = Color.FromRgb(0x1A, 0x1A, 0x1A);
        Color targetInk = TryFindResource("TextFillColorPrimaryBrush") is SolidColorBrush textBrush
            ? textBrush.Color
            : splashInk;

        var v1Brush = new SolidColorBrush(splashInk);
        var v2Brush = new SolidColorBrush(splashInk);
        SplashV1.Foreground = v1Brush;
        SplashV2.Foreground = v2Brush;

        TimeSpan holdDuration = TimeSpan.FromMilliseconds(200);
        TimeSpan flightDuration = TimeSpan.FromMilliseconds(720);
        TimeSpan textFadeDuration = TimeSpan.FromMilliseconds(480);

        var flightEase = new QuarticEase { EasingMode = EasingMode.EaseInOut };
        var fadeEase = new CubicEase { EasingMode = EasingMode.EaseInOut };

        DoubleAnimation MakeDoubleAnim(double from, double to, TimeSpan begin, TimeSpan duration, IEasingFunction easing) =>
            new()
            {
                From = from,
                To = to,
                BeginTime = begin,
                Duration = duration,
                EasingFunction = easing
            };

        ColorAnimation MakeColorAnim() =>
            new()
            {
                From = splashInk,
                To = targetInk,
                BeginTime = holdDuration,
                Duration = flightDuration,
                EasingFunction = fadeEase
            };

        // Straight-line nonlinear motion of both 'v's into "( • vv • )"
        SplashV1Translate.BeginAnimation(TranslateTransform.XProperty, MakeDoubleAnim(0, dx1, holdDuration, flightDuration, flightEase));
        DoubleAnimation flightY1 = MakeDoubleAnim(0, dy1, holdDuration, flightDuration, flightEase);
        flightY1.Completed += (_, _) => CompleteSplashImmediately();
        SplashV1Translate.BeginAnimation(TranslateTransform.YProperty, flightY1);
        SplashV1Scale.BeginAnimation(ScaleTransform.ScaleXProperty, MakeDoubleAnim(1, targetScale, holdDuration, flightDuration, flightEase));
        SplashV1Scale.BeginAnimation(ScaleTransform.ScaleYProperty, MakeDoubleAnim(1, targetScale, holdDuration, flightDuration, flightEase));

        SplashV2Translate.BeginAnimation(TranslateTransform.XProperty, MakeDoubleAnim(0, dx2, holdDuration, flightDuration, flightEase));
        SplashV2Translate.BeginAnimation(TranslateTransform.YProperty, MakeDoubleAnim(0, dy2, holdDuration, flightDuration, flightEase));
        SplashV2Scale.BeginAnimation(ScaleTransform.ScaleXProperty, MakeDoubleAnim(1, targetScale, holdDuration, flightDuration, flightEase));
        SplashV2Scale.BeginAnimation(ScaleTransform.ScaleYProperty, MakeDoubleAnim(1, targetScale, holdDuration, flightDuration, flightEase));

        // Smooth color alignment to active theme foreground
        v1Brush.BeginAnimation(SolidColorBrush.ColorProperty, MakeColorAnim());
        v2Brush.BeginAnimation(SolidColorBrush.ColorProperty, MakeColorAnim());

        // Gradually increase transparency (fade out) of white background and remaining wordmark characters
        SplashWhiteBackdrop.BeginAnimation(OpacityProperty, MakeDoubleAnim(1.0, 0.0, holdDuration, flightDuration, fadeEase));
        SplashRest1.BeginAnimation(OpacityProperty, MakeDoubleAnim(1.0, 0.0, holdDuration, textFadeDuration, fadeEase));
        SplashRest2.BeginAnimation(OpacityProperty, MakeDoubleAnim(1.0, 0.0, holdDuration, textFadeDuration, fadeEase));
    }

    private void CompleteSplashImmediately()
    {
        HeaderKaomoji.SetFlightTargetsHidden(false);
        SplashOverlay.Visibility = Visibility.Collapsed;
        SplashOverlay.IsHitTestVisible = false;
    }

    private void OnMainWindowDataContextChanged(object sender, DependencyPropertyChangedEventArgs e)
    {
        if (_subscribedViewModel is not null)
        {
            _subscribedViewModel.GuidanceRequested -= OnGuidanceRequested;
            _subscribedViewModel = null;
        }

        if (e.NewValue is MainViewModel vm)
        {
            _subscribedViewModel = vm;
            _subscribedViewModel.GuidanceRequested += OnGuidanceRequested;
            UpdateTelemetryWindowActivity();
        }
    }

    private void OnMainWindowClosed(object? sender, EventArgs e)
    {
        if (_subscribedViewModel is not null)
        {
            _subscribedViewModel.GuidanceRequested -= OnGuidanceRequested;
            _subscribedViewModel = null;
        }
    }

    private void OnGuidanceRequested(object? sender, GuidanceTarget target)
    {
        if (target == GuidanceTarget.MasterSwitch)
        {
            GuidanceAnimationHelper.PlayOutlinePulse(MasterSwitchGuideOverlay);
        }
    }

    private void OnNavItemChecked(object sender, RoutedEventArgs e)
    {
        PageScrollViewer?.ScrollToHome();
    }

    private void OnWindowStateChanged(object? sender, EventArgs e)
    {
        UpdateMaximizedLayoutState();
        UpdateTelemetryWindowActivity();
    }

    private void OnWindowIsVisibleChanged(object sender, DependencyPropertyChangedEventArgs e)
    {
        UpdateTelemetryWindowActivity();
    }

    private void UpdateTelemetryWindowActivity()
    {
        bool active = IsVisible && WindowState != WindowState.Minimized;
        _subscribedViewModel?.SetWindowActiveForTelemetry(active);
    }

    private void UpdateMaximizedLayoutState()
    {
        bool isMaximized = WindowState == WindowState.Maximized;
        RootLayoutGrid.Margin = isMaximized ? new Thickness(7) : new Thickness(0);
    }
}
