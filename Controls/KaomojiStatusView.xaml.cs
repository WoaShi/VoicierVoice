using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using VoiceLockStudio.ViewModels;

namespace VoiceLockStudio.Controls;

public partial class KaomojiStatusView : UserControl
{
    public const double KaomojiFontSize = 22.0;

    public static readonly DependencyProperty IsPipelineActiveProperty =
        DependencyProperty.Register(
            nameof(IsPipelineActive),
            typeof(bool),
            typeof(KaomojiStatusView),
            new PropertyMetadata(false, OnStatusPropertyChanged));

    public static readonly DependencyProperty StatusKindProperty =
        DependencyProperty.Register(
            nameof(StatusKind),
            typeof(PipelineStatusKind),
            typeof(KaomojiStatusView),
            new PropertyMetadata(PipelineStatusKind.Idle, OnStatusPropertyChanged));

    private bool _hideFlightTargets;

    public KaomojiStatusView()
    {
        InitializeComponent();
        UpdateVisualState();
    }

    public bool IsPipelineActive
    {
        get => (bool)GetValue(IsPipelineActiveProperty);
        set => SetValue(IsPipelineActiveProperty, value);
    }

    public PipelineStatusKind StatusKind
    {
        get => (PipelineStatusKind)GetValue(StatusKindProperty);
        set => SetValue(StatusKindProperty, value);
    }

    public void SetFlightTargetsHidden(bool hidden)
    {
        _hideFlightTargets = hidden;
        UpdateFlightTargetOpacity();
    }

    public bool TryGetMouthVCenters(Visual ancestor, out Point v1Center, out Point v2Center)
    {
        v1Center = default;
        v2Center = default;

        if (MouthV1.ActualWidth <= 0 || MouthV2.ActualWidth <= 0 || !ancestor.IsAncestorOf(MouthV1))
        {
            return false;
        }

        v1Center = MouthV1.TransformToAncestor(ancestor)
            .Transform(new Point(MouthV1.ActualWidth / 2.0, MouthV1.ActualHeight / 2.0));
        v2Center = MouthV2.TransformToAncestor(ancestor)
            .Transform(new Point(MouthV2.ActualWidth / 2.0, MouthV2.ActualHeight / 2.0));
        return true;
    }

    private static void OnStatusPropertyChanged(DependencyObject d, DependencyPropertyChangedEventArgs e)
    {
        if (d is KaomojiStatusView view)
        {
            view.UpdateVisualState();
        }
    }

    private void UpdateVisualState()
    {
        bool active = IsPipelineActive;
        bool isolatingSpeech = active && StatusKind == PipelineStatusKind.Running;

        if (isolatingSpeech)
        {
            CalmPanel.Visibility = Visibility.Collapsed;
            SpeakingPanel.Visibility = Visibility.Visible;
        }
        else
        {
            SpeakingPanel.Visibility = Visibility.Collapsed;
            CalmPanel.Visibility = Visibility.Visible;

            Visibility headphoneVisibility = active ? Visibility.Visible : Visibility.Collapsed;
            LeftHeadphone.Visibility = headphoneVisibility;
            RightHeadphone.Visibility = headphoneVisibility;
        }

        UpdateFlightTargetOpacity();
    }

    private void UpdateFlightTargetOpacity()
    {
        double opacity = _hideFlightTargets ? 0.0 : 1.0;
        MouthV1.Opacity = opacity;
        MouthV2.Opacity = opacity;
    }
}
