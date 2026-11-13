using System.Windows;
using System.Windows.Controls;
using VoiceLockStudio.Controls;
using VoiceLockStudio.ViewModels;

namespace VoiceLockStudio.Views;

public partial class PipelineView : UserControl
{
    private MainViewModel? _subscribedViewModel;

    public PipelineView()
    {
        InitializeComponent();
        Loaded += OnViewLoaded;
        Unloaded += OnViewUnloaded;
        DataContextChanged += OnViewDataContextChanged;
    }

    private void OnViewLoaded(object sender, RoutedEventArgs e)
    {
        AttachViewModel(DataContext as MainViewModel);
    }

    private void OnViewDataContextChanged(object sender, DependencyPropertyChangedEventArgs e)
    {
        AttachViewModel(e.NewValue as MainViewModel);
    }

    private void OnViewUnloaded(object sender, RoutedEventArgs e)
    {
        AttachViewModel(null);
    }

    private void AttachViewModel(MainViewModel? next)
    {
        if (ReferenceEquals(_subscribedViewModel, next))
        {
            return;
        }

        if (_subscribedViewModel is not null)
        {
            _subscribedViewModel.GuidanceRequested -= OnGuidanceRequested;
        }

        _subscribedViewModel = next;

        if (_subscribedViewModel is not null)
        {
            _subscribedViewModel.GuidanceRequested += OnGuidanceRequested;
        }
    }

    private void OnGuidanceRequested(object? sender, GuidanceTarget target)
    {
        if (target == GuidanceTarget.AudioRouting)
        {
            RoutingCardHost.BringIntoView();
            GuidanceAnimationHelper.PlayOutlinePulse(RoutingGuideOverlay);
        }
    }
}
