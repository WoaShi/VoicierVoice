using System.Windows;
using System.Windows.Media;
using VoiceLockStudio.Services.System;

namespace VoiceLockStudio;

public partial class MainWindow : Window
{
    public MainWindow()
    {
        InitializeComponent();
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
    }
}
