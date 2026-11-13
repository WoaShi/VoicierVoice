using System.Windows;
using System.Windows.Automation;
using System.Windows.Controls;

namespace VoiceLockStudio.Controls;

public partial class InfoHintBadge : UserControl
{
    public static readonly DependencyProperty HintTextProperty = DependencyProperty.Register(
        nameof(HintText),
        typeof(string),
        typeof(InfoHintBadge),
        new PropertyMetadata(string.Empty, OnHintTextChanged));

    private ToolTip? _hintToolTip;
    private TextBlock? _hintTextBlock;

    public InfoHintBadge()
    {
        InitializeComponent();
        _hintToolTip = BadgeSurface.Resources["HintToolTipResource"] as ToolTip;
        _hintTextBlock = _hintToolTip?.Content as TextBlock;
        ApplyHintState(HintText);
    }

    public string HintText
    {
        get => (string)GetValue(HintTextProperty);
        set => SetValue(HintTextProperty, value);
    }

    private static void OnHintTextChanged(DependencyObject d, DependencyPropertyChangedEventArgs e)
    {
        if (d is InfoHintBadge badge)
        {
            badge.ApplyHintState(e.NewValue as string);
        }
    }

    private void ApplyHintState(string? text)
    {
        string trimmed = text?.Trim() ?? string.Empty;
        bool hasHint = trimmed.Length > 0;

        Visibility = hasHint ? Visibility.Visible : Visibility.Collapsed;
        AutomationProperties.SetName(BadgeSurface, trimmed);

        if (_hintTextBlock is not null)
        {
            _hintTextBlock.Text = trimmed;
        }

        BadgeSurface.ToolTip = hasHint ? _hintToolTip : null;
    }
}
