using System.Globalization;
using System.Windows;
using System.Windows.Data;
using System.Windows.Media;
using VoiceLockStudio.ViewModels;

namespace VoiceLockStudio.Converters;

public sealed class StatusKindToBrushConverter : IValueConverter
{
    public object Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
    {
        PipelineStatusKind kind = value is PipelineStatusKind k ? k : PipelineStatusKind.Idle;
        bool isBackground = string.Equals(parameter as string, "Background", StringComparison.OrdinalIgnoreCase);

        string resourceKey = (kind, isBackground) switch
        {
            (PipelineStatusKind.Running, true) => "SystemFillColorSuccessBackgroundBrush",
            (PipelineStatusKind.Running, false) => "SystemFillColorSuccessBrush",
            (PipelineStatusKind.Sleeping, true) => "SystemFillColorCautionBackgroundBrush",
            (PipelineStatusKind.Sleeping, false) => "SystemFillColorCautionBrush",
            (PipelineStatusKind.Faulted, true) => "SystemFillColorCriticalBackgroundBrush",
            (PipelineStatusKind.Faulted, false) => "SystemFillColorCriticalBrush",
            (_, true) => "SystemFillColorNeutralBackgroundBrush",
            (_, false) => "SystemFillColorNeutralBrush"
        };

        if (Application.Current?.TryFindResource(resourceKey) is Brush brush)
        {
            return brush;
        }

        return isBackground ? Brushes.Transparent : Brushes.Gray;
    }

    public object ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture)
    {
        return Binding.DoNothing;
    }
}
