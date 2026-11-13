using System.Globalization;
using System.Windows;
using System.Windows.Data;

namespace VoiceLockStudio.Converters;

public sealed class BoolToVisibilityConverter : IValueConverter
{
    public object Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
    {
        bool flag = value is bool b && b;
        bool invert = string.Equals(parameter as string, "Inverse", StringComparison.OrdinalIgnoreCase);
        if (invert)
        {
            flag = !flag;
        }

        return flag ? Visibility.Visible : Visibility.Collapsed;
    }

    public object ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture)
    {
        bool visible = value is Visibility v && v == Visibility.Visible;
        bool invert = string.Equals(parameter as string, "Inverse", StringComparison.OrdinalIgnoreCase);
        return invert ? !visible : visible;
    }
}
