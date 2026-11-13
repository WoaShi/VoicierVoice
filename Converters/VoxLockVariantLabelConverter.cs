using System.Globalization;
using System.Windows.Data;
using VoiceLockStudio.Models;
using VoiceLockStudio.Services.Localization;

namespace VoiceLockStudio.Converters;

/// <summary>
/// Formats <see cref="VoxLockModelVariant"/> values via <see cref="AppStrings.Current"/>.
/// Supports both single-value binding and MultiBinding (variant + <see cref="AppStrings.Language"/>)
/// so ComboBox selections refresh immediately upon runtime language switches.
/// </summary>
public sealed class VoxLockVariantLabelConverter : IValueConverter, IMultiValueConverter
{
    public object Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
    {
        if (value is VoxLockModelVariant variant)
        {
            return AppStrings.Current.FormatVariantLabel(variant);
        }

        return value?.ToString() ?? string.Empty;
    }

    public object ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture)
    {
        return Binding.DoNothing;
    }

    public object Convert(object[] values, Type targetType, object? parameter, CultureInfo culture)
    {
        if (values is { Length: > 0 } && values[0] is VoxLockModelVariant variant)
        {
            return AppStrings.Current.FormatVariantLabel(variant);
        }

        return values is { Length: > 0 } ? values[0]?.ToString() ?? string.Empty : string.Empty;
    }

    public object[] ConvertBack(object value, Type[] targetTypes, object? parameter, CultureInfo culture)
    {
        return [Binding.DoNothing];
    }
}
