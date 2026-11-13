using System.Globalization;
using System.Windows.Data;

namespace VoiceLockStudio.Converters;

/// <summary>
/// Converts between an enum value and a boolean by comparing against <c>ConverterParameter</c>.
/// Supports either an enum instance or an enum member name string as <c>ConverterParameter</c>.
/// </summary>
public sealed class EnumToBoolConverter : IValueConverter
{
    public object Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
    {
        if (value is null || parameter is null)
        {
            return false;
        }

        if (parameter is string paramName)
        {
            Type valueType = value.GetType();
            if (valueType.IsEnum && Enum.TryParse(valueType, paramName, ignoreCase: true, out object? parsed))
            {
                return Equals(value, parsed);
            }

            return string.Equals(value.ToString(), paramName, StringComparison.OrdinalIgnoreCase);
        }

        return Equals(value, parameter);
    }

    public object ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture)
    {
        if (value is not true || parameter is null)
        {
            return Binding.DoNothing;
        }

        if (parameter is string paramName)
        {
            Type underlying = Nullable.GetUnderlyingType(targetType) ?? targetType;
            if (underlying.IsEnum && Enum.TryParse(underlying, paramName, ignoreCase: true, out object? parsed))
            {
                return parsed;
            }

            return Binding.DoNothing;
        }

        return parameter;
    }
}
