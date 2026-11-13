using System.Globalization;
using System.Windows.Data;
using VoiceLockStudio.Models;

namespace VoiceLockStudio.Converters;

public sealed class VoxLockVariantLabelConverter : IValueConverter
{
    public object Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
    {
        if (value is VoxLockModelVariant variant)
        {
            return variant switch
            {
                VoxLockModelVariant.Int8 => "INT8 (推荐 · 平衡)",
                VoxLockModelVariant.Int8Full => "INT8 Full (极速 · 最小算力)",
                VoxLockModelVariant.Fp16 => "FP16 (高精度)",
                _ => variant.ToString()
            };
        }

        return value?.ToString() ?? string.Empty;
    }

    public object ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture)
    {
        return Binding.DoNothing;
    }
}
