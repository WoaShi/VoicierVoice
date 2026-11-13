using System.Windows;
using System.Windows.Controls;
using System.Windows.Media.Animation;

namespace VoiceLockStudio.Controls;

/// <summary>
/// Plays a smooth Fluent fade-in / pulse / fade-out outline highlight over a target card
/// or control when triggered by a Quick Start guidance button.
/// </summary>
public static class GuidanceAnimationHelper
{
    public static void PlayOutlinePulse(Border? overlayBorder)
    {
        if (overlayBorder is null)
        {
            return;
        }

        overlayBorder.BeginAnimation(UIElement.OpacityProperty, null);

        var animation = new DoubleAnimationUsingKeyFrames();
        var ease = new SineEase { EasingMode = EasingMode.EaseInOut };

        animation.KeyFrames.Add(new EasingDoubleKeyFrame(0.0, KeyTime.FromTimeSpan(TimeSpan.Zero)));
        animation.KeyFrames.Add(new EasingDoubleKeyFrame(1.0, KeyTime.FromTimeSpan(TimeSpan.FromMilliseconds(260)), ease));
        animation.KeyFrames.Add(new EasingDoubleKeyFrame(0.35, KeyTime.FromTimeSpan(TimeSpan.FromMilliseconds(750)), ease));
        animation.KeyFrames.Add(new EasingDoubleKeyFrame(1.0, KeyTime.FromTimeSpan(TimeSpan.FromMilliseconds(1220)), ease));
        animation.KeyFrames.Add(new EasingDoubleKeyFrame(0.35, KeyTime.FromTimeSpan(TimeSpan.FromMilliseconds(1700)), ease));
        animation.KeyFrames.Add(new EasingDoubleKeyFrame(0.0, KeyTime.FromTimeSpan(TimeSpan.FromMilliseconds(2300)), ease));

        overlayBorder.BeginAnimation(UIElement.OpacityProperty, animation);
    }
}
