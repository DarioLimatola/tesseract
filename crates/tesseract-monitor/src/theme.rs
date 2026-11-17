//! Colours of the monitor, for light and dark mode.
//!
//! Categorical colours are assigned in a fixed order and never cycled;
//! polarity (positive or negative connection) uses a blue and red pair
//! with a neutral middle; magnitude uses one blue ramp. Text is never
//! painted in a series colour.

use eframe::egui::Color32;

/// The colours one mode uses.
#[derive(Clone, Copy, Debug)]
pub struct Theme {
    /// Background of charts and maps.
    pub surface: Color32,
    /// Main text.
    pub text: Color32,
    /// Secondary text: axis labels, captions.
    pub text_secondary: Color32,
    /// Grid lines and separators.
    pub grid: Color32,
    /// Categorical series colours, in fixed order.
    pub series: [Color32; 3],
    /// Positive connection.
    pub positive: Color32,
    /// Negative connection.
    pub negative: Color32,
    /// Neutral: nothing there.
    pub neutral: Color32,
    /// A dormant part.
    pub dormant: Color32,
    /// Status: good.
    pub good: Color32,
    /// Status: critical.
    pub critical: Color32,
    /// One-hue ramp for magnitude, from "little" to "much".
    pub ramp: [Color32; 5],
}

const fn rgb(hex: u32) -> Color32 {
    Color32::from_rgb((hex >> 16) as u8, (hex >> 8) as u8, hex as u8)
}

impl Theme {
    /// Colours for the light surface.
    pub const LIGHT: Theme = Theme {
        surface: rgb(0xfcfcfb),
        text: rgb(0x0b0b0b),
        text_secondary: rgb(0x52514e),
        grid: rgb(0xe3e2dd),
        series: [rgb(0x2a78d6), rgb(0xeb6834), rgb(0x1baf7a)],
        positive: rgb(0x2a78d6),
        negative: rgb(0xe34948),
        neutral: rgb(0xf0efec),
        dormant: rgb(0xa9a8a1),
        good: rgb(0x0ca30c),
        critical: rgb(0xd03b3b),
        ramp: [
            rgb(0xb7d3f6),
            rgb(0x6da7ec),
            rgb(0x2a78d6),
            rgb(0x184f95),
            rgb(0x0d366b),
        ],
    };

    /// Colours for the dark surface.
    pub const DARK: Theme = Theme {
        surface: rgb(0x1a1a19),
        text: rgb(0xf0efec),
        text_secondary: rgb(0xc3c2b7),
        grid: rgb(0x383835),
        series: [rgb(0x3987e5), rgb(0xd95926), rgb(0x199e70)],
        positive: rgb(0x3987e5),
        negative: rgb(0xe66767),
        neutral: rgb(0x2a2a28),
        dormant: rgb(0x6f6e69),
        good: rgb(0x0ca30c),
        critical: rgb(0xd03b3b),
        ramp: [
            rgb(0x184f95),
            rgb(0x256abf),
            rgb(0x3987e5),
            rgb(0x86b6ef),
            rgb(0xcde2fb),
        ],
    };

    /// The theme for the current mode.
    pub fn of(dark_mode: bool) -> Theme {
        if dark_mode { Self::DARK } else { Self::LIGHT }
    }

    /// Colour of a magnitude between 0 and 1 on the ramp.
    pub fn magnitude(&self, share: f64) -> Color32 {
        let i = (share.clamp(0.0, 1.0) * 4.999) as usize;
        self.ramp[i.min(4)]
    }

    /// A colour moved towards the surface: `keep` of 255 leaves it as is.
    pub fn faded(&self, color: Color32, keep: u8) -> Color32 {
        let mix = |a: u8, b: u8| {
            ((u16::from(a) * u16::from(keep) + u16::from(b) * (255 - u16::from(keep))) / 255) as u8
        };
        Color32::from_rgb(
            mix(color.r(), self.surface.r()),
            mix(color.g(), self.surface.g()),
            mix(color.b(), self.surface.b()),
        )
    }
}

/// A short form of a count: `950`, `12.3k`, `2.56M`.
pub fn compact(value: f64) -> String {
    let a = value.abs();
    if a >= 1e9 {
        format!("{:.2}G", value / 1e9)
    } else if a >= 1e6 {
        format!("{:.2}M", value / 1e6)
    } else if a >= 1e4 {
        format!("{:.1}k", value / 1e3)
    } else if a >= 100.0 || value.fract() == 0.0 {
        format!("{value:.0}")
    } else if a >= 1.0 {
        format!("{value:.1}")
    } else {
        format!("{value:.2}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compact_numbers() {
        assert_eq!(compact(0.0), "0");
        assert_eq!(compact(0.25), "0.25");
        assert_eq!(compact(7.5), "7.5");
        assert_eq!(compact(950.0), "950");
        assert_eq!(compact(12_300.0), "12.3k");
        assert_eq!(compact(2_560_000.0), "2.56M");
    }

    #[test]
    fn ramp_covers_the_range() {
        let t = Theme::LIGHT;
        assert_eq!(t.magnitude(-1.0), t.ramp[0]);
        assert_eq!(t.magnitude(0.0), t.ramp[0]);
        assert_eq!(t.magnitude(1.0), t.ramp[4]);
        assert_eq!(t.magnitude(9.0), t.ramp[4]);
        assert_eq!(t.faded(t.positive, 255), t.positive);
        assert_eq!(t.faded(t.positive, 0), t.surface);
    }
}
