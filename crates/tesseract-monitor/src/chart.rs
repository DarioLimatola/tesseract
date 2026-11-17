//! A small line chart drawn with the window's painter.
//!
//! One y-axis, thin lines, a recessive grid, a legend when there are two
//! or more series, a direct label at the end of each line, dashed
//! reference lines with their own labels, and a crosshair with a tooltip
//! under the pointer.

use eframe::egui::{
    self, Align2, Color32, FontId, Pos2, Rect, Sense, Shape, Stroke, Ui, pos2, vec2,
};

use crate::theme::{Theme, compact};

/// One line of a chart.
pub struct Series {
    /// Name, shown in the legend, at the end of the line and in the tooltip.
    pub name: String,
    /// Colour of the line.
    pub color: Color32,
    /// Points as `[x, y]`, in increasing x.
    pub points: Vec<[f64; 2]>,
}

/// A horizontal reference line, such as a baseline or a threshold.
pub struct RefLine {
    /// Label written on the line.
    pub label: String,
    /// Its y value.
    pub y: f64,
}

/// How values are written.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Unit {
    /// A share between 0 and 1, written as a percentage.
    Percent,
    /// A count, written in compact form.
    Count,
}

/// A chart to draw.
pub struct Chart {
    /// Title, at the top left.
    pub title: String,
    /// Label of the x-axis.
    pub x_label: String,
    /// The lines.
    pub series: Vec<Series>,
    /// Reference lines.
    pub ref_lines: Vec<RefLine>,
    /// How values are written.
    pub unit: Unit,
    /// Largest x shown, when it should extend past the data.
    pub x_max: Option<f64>,
    /// Height in points.
    pub height: f32,
}

fn write(unit: Unit, v: f64) -> String {
    match unit {
        Unit::Percent => format!("{:.1}%", v * 100.0),
        Unit::Count => compact(v),
    }
}

/// A round upper bound for an axis that starts at zero, chosen so that
/// its quarters are round too.
fn nice_max(v: f64) -> f64 {
    if v <= 0.0 || !v.is_finite() {
        return 1.0;
    }
    let mag = 10f64.powf(v.log10().floor());
    for m in [1.0, 2.0, 4.0, 8.0, 10.0] {
        if v <= m * mag {
            return m * mag;
        }
    }
    10.0 * mag
}

impl Chart {
    /// Draws the chart across the available width.
    pub fn show(&self, ui: &mut Ui, theme: &Theme) {
        let size = vec2(ui.available_width().max(160.0), self.height);
        let (response, painter) = ui.allocate_painter(size, Sense::hover());
        let rect = response.rect;
        painter.rect_filled(rect, 6.0, theme.surface);
        let small = FontId::proportional(11.0);
        let body = FontId::proportional(13.0);

        // Title and legend share the top row.
        painter.text(
            rect.left_top() + vec2(10.0, 7.0),
            Align2::LEFT_TOP,
            &self.title,
            body.clone(),
            theme.text,
        );
        if self.series.len() >= 2 {
            let mut x = rect.right() - 10.0;
            for s in self.series.iter().rev() {
                let text = painter.text(
                    pos2(x, rect.top() + 8.0),
                    Align2::RIGHT_TOP,
                    &s.name,
                    small.clone(),
                    theme.text_secondary,
                );
                let swatch = Rect::from_min_size(
                    pos2(text.left() - 14.0, text.center().y - 1.0),
                    vec2(10.0, 2.0),
                );
                painter.rect_filled(swatch, 1.0, s.color);
                x = swatch.left() - 12.0;
            }
        }

        let plot = Rect::from_min_max(
            rect.left_top() + vec2(48.0, 30.0),
            rect.right_bottom() - vec2(92.0, 24.0),
        );
        if plot.width() < 20.0 || plot.height() < 20.0 {
            return;
        }

        // Ranges: x from zero, y from zero (one axis, anchored).
        let data_x = self
            .series
            .iter()
            .flat_map(|s| s.points.iter().map(|p| p[0]))
            .fold(0.0f64, f64::max);
        let x_max = self.x_max.unwrap_or(0.0).max(data_x).max(1e-9);
        let y_max = match self.unit {
            Unit::Percent => 1.0,
            Unit::Count => nice_max(
                self.series
                    .iter()
                    .flat_map(|s| s.points.iter().map(|p| p[1]))
                    .chain(self.ref_lines.iter().map(|r| r.y))
                    .fold(0.0f64, f64::max),
            ),
        };
        let to_screen = |x: f64, y: f64| -> Pos2 {
            pos2(
                plot.left() + (x / x_max) as f32 * plot.width(),
                plot.bottom() - (y / y_max).clamp(0.0, 1.0) as f32 * plot.height(),
            )
        };

        // Grid and axis labels.
        for i in 0..=4 {
            let y = y_max * f64::from(i) / 4.0;
            let p = to_screen(0.0, y);
            painter.hline(plot.x_range(), p.y, Stroke::new(1.0, theme.grid));
            painter.text(
                pos2(plot.left() - 6.0, p.y),
                Align2::RIGHT_CENTER,
                write(self.unit, y),
                small.clone(),
                theme.text_secondary,
            );
        }
        painter.text(
            pos2(plot.left(), plot.bottom() + 5.0),
            Align2::LEFT_TOP,
            "0",
            small.clone(),
            theme.text_secondary,
        );
        painter.text(
            pos2(plot.right(), plot.bottom() + 5.0),
            Align2::RIGHT_TOP,
            format!("{} {}", compact(x_max), self.x_label),
            small.clone(),
            theme.text_secondary,
        );

        // Reference lines: dashed, labelled on the line. Labels of lines
        // that are close together are written side by side.
        let mut refs: Vec<&RefLine> = self.ref_lines.iter().collect();
        refs.sort_by(|a, b| a.y.total_cmp(&b.y));
        let mut previous: Option<Rect> = None;
        for r in refs {
            let a = to_screen(0.0, r.y);
            let b = pos2(plot.right(), a.y);
            painter.extend(Shape::dashed_line(
                &[a, b],
                Stroke::new(1.0, theme.text_secondary),
                5.0,
                4.0,
            ));
            let x = match previous {
                Some(p) if (p.bottom() - (a.y - 2.0)).abs() < 13.0 => p.right() + 12.0,
                _ => plot.left() + 4.0,
            };
            previous = Some(painter.text(
                pos2(x, a.y - 2.0),
                Align2::LEFT_BOTTOM,
                format!("{} {}", r.label, write(self.unit, r.y)),
                small.clone(),
                theme.text_secondary,
            ));
        }

        // Lines.
        for s in &self.series {
            let points: Vec<Pos2> = s.points.iter().map(|p| to_screen(p[0], p[1])).collect();
            match points.len() {
                0 => {}
                1 => {
                    painter.circle_filled(points[0], 4.0, s.color);
                }
                _ => {
                    painter.line(points, Stroke::new(2.0, s.color));
                }
            }
        }

        // Direct labels at the line ends, pushed apart when they collide.
        let mut ends: Vec<(f32, &Series, f64)> = self
            .series
            .iter()
            .filter_map(|s| s.points.last().map(|p| (to_screen(p[0], p[1]).y, s, p[1])))
            .collect();
        ends.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut last_y = f32::NEG_INFINITY;
        for (y, s, value) in ends {
            let y = y.max(last_y + 13.0).min(rect.bottom() - 8.0);
            last_y = y;
            let label = if self.series.len() >= 2 {
                format!("{} {}", s.name, write(self.unit, value))
            } else {
                write(self.unit, value)
            };
            painter.text(
                pos2(plot.right() + 6.0, y),
                Align2::LEFT_CENTER,
                label,
                small.clone(),
                theme.text,
            );
        }

        // Hover: crosshair at the nearest x, markers, and a tooltip.
        let Some(pointer) = response
            .hover_pos()
            .filter(|p| plot.expand(4.0).contains(*p))
        else {
            return;
        };
        let wanted = f64::from((pointer.x - plot.left()) / plot.width()) * x_max;
        let nearest = self
            .series
            .iter()
            .flat_map(|s| s.points.iter().map(|p| p[0]))
            .min_by(|a, b| (a - wanted).abs().total_cmp(&(b - wanted).abs()));
        let Some(x) = nearest else {
            return;
        };
        let sx = to_screen(x, 0.0).x;
        painter.vline(sx, plot.y_range(), Stroke::new(1.0, theme.text_secondary));
        let mut lines = vec![format!("{} {}", compact(x), self.x_label)];
        for s in &self.series {
            if let Some(p) = s.points.iter().find(|p| p[0] == x) {
                let at = to_screen(p[0], p[1]);
                painter.circle_filled(at, 6.0, theme.surface);
                painter.circle_filled(at, 4.0, s.color);
                lines.push(format!("{}: {}", s.name, write(self.unit, p[1])));
            }
        }
        for r in &self.ref_lines {
            lines.push(format!("{}: {}", r.label, write(self.unit, r.y)));
        }
        response.on_hover_ui_at_pointer(|ui| {
            for (i, line) in lines.iter().enumerate() {
                if i == 0 {
                    ui.label(egui::RichText::new(line).strong());
                } else {
                    ui.label(line);
                }
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nice_upper_bounds() {
        assert_eq!(nice_max(0.0), 1.0);
        assert!((nice_max(0.7) - 0.8).abs() < 1e-12);
        assert_eq!(nice_max(3.0), 4.0);
        assert_eq!(nice_max(2560.0), 4000.0);
        assert_eq!(nice_max(2_560_000.0), 4_000_000.0);
        assert_eq!(nice_max(f64::NAN), 1.0);
    }

    #[test]
    fn values_are_written_by_unit() {
        assert_eq!(write(Unit::Percent, 0.9953), "99.5%");
        assert_eq!(write(Unit::Count, 4096.0), "4096");
    }
}
