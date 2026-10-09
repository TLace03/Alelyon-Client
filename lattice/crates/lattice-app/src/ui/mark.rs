//! The Lattice mark: a diamond of four panes with the top pane lit, drawn on a
//! small canvas. The geometry is `MARK` in `lattice_desktop/src/brand/mark.ts`
//! (1024-unit icon canvas), which is the only other copy.

use iced::widget::canvas::{
    self, Cache, Frame, Geometry, Gradient, LineCap, LineJoin, Path, Stroke, Style, gradient,
};
use iced::{Color, Point, Rectangle, mouse};

use crate::app::Message;
use crate::perf;

/// The mark's own bounds in icon units, with the padding `markViewBox` adds.
const LEFT: f32 = 222.0 - 26.0 - 8.0;
const TOP: f32 = 232.0 - 26.0 - 8.0;
const SIDE: f32 = 632.0 + 16.0;

const OUTLINE: [(f32, f32); 4] = [
    (512.0, 232.0),
    (802.0, 522.0),
    (512.0, 812.0),
    (222.0, 522.0),
];
const STRIPS: [((f32, f32), (f32, f32)); 2] = [
    ((367.0, 377.0), (657.0, 667.0)),
    ((657.0, 377.0), (367.0, 667.0)),
];
const PANE: [(f32, f32); 4] = [
    (512.0, 232.0),
    (657.0, 377.0),
    (512.0, 522.0),
    (367.0, 377.0),
];
const FACET: [(f32, f32); 3] = [(512.0, 232.0), (512.0, 522.0), (367.0, 377.0)];
const STROKE: f32 = 52.0;

const STRIP_TOP: Color = Color::from_rgb8(0xf6, 0xdd, 0x9c);
const STRIP: Color = Color::from_rgb8(0xe6, 0xc4, 0x6a);
const STRIP_BOTTOM: Color = Color::from_rgb8(0xb0, 0x88, 0x38);
const PANE_FILL: Color = Color::from_rgb8(0xe9, 0xc9, 0x79);
const FACET_FILL: Color = Color::from_rgb8(0xf8, 0xe4, 0xab);

pub struct Mark<'a> {
    pub cache: &'a Cache,
}

fn polygon(points: &[(f32, f32)], scale: f32) -> Path {
    Path::new(|b| {
        for (i, (x, y)) in points.iter().enumerate() {
            let p = Point::new((x - LEFT) * scale, (y - TOP) * scale);
            if i == 0 {
                b.move_to(p);
            } else {
                b.line_to(p);
            }
        }
        b.close();
    })
}

impl canvas::Program<Message> for Mark<'_> {
    type State = ();

    fn draw(
        &self,
        _state: &(),
        renderer: &iced::Renderer,
        _theme: &iced::Theme,
        bounds: Rectangle,
        _cursor: mouse::Cursor,
    ) -> Vec<Geometry> {
        let side = bounds.width.min(bounds.height);
        let geometry = self
            .cache
            .draw(renderer, bounds.size(), |frame: &mut Frame| {
                perf::cache_rebuilt();
                let scale = side / SIDE;
                frame.fill(&polygon(&PANE, scale), PANE_FILL);
                frame.fill(&polygon(&FACET, scale), FACET_FILL);
                let top = (232.0 - 26.0 - TOP) * scale;
                let bottom = (812.0 + 26.0 - TOP) * scale;
                let gold: Gradient =
                    gradient::Linear::new(Point::new(0.0, top), Point::new(0.0, bottom))
                        .add_stop(0.0, STRIP_TOP)
                        .add_stop(0.5, STRIP)
                        .add_stop(1.0, STRIP_BOTTOM)
                        .into();
                let stroke = Stroke {
                    style: Style::Gradient(gold),
                    width: STROKE * scale,
                    line_cap: LineCap::Round,
                    line_join: LineJoin::Round,
                    ..Stroke::default()
                };
                frame.stroke(&polygon(&OUTLINE, scale), stroke);
                let strips = Path::new(|b| {
                    for ((x1, y1), (x2, y2)) in STRIPS {
                        b.move_to(Point::new((x1 - LEFT) * scale, (y1 - TOP) * scale));
                        b.line_to(Point::new((x2 - LEFT) * scale, (y2 - TOP) * scale));
                    }
                });
                frame.stroke(&strips, stroke);
            });
        vec![geometry]
    }
}
