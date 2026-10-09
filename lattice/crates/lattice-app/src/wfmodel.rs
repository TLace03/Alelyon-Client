//! The waterfall's rows and the arithmetic around them: geometry, scrolling,
//! virtualization, hit testing and keyboard navigation.
//!
//! All of it is pure and in logical pixels, so the canvas that draws it and the
//! tests that check it use the same functions.
//!
//! Invariants:
//! - A row's text is computed once when the tree changes ([`build_rows`]); the
//!   canvas only draws it.
//! - Virtualization: [`visible_range`] is exactly the rows that intersect the
//!   viewport (plus none), so drawing cost depends on the window's height, not
//!   on the number of spans.
//! - Scrolling never leaves the content: [`clamp_scroll`] keeps the offset in
//!   `0..=max_scroll`, and [`ensure_visible`] moves it the least distance that
//!   brings a row fully into view.
//! - Keyboard navigation follows the tree view convention: Up/Down move
//!   through the visible rows; Left collapses an open row, else goes to the
//!   parent; Right expands a closed row, else goes to the first child.

use std::ops::Range;

use crate::runstate::RunState;
use crate::spans::{Row, SpanKind, guardrail_triggered, kind_of, title_of};

pub const ROW_H: f32 = 28.0;
pub const HEADER_H: f32 = 30.0;
pub const INDENT: f32 = 14.0;
pub const SCROLLBAR_W: f32 = 10.0;
/// The gap between the tree column and the bar track.
pub const TRACK_GAP: f32 = 14.0;

/// One visible row, with everything the canvas draws already worked out.
#[derive(Clone, Debug, PartialEq)]
pub struct WfRow {
    pub span_id: String,
    pub depth: u16,
    pub has_children: bool,
    pub collapsed: bool,
    pub kind: SpanKind,
    pub title: String,
    /// Seconds since the epoch.
    pub start: f64,
    /// `None`: still open, in a run that is working: drawn to "now".
    pub end: Option<f64>,
    pub error: bool,
    pub triggered: bool,
}

/// The visible rows of `state`'s spans. `rows` came from
/// [`crate::spans::visible_rows`] over `state.spans` and its tree.
pub fn build_rows(state: &RunState, rows: &[Row]) -> Vec<WfRow> {
    rows.iter()
        .map(|row| {
            let entry = &state.spans[row.index];
            let kind = kind_of(&entry.rec);
            WfRow {
                span_id: entry.rec.id.clone(),
                depth: row.depth,
                has_children: row.has_children,
                collapsed: row.collapsed,
                kind,
                title: title_of(&entry.rec),
                start: entry.start,
                end: state.display_end(entry),
                error: entry.rec.error.is_some(),
                triggered: kind == SpanKind::Guardrail && guardrail_triggered(&entry.rec),
            }
        })
        .collect()
}

/// Where the columns fall for a canvas `width` pixels wide.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Metrics {
    pub tree_w: f32,
    pub track_x: f32,
    pub track_w: f32,
}

impl Metrics {
    pub fn new(width: f32) -> Self {
        let tree_w = if width < 460.0 {
            (width * 0.5).max(96.0)
        } else {
            (width * 0.42).clamp(190.0, 420.0)
        };
        let track_x = tree_w + TRACK_GAP;
        let track_w = (width - track_x - SCROLLBAR_W - 10.0).max(40.0);
        Self {
            tree_w,
            track_x,
            track_w,
        }
    }
}

/// Where a row's chevron is centred, from the left of the canvas.
pub fn chevron_x(depth: u16) -> f32 {
    12.0 + f32::from(depth) * INDENT
}

/// True when a click at `x` lands on the chevron of a row of `depth`.
pub fn on_chevron(x: f32, depth: u16) -> bool {
    (x - chevron_x(depth)).abs() <= 9.0
}

pub fn content_height(rows: usize) -> f32 {
    rows as f32 * ROW_H
}

/// The height available to rows, under the axis header.
pub fn viewport_height(canvas_height: f32) -> f32 {
    (canvas_height - HEADER_H).max(0.0)
}

pub fn max_scroll(rows: usize, canvas_height: f32) -> f32 {
    (content_height(rows) - viewport_height(canvas_height)).max(0.0)
}

pub fn clamp_scroll(scroll: f32, rows: usize, canvas_height: f32) -> f32 {
    if scroll.is_nan() {
        0.0
    } else {
        scroll.clamp(0.0, max_scroll(rows, canvas_height))
    }
}

/// The shortest the scrollbar's thumb gets.
pub const MIN_THUMB: f32 = 28.0;

fn thumb_height(rows: usize, canvas_height: f32) -> f32 {
    let view = viewport_height(canvas_height);
    let content = content_height(rows);
    (view * view / content.max(1.0)).clamp(MIN_THUMB.min(view), view.max(0.0))
}

/// The scrollbar thumb as `(top, height)` in canvas coordinates, or `None` when
/// every row fits and there is nothing to scroll.
pub fn thumb(scroll: f32, rows: usize, canvas_height: f32) -> Option<(f32, f32)> {
    let view = viewport_height(canvas_height);
    if view <= 0.0 || content_height(rows) <= view {
        return None;
    }
    let height = thumb_height(rows, canvas_height);
    let fraction = (scroll / max_scroll(rows, canvas_height)).clamp(0.0, 1.0);
    Some((HEADER_H + (view - height) * fraction, height))
}

/// The scroll offset that puts the thumb's top at `top` (canvas coordinates):
/// what dragging the thumb means.
pub fn scroll_from_thumb_top(top: f32, rows: usize, canvas_height: f32) -> f32 {
    let view = viewport_height(canvas_height);
    if view <= 0.0 || content_height(rows) <= view {
        return 0.0;
    }
    let travel = (view - thumb_height(rows, canvas_height)).max(1e-3);
    ((top - HEADER_H) / travel).clamp(0.0, 1.0) * max_scroll(rows, canvas_height)
}

/// The rows that intersect the viewport: nothing else is drawn.
pub fn visible_range(scroll: f32, canvas_height: f32, rows: usize) -> Range<usize> {
    let view = viewport_height(canvas_height);
    let first = (scroll.max(0.0) / ROW_H).floor() as usize;
    let last = (((scroll.max(0.0) + view) / ROW_H).ceil() as usize).min(rows);
    first.min(rows)..last.max(first.min(rows))
}

/// The row under a point `y` pixels from the canvas's top, if any.
pub fn row_at(y: f32, scroll: f32, rows: usize) -> Option<usize> {
    if y < HEADER_H {
        return None;
    }
    let index = ((y - HEADER_H + scroll) / ROW_H).floor();
    (index >= 0.0 && (index as usize) < rows).then_some(index as usize)
}

/// The smallest scroll change that brings `row` fully into view.
pub fn ensure_visible(scroll: f32, row: usize, rows: usize, canvas_height: f32) -> f32 {
    let top = row as f32 * ROW_H;
    let bottom = top + ROW_H;
    let view = viewport_height(canvas_height);
    if view <= 0.0 {
        // The canvas has not reported its size yet: there is no viewport to bring
        // the row into, and guessing would scroll the first row under the header.
        return scroll;
    }
    let target = if top < scroll {
        top
    } else if bottom > scroll + view {
        bottom - view
    } else {
        scroll
    };
    clamp_scroll(target, rows, canvas_height)
}

/// A key the waterfall understands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Nav {
    Up,
    Down,
    Left,
    Right,
    Home,
    End,
    PageUp,
    PageDown,
}

/// What a key does.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NavResult {
    /// Select the row at this index.
    Select(usize),
    /// Collapse or expand this span.
    Toggle(String),
    Nothing,
}

/// The effect of `key` on a list of visible `rows` with `selected` chosen;
/// `page` is how many rows fit in the viewport.
pub fn navigate(rows: &[WfRow], selected: Option<usize>, key: Nav, page: usize) -> NavResult {
    if rows.is_empty() {
        return NavResult::Nothing;
    }
    let last = rows.len() - 1;
    let page = page.max(1);
    let Some(current) = selected.filter(|&i| i <= last) else {
        return match key {
            Nav::Up | Nav::End | Nav::PageUp => NavResult::Select(last),
            _ => NavResult::Select(0),
        };
    };
    let row = &rows[current];
    match key {
        Nav::Up => NavResult::Select(current.saturating_sub(1)),
        Nav::Down => NavResult::Select((current + 1).min(last)),
        Nav::Home => NavResult::Select(0),
        Nav::End => NavResult::Select(last),
        Nav::PageUp => NavResult::Select(current.saturating_sub(page)),
        Nav::PageDown => NavResult::Select((current + page).min(last)),
        Nav::Left => {
            if row.has_children && !row.collapsed {
                NavResult::Toggle(row.span_id.clone())
            } else {
                (0..current)
                    .rev()
                    .find(|&i| rows[i].depth < row.depth)
                    .map_or(NavResult::Nothing, NavResult::Select)
            }
        }
        Nav::Right => {
            if row.has_children && row.collapsed {
                NavResult::Toggle(row.span_id.clone())
            } else if row.has_children && current < last {
                NavResult::Select(current + 1)
            } else {
                NavResult::Nothing
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::demo::DemoService;
    use crate::spans::{SpanTree, visible_rows};
    use lattice_protocol::RunService;
    use std::collections::HashSet;

    fn row(id: &str, depth: u16, has_children: bool, collapsed: bool) -> WfRow {
        WfRow {
            span_id: id.into(),
            depth,
            has_children,
            collapsed,
            kind: SpanKind::Agent,
            title: id.into(),
            start: 0.0,
            end: Some(1.0),
            error: false,
            triggered: false,
        }
    }

    /// task > (agent > (turn > tool), turn2)
    fn sample() -> Vec<WfRow> {
        vec![
            row("task", 0, true, false),
            row("agent", 1, true, false),
            row("turn", 2, true, false),
            row("tool", 3, false, false),
            row("turn2", 2, false, false),
        ]
    }

    #[test]
    fn rows_of_a_demo_run_carry_kind_title_and_error_flags() {
        let service = DemoService::instant();
        let failed = service.runs()[1].clone();
        let state = RunState::from_detail(service.run(&failed.id).unwrap());
        let tree = SpanTree::build(&state.spans);
        let rows = visible_rows(&state.spans, &tree, &HashSet::new());
        let wf = build_rows(&state, &rows);
        assert_eq!(wf.len(), state.spans.len());
        assert_eq!(wf[0].kind, SpanKind::Task);
        assert_eq!(wf[0].title, "Task");
        assert_eq!(wf[0].depth, 0);
        assert!(
            wf.iter().any(|r| r.error),
            "the failed run shows an error row"
        );
        assert!(
            wf.iter().all(|r| r.end.is_some()),
            "a finished run has no open row"
        );
        assert!(
            wf.iter()
                .any(|r| r.kind == SpanKind::Model && !r.title.is_empty())
        );
    }

    #[test]
    fn a_refused_run_shows_its_triggered_guardrail() {
        let service = DemoService::instant();
        let refused = service.runs()[2].clone();
        let state = RunState::from_detail(service.run(&refused.id).unwrap());
        let tree = SpanTree::build(&state.spans);
        let wf = build_rows(&state, &visible_rows(&state.spans, &tree, &HashSet::new()));
        let guard = wf.iter().find(|r| r.kind == SpanKind::Guardrail).unwrap();
        assert!(guard.triggered && guard.error);
    }

    #[test]
    fn columns_adapt_to_a_narrow_canvas() {
        let wide = Metrics::new(1000.0);
        assert!((wide.tree_w - 420.0).abs() < 1e-3, "{wide:?}");
        assert!(wide.track_x > wide.tree_w && wide.track_w > 300.0);
        let mid = Metrics::new(700.0);
        assert!((mid.tree_w - 294.0).abs() < 1e-3);
        // The centre column can be only ~264 px wide with the detail column open at 1000 px.
        let narrow = Metrics::new(264.0);
        assert!(narrow.track_w >= 40.0);
        assert!(narrow.tree_w < 264.0);
        for width in [0.0, 50.0, 200.0, 459.0, 460.0, 4000.0] {
            let m = Metrics::new(width);
            assert!(m.track_w >= 40.0 && m.tree_w >= 96.0 && m.track_x.is_finite());
        }
    }

    #[test]
    fn only_the_rows_that_intersect_the_viewport_are_drawn() {
        // 1000 rows in a canvas 330 px tall: 300 px of rows = 10.7 rows.
        assert_eq!(visible_range(0.0, 330.0, 1000), 0..11);
        // Scrolled half a row: row 0 still touches the viewport top, row 11 now touches the bottom.
        assert_eq!(visible_range(14.0, 330.0, 1000), 0..12);
        assert_eq!(visible_range(28.0, 330.0, 1000), 1..12);
        // Never past the end, never empty for a non-empty list.
        assert_eq!(visible_range(0.0, 330.0, 5), 0..5);
        assert_eq!(visible_range(10_000.0, 330.0, 5), 5..5);
        assert_eq!(
            visible_range(0.0, 10.0, 5),
            0..0,
            "a canvas shorter than the header shows no rows"
        );
        // The number drawn depends on the window, not on the span count.
        for rows in [100usize, 10_000, 1_000_000] {
            assert!(visible_range(1234.0, 630.0, rows).len() <= 22);
        }
    }

    #[test]
    fn scrolling_stays_inside_the_content() {
        assert_eq!(max_scroll(10, 330.0), 0.0, "ten rows fit");
        assert_eq!(max_scroll(100, 330.0), 100.0 * ROW_H - 300.0);
        assert_eq!(clamp_scroll(-5.0, 100, 330.0), 0.0);
        assert_eq!(clamp_scroll(1e9, 100, 330.0), max_scroll(100, 330.0));
        assert_eq!(clamp_scroll(f32::NAN, 100, 330.0), 0.0);
        assert_eq!(
            clamp_scroll(50.0, 3, 330.0),
            0.0,
            "content shorter than the view never scrolls"
        );
    }

    #[test]
    fn the_thumb_tracks_the_scroll_and_dragging_inverts_it() {
        assert_eq!(thumb(0.0, 5, 330.0), None, "everything fits: no thumb");
        let rows = 200;
        let canvas = 630.0; // 600 px of rows
        let (top, height) = thumb(0.0, rows, canvas).unwrap();
        assert_eq!(top, HEADER_H, "at the top of the track");
        assert!((MIN_THUMB..600.0).contains(&height));
        let (bottom_top, bottom_height) = thumb(max_scroll(rows, canvas), rows, canvas).unwrap();
        assert!(
            (bottom_top + bottom_height - canvas).abs() < 1e-3,
            "at the bottom of the track"
        );
        for scroll in [0.0, 100.0, 1234.0, max_scroll(rows, canvas)] {
            let (top, _) = thumb(scroll, rows, canvas).unwrap();
            assert!(
                (scroll_from_thumb_top(top, rows, canvas) - scroll).abs() < 0.5,
                "{scroll}"
            );
        }
        // Dragging past either end stays inside the content.
        assert_eq!(scroll_from_thumb_top(-500.0, rows, canvas), 0.0);
        assert_eq!(
            scroll_from_thumb_top(1e6, rows, canvas),
            max_scroll(rows, canvas)
        );
        assert_eq!(scroll_from_thumb_top(50.0, 3, canvas), 0.0);
        // A very long list still has a grabbable thumb.
        assert!(thumb(0.0, 1_000_000, canvas).unwrap().1 >= MIN_THUMB);
    }

    #[test]
    fn rows_are_hit_below_the_header_and_within_the_list() {
        assert_eq!(row_at(10.0, 0.0, 10), None, "the header is not a row");
        assert_eq!(row_at(HEADER_H + 1.0, 0.0, 10), Some(0));
        assert_eq!(row_at(HEADER_H + ROW_H + 1.0, 0.0, 10), Some(1));
        assert_eq!(row_at(HEADER_H + 1.0, ROW_H * 3.0, 10), Some(3));
        assert_eq!(
            row_at(HEADER_H + ROW_H * 10.5, 0.0, 10),
            None,
            "below the last row"
        );
    }

    #[test]
    fn the_chevron_is_a_small_target_that_moves_with_depth() {
        assert!(on_chevron(chevron_x(0), 0));
        assert!(on_chevron(chevron_x(3) + 8.0, 3));
        assert!(!on_chevron(chevron_x(3) + 30.0, 3));
        assert!(
            !on_chevron(chevron_x(0), 2),
            "a click at depth 0's chevron is not depth 2's"
        );
    }

    #[test]
    fn an_unknown_viewport_never_scrolls() {
        // Before the canvas reports its size its height is 0: selecting the first
        // row at launch (`--select-latest`) must leave the list at the top.
        assert_eq!(ensure_visible(0.0, 0, 30, 0.0), 0.0);
        assert_eq!(ensure_visible(0.0, 5, 30, 0.0), 0.0);
        assert_eq!(
            ensure_visible(0.0, 5, 30, HEADER_H),
            0.0,
            "a canvas as tall as its header has no rows either"
        );
    }

    #[test]
    fn keyboard_selection_scrolls_the_minimum() {
        let canvas = 330.0; // 300 px = 10.7 rows
        assert_eq!(ensure_visible(0.0, 3, 100, canvas), 0.0, "already visible");
        assert_eq!(
            ensure_visible(0.0, 10, 100, canvas),
            11.0 * ROW_H - 300.0,
            "one row below: scroll just enough"
        );
        assert_eq!(
            ensure_visible(500.0, 2, 100, canvas),
            2.0 * ROW_H,
            "above: bring its top to the top"
        );
        assert_eq!(
            ensure_visible(0.0, 99, 100, canvas),
            max_scroll(100, canvas)
        );
    }

    #[test]
    fn up_and_down_move_through_visible_rows_and_stop_at_the_ends() {
        let rows = sample();
        assert_eq!(navigate(&rows, Some(1), Nav::Down, 5), NavResult::Select(2));
        assert_eq!(navigate(&rows, Some(1), Nav::Up, 5), NavResult::Select(0));
        assert_eq!(navigate(&rows, Some(0), Nav::Up, 5), NavResult::Select(0));
        assert_eq!(navigate(&rows, Some(4), Nav::Down, 5), NavResult::Select(4));
        assert_eq!(
            navigate(&rows, None, Nav::Down, 5),
            NavResult::Select(0),
            "nothing selected: Down picks the first row"
        );
        assert_eq!(
            navigate(&rows, None, Nav::Up, 5),
            NavResult::Select(4),
            "nothing selected: Up picks the last row"
        );
        assert_eq!(navigate(&rows, Some(2), Nav::Home, 5), NavResult::Select(0));
        assert_eq!(navigate(&rows, Some(2), Nav::End, 5), NavResult::Select(4));
        assert_eq!(
            navigate(&rows, Some(4), Nav::PageUp, 2),
            NavResult::Select(2)
        );
        assert_eq!(
            navigate(&rows, Some(3), Nav::PageDown, 9),
            NavResult::Select(4)
        );
        assert_eq!(navigate(&[], Some(0), Nav::Down, 5), NavResult::Nothing);
        assert_eq!(
            navigate(&rows, Some(99), Nav::Down, 5),
            NavResult::Select(0),
            "a stale selection is treated as none"
        );
    }

    #[test]
    fn left_collapses_an_open_row_else_goes_to_the_parent() {
        let rows = sample();
        assert_eq!(
            navigate(&rows, Some(1), Nav::Left, 5),
            NavResult::Toggle("agent".into()),
            "open with children: collapse"
        );
        assert_eq!(
            navigate(&rows, Some(3), Nav::Left, 5),
            NavResult::Select(2),
            "a leaf goes to its parent"
        );
        assert_eq!(
            navigate(&rows, Some(4), Nav::Left, 5),
            NavResult::Select(1),
            "the parent is the nearest shallower row above"
        );
        assert_eq!(
            navigate(&rows, Some(0), Nav::Left, 5),
            NavResult::Toggle("task".into())
        );
        let mut closed = sample();
        closed[0].collapsed = true;
        assert_eq!(
            navigate(&closed, Some(0), Nav::Left, 5),
            NavResult::Nothing,
            "a closed root has nowhere to go"
        );
    }

    #[test]
    fn right_expands_a_closed_row_else_goes_to_the_first_child() {
        let mut rows = sample();
        assert_eq!(
            navigate(&rows, Some(1), Nav::Right, 5),
            NavResult::Select(2),
            "open: the first child is the next row"
        );
        rows[1].collapsed = true;
        assert_eq!(
            navigate(&rows, Some(1), Nav::Right, 5),
            NavResult::Toggle("agent".into()),
            "closed: expand"
        );
        assert_eq!(
            navigate(&rows, Some(3), Nav::Right, 5),
            NavResult::Nothing,
            "a leaf has no children"
        );
    }
}
