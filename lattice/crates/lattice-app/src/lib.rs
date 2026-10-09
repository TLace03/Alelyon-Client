//! The native Lattice window: agent runs and their traces, drawn on the GPU by
//! iced over wgpu, and only when something changed.
//!
//! The window depends on the data contract (`lattice-protocol`) and on nothing
//! else of the runtime: it is handed a [`RunService`] and draws what that
//! service reports. `lattice` (the binary) hands it the real service
//! (`lattice-core`'s `CoreService`) or, with `--demo`, the in-memory
//! [`demo::DemoService`].
//!
//! The compute budget for the interface ("rounding error") is kept as
//! rules, each stated where it is enforced (`app.rs` has the list):
//! conditional rendering, conditional subscriptions, one message per batch of
//! events, cached canvas geometry that hover cannot invalidate, virtualized
//! rows, parse-once text, and `LATTICE_PERF=1` counters (views, cache rebuilds,
//! frames) that prove it.
//!
//! Graphics: OpenGL is requested unless `WGPU_BACKEND` says otherwise, and
//! iced falls back to its software renderer (tiny-skia) when no adapter fits.
//! No adapter is pinned by id. Why OpenGL and not Vulkan: see
//! [`backend_to_request`].

#![deny(unsafe_code)]

pub mod app;
pub mod axis;
pub mod cli;
pub mod clock;
pub mod demo;
pub mod detail;
pub mod graph;
pub mod newrun;
pub mod perf;
pub mod runlist;
pub mod runstate;
pub mod screenshot;
pub mod spans;
#[cfg(test)]
mod testkit;
pub mod textmetrics;
pub mod theme;
mod ui;
pub mod wfmodel;

#[cfg(test)]
mod apptests;

#[cfg(test)]
mod coretests;

#[cfg(test)]
mod perftests;

use std::ffi::OsStr;
use std::sync::Arc;

use iced::{Size, window};
use lattice_protocol::RunService;

pub use app::{App, Options, ScreenshotRequest};

/// A banner for a window that is not running on real data. The `lattice` binary
/// no longer shows it (both of its services are honest about what they are);
/// it stays for a caller that runs the window over a service that is not real.
pub const BANNER: &str = "Demonstration data — the agent runtime is not connected yet";

/// The value to give `WGPU_BACKEND`, if the environment has not chosen one.
///
/// iced 0.14 has no settings field for the wgpu backends: its compositor reads
/// `wgpu::Backends::from_env()`, which is the `WGPU_BACKEND` variable
/// (`iced_wgpu::window::Compositor::with_backend`). So the preference is made
/// through that variable, and only when nobody has set it: an explicit
/// `WGPU_BACKEND` (or `ICED_BACKEND`) always wins.
///
/// OpenGL, not Vulkan, because of a measurement (2026-09-30, AMD Radeon RX 9070
/// XT, Windows 11; the native Lattice's review). After the window
/// presented its first frame in response to input, a thread of the AMD driver
/// (`amdvlk64.dll` under Vulkan, `amdxc64.dll` under DirectX 12) busy-waited a
/// whole core for as long as the window stayed open, in every present mode
/// (fifo, fifo_relaxed, mailbox, immediate), while the window itself drew
/// nothing. Under OpenGL the same window used 0.0% of a core idle, before and
/// after input. The interface draws rectangles and text, so OpenGL costs it
/// nothing it needs; OpenGL and Vulkan are the two backends it allows.
pub fn backend_to_request(current: Option<&OsStr>) -> Option<&'static str> {
    current.is_none().then_some("gl")
}

/// Ask for OpenGL unless the environment already chose a backend.
#[allow(unsafe_code)]
fn prefer_opengl() {
    if let Some(backend) = backend_to_request(std::env::var_os("WGPU_BACKEND").as_deref()) {
        // SAFETY: `set_var` is unsound only when another thread reads or writes the
        // environment at the same time. This runs first thing in `run`, before iced,
        // the perf reporter or any thread of ours exists, and the process's `main`
        // does nothing else before calling it. (On Windows, the target, the
        // environment is process-wide and safe to set at any time.)
        unsafe { std::env::set_var("WGPU_BACKEND", backend) };
        if perf::enabled() {
            eprintln!("perf: WGPU_BACKEND={backend} (requested by lattice)");
        }
    }
}

/// The window: 1440 x 900, no smaller than 1000 x 640.
pub fn window_settings() -> window::Settings {
    window::Settings {
        size: Size::new(1440.0, 900.0),
        min_size: Some(Size::new(1000.0, 640.0)),
        ..window::Settings::default()
    }
}

/// Open the window over `service` and run until it closes.
pub fn run(
    service: Arc<dyn RunService>,
    options: Options,
) -> Result<(), Box<dyn std::error::Error>> {
    prefer_opengl();
    perf::start_reporter();
    theme::set_fonts(theme::Fonts::detect());
    let screenshot_requested = options.screenshot.is_some();
    iced::application(
        move || App::boot(service.clone(), options.clone()),
        App::update,
        ui::view,
    )
    .title(|_: &App| "Lattice".to_string())
    .subscription(App::subscription)
    .theme(App::theme)
    .style(|_: &App, theme: &iced::Theme| theme::app_style(theme))
    .window(window_settings())
    .default_font(theme::fonts().ui)
    .antialiasing(true)
    .run()?;
    if screenshot_requested && screenshot::failed() {
        return Err("the screenshot could not be written".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opengl_is_requested_only_when_the_environment_has_not_chosen() {
        assert_eq!(backend_to_request(None), Some("gl"));
        assert_eq!(backend_to_request(Some(OsStr::new("dx12"))), None);
        assert_eq!(backend_to_request(Some(OsStr::new("vulkan"))), None);
        assert_eq!(
            backend_to_request(Some(OsStr::new(""))),
            None,
            "an explicitly empty value is a choice too"
        );
    }

    #[test]
    fn the_window_is_1440_by_900_with_a_1000_by_640_floor() {
        let settings = window_settings();
        assert_eq!(settings.size, Size::new(1440.0, 900.0));
        assert_eq!(settings.min_size, Some(Size::new(1000.0, 640.0)));
    }
}
