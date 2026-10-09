//! The command line of the `lattice` binary.
//!
//! ```text
//! lattice [--demo | --dev-model] [--demo-run] [--select-latest]
//!         [--view <timeline|graph>] [--screenshot <path.png> [--after-ms <N>]]
//! ```
//!
//! Invariants: parsing is pure (no environment, no files), every flag that
//! needs a value refuses to run without one, and an unknown argument is an error
//! rather than being ignored, so a mistyped flag in a verification script fails
//! loudly instead of silently measuring the wrong thing. Flags that cannot work
//! together (`--demo` and `--dev-model`) or that need another (`--demo-run` needs
//! a service that has a model to run) are refused, not guessed at.

use std::path::PathBuf;

use crate::app::ViewMode;

/// The default wait before a screenshot is taken, in milliseconds.
pub const DEFAULT_AFTER_MS: u64 = 1500;

pub const USAGE: &str = "\
Lattice: the native window for agent runs and their traces.

USAGE:
    lattice [OPTIONS]

OPTIONS:
    --demo                 Use the in-memory demonstration data instead of the
                           real agent runtime.
    --dev-model            Offer the scripted development model (`dev:scripted`)
                           in the model list of the real runtime. It is a script,
                           not a language model.
    --demo-run             Start a run when the window opens: the demonstration
                           run with --demo, a development-model run with
                           --dev-model.
    --select-latest        Select the newest run and its first span at launch (with
                           --demo-run: the first span of the new run, once it has
                           one).
    --view <timeline|graph>
                           Open the selected run on this tab (default timeline).
    --screenshot <PATH>    After a delay, save the window as a PNG and exit.
    --after-ms <N>         The delay for --screenshot (default 1500).
    -h, --help             Show this text.

ENVIRONMENT:
    LATTICE_PERF=1         Print `perf: views=<n> rebuilds=<n> frames=<n> (last 10 s)`
                           to stderr every 10 s (only when non-zero, and once for
                           the first idle interval). `frames` counts the frames
                           the window drew.
    WGPU_BACKEND           Graphics backend; set to `gl` (OpenGL) when unset.
";

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Cli {
    pub demo: bool,
    pub dev_model: bool,
    pub demo_run: bool,
    pub select_latest: bool,
    pub view: Option<ViewMode>,
    pub screenshot: Option<PathBuf>,
    pub after_ms: Option<u64>,
    pub help: bool,
}

impl Cli {
    /// The delay before the screenshot.
    pub fn screenshot_delay_ms(&self) -> u64 {
        self.after_ms.unwrap_or(DEFAULT_AFTER_MS)
    }
}

/// Parse the arguments after the program name.
pub fn parse(args: impl IntoIterator<Item = String>) -> Result<Cli, String> {
    let mut cli = Cli::default();
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--demo" => cli.demo = true,
            "--dev-model" => cli.dev_model = true,
            "--demo-run" => cli.demo_run = true,
            "--select-latest" => cli.select_latest = true,
            "--view" => {
                let value = args.next().ok_or("--view needs a tab: timeline or graph")?;
                cli.view = Some(match value.as_str() {
                    "timeline" => ViewMode::Timeline,
                    "graph" => ViewMode::Graph,
                    other => {
                        return Err(format!("--view needs timeline or graph, not {other:?}"));
                    }
                });
            }
            "-h" | "--help" => cli.help = true,
            "--screenshot" => {
                let path = args
                    .next()
                    .filter(|p| !p.starts_with("--"))
                    .ok_or("--screenshot needs a path, for example --screenshot shot.png")?;
                cli.screenshot = Some(PathBuf::from(path));
            }
            "--after-ms" => {
                let value = args
                    .next()
                    .ok_or("--after-ms needs a number of milliseconds")?;
                cli.after_ms = Some(value.parse::<u64>().map_err(|_| {
                    format!("--after-ms needs a whole number of milliseconds, not {value:?}")
                })?);
            }
            other => return Err(format!("unknown argument {other:?}; try --help")),
        }
    }
    if cli.after_ms.is_some() && cli.screenshot.is_none() {
        return Err("--after-ms only applies together with --screenshot".to_string());
    }
    if cli.demo && cli.dev_model {
        return Err(
            "--dev-model belongs to the real runtime; it does not apply with --demo".to_string(),
        );
    }
    if cli.demo_run && !cli.demo && !cli.dev_model {
        return Err(
            "--demo-run needs --demo, or --dev-model (the real runtime has no model to run without one)".to_string(),
        );
    }
    Ok(cli)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> Result<Cli, String> {
        parse(list.iter().map(|s| s.to_string()))
    }

    #[test]
    fn no_arguments_is_the_default() {
        assert_eq!(args(&[]).unwrap(), Cli::default());
    }

    #[test]
    fn the_verification_invocation_parses() {
        let cli = args(&[
            "--demo",
            "--select-latest",
            "--screenshot",
            "out/shot.png",
            "--after-ms",
            "2500",
        ])
        .unwrap();
        assert!(cli.demo && cli.select_latest && !cli.demo_run);
        assert_eq!(
            cli.screenshot.as_deref(),
            Some(std::path::Path::new("out/shot.png"))
        );
        assert_eq!(cli.screenshot_delay_ms(), 2500);
    }

    #[test]
    fn the_screenshot_delay_defaults_to_one_and_a_half_seconds() {
        let cli = args(&["--screenshot", "a.png"]).unwrap();
        assert_eq!(cli.screenshot_delay_ms(), 1500);
    }

    #[test]
    fn flags_that_need_a_value_refuse_to_run_without_one() {
        assert!(args(&["--screenshot"]).is_err());
        assert!(args(&["--screenshot", "--demo"]).is_err());
        assert!(args(&["--screenshot", "a.png", "--after-ms"]).is_err());
        assert!(args(&["--screenshot", "a.png", "--after-ms", "soon"]).is_err());
        assert!(args(&["--screenshot", "a.png", "--after-ms", "-5"]).is_err());
    }

    #[test]
    fn an_unknown_argument_is_an_error_not_ignored() {
        let error = args(&["--demo", "--fullscreen"]).unwrap_err();
        assert!(error.contains("--fullscreen"), "{error}");
    }

    #[test]
    fn after_ms_without_a_screenshot_is_refused() {
        assert!(args(&["--after-ms", "100"]).is_err());
    }

    #[test]
    fn help_is_recognised() {
        assert!(args(&["--help"]).unwrap().help);
        assert!(args(&["-h"]).unwrap().help);
        assert!(USAGE.contains("--screenshot") && USAGE.contains("LATTICE_PERF"));
        assert!(USAGE.contains("--dev-model") && USAGE.contains("--demo"));
        assert!(!USAGE.contains("not connected"), "the banner is gone");
        assert!(
            USAGE.contains("--view <timeline|graph>"),
            "--view is documented"
        );
        assert!(
            USAGE.contains("frames=<n>"),
            "the perf line names its frame count"
        );
    }

    #[test]
    fn view_opens_a_tab_and_defaults_to_none() {
        assert_eq!(
            args(&[]).unwrap().view,
            None,
            "the app's default is Timeline"
        );
        assert_eq!(
            args(&["--view", "graph"]).unwrap().view,
            Some(ViewMode::Graph)
        );
        assert_eq!(
            args(&["--view", "timeline"]).unwrap().view,
            Some(ViewMode::Timeline)
        );
        // It combines with the other launch flags, in any order.
        let cli = args(&[
            "--demo",
            "--select-latest",
            "--view",
            "graph",
            "--screenshot",
            "g.png",
        ])
        .unwrap();
        assert!(cli.demo && cli.select_latest);
        assert_eq!(cli.view, Some(ViewMode::Graph));
        assert_eq!(
            args(&["--view", "graph", "--demo"]).unwrap().view,
            Some(ViewMode::Graph)
        );
    }

    #[test]
    fn view_refuses_a_missing_or_unknown_tab() {
        let missing = args(&["--view"]).unwrap_err();
        assert!(missing.contains("--view"), "{missing}");
        let unknown = args(&["--view", "waterfall"]).unwrap_err();
        assert!(
            unknown.contains("waterfall") && unknown.contains("graph"),
            "{unknown}"
        );
        // A flag is not a tab: the next option is not swallowed as the value.
        let swallowed = args(&["--view", "--demo"]).unwrap_err();
        assert!(swallowed.contains("--demo"), "{swallowed}");
        assert!(args(&["--view", "Graph"]).is_err(), "case matters");
    }

    #[test]
    fn the_real_runtime_is_the_default_and_the_development_model_is_opt_in() {
        let default = args(&[]).unwrap();
        assert!(!default.demo && !default.dev_model);
        assert!(args(&["--dev-model"]).unwrap().dev_model);
        assert!(
            args(&["--dev-model", "--select-latest", "--screenshot", "s.png"])
                .unwrap()
                .dev_model
        );
    }

    #[test]
    fn flags_that_cannot_work_together_or_have_nothing_to_run_are_refused() {
        let both = args(&["--demo", "--dev-model"]).unwrap_err();
        assert!(
            both.contains("--dev-model") && both.contains("--demo"),
            "{both}"
        );
        assert!(args(&["--dev-model", "--demo"]).is_err(), "in either order");
        let nothing = args(&["--demo-run"]).unwrap_err();
        assert!(
            nothing.contains("--demo-run") && nothing.contains("--dev-model"),
            "{nothing}"
        );
        assert!(args(&["--demo", "--demo-run"]).unwrap().demo_run);
        assert!(args(&["--dev-model", "--demo-run"]).unwrap().demo_run);
    }
}
