//! `lattice`: the native window.
//!
//! By default it runs over the real service (`lattice-core`'s `CoreService`: the
//! agent runtime, the model registry, the run store); `--dev-model` adds the
//! scripted development model to that service's model list, and `--demo` runs
//! over the in-memory demonstration service instead (for screenshots and tests).

use std::process::ExitCode;
use std::sync::Arc;

use lattice_app::demo::{self, DemoService};
use lattice_app::{Options, ScreenshotRequest, cli};
use lattice_core::{CoreConfig, CoreService};
use lattice_protocol::{RunService, StartRun};

fn main() -> ExitCode {
    let cli = match cli::parse(std::env::args().skip(1)) {
        Ok(cli) => cli,
        Err(error) => {
            eprintln!("lattice: {error}");
            return ExitCode::from(2);
        }
    };
    if cli.help {
        print!("{}", cli::USAGE);
        return ExitCode::SUCCESS;
    }
    let after_ms = cli.screenshot_delay_ms();
    // The service, and the agent and model `--demo-run` starts a run with.
    let (service, agent, model): (Arc<dyn RunService>, &str, &str) = if cli.demo {
        (
            Arc::new(DemoService::new()),
            demo::ASSISTANT,
            demo::LOCAL_MODEL,
        )
    } else {
        (
            Arc::new(CoreService::new(CoreConfig::from_process(cli.dev_model))),
            lattice_core::catalog::ASSISTANT_ID,
            lattice_core::choices::DEV_MODEL_ID,
        )
    };
    let options = Options {
        // Neither service is pretending: the demonstration data is announced by
        // `--demo` itself, and the real service has nothing to apologise for.
        banner: None,
        select_latest: cli.select_latest,
        view: cli.view,
        screenshot: cli
            .screenshot
            .map(|path| ScreenshotRequest { path, after_ms }),
        start_run: cli.demo_run.then(|| StartRun {
            task: "Which model should I use to translate a contract?".into(),
            agent: agent.into(),
            model: model.into(),
        }),
    };
    match lattice_app::run(service, options) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("lattice: {error}");
            ExitCode::FAILURE
        }
    }
}
