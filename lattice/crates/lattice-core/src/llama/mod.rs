//! The local model runtime: llama.cpp's `llama-server`, which the core starts
//! and owns (ADR-0041, "Local inference without Ollama";
//! the chat core's spec §22, rules LR1–LR10).
//!
//! A port of the Python runtime's `llama_server.py`,
//! `gguf_header.py` and `local_model.selected_model` as
//! ADR-0041 phase 1 wrote them:
//! - [`files`]: where things are (`~/.alelyon/llama`, the models folder), the
//!   binary (LR3), the GGUF models a name may mean (LR4, never guessed), and
//!   the model Local uses (`analyst_model.json`, no override, no default);
//! - [`gguf`]: what a GGUF file says about itself, from its header (LR4);
//! - [`settings`]: how the server runs (`settings.json`, read-only, LR5) and
//!   the binary's recorded SHA-256 (`manifest.json`, LR3);
//! - [`server`]: [`server::LlamaRuntime`], the one server this process runs
//!   (LR2, LR6, LR7, LR10): started on the first Local turn, on 127.0.0.1 with
//!   a token per launch, inside a Job Object; stopped by a single timer after
//!   `idle_seconds` without a request; restarted to switch models; ended
//!   with the core;
//! - [`probes`]: the probe records: Python's `grammar-probes.json`, read
//!   only, and the tool-call probe's `tool-probes.json` beside it, written
//!   only through `fsx` (LR8, LR8′, LR9);
//!
//! What "Local" is here: the core's own managed server and nothing else (LR1).
//! An endpoint a person names is never Local, even on a loopback address.
//! Nothing in this module calls Ollama, names its endpoints or reads an
//! `OLLAMA_*` variable; those variables are removed from the server's
//! environment (LR6), and a path holding `ollama` is refused as a binary.
//! The Vulkan loader's device-selection names are the one addition to the
//! server's X7 environment, each only when it is set (LR6′).

use futures::future::BoxFuture;
use lattice_agents::SecretString;

pub mod bench;
pub mod files;
pub mod gguf;
pub mod probes;
pub mod server;
pub mod settings;

use files::{BinaryProblem, LocalModel, ModelProblem};

/// The explicit offline sentence for a binary that cannot be started.
/// PROVISIONAL (spec §22.3, row C4).
pub fn binary_sentence(problem: BinaryProblem) -> &'static str {
    match problem {
        BinaryProblem::Missing => {
            "The local model is offline: llama.cpp's server is not installed for Lattice on this machine."
        }
        BinaryProblem::Ollama => {
            "The local model is offline: Lattice does not start a server from an Ollama install."
        }
        BinaryProblem::NotAbsolute => {
            "The local model is offline: the llama.cpp server Lattice is set to use is not a full path."
        }
        BinaryProblem::NotExe => {
            "The local model is offline: the llama.cpp server Lattice is set to use is not a program (.exe)."
        }
        BinaryProblem::NotLocal => {
            "The local model is offline: the llama.cpp server Lattice is set to use is not on a drive of this machine."
        }
    }
}

/// The explicit offline sentence for a model name that means no file.
/// PROVISIONAL.
pub fn model_sentence(problem: ModelProblem) -> &'static str {
    match problem {
        ModelProblem::NoneChosen => {
            "The local model is offline: no model is chosen; put a GGUF file in the models folder and choose it."
        }
        ModelProblem::NotFound => {
            "The local model is offline: the chosen model is not a GGUF file in the models folder."
        }
    }
}

/// Why the managed server could not serve a turn.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LlamaError {
    Binary(BinaryProblem),
    /// The runtime was shut down.
    Closed,
    /// No free loopback port.
    Port,
    /// The token's temporary folder or file could not be made.
    Token,
    /// The binary could not be started.
    Spawn,
    /// The server exited while it loaded the model.
    ExitedWhileLoading,
    /// The server did not answer `/health` in time.
    NotReady,
    /// Another process listened on the port chosen for the server, twice
    /// (LR7a): nothing was sent to it.
    PortTaken,
    /// The server runs another model, and this request may not switch it
    /// (an editor's completion never stops a chat's answer).
    Serving,
}

impl LlamaError {
    /// One explicit offline sentence. PROVISIONAL.
    pub fn sentence(self) -> &'static str {
        match self {
            Self::Binary(problem) => binary_sentence(problem),
            Self::Closed => "The local model is offline: Lattice is closing.",
            Self::Port => "The local model is offline: no free port was found on this machine.",
            Self::Token => "The local model is offline: Lattice could not make the server's key.",
            Self::Spawn => "The local model is offline: llama.cpp's server could not be started.",
            Self::ExitedWhileLoading => {
                "The local model is offline: llama.cpp's server stopped while it loaded the model."
            }
            Self::PortTaken => {
                "The local model is offline: another program took the port Lattice chose for llama.cpp's server."
            }
            Self::NotReady => {
                "The local model is offline: llama.cpp's server did not become ready in time."
            }
            Self::Serving => "The local model server is running another model now.",
        }
    }
}

/// How a client reaches the running server.
#[derive(Clone, Debug)]
pub struct ManagedEndpoint {
    /// `http://127.0.0.1:<port>`.
    pub base_url: String,
    /// The served model's name (`--alias`): its file stem.
    pub alias: String,
    /// This launch's token; it reaches only the request header.
    pub token: SecretString,
    /// The binary's SHA-256 from the install manifest, for the turn's
    /// provenance; `None` when no manifest records it (UNMEASURED).
    pub binary_sha256: Option<String>,
    /// Which launch this is.
    pub generation: u64,
}

impl ManagedEndpoint {
    /// The OpenAI-compatible root: `<base_url>/v1`.
    pub fn api_base(&self) -> String {
        format!("{}/v1", self.base_url)
    }
}

/// Holds the server in use: while any lease is alive, no idle stop happens.
/// Dropping the last one arms the idle timer.
pub struct Lease(Option<Box<dyn FnOnce() + Send>>);

impl Lease {
    pub(crate) fn new(release: impl FnOnce() + Send + 'static) -> Self {
        Self(Some(Box::new(release)))
    }

    /// A lease that holds nothing (a runtime that is not a real server).
    pub fn detached() -> Self {
        Self(None)
    }
}

impl Drop for Lease {
    fn drop(&mut self) {
        if let Some(release) = self.0.take() {
            release();
        }
    }
}

impl std::fmt::Debug for Lease {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Lease")
    }
}

/// A running server for one model, and the lease that keeps it so.
#[derive(Debug)]
pub struct Opened {
    pub endpoint: ManagedEndpoint,
    pub lease: Lease,
}

/// What the chat asks of the local runtime. [`server::LlamaRuntime`] is the
/// one implementation that ships; tests give the chat their own.
pub trait ManagedRuntime: Send + Sync {
    /// The model the server runs now, if one runs (memory only).
    fn running(&self) -> Option<String>;
    /// The last start's failure, when it failed (memory only).
    fn failed(&self) -> Option<String>;
    /// A server running `model`, started or restarted as needed, and a lease.
    fn open(&self, model: LocalModel) -> BoxFuture<'static, Result<Opened, LlamaError>>;
    /// As [`open`](Self::open), but never a switch: refused with
    /// [`LlamaError::Serving`] while the server runs another model. A runtime
    /// that cannot tell answers from what [`running`](Self::running) says.
    fn open_unswitched(&self, model: LocalModel) -> BoxFuture<'static, Result<Opened, LlamaError>> {
        match self.running() {
            Some(name) if name != model.name => Box::pin(async { Err(LlamaError::Serving) }),
            _ => self.open(model),
        }
    }
    /// What the running server's `/props` says (LR8), read with its token;
    /// `None` when it cannot be read, or by a runtime that cannot read it.
    fn props(&self, endpoint: &ManagedEndpoint) -> BoxFuture<'static, Option<server::Props>> {
        let _ = endpoint;
        Box::pin(async { None })
    }
}
