//! Where `serve` finds what it runs that is not part of this crate: the recogniser program (whisper.cpp's
//! `whisper-server`) and the speech model. Neither is shipped or downloaded here; a person puts them in place, and
//! what is missing is said in words that name the place, so a window can show it as a step to take rather than as
//! an error.
//!
//! - The model: `--model`, else `ANGEL_EARS_MODEL`, else [`MODEL_FILE`] in [`models_dir`] (`~/.alelyon/angel/models`).
//! - The recogniser: `--whisper-exe`, else `ANGEL_WHISPER_SERVER`, else [`SERVER_PROGRAM`] beside the engine's own
//!   program, or in a `whisper` folder beside it.
//!
//! `--attach <host:port>` uses a recogniser that is already running, and then neither is needed.

use std::path::{Path, PathBuf};

/// The speech model `serve` starts the recogniser with: whisper large-v3-turbo, quantised to q5_0, in whisper.cpp's
/// GGML format.
pub const MODEL_FILE: &str = "ggml-large-v3-turbo-q5_0.bin";
/// The variable that names another model file.
pub const MODEL_ENV: &str = "ANGEL_EARS_MODEL";
/// The variable that names the recogniser program.
pub const SERVER_ENV: &str = "ANGEL_WHISPER_SERVER";
/// whisper.cpp's HTTP server program, as its build names it.
#[cfg(windows)]
pub const SERVER_PROGRAM: &str = "whisper-server.exe";
#[cfg(not(windows))]
pub const SERVER_PROGRAM: &str = "whisper-server";

/// The folder the model is looked for in when nothing names it: `<home>/.alelyon/angel/models`.
pub fn models_dir(home: &Path) -> PathBuf {
    home.join(".alelyon").join("angel").join("models")
}

/// What the rule reads, so it can be tested without changing the environment.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Given {
    /// `--attach`: a recogniser already running, so nothing else is needed.
    pub attach: bool,
    /// `--model`.
    pub model: Option<PathBuf>,
    /// `--whisper-exe`.
    pub server: Option<PathBuf>,
    /// `ANGEL_EARS_MODEL`.
    pub model_env: Option<PathBuf>,
    /// `ANGEL_WHISPER_SERVER`.
    pub server_env: Option<PathBuf>,
    /// The user's home folder (`USERPROFILE`, else `HOME`).
    pub home: Option<PathBuf>,
    /// The folder the engine's program is in.
    pub engine_dir: Option<PathBuf>,
}

impl Given {
    /// The rule's inputs from this process's environment, for an engine program at `engine` (None: this program).
    pub fn from_env(engine: Option<&Path>) -> Given {
        let var = |key: &str| std::env::var_os(key).filter(|v| !v.is_empty()).map(PathBuf::from);
        let engine = engine.map(Path::to_path_buf).or_else(|| std::env::current_exe().ok());
        Given {
            model_env: var(MODEL_ENV),
            server_env: var(SERVER_ENV),
            home: var("USERPROFILE").or_else(|| var("HOME")),
            engine_dir: engine.and_then(|e| e.parent().map(Path::to_path_buf)),
            ..Given::default()
        }
    }

    /// The same, with an engine's command-line options (`--attach`, `--model`, `--whisper-exe`) read from `args`, as
    /// `serve` would read them.
    pub fn with_args<S: AsRef<str>>(mut self, args: &[S]) -> Given {
        let args: Vec<&str> = args.iter().map(AsRef::as_ref).collect();
        let value = |name: &str| args.iter().rposition(|a| *a == name).and_then(|i| args.get(i + 1)).map(PathBuf::from);
        self.attach = args.contains(&"--attach");
        self.model = value("--model").or(self.model);
        self.server = value("--whisper-exe").or(self.server);
        self
    }

    /// The model file: named, or [`MODEL_FILE`] in [`models_dir`]. None when nothing names it and no home is known.
    pub fn model_path(&self) -> Option<PathBuf> {
        self.model.clone().or_else(|| self.model_env.clone()).or_else(|| self.home.as_deref().map(|h| models_dir(h).join(MODEL_FILE)))
    }

    /// Where the recogniser program is looked for, in order: the one named, else beside the engine and in a
    /// `whisper` folder beside it.
    pub fn server_candidates(&self) -> Vec<PathBuf> {
        if let Some(named) = self.server.clone().or_else(|| self.server_env.clone()) {
            return vec![named];
        }
        match &self.engine_dir {
            Some(dir) => vec![dir.join(SERVER_PROGRAM), dir.join("whisper").join(SERVER_PROGRAM)],
            None => Vec::new(),
        }
    }

    /// The recogniser program: the first candidate that `exists`, else the first candidate (so a failure to start
    /// names it).
    pub fn server_path(&self, exists: impl Fn(&Path) -> bool) -> Option<PathBuf> {
        let candidates = self.server_candidates();
        candidates.iter().find(|p| exists(p)).or(candidates.first()).cloned()
    }

    fn named_model(&self) -> bool {
        self.model.is_some() || self.model_env.is_some()
    }

    fn named_server(&self) -> bool {
        self.server.is_some() || self.server_env.is_some()
    }

    /// What `serve` would not find, with `exists` saying whether a file is there. Empty: it can start.
    pub fn missing(&self, exists: impl Fn(&Path) -> bool) -> Vec<Missing> {
        if self.attach {
            return Vec::new();
        }
        let mut out = Vec::new();
        match self.model_path() {
            Some(path) if exists(&path) => {}
            Some(path) => out.push(Missing::Model { path, named: self.named_model() }),
            None => out.push(Missing::NoHome),
        }
        let candidates = self.server_candidates();
        if !candidates.iter().any(|p| exists(p)) {
            out.push(Missing::Server { looked: candidates, named: self.named_server() });
        }
        out
    }
}

/// Something `serve` needs that is not in place.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Missing {
    /// The model file is not at `path`; `named` when `--model` or `ANGEL_EARS_MODEL` named that path.
    Model { path: PathBuf, named: bool },
    /// The recogniser program is in none of the places `looked`; `named` when it was named.
    Server { looked: Vec<PathBuf>, named: bool },
    /// No home folder is known, so the model's folder is not either.
    NoHome,
}

impl Missing {
    /// What to do about it, in words a person can act on.
    pub fn explain(&self) -> String {
        match self {
            Missing::Model { path, named: true } => {
                format!("The speech model is not at {}, where {MODEL_ENV} or --model names it. Put the file there, or name another.", path.display())
            }
            Missing::Model { path, named: false } => {
                let folder = path.parent().map(|p| p.display().to_string()).unwrap_or_default();
                format!(
                    "The speech model is not on this PC yet. Put {MODEL_FILE} (whisper large-v3-turbo, q5_0, in whisper.cpp's \
                     format) in {folder}, or set {MODEL_ENV} to the file's path."
                )
            }
            Missing::Server { looked, named: true } => {
                let at = looked.first().map(|p| p.display().to_string()).unwrap_or_default();
                format!("The recogniser program is not at {at}, where {SERVER_ENV} or --whisper-exe names it.")
            }
            Missing::Server { looked, named: false } => {
                let beside = looked.first().and_then(|p| p.parent()).map(|p| p.display().to_string()).unwrap_or_else(|| "the engine's folder".into());
                format!(
                    "The recogniser program is not beside the speech engine. Put whisper.cpp's {SERVER_PROGRAM} (with the \
                     libraries its build made) in {beside}, or in a whisper folder there, or set {SERVER_ENV} to its path."
                )
            }
            Missing::NoHome => format!("No home folder is set (USERPROFILE), so the speech model's folder is unknown; set {MODEL_ENV} to the model's path."),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn given() -> Given {
        Given { home: Some(PathBuf::from("H")), engine_dir: Some(PathBuf::from("E")), ..Given::default() }
    }

    #[test]
    fn the_model_is_the_named_file_or_the_approved_one_in_the_models_folder() {
        let g = given();
        assert_eq!(g.model_path(), Some(models_dir(Path::new("H")).join(MODEL_FILE)));
        let env = Given { model_env: Some("env.bin".into()), ..given() };
        assert_eq!(env.model_path(), Some(PathBuf::from("env.bin")));
        let flag = Given { model: Some("flag.bin".into()), ..env };
        assert_eq!(flag.model_path(), Some(PathBuf::from("flag.bin")), "--model wins over the variable");
        assert_eq!(Given::default().model_path(), None, "no home, nothing named");
    }

    #[test]
    fn the_recogniser_is_the_named_program_or_one_beside_the_engine() {
        let g = given();
        let beside = PathBuf::from("E").join(SERVER_PROGRAM);
        let folder = PathBuf::from("E").join("whisper").join(SERVER_PROGRAM);
        assert_eq!(g.server_candidates(), [beside.clone(), folder.clone()]);
        assert_eq!(g.server_path(|p| p == folder), Some(folder.clone()), "the whisper folder when only it has one");
        assert_eq!(g.server_path(|_| false), Some(beside), "the first place, so a failure names it");
        let named = Given { server_env: Some("w.exe".into()), ..given() };
        assert_eq!(named.server_candidates(), [PathBuf::from("w.exe")], "a named program is the only place looked");
        let flag = Given { server: Some("f.exe".into()), ..named };
        assert_eq!(flag.server_candidates(), [PathBuf::from("f.exe")]);
    }

    #[test]
    fn what_is_missing_is_named_with_where_to_put_it() {
        let g = given();
        let missing = g.missing(|_| false);
        assert_eq!(missing.len(), 2, "{missing:?}");
        let model = missing[0].explain();
        assert!(model.contains(MODEL_FILE) && model.contains(&models_dir(Path::new("H")).display().to_string()) && model.contains(MODEL_ENV), "{model}");
        let server = missing[1].explain();
        assert!(server.contains(SERVER_PROGRAM) && server.contains(SERVER_ENV) && server.contains("whisper folder"), "{server}");
        for words in [&model, &server] {
            for bad in ["error", "failed", "download"] {
                assert!(!words.to_lowercase().contains(bad), "{words}");
            }
        }
        let model_file = models_dir(Path::new("H")).join(MODEL_FILE);
        let only_server = g.missing(|p| p == model_file);
        assert!(matches!(only_server.as_slice(), [Missing::Server { named: false, .. }]), "{only_server:?}");
        assert!(g.missing(|_| true).is_empty(), "everything in place");
        assert!(Given { attach: true, ..given() }.missing(|_| false).is_empty(), "--attach needs neither");
        assert_eq!(Given { home: None, ..given() }.missing(|_| true), [Missing::NoHome]);
        let nowhere = Given { home: Some("H".into()), ..Given::default() }.missing(|_| true);
        assert!(matches!(nowhere.as_slice(), [Missing::Server { looked, .. }] if looked.is_empty()), "no engine folder, nowhere to look: {nowhere:?}");
        let named = Given { model_env: Some("m.bin".into()), ..given() }.missing(|_| false);
        assert!(named[0].explain().contains("m.bin") && named[0].explain().contains(MODEL_ENV));
    }

    #[test]
    fn an_engines_options_are_read_as_serve_reads_them() {
        let g = given().with_args(&["--attach", "127.0.0.1:18178"]);
        assert!(g.attach);
        let g = given().with_args(&["--model", "a.bin", "--whisper-exe", "w.exe", "--model", "b.bin"]);
        assert_eq!((g.model.clone(), g.server.clone()), (Some(PathBuf::from("b.bin")), Some(PathBuf::from("w.exe"))), "the last one, as serve's options");
        assert!(!g.attach);
        assert_eq!(given().with_args::<&str>(&[]), given());
    }
}
