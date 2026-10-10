//! The ports the shell provides to the core (the chat core's spec §2.5).
//! Not a port of Python code.
//!
//! The core never draws, opens a dialog or flashes the taskbar. When a
//! decision widens authority or sends data off the machine, the core itself
//! calls [`ConfirmPort::confirm`]; the shell answers with a NATIVE dialog that
//! page script cannot click (§9.4), and answers `true` only for the reader's
//! click. The page can only ask for the action that leads here. Tests use
//! recording fakes.

use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};

use futures::future::BoxFuture;
use lattice_protocol::conversation::CommandMode;

/// A decision only the reader may make, through a native dialog.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ConfirmRequest {
    /// Trust a folder: rules files, Agent mode, and git writing snapshot
    /// objects and refs into its `.git` (FT1).
    TrustFolder {
        path: String,
        will_read: Vec<String>,
        git_writes: bool,
    },
    /// Attach a folder whose path came from the page (§6.1).
    AttachFolder { path: String },
    /// Every approval of `run_command` (§9.3).
    RunCommand {
        text: String,
        cwd: String,
        mode: CommandMode,
        /// It keeps running after the agent's call returns.
        background: bool,
    },
    /// Every Keep of an authority file (ST4).
    KeepAuthority {
        path: String,
        added: u32,
        removed: u32,
    },
    /// A standing approval, Exact only (X3).
    AllowAlways {
        workspace: String,
        argv: Vec<String>,
        program: String,
        cwd: String,
    },
    /// Send a withheld result once (T3).
    ReleaseWithheld {
        conversation: String,
        label: String,
        call: String,
    },
    /// The first remote send in a conversation, or the first after its label
    /// or its workspace changed (T1).
    FirstRemoteSend {
        conversation: String,
        label: String,
        earlier_local_results: u32,
    },
    /// A send with images to a model off this PC: asked every time, as the
    /// secret checks read words, not pictures (`convo::images`).
    SendImages {
        conversation: String,
        label: String,
        images: u32,
    },
    /// Enable an MCP server (§12): its entry's exact hash, shown as its
    /// command line, the program that line starts, the folder it runs in and
    /// the variable **names** it gets (never their values).
    EnableMcpServer {
        name: String,
        /// The file that declares it ("your MCP settings", or a folder's).
        from: String,
        command_line: String,
        program: String,
        cwd: String,
        env_names: Vec<String>,
    },
    /// Every approval of an MCP tool's call that is not allowed always
    /// (§12), as for `run_command`.
    McpCall {
        server: String,
        tool: String,
        /// At most 2 KiB of the arguments.
        arguments: String,
    },
    /// One of the labs' agents (`crate::acp`) asks to act: what, as it says,
    /// and at most 2 KiB of the details it gives.
    /// The agent's commit (`convo::git_tools`): the reader's own git, its
    /// hooks run.
    GitCommit {
        branch: String,
        message: String,
        /// The files, at most 40 and then how many more.
        files: Vec<String>,
    },
    /// The agent's push, and a pull request when GitHub's CLI is there.
    GitPush {
        branch: String,
        remote: String,
        base: String,
        title: String,
        pull_request: bool,
    },
    AgentPermission {
        agent: String,
        action: String,
        detail: String,
        /// The change it would make, as lines (`acp::session::diff_lines`);
        /// empty when the agent did not say.
        diff: Vec<String>,
    },
    /// "Allow always" for one tool of one server (§12).
    AllowMcpTool {
        server: String,
        tool: String,
        from: String,
    },
    /// An action in the agent's browser whose effect another person will
    /// see, moves money, deletes something or changes an account.
    BrowserAct {
        site: String,
        action: String,
        /// What the agent says it is (its own words).
        what: String,
        /// The effect, in words.
        effect: String,
    },
    /// Switch auto mode on: the agent on the whole desktop.
    AutoMode {
        /// The keys that stop it.
        stop_keys: String,
    },
    /// An action on the whole desktop, in auto mode, that moves money or
    /// changes an account.
    DesktopAct {
        app: String,
        action: String,
        what: String,
        effect: String,
    },
}

/// A decision that widens authority or sends data off the machine.
pub trait ConfirmPort: Send + Sync {
    /// `true` only for the reader's click in a native dialog.
    fn confirm(&self, request: ConfirmRequest) -> BoxFuture<'static, bool>;
}

/// Tell the reader something needs them (a taskbar flash when unfocused).
/// Never carries text from the conversation, and never causes an action.
pub trait AttentionPort: Send + Sync {
    fn attention(&self, conversation: &str);
}

// ------------------------------------------------------- CP1–CP3 (§2.5)
//
// The page can make a dialog open at any moment, so a dialog must not be
// confirmable by a keystroke or a click meant for the composer, and must not
// wear the reader down. The core states each dialog's facts in a `Dialog`,
// which the shell draws as they are:
// - CP1: the default button is the refusing one, and input is ignored for
//   `IGNORE_INPUT_FOR` after the dialog appears;
// - CP2: text that came from the model or the folder (a command, a path, a
//   name) is shown with every character that is not printable ASCII escaped
//   (`\u{2013}`), so the reader sees what will run;
// - CP3: a request the reader refused is not asked again in the same session
//   when the page asks for it again ([`Confirmer`]); only an action the shell
//   handles in Rust (a native menu item, a drop, the folder picker) or a
//   fresh pending call asks again.

/// How long a dialog ignores input after it appears (CP1).
pub const IGNORE_INPUT_FOR: std::time::Duration = std::time::Duration::from_millis(500);

/// Which button a dialog selects first.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DefaultButton {
    Refuse,
    Confirm,
}

/// A native dialog's facts, for the shell to draw.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Dialog {
    pub title: String,
    /// Each line already escaped (CP2).
    pub lines: Vec<String>,
    pub confirm: &'static str,
    pub refuse: &'static str,
    /// Always [`DefaultButton::Refuse`] (CP1).
    pub default: DefaultButton,
    /// Always [`IGNORE_INPUT_FOR`] (CP1).
    pub ignore_input_for: std::time::Duration,
}

/// CP2: `text` with every character that is not printable ASCII (a control
/// character, a bidirectional mark, a Unicode dash or quote, any other
/// non-ASCII character) shown as `\u{…}`.
pub fn escape_for_dialog(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        if (' '..='~').contains(&c) {
            out.push(c);
        } else {
            out.push_str(&format!("\\u{{{:04x}}}", u32::from(c)));
        }
    }
    out
}

impl ConfirmRequest {
    /// The dialog the shell shows for this request.
    pub fn dialog(&self) -> Dialog {
        let e = escape_for_dialog;
        let (title, lines, confirm, refuse): (String, Vec<String>, &'static str, &'static str) = match self {
            Self::TrustFolder {
                path,
                will_read,
                git_writes,
            } => {
                let mut lines = vec![format!("Folder: {}", e(path))];
                if will_read.is_empty() {
                    lines.push("No rules files were found.".to_owned());
                } else {
                    lines.push("Lattice will read these rules files, as text for the model; they cannot grant anything:".to_owned());
                    lines.extend(will_read.iter().map(|file| format!("  {}", e(file))));
                }
                if *git_writes {
                    lines.push("Agent mode will run git here, with hooks and filters off, and will write snapshot objects and refs into its .git folder.".to_owned());
                }
                ("Trust this folder?".to_owned(), lines, "Trust", "Do not trust")
            }
            Self::AttachFolder { path } => (
                "Attach this folder?".to_owned(),
                vec![
                    format!("Folder: {}", e(path)),
                    "The path came from the page, not from the folder picker.".to_owned(),
                ],
                "Attach",
                "Cancel",
            ),
            Self::RunCommand {
                text,
                cwd,
                mode,
                background,
            } => (
                "Run this command?".to_owned(),
                [
                    format!("Command: {}", e(text)),
                    format!("In: {}", e(cwd)),
                    match mode {
                        CommandMode::Direct => "Runs this program directly, with no shell.",
                        CommandMode::PowerShell => "Runs in Windows PowerShell 5.1.",
                    }
                    .to_owned(),
                ]
                .into_iter()
                .chain(background.then(|| {
                    "Keeps running in the background until you stop it, it ends, or Lattice closes.".to_owned()
                }))
                .chain([
                    "Runs as you, with your permissions. It can change files outside the folder and use the network; Lattice cannot prevent either on Windows yet.".to_owned(),
                ])
                .collect(),
                "Run",
                "Do not run",
            ),
            Self::KeepAuthority {
                path,
                added,
                removed,
            } => (
                format!("Keep this change to {}?", e(path)),
                vec![
                    format!("+{added} -{removed}"),
                    "This file changes what Lattice, git, a package manager or a command does.".to_owned(),
                ],
                "Keep",
                "Do not keep",
            ),
            Self::AllowAlways {
                workspace,
                argv,
                program,
                cwd,
            } => (
                "Always allow this exact command here?".to_owned(),
                vec![
                    format!("Program: {}", e(program)),
                    format!(
                        "Arguments: {}",
                        argv.iter().map(|arg| e(arg)).collect::<Vec<_>>().join(" ")
                    ),
                    format!("In: {}", e(cwd)),
                    format!("Folder: {}", e(workspace)),
                ],
                "Allow always",
                "Ask each time",
            ),
            Self::ReleaseWithheld {
                conversation,
                label,
                call,
            } => (
                "Send this withheld output once?".to_owned(),
                vec![
                    format!("To: {}", e(label)),
                    format!("Conversation: {}", e(conversation)),
                    format!("Call: {}", e(call)),
                    "It looks like it contains a secret.".to_owned(),
                ],
                "Send once",
                "Keep withheld",
            ),
            Self::FirstRemoteSend {
                conversation,
                label,
                earlier_local_results,
            } => {
                let mut lines = vec![
                    format!("Conversation: {}", e(conversation)),
                    "This conversation, and the files, search results and command output the agent reads, go off this machine.".to_owned(),
                ];
                if *earlier_local_results > 0 {
                    lines.push(format!(
                        "{earlier_local_results} earlier tool results read on this machine will be sent."
                    ));
                }
                (format!("Send to {}?", e(label)), lines, "Send", "Do not send")
            }
            Self::SendImages {
                conversation,
                label,
                images,
            } => (
                format!("Send {images} image{} to {}?", if *images == 1 { "" } else { "s" }, e(label)),
                vec![
                    format!("Conversation: {}", e(conversation)),
                    "The attached images go off this machine. Lattice cannot check a picture for passwords, keys or other secrets: look at them first.".to_owned(),
                ],
                "Send",
                "Do not send",
            ),
            Self::EnableMcpServer {
                name,
                from,
                command_line,
                program,
                cwd,
                env_names,
            } => {
                let mut lines = vec![
                    format!("Declared in: {}", e(from)),
                    format!("Command: {}", e(command_line)),
                    format!("Runs: {}", e(program)),
                    format!("In: {}", e(cwd)),
                ];
                if !env_names.is_empty() {
                    lines.push(format!(
                        "Environment names: {}",
                        env_names.iter().map(|name| e(name)).collect::<Vec<_>>().join(", ")
                    ));
                }
                lines.push("This server is a program on your machine. It runs as you and may use the network or change files; Lattice asks before each of its tools runs.".to_owned());
                (
                    format!("Enable the MCP server {}?", e(name)),
                    lines,
                    "Enable",
                    "Do not enable",
                )
            }
            Self::McpCall {
                server,
                tool,
                arguments,
            } => (
                format!("Use {} from {}?", e(tool), e(server)),
                vec![
                    format!("Arguments: {}", e(arguments)),
                    "The server runs as you. It may change files outside the folder and use the network; Lattice cannot prevent either.".to_owned(),
                ],
                "Use it",
                "Do not use it",
            ),
            Self::GitCommit {
                branch,
                message,
                files,
            } => {
                let mut lines = vec![format!("On branch {}, {} file(s):", e(branch), files.len())];
                lines.extend(files.iter().map(|file| format!("  {}", e(file))));
                lines.push("Message:".to_owned());
                lines.extend(message.lines().take(12).map(|line| format!("  {}", e(line))));
                lines.push("It runs your own git, as a commit by hand does: the repository's hooks run, with your identity and signing.".to_owned());
                ("Commit these changes?".to_owned(), lines, "Commit", "Do not commit")
            }
            Self::GitPush {
                branch,
                remote,
                base,
                title,
                pull_request,
            } => {
                let mut lines = vec![format!("Push branch {} to {}.", e(branch), e(remote))];
                if *pull_request {
                    lines.push(format!("Then open a pull request into {}: {}", e(base), e(title)));
                    lines.push("It uses your own git and GitHub CLI sign-ins; the branch and the pull request are seen by everyone with access to the repository.".to_owned());
                } else {
                    lines.push("GitHub's CLI is not installed, so no pull request is opened.".to_owned());
                }
                (
                    if *pull_request { "Push and open a pull request?".to_owned() } else { "Push this branch?".to_owned() },
                    lines,
                    "Push",
                    "Do not push",
                )
            }
            Self::AgentPermission {
                agent,
                action,
                detail,
                diff,
            } => {
                let mut lines = vec![format!("{}: {}", e(agent), e(action))];
                if !diff.is_empty() {
                    lines.push("The change (nothing is written unless you allow it):".to_owned());
                    lines.extend(diff.iter().map(|line| e(line)));
                } else if !detail.is_empty() {
                    lines.push(format!("Details: {}", e(detail)));
                }
                lines.push("It runs as you, with its own tools, on your account with its maker. Yes allows this once.".to_owned());
                (format!("Let {} do this?", e(agent)), lines, "Allow once", "Do not allow")
            }
            Self::BrowserAct {
                site,
                action,
                what,
                effect,
            } => (
                format!("Let the agent act on {}?", e(site)),
                vec![
                    format!("It will: {}", e(action)),
                    format!("It says this is: {}", e(what)),
                    format!("Its effect: {}", e(effect)),
                    "Check the agent's browser window before you allow it: what a page says can mislead the agent.".to_owned(),
                ],
                "Allow this once",
                "Do not allow",
            ),
            Self::AutoMode { stop_keys } => (
                "Let the agent control your whole desktop?".to_owned(),
                vec![
                    "In Agent mode it will see your screen and use your mouse and keyboard in any program, as you would.".to_owned(),
                    "It posts, sends, edits and deletes without asking first; a purchase, a payment, and a sign-in, security or account change still ask you.".to_owned(),
                    "It never acts on Alelyon itself, a password manager or Windows' sign-in and permission prompts, and never types a password.".to_owned(),
                    "Pictures of your screen go to the model you choose: with a model off this PC, they leave it.".to_owned(),
                    format!("{} stops it at once, from any program.", e(stop_keys)),
                ],
                "Switch auto mode on",
                "Do not",
            ),
            Self::DesktopAct {
                app,
                action,
                what,
                effect,
            } => (
                format!("Let the agent act in {}?", e(app)),
                vec![
                    format!("It will: {}", e(action)),
                    format!("It says this is: {}", e(what)),
                    format!("Its effect: {}", e(effect)),
                    "Look at your screen before you allow it: what a program shows can mislead the agent.".to_owned(),
                ],
                "Allow this once",
                "Do not allow",
            ),
            Self::AllowMcpTool { server, tool, from } => (
                format!("Always allow {} from {}?", e(tool), e(server)),
                vec![
                    format!("Declared in: {}", e(from)),
                    "Its calls will run without asking, until you stop allowing it in Tools or the server's entry changes.".to_owned(),
                ],
                "Allow always",
                "Ask each time",
            ),
        };
        Dialog {
            title,
            lines,
            confirm,
            refuse,
            default: DefaultButton::Refuse,
            ignore_input_for: IGNORE_INPUT_FOR,
        }
    }
}

/// Who started the action that leads to a dialog (CP3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Initiated {
    /// The page asked: a refused request is not asked again this session.
    Page,
    /// The shell's own Rust code (a native menu item, a drop, the folder
    /// picker): the reader started it, so it asks again.
    Native,
}

/// CP3 over a [`ConfirmPort`]: the requests the reader refused this session,
/// by a key the caller names (the request's kind and target, and for a call
/// its call id, so a fresh pending call is a new key).
pub struct Confirmer {
    port: Arc<dyn ConfirmPort>,
    refused: Arc<Mutex<BTreeSet<String>>>,
}

impl Confirmer {
    pub fn new(port: Arc<dyn ConfirmPort>) -> Self {
        Self {
            port,
            refused: Arc::default(),
        }
    }

    /// Ask the reader, unless the page asks again for what the reader
    /// refused under `key` this session (then `false`, with no dialog).
    pub fn ask(
        &self,
        key: &str,
        request: ConfirmRequest,
        initiated: Initiated,
    ) -> BoxFuture<'static, bool> {
        let already = self
            .refused
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .contains(key);
        if already && initiated == Initiated::Page {
            return Box::pin(async { false });
        }
        let answer = self.port.confirm(request);
        let key = key.to_owned();
        let refused = Arc::clone(&self.refused);
        Box::pin(async move {
            let confirmed = answer.await;
            let mut set = refused
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if confirmed {
                set.remove(&key);
            } else {
                set.insert(key);
            }
            confirmed
        })
    }
}

/// A fake for tests: answers as told and records every request.
#[cfg(test)]
pub(crate) mod fake {
    use std::sync::{Arc, Mutex};

    use futures::future::BoxFuture;

    use super::{ConfirmPort, ConfirmRequest};

    #[derive(Clone, Default)]
    pub(crate) struct RecordingConfirm {
        pub(crate) answer: Arc<Mutex<bool>>,
        pub(crate) asked: Arc<Mutex<Vec<ConfirmRequest>>>,
    }

    impl RecordingConfirm {
        pub(crate) fn answering(answer: bool) -> Self {
            Self {
                answer: Arc::new(Mutex::new(answer)),
                asked: Arc::default(),
            }
        }

        pub(crate) fn asked(&self) -> Vec<ConfirmRequest> {
            self.asked.lock().unwrap().clone()
        }
    }

    impl ConfirmPort for RecordingConfirm {
        fn confirm(&self, request: ConfirmRequest) -> BoxFuture<'static, bool> {
            self.asked.lock().unwrap().push(request);
            let answer = *self.answer.lock().unwrap();
            Box::pin(async move { answer })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::fake::RecordingConfirm;
    use super::*;

    /// Every request, each string holding a Unicode dash, a bidirectional
    /// override and a control character.
    fn every_request() -> Vec<ConfirmRequest> {
        let s = |text: &str| format!("{text}\u{2013}\u{202e}\t");
        vec![
            ConfirmRequest::TrustFolder {
                path: s("C:\\w"),
                will_read: vec![s("AGENTS.md")],
                git_writes: true,
            },
            ConfirmRequest::AttachFolder { path: s("C:\\p") },
            ConfirmRequest::RunCommand {
                text: s("rm -rf x"),
                cwd: s("src"),
                mode: CommandMode::PowerShell,
                background: false,
            },
            ConfirmRequest::KeepAuthority {
                path: s(".github/ci.yml"),
                added: 3,
                removed: 1,
            },
            ConfirmRequest::AllowAlways {
                workspace: s("C:\\w"),
                argv: vec![s("cargo"), s("test")],
                program: s("C:\\cargo.exe"),
                cwd: s("."),
            },
            ConfirmRequest::ReleaseWithheld {
                conversation: s("c1"),
                label: s("Hosted"),
                call: s("call_1"),
            },
            ConfirmRequest::FirstRemoteSend {
                conversation: s("c1"),
                label: s("Hosted"),
                earlier_local_results: 2,
            },
            ConfirmRequest::SendImages {
                conversation: s("c1"),
                label: s("Hosted"),
                images: 2,
            },
            ConfirmRequest::EnableMcpServer {
                name: s("srv"),
                from: s(".lattice/mcp.json"),
                command_line: s("node x.js"),
                program: s("C:\\node.exe"),
                cwd: s("C:\\w"),
                env_names: vec![s("TOKEN")],
            },
            ConfirmRequest::McpCall {
                server: s("srv"),
                tool: s("write"),
                arguments: s("{\"path\": \"a\"}"),
            },
            ConfirmRequest::AllowMcpTool {
                server: s("srv"),
                tool: s("read"),
                from: s("your MCP settings"),
            },
            ConfirmRequest::AgentPermission {
                agent: s("Claude Code (your subscription)"),
                action: s("Edit a.txt"),
                detail: s("{\"path\": \"a.txt\"}"),
                diff: vec![s("a.txt:"), s("- hello"), s("+ goodbye")],
            },
            ConfirmRequest::BrowserAct {
                site: s("x.com"),
                action: s("click at (612, 304)"),
                what: s("the Post button"),
                effect: s("something another person will see"),
            },
            ConfirmRequest::AutoMode {
                stop_keys: s("Ctrl+Alt+End"),
            },
            ConfirmRequest::DesktopAct {
                app: s("Checkout - Amazon.com"),
                action: s("click at (980, 512)"),
                what: s("the Place your order button"),
                effect: s("a purchase or an order"),
            },
        ]
    }

    /// CP1: every dialog's default button refuses, and it ignores input for
    /// half a second.
    /// Mutant: the confirming button as the default.
    #[test]
    fn cp1_the_refusing_button_is_the_default() {
        for request in every_request() {
            let dialog = request.dialog();
            assert_eq!(dialog.default, DefaultButton::Refuse, "{request:?}");
            assert_eq!(dialog.ignore_input_for, IGNORE_INPUT_FOR);
            assert_eq!(IGNORE_INPUT_FOR.as_millis(), 500);
        }
    }

    /// CP2: what came from the model or the folder is shown escaped: no
    /// character outside printable ASCII reaches a dialog, and the escapes
    /// name each character.
    /// Mutant: text passed through unescaped.
    #[test]
    fn cp2_every_dialog_shows_only_printable_ascii() {
        for request in every_request() {
            let dialog = request.dialog();
            for line in std::iter::once(&dialog.title).chain(&dialog.lines) {
                assert!(
                    line.chars().all(|c| (' '..='~').contains(&c)),
                    "{request:?}: {line:?}"
                );
            }
            let all = format!("{} {}", dialog.title, dialog.lines.join(" "));
            assert!(all.contains("\\u{2013}\\u{202e}\\u{0009}"), "{all}");
        }
        assert_eq!(escape_for_dialog("a\u{2013}b"), "a\\u{2013}b");
        assert_eq!(escape_for_dialog("C:\\x y"), "C:\\x y");
    }

    /// CP3: a request the reader refused is not asked again when the page
    /// asks; a native action, or a fresh call (a new key), asks again.
    /// Mutant: the refusal forgotten.
    #[test]
    fn cp3_a_refused_request_is_not_asked_again_from_the_page() {
        let port = RecordingConfirm::answering(false);
        let confirmer = Confirmer::new(Arc::new(port.clone()));
        let request = || ConfirmRequest::AttachFolder {
            path: "C:\\p".into(),
        };
        let ask = |key: &str, initiated| {
            futures::executor::block_on(confirmer.ask(key, request(), initiated))
        };
        assert!(!ask("attach:p", Initiated::Page));
        assert_eq!(port.asked().len(), 1);
        assert!(!ask("attach:p", Initiated::Page));
        assert_eq!(port.asked().len(), 1, "the page cannot ask again");
        assert!(!ask("attach:p", Initiated::Native));
        assert_eq!(port.asked().len(), 2, "the reader's own action asks again");
        assert!(!ask("run:c1:call_2", Initiated::Page));
        assert_eq!(port.asked().len(), 3, "a fresh call asks");
        *port.answer.lock().unwrap() = true;
        assert!(ask("attach:p", Initiated::Native));
        assert_eq!(port.asked().len(), 4);
        assert!(
            ask("attach:p", Initiated::Page),
            "confirmed since: asked again"
        );
        assert_eq!(port.asked().len(), 5);
    }
}
