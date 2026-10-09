//! The conversation payloads against their JSON goldens (spec §13.6).
//!
//! `tests/goldens/conversation/` holds one file per payload and per event kind.
//! Each test value here is serialised (`serde_json::to_string_pretty`, then a
//! newline) and compared with its file byte for byte, after the file's line
//! ends are read as LF (a Windows checkout may hand back CRLF); each file is
//! also read back into its type and must equal the value. The TypeScript
//! interface will read the same files with its type guards, so a change to a
//! payload's shape is a reviewed diff of a golden, on both sides.
//!
//! `LATTICE_BLESS_GOLDENS=1` writes the files instead of comparing them; a
//! blessed change still has to pass review as a diff.

use std::collections::BTreeSet;
use std::fmt::Debug;
use std::path::{Path, PathBuf};

use lattice_protocol::chat::{ChatTurn, Fact, Role, Shown, Stage};
use lattice_protocol::conversation::*;
use lattice_protocol::{ChatChoice, Locality};
use serde::Serialize;
use serde::de::DeserializeOwned;

/// Reads a golden's text back into the fixture's type and compares.
type RoundTrip = Box<dyn Fn(&str) -> Result<(), String>>;

struct Fixture {
    name: String,
    text: String,
    round_trip: RoundTrip,
}

fn fixture<T>(name: &str, value: T) -> Fixture
where
    T: Serialize + DeserializeOwned + PartialEq + Debug + 'static,
{
    let text = serde_json::to_string_pretty(&value).expect("a payload serialises") + "\n";
    Fixture {
        name: name.to_owned(),
        text,
        round_trip: Box::new(move |text| {
            let back: T = serde_json::from_str(text).map_err(|e| e.to_string())?;
            if back == value {
                Ok(())
            } else {
                Err(format!("read back as {back:?}, not {value:?}"))
            }
        }),
    }
}

fn dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("goldens")
        .join("conversation")
}

fn shown() -> Shown {
    Shown {
        locality: Locality::Local,
        label: "Local".into(),
    }
}

fn badge() -> WorkspaceBadge {
    WorkspaceBadge {
        id: "0f1e2d3c4b5a6978".into(),
        name: "project".into(),
        path: "<workspace>".into(),
        trusted: true,
    }
}

fn summary() -> ConversationSummary {
    ConversationSummary {
        id: "0123456789ab".into(),
        title: "Fix the build".into(),
        created: 1_790_000_000.5,
        updated: 1_790_000_100.25,
        turns: 4,
        pinned_provider: "local".into(),
        workspace: Some(badge()),
        mode: Mode::Agent,
        origin: Origin::Native,
        running: true,
        needs_you: true,
    }
}

fn user_turn() -> ChatTurn {
    ChatTurn {
        id: "a1b2c3d4e5f6".into(),
        ts: 1_790_000_100.25,
        role: Role::User,
        text: "Why does the build fail?".into(),
        tools: vec![],
        facts: vec![],
        unsupported: vec![],
        provider: "".into(),
        error: "".into(),
        constrained: false,
        truncated: false,
        cancelled: false,
        prompt_tokens: None,
        completion_tokens: None,
    }
}

fn answer_turn() -> ChatTurn {
    ChatTurn {
        id: "f6e5d4c3b2a1".into(),
        ts: 1_790_000_160.75,
        role: Role::Assistant,
        text: "A missing import; it is staged for review.".into(),
        tools: vec![],
        facts: vec![Fact::default()],
        unsupported: vec![],
        provider: "ollama:qwen3:8b".into(),
        error: "".into(),
        constrained: false,
        truncated: false,
        cancelled: false,
        prompt_tokens: Some(1200),
        completion_tokens: None,
    }
}

fn command_detail() -> ApprovalDetail {
    ApprovalDetail::Command {
        text: "cargo test -p lattice-core \\u{2013}quiet".into(),
        cwd: "crates".into(),
        mode: CommandMode::PowerShell,
        timeout_s: 600,
        remote: None,
        staged_waiting: 0,
        background: false,
    }
}

fn change_view() -> ChangeView {
    ChangeView {
        id: "ch_00112233445566aa".into(),
        path: "src/lib.rs".into(),
        kind: ChangeKind::Edit,
        authority: false,
        state: ChangeState::Pending,
        origin: ChangeOrigin::Agent {
            turn: "a1b2c3d4e5f6".into(),
            call: "call_7".into(),
        },
        added: 3,
        removed: 1,
        binary: false,
    }
}

/// One sample of every event kind, and its `type`. The match is exhaustive, so
/// a new kind does not compile until it is named here and given a sample.
fn event_type(kind: &ConversationEventKind) -> &'static str {
    use ConversationEventKind as K;
    match kind {
        K::TurnStarted { .. } => "turn_started",
        K::Stage { .. } => "stage",
        K::Delta { .. } => "delta",
        K::DeltasDropped { .. } => "deltas_dropped",
        K::ToolCall { .. } => "tool_call",
        K::ToolOutput { .. } => "tool_output",
        K::Withheld { .. } => "withheld",
        K::ApprovalRequested { .. } => "approval_requested",
        K::ApprovalResolved { .. } => "approval_resolved",
        K::Question { .. } => "question",
        K::QuestionAnswered { .. } => "question_answered",
        K::TaskSuggested { .. } => "task_suggested",
        K::TaskSettled { .. } => "task_settled",
        K::ArtifactSaved { .. } => "artifact_saved",
        K::TodosUpdated { .. } => "todos_updated",
        K::PlanProposed { .. } => "plan_proposed",
        K::PlanSettled { .. } => "plan_settled",
        K::ImagesAttached { .. } => "images_attached",
        K::Staged { .. } => "staged",
        K::Reviewed { .. } => "reviewed",
        K::Conflict { .. } => "conflict",
        K::Checkpoint { .. } => "checkpoint",
        K::CommandStarted { .. } => "command_started",
        K::CommandProgress { .. } => "command_progress",
        K::CommandExited { .. } => "command_exited",
        K::CommandEffect { .. } => "command_effect",
        K::Steered { .. } => "steered",
        K::Queued { .. } => "queued",
        K::Dequeued { .. } => "dequeued",
        K::Notice { .. } => "notice",
        K::TurnSaved { .. } => "turn_saved",
        K::Error { .. } => "error",
        K::TurnEnded { .. } => "turn_ended",
    }
}

const EVENT_TYPES: [&str; 33] = [
    "turn_started",
    "stage",
    "delta",
    "deltas_dropped",
    "tool_call",
    "tool_output",
    "withheld",
    "approval_requested",
    "approval_resolved",
    "question",
    "question_answered",
    "staged",
    "reviewed",
    "conflict",
    "checkpoint",
    "command_started",
    "command_progress",
    "command_exited",
    "command_effect",
    "steered",
    "queued",
    "dequeued",
    "notice",
    "turn_saved",
    "error",
    "turn_ended",
    "task_suggested",
    "task_settled",
    "artifact_saved",
    "todos_updated",
    "plan_proposed",
    "plan_settled",
    "images_attached",
];

fn sample_events() -> Vec<ConversationEventKind> {
    use ConversationEventKind as K;
    vec![
        K::TurnStarted {
            turn: "a1b2c3d4e5f6".into(),
            kind: TurnKind::Agent,
            mode: Mode::Agent,
            label: "Local".into(),
            locality: Locality::Local,
        },
        K::Stage {
            stage: Stage::Writing,
            detail: "writing the answer".into(),
        },
        K::Delta {
            text: "The build fails because".into(),
        },
        K::DeltasDropped { n: 12 },
        K::ToolCall {
            call_id: "call_1".into(),
            tool: "read_file".into(),
            summary: "Read src/lib.rs".into(),
            target: Some("src/lib.rs".into()),
        },
        K::ToolOutput {
            call_id: "call_1".into(),
            preview: "src/lib.rs (412 lines; lines 1-200 shown)".into(),
            withheld: false,
            truncated: true,
        },
        K::Withheld {
            call_id: "call_2".into(),
        },
        K::ApprovalRequested {
            call_id: "call_3".into(),
            kind: ApprovalKind::Command,
            detail: command_detail(),
            allow_always_offer: true,
        },
        K::ApprovalResolved {
            call_id: "call_3".into(),
            approved: true,
            by: DecidedBy::Reader,
        },
        K::Question {
            call_id: "call_4".into(),
            text: "Which crate should I change?".into(),
            options: vec!["lattice-core".into(), "lattice-sys".into()],
        },
        K::QuestionAnswered {
            call_id: "call_4".into(),
        },
        K::Staged {
            change: "ch_00112233445566aa".into(),
            path: "src/lib.rs".into(),
            added: 3,
            removed: 1,
            authority: false,
        },
        K::Reviewed {
            change: "ch_00112233445566aa".into(),
            result: ReviewResult::Kept {
                changed_after: false,
            },
        },
        K::Conflict {
            change: "ch_00112233445566bb".into(),
            path: "Cargo.toml".into(),
            reason: "The file changed outside Lattice.".into(),
        },
        K::Checkpoint {
            id: 3,
            kind: CheckpointKind::Git,
            reason: CheckpointReason::BeforeCommand {
                call_id: "call_3".into(),
            },
        },
        K::CommandStarted {
            call_id: "call_3".into(),
            mode: CommandMode::Direct,
        },
        K::CommandProgress {
            call_id: "call_3".into(),
            bytes: 65_536,
            lines: 812,
            tail_preview: "test result: ok.".into(),
        },
        K::CommandExited {
            call_id: "call_3".into(),
            code: Some(0),
            duration_ms: 41_250,
            reason: ExitReason::Exited,
        },
        K::CommandEffect {
            call_id: "call_3".into(),
            before: 3,
            after: 4,
            files: vec![
                ChangedPath {
                    path: "target/report.txt".into(),
                    change: PathChange::Added,
                },
                ChangedPath {
                    path: "src/gen.rs".into(),
                    change: PathChange::Modified,
                },
            ],
        },
        K::Steered {
            text: "Only touch the tests.".into(),
        },
        K::Queued {
            queued_id: "q_1".into(),
            text: "Then run clippy.".into(),
            position: 1,
        },
        K::Dequeued {
            queued_id: "q_1".into(),
            outcome: DequeueOutcome::Sent {
                turn: "0a0b0c0d0e0f".into(),
            },
        },
        K::Notice {
            text: "This turn's tool history could not be saved.".into(),
        },
        K::TurnSaved {
            turn: Box::new(answer_turn()),
            saved: true,
        },
        K::Error {
            message: "The model did not answer.".into(),
        },
        K::TurnEnded {
            turn: "a1b2c3d4e5f6".into(),
            status: TurnStatus::MaxTurns,
        },
        K::TaskSuggested {
            task: "task_0011223344556677".into(),
            title: "Fix the stale README badge".into(),
            summary: "The README's build badge still names the old workflow.".into(),
            prompt: "In README.md, the build badge points at .github/workflows/ci.yml, which was renamed to build.yml. Point the badge at build.yml and check that nothing else names ci.yml.".into(),
        },
        K::TaskSettled {
            task: "task_0011223344556677".into(),
            outcome: TaskOutcome::Started {
                conversation: "0a0b0c0d0e0f".into(),
            },
        },
        K::ArtifactSaved {
            name: "plan".into(),
            title: "Plan: natural sort in the explorer".into(),
            kind: ArtifactKind::Markdown,
            version: 2,
            bytes: 1_843,
        },
        K::TodosUpdated {
            items: vec![
                TodoItem {
                    content: "Read how the explorer sorts its rows".into(),
                    status: TodoStatus::Done,
                },
                TodoItem {
                    content: "Compare digit runs by value".into(),
                    status: TodoStatus::InProgress,
                },
                TodoItem {
                    content: "Add a test for file2 before file10".into(),
                    status: TodoStatus::Pending,
                },
            ],
        },
        K::PlanProposed {
            plan: "plan_8899aabbccddeeff".into(),
            title: "Natural sort in the explorer".into(),
            text: "1. Compare digit runs by value in `tree::order`.\n2. Add a test: file2 before file10.".into(),
        },
        K::PlanSettled {
            plan: "plan_8899aabbccddeeff".into(),
            outcome: PlanOutcome::Approved,
        },
        K::ImagesAttached {
            turn: "a1b2c3d4e5f6".into(),
            count: 2,
            bytes: 184_320,
        },
    ]
}

fn event(seq: u64, kind: ConversationEventKind) -> ConversationEvent {
    ConversationEvent {
        seq,
        at: 1_790_000_100.0 + seq as f64 / 4.0,
        kind,
    }
}

fn fixtures() -> Vec<Fixture> {
    let mut all = vec![
        fixture(
            "archive_key",
            ArchiveKey {
                id: "0123456789ab".into(),
                created: 1_790_000_000.5,
            },
        ),
        fixture("conversation_summary", summary()),
        fixture(
            "archived_summary",
            ArchivedSummary {
                key: ArchiveKey {
                    id: "ba9876543210".into(),
                    created: 1_780_000_000.0,
                },
                title: "Old question".into(),
                updated: 1_780_000_500.0,
                turns: 2,
                evicted_at: 1_790_000_000.0,
                file: "ba9876543210-2.jsonl".into(),
                reason: ArchiveReason::Reader,
                transcript_missing: false,
            },
        ),
        fixture(
            "chat_list",
            ChatList {
                conversations: vec![summary()],
                archived: vec![],
                store_dir: "<globals>/lattice_chat".into(),
                installed: false,
                index_unreadable: false,
                archive_unreadable: false,
                shared_writes: false,
                limit: 60,
            },
        ),
        fixture(
            "send_request",
            SendRequest {
                conversation: None,
                text: "Why does the build fail?".into(),
                choice: "local".into(),
                shown: shown(),
                mode: Mode::Agent,
                workspace: Some("0f1e2d3c4b5a6978".into()),
                edit_of: None,
                project: None,
                images: Vec::new(),
            },
        ),
        // A new chat in a project, with no folder.
        fixture(
            "send_request_project",
            SendRequest {
                conversation: None,
                text: "Draft the investor summary.".into(),
                choice: "local".into(),
                shown: shown(),
                mode: Mode::Ask,
                workspace: None,
                edit_of: None,
                project: Some("p_1a2b3c4d5e6f".into()),
                images: Vec::new(),
            },
        ),
        // A message with an image attached (a 1x1 PNG).
        fixture(
            "send_request_images",
            SendRequest {
                conversation: Some("0123456789ab".into()),
                text: "Why is this button grey?".into(),
                choice: "local".into(),
                shown: shown(),
                mode: Mode::Agent,
                workspace: Some("0f1e2d3c4b5a6978".into()),
                edit_of: None,
                project: None,
                images: vec![UserImage {
                    media_type: "image/png".into(),
                    base64: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGBgAAAABQABpfZFQAAAAABJRU5ErkJggg==".into(),
                }],
            },
        ),
        fixture(
            "regenerate_request",
            RegenerateRequest {
                conversation: "0123456789ab".into(),
                choice: "endpoint:lab".into(),
                shown: Shown {
                    locality: Locality::Remote,
                    label: "Lab GPU".into(),
                },
                mode: Mode::Ask,
            },
        ),
        fixture(
            "accepted_started",
            Accepted::Started {
                conversation: Box::new(summary()),
                user_turn: Box::new(user_turn()),
                turn_kind: TurnKind::Agent,
            },
        ),
        fixture(
            "accepted_queued",
            Accepted::Queued {
                conversation: "0123456789ab".into(),
                queued_id: "q_1".into(),
                position: 1,
            },
        ),
        fixture(
            "snapshot",
            Snapshot {
                conversation: summary(),
                turns: vec![user_turn(), answer_turn()],
                events: vec![event(
                    7,
                    ConversationEventKind::ToolCall {
                        call_id: "call_1".into(),
                        tool: "grep".into(),
                        summary: "Search for \"fn main\"".into(),
                        target: None,
                    },
                )],
                last_seq: 7,
                queued: vec![QueuedMessage {
                    queued_id: "q_1".into(),
                    text: "Then run clippy.".into(),
                    shown: shown(),
                }],
                answering_elsewhere: false,
            },
        ),
        fixture(
            "core_status",
            CoreStatus {
                runtime: "lattice-agents 0.1.0".into(),
                agent_turns: 1,
                shared_writes: false,
                refusal: None,
            },
        ),
        fixture(
            "chat_choice",
            ChatChoice {
                id: "auto".into(),
                label: "Auto".into(),
                detail: "A model on this machine when one is ready.".into(),
                locality: Locality::Local,
                ready: true,
                refusal: None,
            },
        ),
        fixture(
            "artifact_view",
            ArtifactView {
                name: "plan".into(),
                title: "Plan: natural sort in the explorer".into(),
                kind: ArtifactKind::Markdown,
                version: 2,
                versions: 2,
                text: "## Steps\n\n1. Compare digit runs by length first.\n".into(),
            },
        ),
        fixture("decision_approve", Decision::Approve),
        fixture(
            "decision_reject",
            Decision::Reject {
                note: Some("Use the existing helper.".into()),
            },
        ),
        fixture("decision_release_withheld", Decision::ReleaseWithheld),
        fixture("match_scope", MatchScope::Exact),
        fixture(
            "allow_entry",
            AllowEntry {
                id: "e_0123456789abcdef".into(),
                workspace: "0f1e2d3c4b5a6978".into(),
                argv: vec!["cargo".into(), "test".into()],
                program: "<tools>\\cargo.exe".into(),
                cwd: "".into(),
                created_at: 1_790_000_000.0,
                scope: MatchScope::Exact,
                revoked_at: None,
            },
        ),
        fixture(
            "approval_detail_mcp",
            ApprovalDetail::Mcp {
                server: "files".into(),
                tool: "read".into(),
                arguments_preview: "{\"path\": \"notes.md\"}".into(),
            },
        ),
        fixture(
            "approval_detail_browser",
            ApprovalDetail::Browser {
                site: "x.com".into(),
                action: "click at (612, 304)".into(),
                what: "the Post button".into(),
                effect: "share".into(),
            },
        ),
        fixture(
            "approval_detail_desktop",
            ApprovalDetail::Desktop {
                app: "Checkout - Amazon.com".into(),
                action: "click at (980, 512)".into(),
                what: "the Place your order button".into(),
                effect: "buy".into(),
            },
        ),
        fixture(
            "decided_by_standing",
            DecidedBy::Standing {
                entry: "e_0123456789abcdef".into(),
            },
        ),
        fixture(
            "decided_by_policy",
            DecidedBy::Policy {
                rule: "read tools".into(),
            },
        ),
        fixture(
            "workspace_view",
            WorkspaceView {
                workspace: badge(),
                trust: TrustState::Trusted,
                rules_found: vec!["AGENTS.md".into(), ".lattice/rules/style.md".into()],
                git_top_level: Some("<workspace>".into()),
            },
        ),
        fixture("trust_state", TrustState::Revoked),
        fixture(
            "ranked_path",
            RankedPath {
                path: "src/main.rs".into(),
                score: 42,
                positions: vec![4, 5, 6, 7],
            },
        ),
        fixture(
            "view_ref_file",
            ViewRef::File {
                workspace: "0f1e2d3c4b5a6978".into(),
                path: "src/lib.rs".into(),
            },
        ),
        fixture(
            "view_ref_staged",
            ViewRef::Staged {
                conversation: "0123456789ab".into(),
                change: "ch_00112233445566aa".into(),
            },
        ),
        fixture(
            "view_ref_output",
            ViewRef::Output {
                conversation: "0123456789ab".into(),
                call_id: "call_3".into(),
            },
        ),
        fixture(
            "lines",
            Lines {
                from: 1,
                total: 412,
                lines: vec!["//! A crate.".into(), "".into()],
                truncated: false,
            },
        ),
        fixture(
            "change_set",
            ChangeSet {
                changes: vec![
                    change_view(),
                    ChangeView {
                        id: "ch_00112233445566cc".into(),
                        path: ".github/workflows/ci.yml".into(),
                        kind: ChangeKind::Create,
                        authority: true,
                        state: ChangeState::Conflict {
                            reason: "The file appeared meanwhile.".into(),
                        },
                        origin: ChangeOrigin::Restore { to: 2 },
                        added: 20,
                        removed: 0,
                        binary: false,
                    },
                    ChangeView {
                        id: "ch_00112233445566dd".into(),
                        path: "target/report.txt".into(),
                        kind: ChangeKind::CommandUndo,
                        authority: false,
                        state: ChangeState::OnDisk,
                        origin: ChangeOrigin::Command {
                            call: "call_3".into(),
                        },
                        added: 0,
                        removed: 0,
                        binary: true,
                    },
                ],
                waiting: 2,
                command_running: false,
            },
        ),
        fixture(
            "file_diff",
            FileDiff {
                change: "ch_00112233445566aa".into(),
                path: "src/lib.rs".into(),
                against: Against::Previous,
                snapshot: 3,
                binary: false,
                hunks: vec![Hunk {
                    id: "9a8b7c6d5e4f3a2b".into(),
                    state: HunkState::Pending,
                    old_start: 10,
                    old_lines: 3,
                    new_start: 10,
                    new_lines: 4,
                    section: "fn main() {".into(),
                    lines: vec![
                        DiffLine {
                            kind: DiffLineKind::Context,
                            text: "use std::fs;".into(),
                        },
                        DiffLine {
                            kind: DiffLineKind::Remove,
                            text: "use std::io;".into(),
                        },
                        DiffLine {
                            kind: DiffLineKind::Add,
                            text: "use std::io::{self, Read};".into(),
                        },
                    ],
                }],
                truncated: false,
            },
        ),
        fixture(
            "review_op_keep_hunks",
            ReviewOp::Keep {
                change: "ch_00112233445566aa".into(),
                hunks: Some(vec!["9a8b7c6d5e4f3a2b".into()]),
            },
        ),
        fixture(
            "review_op_undo",
            ReviewOp::Undo {
                change: "ch_00112233445566aa".into(),
                hunks: None,
                note: Some("Keep the old import.".into()),
            },
        ),
        fixture("review_op_keep_all", ReviewOp::KeepAll),
        fixture("review_op_undo_all", ReviewOp::UndoAll { note: None }),
        fixture(
            "review_outcome",
            ReviewOutcome {
                results: vec![
                    ChangeResult {
                        change: "ch_00112233445566aa".into(),
                        path: "src/lib.rs".into(),
                        result: ReviewResult::PartlyKept { remaining_hunks: 1 },
                    },
                    ChangeResult {
                        change: "ch_00112233445566cc".into(),
                        path: ".github/workflows/ci.yml".into(),
                        result: ReviewResult::Skipped {
                            reason: "Authority files are kept one at a time.".into(),
                        },
                    },
                    ChangeResult {
                        change: "ch_00112233445566ee".into(),
                        path: "README.md".into(),
                        result: ReviewResult::Rebased,
                    },
                    ChangeResult {
                        change: "ch_00112233445566ff".into(),
                        path: "big.bin".into(),
                        result: ReviewResult::Failed {
                            reason: "That file is too large for Lattice to keep a copy of, so nothing was written.".into(),
                        },
                    },
                ],
            },
        ),
        fixture(
            "checkpoint_view",
            CheckpointView {
                id: 4,
                kind: CheckpointKind::Copies,
                reason: CheckpointReason::BeforeKeep,
                at_turn: Some("a1b2c3d4e5f6".into()),
                exposed: true,
                omitted: 2,
                bytes: 18_432,
            },
        ),
        fixture(
            "follow_batch",
            FollowBatch {
                conversation: "0123456789ab".into(),
                first_seq: 8,
                last_seq: 9,
                urgent: true,
                events: vec![
                    event(
                        8,
                        ConversationEventKind::Delta {
                            text: "Here is".into(),
                        },
                    ),
                    event(
                        9,
                        ConversationEventKind::ApprovalRequested {
                            call_id: "call_3".into(),
                            kind: ApprovalKind::Command,
                            detail: command_detail(),
                            allow_always_offer: false,
                        },
                    ),
                ],
            },
        ),
    ];
    for (index, kind) in sample_events().into_iter().enumerate() {
        let name = format!("event_{}", event_type(&kind));
        all.push(fixture(&name, event(index as u64 + 1, kind)));
    }
    all
}

fn read_lf(path: &Path) -> Option<String> {
    let text = std::fs::read_to_string(path).ok()?;
    Some(text.replace("\r\n", "\n"))
}

#[test]
fn every_payload_matches_its_golden_byte_for_byte() {
    let bless = std::env::var_os("LATTICE_BLESS_GOLDENS").is_some_and(|v| v == "1");
    let fixtures = fixtures();
    if bless {
        std::fs::create_dir_all(dir()).unwrap();
        for fixture in &fixtures {
            std::fs::write(dir().join(format!("{}.json", fixture.name)), &fixture.text).unwrap();
        }
    }
    let mut problems = Vec::new();
    for fixture in &fixtures {
        let path = dir().join(format!("{}.json", fixture.name));
        match read_lf(&path) {
            None => problems.push(format!("{}: missing", fixture.name)),
            Some(on_disk) if on_disk != fixture.text => {
                let line = on_disk
                    .lines()
                    .zip(fixture.text.lines())
                    .position(|(a, b)| a != b)
                    .map_or_else(|| "the length".to_owned(), |n| format!("line {}", n + 1));
                problems.push(format!("{}: differs at {line}", fixture.name));
            }
            Some(_) => {}
        }
    }
    assert!(
        problems.is_empty(),
        "payloads no longer match their goldens:\n{}",
        problems.join("\n")
    );
    println!("{} goldens match", fixtures.len());
}

#[test]
fn every_golden_reads_back_as_its_value() {
    for fixture in fixtures() {
        let path = dir().join(format!("{}.json", fixture.name));
        let text = read_lf(&path).unwrap_or_else(|| panic!("{} is missing", path.display()));
        (fixture.round_trip)(&text).unwrap_or_else(|e| panic!("{}: {e}", fixture.name));
    }
}

#[test]
fn every_event_kind_has_a_golden_and_events_are_flat() {
    let names: BTreeSet<String> = fixtures().into_iter().map(|f| f.name).collect();
    for kind in EVENT_TYPES {
        assert!(
            names.contains(&format!("event_{kind}")),
            "no golden for the event kind {kind}"
        );
    }
    let samples = sample_events();
    assert_eq!(samples.len(), EVENT_TYPES.len(), "one sample per kind");
    for (index, kind) in samples.into_iter().enumerate() {
        let expected = event_type(&kind);
        let typed = event(index as u64 + 1, kind);
        // The typed text, not a `Value`: a key written twice by `flatten`
        // shows here, and reads back as an error.
        let text = serde_json::to_string(&typed).unwrap();
        let back: ConversationEvent =
            serde_json::from_str(&text).unwrap_or_else(|e| panic!("{text}: {e}"));
        assert_eq!(back, typed);
        let value: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(value["type"], expected);
        assert!(value["seq"].is_u64() && value["at"].is_f64(), "{value}");
        // `type` also tags nested enums (a detail, a result); `seq` and `at`
        // belong to the event alone.
        for key in ["\"seq\":", "\"at\":"] {
            assert_eq!(text.matches(key).count(), 1, "{key} once in {text}");
        }
    }
}

#[test]
fn no_golden_on_disk_is_unaccounted_for() {
    let names: BTreeSet<String> = fixtures()
        .into_iter()
        .map(|f| format!("{}.json", f.name))
        .collect();
    let on_disk: BTreeSet<String> = std::fs::read_dir(dir())
        .expect("the goldens folder exists")
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    let stale: Vec<_> = on_disk.difference(&names).collect();
    assert!(stale.is_empty(), "goldens no fixture produces: {stale:?}");
}
