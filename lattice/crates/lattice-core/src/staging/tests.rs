//! The staging primitives in memory and against a sidecar in a temporary
//! folder (the chat core's spec §7.4.1: the unique-match rule, line ends
//! and the byte-order mark; ST2; a reopened record's staged view).

use std::sync::Arc;

use lattice_protocol::conversation::{ChangeKind, ChangeState, Mode, Origin};

use super::{
    Action, BOM, Base, OpError, Proposal, Staging, added_removed, apply_op, encode, eol_of, text_of,
};
use crate::convo::item::{BaseState, EditOp, Eol, NewState};
use crate::convo::sidecar::{NewMeta, SidecarStore};
use crate::testkit::TempDir;
use crate::tools::read::{Overlay, Staged};

fn op(old: &str, new: &str, all: bool) -> EditOp {
    EditOp {
        old_string: old.into(),
        new_string: new.into(),
        replace_all: all,
    }
}

#[test]
fn line_ends_are_named_by_what_the_bytes_hold() {
    assert_eq!(eol_of(b"a\nb\n"), Eol::Lf);
    assert_eq!(eol_of(b"a\r\nb\r\n"), Eol::Crlf);
    assert_eq!(eol_of(b"a\r\nb\n"), Eol::Mixed);
    assert_eq!(eol_of(b"one line"), Eol::None);
    assert_eq!(eol_of(b""), Eol::None);
    assert_eq!(eol_of(b"\n"), Eol::Lf);
}

#[test]
fn encode_applies_uniform_line_ends_and_the_bom() {
    assert_eq!(encode("a\nb\n", Eol::Crlf, false), b"a\r\nb\r\n");
    assert_eq!(encode("a\r\nb\n", Eol::Crlf, false), b"a\r\nb\r\n");
    assert_eq!(encode("a\r\nb\r\n", Eol::Lf, false), b"a\nb\n");
    assert_eq!(encode("a\r\nb\n", Eol::Mixed, false), b"a\r\nb\n");
    assert_eq!(encode("x", Eol::None, true), [BOM, b"x"].concat());
    assert_eq!(
        encode("\u{feff}x", Eol::None, true),
        [BOM, b"x"].concat(),
        "a BOM in the content is not doubled"
    );
}

/// §7.4.1: the unique-match rule and its sentences.
#[test]
fn an_edit_needs_one_exact_match_unless_replace_all() {
    assert_eq!(apply_op(b"a b a", &op("b", "c", false)).unwrap(), b"a c a");
    assert_eq!(
        apply_op(b"a b a", &op("x", "c", false)),
        Err(OpError::NotFound)
    );
    assert_eq!(
        apply_op(b"a b a", &op("a", "c", false)),
        Err(OpError::Many(2))
    );
    assert_eq!(
        OpError::Many(2).sentence(),
        "old_string occurs 2 times; give more context, or set replace_all"
    );
    assert_eq!(OpError::NotFound.sentence(), "old_string was not found");
    assert_eq!(apply_op(b"a b a", &op("a", "c", true)).unwrap(), b"c b c");
    assert_eq!(
        apply_op(b"a", &op("", "c", false)),
        Err(OpError::NotFound),
        "an empty old_string matches nothing"
    );
}

/// §7.4.1: a CRLF file matches on its LF form and stays CRLF; a mixed one
/// matches raw; the BOM stays.
/// Mutant: matching always on the raw text (the LF-form edit is not found).
#[test]
fn crlf_files_match_on_their_lf_form_and_stay_crlf() {
    let crlf = b"one\r\ntwo\r\nthree\r\n";
    assert_eq!(
        apply_op(crlf, &op("one\ntwo", "uno\ndos\ndos", false)).unwrap(),
        b"uno\r\ndos\r\ndos\r\nthree\r\n"
    );
    assert_eq!(
        apply_op(crlf, &op("one\r\ntwo", "1\r\n2", false)).unwrap(),
        b"1\r\n2\r\nthree\r\n",
        "CRLF in the strings is read as LF too"
    );
    let mixed = b"one\r\ntwo\nthree\n";
    assert_eq!(
        apply_op(mixed, &op("one\ntwo", "x", false)),
        Err(OpError::NotFound),
        "mixed line ends match raw"
    );
    assert_eq!(
        apply_op(mixed, &op("one\r\ntwo", "x", false)).unwrap(),
        b"x\nthree\n"
    );
    let bom = [BOM, b"a\r\nb\r\n"].concat();
    assert_eq!(
        apply_op(&bom, &op("a\nb", "c\nd", false)).unwrap(),
        [BOM, b"c\r\nd\r\n"].concat()
    );
}

#[test]
fn only_utf8_text_is_edited() {
    assert!(matches!(text_of(b"\xFF\xFEa\0"), Err(OpError::NotText(_))));
    assert!(matches!(text_of(b"a\0b"), Err(OpError::NotText(_))));
    assert!(matches!(text_of(b"\xC3("), Err(OpError::NotText(_))));
    assert_eq!(text_of(&[BOM, b"x"].concat()).unwrap(), ("x", true));
}

#[test]
fn added_and_removed_lines_count_the_line_diff() {
    assert_eq!(added_removed("a\nb\nc\n", "a\nB\nc\nd\n"), (2, 1));
    assert_eq!(added_removed("", "x\ny\n"), (2, 0));
    assert_eq!(added_removed("x\n", ""), (0, 1));
}

#[test]
fn a_large_base_is_hashed_in_pieces() {
    let bytes = [BOM.to_vec(), b"a\nb\nc".repeat(1000)].concat();
    let whole = Base::of_bytes(bytes.clone());
    let pieces = Base::of_reader(&bytes[..]).unwrap();
    match (&whole.state, &pieces.state) {
        (
            BaseState::Present {
                sha256: a,
                bytes: n,
                bom: true,
                ..
            },
            BaseState::Present {
                sha256: b,
                bytes: m,
                bom: true,
                eol: Eol::None,
            },
        ) => {
            assert_eq!(a, b);
            assert_eq!(n, m);
        }
        other => panic!("{other:?}"),
    }
    assert_eq!(whole.lines, pieces.lines);
    assert_eq!(pieces.bytes, None);
}

struct Record {
    _dir: TempDir,
    store: SidecarStore,
    staging: Staging,
}

fn record() -> Record {
    let dir = TempDir::new("staging");
    let store = SidecarStore::new(dir.path().join("chat"));
    let (sidecar, _) = store
        .open_for_writing(
            "c0ffee000001",
            1.5,
            NewMeta {
                workspace: None,
                mode: Mode::Agent,
                origin: Origin::Native,
            },
        )
        .unwrap();
    let staging = Staging::new(Arc::new(sidecar));
    Record {
        _dir: dir,
        store,
        staging,
    }
}

fn proposal(path: &str, base: Base, new: Option<&[u8]>, action: Action) -> Proposal {
    Proposal {
        path: path.into(),
        authority: false,
        base,
        new: new.map(<[u8]>::to_vec),
        action,
    }
}

/// ST2: a later staging of the same path (in any case) updates the same
/// change, appends its op, and keeps the base; an overwrite drops the ops.
/// Mutant: every staging a new change.
#[test]
fn st2_one_live_change_per_path() {
    let r = record();
    let (turn, call) = ("a1b2c3d4e5f6".to_owned(), "call_1".to_owned());
    let base = Base::of_bytes(b"a\n".to_vec());
    let first = r
        .staging
        .record(
            proposal(
                "src/A.txt",
                base.clone(),
                Some(b"b\n"),
                Action::Edit(op("a", "b", false)),
            ),
            &turn,
            &call,
        )
        .unwrap();
    let second = r
        .staging
        .record(
            proposal(
                "src/a.txt",
                Base::absent(),
                Some(b"c\n"),
                Action::Edit(op("b", "c", false)),
            ),
            &turn,
            &call,
        )
        .unwrap();
    assert_eq!(first.change.id, second.change.id);
    assert_eq!(second.change.path, "src/A.txt", "the first spelling stays");
    assert_eq!(second.change.base, base.state, "the base stays");
    assert_eq!(second.change.kind, ChangeKind::Edit);
    assert_eq!(second.change.ops.len(), 2);
    assert_eq!((second.added, second.removed), (1, 1));
    assert_eq!(r.staging.changes().len(), 1);
    let third = r
        .staging
        .record(
            proposal("src/a.txt", Base::absent(), Some(b"d\n"), Action::Write),
            &turn,
            &call,
        )
        .unwrap();
    assert_eq!(third.change.kind, ChangeKind::Overwrite);
    assert!(
        third.change.ops.is_empty(),
        "an overwrite cannot be re-applied"
    );
    let fourth = r
        .staging
        .record(
            proposal(
                "src/a.txt",
                Base::absent(),
                Some(b"e\n"),
                Action::Edit(op("d", "e", false)),
            ),
            &turn,
            &call,
        )
        .unwrap();
    assert_eq!(fourth.change.kind, ChangeKind::Overwrite);
    assert!(fourth.change.ops.is_empty());
    assert_eq!(r.staging.changes().len(), 1);
    assert_eq!(r.staging.waiting(), 1);
}

/// The staged view, and the same view read back from the record by a
/// reopened conversation.
#[test]
fn the_staged_view_survives_a_reopen() {
    let r = record();
    let (turn, call) = ("a1b2c3d4e5f6".to_owned(), "call_1".to_owned());
    r.staging
        .record(
            proposal("new.txt", Base::absent(), Some(b"fresh\n"), Action::Write),
            &turn,
            &call,
        )
        .unwrap();
    let deleted = r
        .staging
        .record(
            proposal(
                "old.txt",
                Base::of_bytes(b"old\n".to_vec()),
                None,
                Action::Delete,
            ),
            &turn,
            &call,
        )
        .unwrap();
    assert_eq!(deleted.change.new, NewState::Deleted);
    assert_eq!((deleted.added, deleted.removed), (0, 1));
    r.staging.note_read("Seen.txt");
    assert!(r.staging.has_read("seen.txt"));
    let check = |staging: &Staging| {
        assert_eq!(
            staging.staged("NEW.txt"),
            Some(Staged::Bytes(b"fresh\n".to_vec()))
        );
        assert_eq!(staging.staged("old.txt"), Some(Staged::Deleted));
        assert_eq!(staging.staged("other.txt"), None);
        assert_eq!(staging.created(), vec!["new.txt".to_owned()]);
    };
    check(&r.staging);
    let items = r.store.read_items("c0ffee000001").unwrap().items;
    let reopened = Staging::from_items(Arc::clone(r.staging.sidecar()), &items);
    check(&reopened);
    assert_eq!(reopened.changes(), r.staging.changes());
    assert!(
        !reopened.has_read("seen.txt"),
        "reads are remembered for the session only"
    );
    assert!(
        reopened
            .changes()
            .iter()
            .all(|change| change.state == ChangeState::Pending)
    );
}
