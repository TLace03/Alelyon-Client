//! The read tools against folders and scratch repositories this file makes
//! (the chat core's spec §7.2, §7.3; TF4, WF6's reads; the caps).
//! Every folder is temporary; git runs only in scratch repositories.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use super::read::{
    GlobArgs, GrepArgs, LineMatch, ListDirArgs, NoOverlay, Overlay, ReadContext, ReadFileArgs,
    SearchArgs, SearchReport, Staged, ToolError, files, glob, grep, list_dir, listing, read_file,
    search,
};
use crate::text::find::Query;
use crate::git::runner::GitRunner;
use crate::git::tests::Scratch;
use crate::state::StateRoot;
use crate::workspace::Workspace;
use crate::workspace::attach::attach_path;

fn attach(scratch: &Scratch, folder: &Path) -> Workspace {
    attach_path(
        folder,
        &scratch.env(),
        &StateRoot::at(scratch.path().join("state")),
        &scratch.runner(),
    )
    .unwrap()
}

struct Tools<'a> {
    workspace: Workspace,
    runner: GitRunner,
    overlay: &'a dyn Overlay,
}

impl<'a> Tools<'a> {
    fn new(scratch: &Scratch, folder: &Path, overlay: &'a dyn Overlay) -> Self {
        Self {
            workspace: attach(scratch, folder),
            runner: scratch.runner(),
            overlay,
        }
    }

    fn ctx(&self) -> ReadContext<'_> {
        ReadContext {
            workspace: &self.workspace,
            runner: &self.runner,
            overlay: self.overlay,
        }
    }

    fn read(&self, path: &str) -> Result<String, ToolError> {
        read_file(
            &self.ctx(),
            &ReadFileArgs {
                path: path.into(),
                ..ReadFileArgs::default()
            },
        )
    }

    fn read_at(&self, path: &str, offset: u64, limit: u64) -> Result<String, ToolError> {
        read_file(
            &self.ctx(),
            &ReadFileArgs {
                path: path.into(),
                offset: Some(offset),
                limit: Some(limit),
            },
        )
    }

    fn glob(&self, pattern: &str) -> Result<String, ToolError> {
        glob(
            &self.ctx(),
            &GlobArgs {
                pattern: pattern.into(),
            },
        )
    }

    fn grep(&self, pattern: &str) -> Result<String, ToolError> {
        grep(
            &self.ctx(),
            &GrepArgs {
                pattern: pattern.into(),
                ..GrepArgs::default()
            },
        )
    }

    fn search(&self, text: &str) -> Result<SearchReport, ToolError> {
        search(
            &self.ctx(),
            &SearchArgs {
                query: Query {
                    text: text.into(),
                    regex: true,
                    ..Query::default()
                },
                max_matches: 1000,
                ..SearchArgs::default()
            },
        )
    }

    fn list(&self, path: Option<&str>, depth: Option<u8>) -> Result<String, ToolError> {
        list_dir(
            &self.ctx(),
            &ListDirArgs {
                path: path.map(str::to_owned),
                depth,
            },
        )
    }
}

const SENTINEL: &str = "lattice-tf4-sentinel";

/// A repository that ignores `FAMEnvironment.env` and `.env`, both holding
/// the sentinel, with a visible file and a link to `.env`.
fn secret_repo(scratch: &Scratch) -> (PathBuf, bool) {
    let repo = scratch.path().join("repo");
    std::fs::create_dir_all(repo.join("src")).unwrap();
    scratch.git(&repo, &["init", "-q"]);
    std::fs::write(repo.join(".gitignore"), "FAMEnvironment.env\n.env\n").unwrap();
    std::fs::write(repo.join("FAMEnvironment.env"), format!("KEY={SENTINEL}\n")).unwrap();
    std::fs::write(repo.join(".env"), format!("KEY={SENTINEL}\n")).unwrap();
    std::fs::write(repo.join("src").join("notes.txt"), "visible line\n").unwrap();
    let linked =
        std::os::windows::fs::symlink_file(repo.join(".env"), repo.join("envlink")).is_ok();
    if !linked {
        println!("UNMEASURED: a symbolic link could not be made here; the link case did not run");
    }
    (repo, linked)
}

/// TF4: in a git workspace, `.env` and `FAMEnvironment.env` (git-ignored),
/// and without git `id_rsa` and `.env` (the built-in defaults), are refused by
/// `read_file` and never reached by `glob` or `grep`. The visible file is the
/// positive control: the same calls do reach it.
/// Mutants: `ls-files` without `--exclude-standard`; the non-git walk
/// unfiltered; the ignore check skipped for reads.
#[test]
fn tf4_ignored_and_secret_files_are_refused_for_every_read_tool() {
    let scratch = Scratch::new("read-tf4");
    let (repo, _) = secret_repo(&scratch);
    let tools = Tools::new(&scratch, &repo, &NoOverlay);
    for path in [".env", "FAMEnvironment.env"] {
        assert_eq!(
            tools.read(path),
            Err(ToolError(
                "git ignores that file, so it is not shown.".to_owned()
            )),
            "{path}"
        );
    }
    let globbed = tools.glob("**/*").unwrap();
    assert!(globbed.contains("src/notes.txt"), "{globbed}");
    assert!(
        !globbed.contains(".env") && !globbed.contains("FAMEnvironment"),
        "{globbed}"
    );
    let found = tools.grep("KEY|visible").unwrap();
    assert!(found.contains("src/notes.txt:1:visible line"), "{found}");
    assert!(!found.contains(SENTINEL), "{found}");
    // A person's search sees what grep sees, and no more.
    let searched = tools.search("KEY|visible").unwrap();
    let paths: Vec<&str> = searched.files.iter().map(|f| f.path.as_str()).collect();
    assert_eq!(paths, ["src/notes.txt"], "{searched:?}");
    assert_eq!(tools.search(SENTINEL).unwrap().matches, 0);

    let plain = scratch.path().join("plain");
    std::fs::create_dir_all(plain.join(".ssh")).unwrap();
    std::fs::write(plain.join("id_rsa"), format!("{SENTINEL}\n")).unwrap();
    std::fs::write(plain.join(".env"), format!("{SENTINEL}\n")).unwrap();
    std::fs::write(plain.join(".ssh").join("config"), format!("{SENTINEL}\n")).unwrap();
    std::fs::write(plain.join("readme.txt"), format!("not the {}\n", "secret")).unwrap();
    let tools = Tools::new(&scratch, &plain, &NoOverlay);
    for path in ["id_rsa", ".env", ".ssh/config"] {
        assert_eq!(
            tools.read(path),
            Err(ToolError(
                "That file may hold secrets, so it is not used.".to_owned()
            )),
            "{path}"
        );
    }
    let globbed = tools.glob("**/*").unwrap();
    assert!(globbed.starts_with("1 paths match"), "{globbed}");
    assert!(globbed.contains("readme.txt"));
    let found = tools.grep(SENTINEL).unwrap();
    assert!(found.starts_with("0 matches"), "{found}");
    assert!(tools.grep("secret").unwrap().contains("readme.txt:1:"));
}

/// WF6's reads: an 8.3 alias and a link to an ignored file are refused by
/// `read_file`, and `grep` does not search through the link (WP8 runs again
/// on its derived path).
/// Mutant: the second WP8 check of a listed path that leads elsewhere skipped.
#[test]
fn wf6_reads_through_aliases_and_links_are_refused() {
    let scratch = Scratch::new("read-wf6");
    let (repo, linked) = secret_repo(&scratch);
    let tools = Tools::new(&scratch, &repo, &NoOverlay);
    for alias in ["FAMENV~1.ENV", "ENV~1"] {
        assert_eq!(
            tools.read(alias),
            Err(ToolError("Use the file's full name.".to_owned())),
            "{alias}"
        );
    }
    if linked {
        assert_eq!(
            tools.read("envlink"),
            Err(ToolError(
                "git ignores that file, so it is not shown.".to_owned()
            ))
        );
        let found = tools.grep("KEY").unwrap();
        assert!(!found.contains(SENTINEL), "{found}");
        assert!(found.contains("1 files could not be read"), "{found}");
    }
}

/// §7.3 `read_file`: the header names the file and its bounds; offset and
/// limit (at most 2,000, stated); lines cut at 2,000 characters; 2 MiB; a
/// binary refused with its size; UTF-8 with its mark removed, UTF-16 with a
/// mark; a folder and a missing file refused.
/// Mutants: the 2 MiB cap not applied; the binary probe skipped.
#[test]
fn read_file_states_its_bounds() {
    let scratch = Scratch::new("read-file");
    let folder = scratch.path().join("f");
    std::fs::create_dir_all(folder.join("sub")).unwrap();
    let text: String = (1..=2500).map(|n| format!("line {n}\n")).collect();
    std::fs::write(folder.join("long.txt"), &text).unwrap();
    std::fs::write(folder.join("wide.txt"), format!("{}\n", "w".repeat(2100))).unwrap();
    std::fs::write(folder.join("big.txt"), vec![b'x'; 2 * 1024 * 1024 + 1]).unwrap();
    std::fs::write(folder.join("exact.txt"), vec![b'y'; 2 * 1024 * 1024]).unwrap();
    std::fs::write(folder.join("bin.dat"), b"ab\0cd").unwrap();
    std::fs::write(folder.join("bom.txt"), b"\xef\xbb\xbfhello\r\nworld\r\n").unwrap();
    let utf16: Vec<u8> = [0xFF, 0xFE]
        .into_iter()
        .chain("h\u{e9}\n".encode_utf16().flat_map(u16::to_le_bytes))
        .collect();
    std::fs::write(folder.join("u16.txt"), utf16).unwrap();
    std::fs::write(folder.join("empty.txt"), b"").unwrap();
    std::fs::write(folder.join("bad.txt"), b"a\xffb\n").unwrap();
    let tools = Tools::new(&scratch, &folder, &NoOverlay);

    let first = tools.read("long.txt").unwrap();
    let lines: Vec<&str> = first.lines().collect();
    assert_eq!(lines[0], "long.txt (2500 lines; lines 1-2000 shown)");
    assert_eq!(lines[1], "1\tline 1");
    assert_eq!(lines.len(), 2001);
    let window = tools.read_at("long.txt", 2499, 10).unwrap();
    assert_eq!(
        window,
        "long.txt (2500 lines; lines 2499-2500 shown)\n2499\tline 2499\n2500\tline 2500"
    );
    let clamped = tools.read_at("long.txt", 1, 5000).unwrap();
    assert!(clamped.starts_with(
        "long.txt (2500 lines; lines 1-2000 shown; at most 2,000 lines are shown at a time)"
    ));
    assert_eq!(
        tools.read_at("long.txt", 3000, 1).unwrap(),
        "long.txt (2500 lines; none from line 3000)"
    );
    let wide = tools.read("wide.txt").unwrap();
    assert!(wide.ends_with(&format!(
        "1\t{} [line cut at 2,000 characters]",
        "w".repeat(2000)
    )));
    assert_eq!(
        tools.read("big.txt"),
        Err(ToolError(
            "That file is 2097153 bytes; Lattice reads files of at most 2 MiB (2,097,152 bytes)."
                .to_owned()
        ))
    );
    assert!(tools.read("exact.txt").is_ok(), "2 MiB exactly is read");
    assert_eq!(
        tools.read("bin.dat"),
        Err(ToolError("binary file, 5 bytes".to_owned()))
    );
    assert_eq!(
        tools.read("bom.txt").unwrap(),
        "bom.txt (2 lines; lines 1-2 shown)\n1\thello\n2\tworld"
    );
    assert_eq!(
        tools.read("u16.txt").unwrap(),
        "u16.txt (1 lines; lines 1-1 shown; UTF-16)\n1\th\u{e9}"
    );
    assert_eq!(tools.read("empty.txt").unwrap(), "empty.txt (empty file)");
    assert!(tools.read("bad.txt").unwrap().contains("not valid UTF-8"));
    assert_eq!(
        tools.read("sub"),
        Err(ToolError("That is a folder; use list_dir.".to_owned()))
    );
    assert_eq!(
        tools.read("nope.txt"),
        Err(ToolError(
            "There is no such file in this folder.".to_owned()
        ))
    );
}

/// §7.3 `list_dir`: folders first, then files, sorted; depth 1 to 3; at most
/// 2,000 entries with "N more not shown".
/// Mutant: the entry cap not applied.
#[test]
fn list_dir_orders_folders_first_and_states_its_cap() {
    let scratch = Scratch::new("read-list");
    let folder = scratch.path().join("f");
    std::fs::create_dir_all(folder.join("b").join("deep")).unwrap();
    std::fs::create_dir_all(folder.join("a")).unwrap();
    std::fs::write(folder.join("z.txt"), b"z").unwrap();
    std::fs::write(folder.join("a").join("one.txt"), b"1").unwrap();
    std::fs::write(folder.join("b").join("deep").join("two.txt"), b"2").unwrap();
    let tools = Tools::new(&scratch, &folder, &NoOverlay);
    assert_eq!(
        tools.list(None, None).unwrap(),
        ". (3 entries; depth 1)\na/\nb/\nz.txt"
    );
    assert_eq!(
        tools.list(None, Some(3)).unwrap(),
        ". (6 entries; depth 3)\na/\nb/\nb/deep/\na/one.txt\nb/deep/two.txt\nz.txt"
    );
    assert_eq!(
        tools.list(Some("b"), Some(2)).unwrap(),
        "b (2 entries; depth 2)\ndeep/\ndeep/two.txt"
    );
    assert_eq!(
        tools.list(None, Some(4)),
        Err(ToolError("depth is 1, 2 or 3.".to_owned()))
    );
    assert_eq!(
        tools.list(Some("z.txt"), None),
        Err(ToolError("That is a file; use read_file.".to_owned()))
    );
    let many = scratch.path().join("many");
    std::fs::create_dir_all(&many).unwrap();
    for n in 0..2005 {
        std::fs::write(many.join(format!("f{n:04}.txt")), b"x").unwrap();
    }
    let tools = Tools::new(&scratch, &many, &NoOverlay);
    let listed = tools.list(None, None).unwrap();
    let lines: Vec<&str> = listed.lines().collect();
    assert_eq!(lines[0], ". (2005 entries; depth 1)");
    assert_eq!(lines.len(), 1 + 2000 + 1);
    assert_eq!(*lines.last().unwrap(), "5 more not shown");
}

/// §7.3 `glob`: `/` is a literal separator, at most 1,000 paths with "N more
/// not shown", patterns at most 256 characters.
/// Mutant: `/` not literal (`*` crossing folders).
#[test]
fn glob_separates_with_slash_and_states_its_cap() {
    let scratch = Scratch::new("read-glob");
    let folder = scratch.path().join("f");
    std::fs::create_dir_all(folder.join("sub")).unwrap();
    std::fs::write(folder.join("top.txt"), b"t").unwrap();
    std::fs::write(folder.join("sub").join("a.txt"), b"a").unwrap();
    for n in 0..1005 {
        std::fs::write(folder.join(format!("m{n:04}.md")), b"m").unwrap();
    }
    let tools = Tools::new(&scratch, &folder, &NoOverlay);
    assert_eq!(tools.glob("*.txt").unwrap(), "1 paths match *.txt\ntop.txt");
    assert_eq!(
        tools.glob("**/*.TXT").unwrap(),
        "2 paths match **/*.TXT\nsub/a.txt\ntop.txt"
    );
    let many = tools.glob("*.md").unwrap();
    let lines: Vec<&str> = many.lines().collect();
    assert_eq!(lines[0], "1005 paths match *.md");
    assert_eq!(lines.len(), 1 + 1000 + 1);
    assert_eq!(*lines.last().unwrap(), "5 more not shown");
    assert_eq!(
        tools.glob(&"a".repeat(257)),
        Err(ToolError(
            "A glob pattern is at most 256 characters.".to_owned()
        ))
    );
    assert_eq!(
        tools.glob("a[b"),
        Err(ToolError("That glob pattern is not valid.".to_owned()))
    );
}

/// §7.3 `grep`: at most 200 matches with "N more matches", 64 KiB of
/// output, context of at most 3 lines, a pattern compiled to at most 1 MiB,
/// no back-references; files over 2 MiB and binaries skipped and counted.
/// Mutants: the match cap not applied; the compiled-size limit not set.
#[test]
fn grep_states_its_caps() {
    let scratch = Scratch::new("read-grep");
    let folder = scratch.path().join("f");
    std::fs::create_dir_all(&folder).unwrap();
    let hits: String = (1..=250).map(|n| format!("hit {n}\n")).collect();
    std::fs::write(folder.join("hits.txt"), hits).unwrap();
    std::fs::write(folder.join("ctx.txt"), "a\nb\nTARGET\nc\nd\ne\nf\nTARGET\n").unwrap();
    std::fs::write(folder.join("big.txt"), vec![b'x'; 2 * 1024 * 1024 + 1]).unwrap();
    std::fs::write(folder.join("bin.dat"), b"hit\0").unwrap();
    let tools = Tools::new(&scratch, &folder, &NoOverlay);
    let found = tools.grep("^hit").unwrap();
    let lines: Vec<&str> = found.lines().collect();
    assert_eq!(
        lines[0],
        "250 matches in 1 files; 200 shown, 50 more matches; 1 files over 2 MiB were not searched; 1 binary files were not searched"
    );
    assert_eq!(lines.len(), 201);
    assert_eq!(lines[1], "hits.txt:1:hit 1");
    let context = grep(
        &tools.ctx(),
        &GrepArgs {
            pattern: "TARGET".into(),
            path: Some("ctx.txt".into()),
            context: Some(1),
            ..GrepArgs::default()
        },
    )
    .unwrap();
    assert_eq!(
        context,
        "2 matches in 1 files\nctx.txt-2-b\nctx.txt:3:TARGET\nctx.txt-4-c\n--\nctx.txt-7-f\nctx.txt:8:TARGET"
    );
    let wide = scratch.path().join("wide");
    std::fs::create_dir_all(&wide).unwrap();
    let long: String = (0..100)
        .map(|n| format!("{n} {}\n", "q".repeat(1990)))
        .collect();
    std::fs::write(wide.join("long.txt"), long).unwrap();
    let tools = Tools::new(&scratch, &wide, &NoOverlay);
    let capped = tools.grep("q").unwrap();
    assert!(
        capped.contains("the output stopped at 64 KiB"),
        "{}",
        &capped[..200]
    );
    assert!(capped.len() <= 64 * 1024 + 300);
    assert_eq!(
        grep(
            &tools.ctx(),
            &GrepArgs {
                pattern: "q".into(),
                context: Some(4),
                ..GrepArgs::default()
            }
        ),
        Err(ToolError("context is at most 3 lines.".to_owned()))
    );
    assert_eq!(
        tools.grep(r"(a)\1"),
        Err(ToolError(
            "That pattern is not a valid regular expression (no back-references or look-around)."
                .to_owned()
        ))
    );
    assert_eq!(
        tools.grep(r"\w{2000}\w{2000}"),
        Err(ToolError(
            "That pattern is too large; Lattice compiles patterns of at most 1 MiB.".to_owned()
        ))
    );
}

/// `search` returns each matching line with where it matches, over the files
/// grep reads, and counts what it skipped and what it found past its cap.
/// Mutants: the cap applied to the count; a file past the cap not counted as
/// matched; ranges past a line's cut kept.
#[test]
fn search_finds_lines_and_ranges_and_counts_what_it_skips() {
    let scratch = Scratch::new("read-search");
    let folder = scratch.path().join("f");
    std::fs::create_dir_all(folder.join("sub")).unwrap();
    std::fs::write(folder.join("a.txt"), "alpha\nBeta alpha\n").unwrap();
    std::fs::write(folder.join("b.rs"), "fn alpha() {}\n").unwrap();
    std::fs::write(folder.join("sub").join("c.md"), "nothing here\n").unwrap();
    std::fs::write(folder.join("big.txt"), vec![b'x'; 2 * 1024 * 1024 + 1]).unwrap();
    std::fs::write(folder.join("bin.dat"), b"alpha\0").unwrap();
    let long = format!("{}alpha", "q".repeat(2000));
    std::fs::write(folder.join("long.txt"), format!("{long}\n")).unwrap();
    let tools = Tools::new(&scratch, &folder, &NoOverlay);
    let ask = |text: &str, glob: Option<&str>, max: usize, whole_word: bool| {
        search(
            &tools.ctx(),
            &SearchArgs {
                query: Query {
                    text: text.into(),
                    whole_word,
                    ..Query::default()
                },
                glob: glob.map(str::to_owned),
                max_matches: max,
            },
        )
        .unwrap()
    };
    let all = ask("ALPHA", None, 100, false);
    assert_eq!(
        all.files.iter().map(|f| f.path.as_str()).collect::<Vec<_>>(),
        ["a.txt", "b.rs"],
        "the long line's match is past its cut, so long.txt shows nothing"
    );
    assert_eq!(
        all.files[0].lines,
        [
            LineMatch { line: 1, text: "alpha".into(), ranges: vec![0..5] },
            LineMatch { line: 2, text: "Beta alpha".into(), ranges: vec![5..10] },
        ]
    );
    assert_eq!(all.files[1].lines[0].ranges, vec![3..8]);
    assert_eq!((all.matches, all.files_matched, all.shown), (4, 3, 3), "long.txt's match is counted");
    assert_eq!((all.too_large, all.binary, all.unreadable, all.truncated), (1, 1, 0, false));
    let only_rust = ask("alpha", Some("*.rs"), 100, false);
    assert_eq!(only_rust.files.len(), 1);
    assert_eq!(only_rust.files[0].path, "b.rs");
    assert_eq!(ask("alph", None, 100, true).matches, 0, "whole words only");
    let capped = ask("alpha", None, 1, false);
    assert_eq!((capped.matches, capped.files_matched, capped.shown), (4, 3, 1));
    assert_eq!(capped.files.len(), 1);
    assert_eq!(
        search(&tools.ctx(), &SearchArgs::default()),
        Err(ToolError("Type what to look for.".to_owned()))
    );
}

/// The overlay as staging will fill it (row E2): a staged file reads with its
/// new bytes, a staged creation is listed, matched and read, a staged
/// deletion disappears.
#[derive(Default)]
struct FakeOverlay {
    entries: Mutex<Vec<(String, Staged)>>,
    created: Vec<String>,
}

impl Overlay for FakeOverlay {
    fn staged(&self, path: &str) -> Option<Staged> {
        self.entries
            .lock()
            .unwrap()
            .iter()
            .find(|(p, _)| p == path)
            .map(|(_, staged)| staged.clone())
    }

    fn created(&self) -> Vec<String> {
        self.created.clone()
    }
}

#[test]
fn the_read_tools_see_the_staged_view() {
    let scratch = Scratch::new("read-overlay");
    let folder = scratch.path().join("f");
    std::fs::create_dir_all(&folder).unwrap();
    std::fs::write(folder.join("a.txt"), "old\n").unwrap();
    std::fs::write(folder.join("gone.txt"), "here\n").unwrap();
    let overlay = FakeOverlay {
        entries: Mutex::new(vec![
            ("a.txt".into(), Staged::Bytes(b"new\n".to_vec())),
            ("gone.txt".into(), Staged::Deleted),
            ("made.txt".into(), Staged::Bytes(b"fresh\n".to_vec())),
        ]),
        created: vec!["made.txt".into()],
    };
    let tools = Tools::new(&scratch, &folder, &overlay);
    assert_eq!(
        tools.read("a.txt").unwrap(),
        "a.txt (1 lines; lines 1-1 shown; staged version)\n1\tnew"
    );
    assert_eq!(
        tools.read("made.txt").unwrap(),
        "made.txt (1 lines; lines 1-1 shown; staged version)\n1\tfresh"
    );
    assert_eq!(
        tools.read("gone.txt"),
        Err(ToolError(
            "There is no such file in this folder.".to_owned()
        ))
    );
    assert_eq!(
        tools.list(None, None).unwrap(),
        ". (2 entries; depth 1)\na.txt\nmade.txt (staged)"
    );
    assert!(tools.glob("*.txt").unwrap().contains("made.txt (staged)"));
    assert_eq!(
        tools.grep("new|fresh|here").unwrap(),
        "2 matches in 2 files\na.txt:1:new\nmade.txt:1:fresh"
    );
}

/// Row D6: `files` ranks the folder's non-ignored files and the staged
/// creations, by the shipping app's quick-open rule; an ignored file is never
/// offered, and the limit holds.
/// Mutant: the picker's list taken from the disk without the ignore rules.
#[test]
fn files_offers_only_what_the_read_tools_see() {
    let scratch = Scratch::new("read-files");
    let (repo, linked) = secret_repo(&scratch);
    for n in 0..60 {
        std::fs::write(repo.join("src").join(format!("env{n:02}.txt")), b"e").unwrap();
    }
    let overlay = FakeOverlay {
        entries: Mutex::new(vec![(
            "src/envnew.txt".into(),
            Staged::Bytes(b"n".to_vec()),
        )]),
        created: vec!["src/envnew.txt".into()],
    };
    let tools = Tools::new(&scratch, &repo, &overlay);
    let all = files(&tools.ctx(), "env", 1000).unwrap();
    let paths: Vec<&str> = all.iter().map(|row| row.path.as_str()).collect();
    assert!(
        paths.contains(&"src/envnew.txt"),
        "a staged creation is offered"
    );
    assert!(
        !paths
            .iter()
            .any(|path| path.ends_with(".env") || path.contains("FAMEnvironment")),
        "{paths:?}"
    );
    assert_eq!(
        all.len(),
        61 + usize::from(linked),
        "60 files, the creation and the link: {paths:?}"
    );
    assert_eq!(files(&tools.ctx(), "env", 0).unwrap().len(), 50);
    assert_eq!(files(&tools.ctx(), "env", 5).unwrap().len(), 5);
    assert!(files(&tools.ctx(), "  ", 10).unwrap().is_empty());
}

/// CENTCOM's Explorer: `listing` is the set `files` ranks, whole and sorted:
/// every path `files` offers is in it, with the staged creations, and an
/// ignored file never is.
/// Mutant: the listing taken from the disk without the ignore rules.
#[test]
fn listing_is_the_whole_set_files_ranks_sorted() {
    let scratch = Scratch::new("read-listing");
    let (repo, linked) = secret_repo(&scratch);
    std::fs::write(repo.join("src").join("b10.txt"), b"x").unwrap();
    std::fs::write(repo.join("src").join("b2.txt"), b"x").unwrap();
    let overlay = FakeOverlay {
        entries: Mutex::new(vec![("src/made.txt".into(), Staged::Bytes(b"n".to_vec()))]),
        created: vec!["src/made.txt".into()],
    };
    let tools = Tools::new(&scratch, &repo, &overlay);
    let listed = listing(&tools.ctx()).unwrap();
    assert!(!listed.truncated && listed.withheld.is_empty(), "{listed:?}");
    let mut sorted = listed.paths.clone();
    sorted.sort();
    assert_eq!(listed.paths, sorted);
    for path in [".gitignore", "src/notes.txt", "src/made.txt", "src/b10.txt", "src/b2.txt"] {
        assert!(listed.paths.iter().any(|p| p == path), "{path} in {:?}", listed.paths);
    }
    assert!(
        !listed
            .paths
            .iter()
            .any(|path| path.ends_with(".env") || path.contains("FAMEnvironment")),
        "{:?}",
        listed.paths
    );
    assert_eq!(listed.paths.len(), 5 + usize::from(linked), "{:?}", listed.paths);
    for row in files(&tools.ctx(), "t", 1000).unwrap() {
        assert!(listed.paths.contains(&row.path), "{}", row.path);
    }
}

/// WP8b (spec §22.7): in the phase D partial-clone fixture, `sub/.gitignore`
/// is skip-worktree and its blob is missing, so git, with lazy fetching off,
/// says `sub/secret.txt` is not ignored. Every read tool refuses what is at
/// or below `sub/`, naming it, and never shows the secret; a listing says
/// what it left out. The sibling `other/`, whose `.gitignore` git reads,
/// is unaffected (the positive control: its own rule still hides `x.tmp`).
/// The workspace carries `ignore_rules_incomplete: ["sub"]`.
/// Mutant: the WP8b check dropped from `PathRules` (the secret is read).
#[test]
fn wp8b_a_folder_whose_gitignore_git_cannot_read_is_refused() {
    let scratch = Scratch::new("read-wp8b");
    let secret = format!("KEY={SENTINEL}\n");
    let clone = crate::git::tests::unreadable_gitignore_clone(&scratch, &secret);
    let tools = Tools::new(&scratch, &clone, &NoOverlay);
    assert_eq!(tools.workspace.ignore_rules_incomplete, ["sub"]);
    let mut outputs = Vec::new();
    for path in ["sub/secret.txt", "sub/visible.txt", "sub"] {
        let read = tools.read(path);
        outputs.push(format!("{read:?}"));
        let error = read.expect_err(path);
        assert!(error.0.contains("sub/"), "{path}: {}", error.0);
        assert!(
            error.0.contains("git cannot read the ignore rules"),
            "{}",
            error.0
        );
    }
    let listed = tools.list(Some("sub"), None);
    outputs.push(format!("{listed:?}"));
    assert!(listed.expect_err("list sub").0.contains("sub/"));
    let root = tools.list(None, Some(3)).unwrap();
    let globbed = tools.glob("**/*").unwrap();
    let grepped = tools.grep(SENTINEL).unwrap();
    let grepped_all = tools.grep("line").unwrap();
    println!("{root}\n--\n{globbed}\n--\n{grepped}\n--\n{grepped_all}");
    for answer in [&root, &globbed, &grepped, &grepped_all] {
        let first = answer.lines().next().unwrap();
        assert!(
            first.contains("not shown, because git cannot read their ignore rules: sub/"),
            "{first}"
        );
        assert!(
            !answer.lines().skip(1).any(|line| line.starts_with("sub/")),
            "{answer}"
        );
    }
    assert!(grepped.starts_with("0 matches"), "{grepped}");
    outputs.extend([root.clone(), globbed.clone(), grepped, grepped_all.clone()]);
    // The positive control: the sibling's rules are read and still apply.
    assert!(tools.read("other/o.txt").unwrap().contains("other line"));
    assert!(
        tools.read("other/x.tmp").is_err(),
        "other/.gitignore hides x.tmp"
    );
    assert!(root.lines().any(|line| line == "other/o.txt"), "{root}");
    assert!(
        grepped_all.contains("other/o.txt:1:other line"),
        "{grepped_all}"
    );
    assert!(
        outputs.iter().all(|output| !output.contains(SENTINEL)),
        "the secret never reached an answer: {outputs:?}"
    );
}

/// A deny-read entry for the current user on one file of a test's own
/// temporary folder (`icacls`, as the verifier's probe did), removed again
/// on drop so the folder can be cleaned up.
struct DenyRead {
    path: PathBuf,
    trustee: String,
}

impl DenyRead {
    fn new(path: &Path) -> Self {
        let trustee = format!("*{}", lattice_sys::fs::current_user_sid().unwrap());
        let output = std::process::Command::new("icacls")
            .arg(path)
            .arg("/deny")
            .arg(format!("{trustee}:(R)"))
            .output()
            .unwrap();
        assert!(output.status.success(), "{output:?}");
        assert_eq!(
            std::fs::read(path).map_err(|error| error.kind()).err(),
            Some(std::io::ErrorKind::PermissionDenied),
            "the file can no longer be read"
        );
        Self {
            path: path.to_path_buf(),
            trustee,
        }
    }
}

impl Drop for DenyRead {
    fn drop(&mut self) {
        let _ = std::process::Command::new("icacls")
            .arg(&self.path)
            .arg("/remove:d")
            .arg(&self.trustee)
            .output();
    }
}

/// WP8b (the verifier's probe, phaseHA/logs/VERIFY-probe-wp8b, case a): the
/// index names `sub/.gitignore` (skip-worktree, its blob missing) while the
/// folder on disk is `SUB`. The derived path then says `SUB/…`, and the
/// incomplete folder is matched without regard to case, as the file system
/// matches it: nothing under `sub/` or `SUB/` is read, listed or grepped.
/// Mutant: the folder compared with case (`at_or_below` without `folded`).
#[test]
fn wp8b_an_incomplete_folder_is_matched_without_regard_to_case() {
    let scratch = Scratch::new("read-wp8b-case");
    let secret = format!("KEY={SENTINEL}\n");
    let clone = crate::git::tests::unreadable_gitignore_clone(&scratch, &secret);
    std::fs::rename(clone.join("sub"), clone.join("SUB-tmp")).unwrap();
    std::fs::rename(clone.join("SUB-tmp"), clone.join("SUB")).unwrap();
    let tools = Tools::new(&scratch, &clone, &NoOverlay);
    assert_eq!(tools.workspace.ignore_rules_incomplete, ["sub"]);
    let mut outputs = Vec::new();
    for path in ["sub/secret.txt", "SUB/secret.txt", "Sub/visible.txt", "SUB"] {
        let read = tools.read(path);
        outputs.push(format!("{read:?}"));
        let error = read.expect_err(path);
        assert!(
            error.0.contains("git cannot read the ignore rules"),
            "{path}: {}",
            error.0
        );
    }
    let grepped = tools.grep(SENTINEL).unwrap();
    let globbed = tools.glob("**/*").unwrap();
    let root = tools.list(None, Some(3)).unwrap();
    for answer in [&grepped, &globbed, &root] {
        assert!(
            !answer
                .lines()
                .skip(1)
                .any(|line| line.to_lowercase().starts_with("sub/")),
            "{answer}"
        );
    }
    assert!(grepped.starts_with("0 matches"), "{grepped}");
    outputs.extend([grepped, globbed, root]);
    // The positive control: the sibling is read as before.
    assert!(tools.read("other/o.txt").unwrap().contains("other line"));
    assert!(
        outputs.iter().all(|output| !output.contains(SENTINEL)),
        "the secret never reached an answer: {outputs:?}"
    );
}

/// WP8b (the verifier's probe, case b): an untracked `.gitignore` that
/// exists but cannot be read is skipped by git with a warning, and its
/// folder's secret would read as not ignored. The folder fails closed on
/// every read and listing, named in the answer. The control: while it can
/// be read, its rule hides the secret and nothing is withheld.
/// Mutant: the per-folder `.gitignore` check dropped from
/// `incomplete_refusal`.
#[cfg(windows)]
#[test]
fn wp8b_an_untracked_gitignore_that_cannot_be_read_fails_closed() {
    let scratch = Scratch::new("read-wp8b-untracked");
    let secret = format!("KEY={SENTINEL}\n");
    let repo = scratch.repo("plainrepo");
    std::fs::create_dir_all(repo.join("sub2")).unwrap();
    let ignore = repo.join("sub2").join(".gitignore");
    std::fs::write(&ignore, b"secret2.txt\n").unwrap();
    std::fs::write(repo.join("sub2").join("secret2.txt"), &secret).unwrap();
    std::fs::write(repo.join("sub2").join("plain.txt"), b"plain line\n").unwrap();
    {
        let tools = Tools::new(&scratch, &repo, &NoOverlay);
        assert!(tools.read("sub2/secret2.txt").is_err(), "the rule applies");
        assert!(tools.read("sub2/plain.txt").unwrap().contains("plain line"));
        let root = tools.list(None, Some(3)).unwrap();
        assert!(!root.contains("not shown"), "{root}");
    }
    let _deny = DenyRead::new(&ignore);
    let tools = Tools::new(&scratch, &repo, &NoOverlay);
    let mut outputs = Vec::new();
    for path in ["sub2/secret2.txt", "sub2/plain.txt", "sub2"] {
        let read = tools.read(path);
        outputs.push(format!("{read:?}"));
        let error = read.expect_err(path);
        assert!(error.0.contains("sub2/"), "{path}: {}", error.0);
        assert!(
            error.0.contains("git cannot read the ignore rules"),
            "{}",
            error.0
        );
    }
    let grepped = tools.grep(SENTINEL).unwrap();
    let root = tools.list(None, Some(3)).unwrap();
    for answer in [&grepped, &root] {
        let first = answer.lines().next().unwrap();
        assert!(
            first.contains("not shown, because git cannot read their ignore rules: sub2/"),
            "{first}"
        );
        assert!(
            !answer.lines().skip(1).any(|line| line.starts_with("sub2/")),
            "{answer}"
        );
    }
    outputs.extend([grepped, root.clone()]);
    // The rest of the folder is unaffected.
    assert!(tools.read("a.txt").is_ok());
    assert!(root.lines().any(|line| line == "a.txt"), "{root}");
    assert!(
        outputs.iter().all(|output| !output.contains(SENTINEL)),
        "the secret never reached an answer: {outputs:?}"
    );
}

/// WP8b: the file `core.excludesFile` names exists but cannot be read, so
/// git would skip its rules: the whole folder fails closed. The control:
/// readable, its rule hides the secret and the rest reads.
/// Mutant: the `core.excludesFile` check dropped from
/// `ignore_rules_incomplete`.
#[cfg(windows)]
#[test]
fn wp8b_an_excludes_file_that_cannot_be_read_fails_closed() {
    let scratch = Scratch::new("read-wp8b-excludes");
    let secret = format!("KEY={SENTINEL}\n");
    let repo = scratch.repo("plainrepo");
    let excludes = scratch.path().join("global-ignore");
    std::fs::write(&excludes, b"secret3.txt\n").unwrap();
    let setting = excludes.to_string_lossy().replace('\\', "/");
    scratch.git(&repo, &["config", "core.excludesFile", &setting]);
    std::fs::write(repo.join("secret3.txt"), &secret).unwrap();
    {
        let tools = Tools::new(&scratch, &repo, &NoOverlay);
        assert_eq!(
            tools.workspace.ignore_rules_incomplete,
            Vec::<String>::new()
        );
        assert!(tools.read("secret3.txt").is_err(), "the rule applies");
        assert!(tools.read("a.txt").is_ok());
    }
    let _deny = DenyRead::new(&excludes);
    let tools = Tools::new(&scratch, &repo, &NoOverlay);
    assert_eq!(tools.workspace.ignore_rules_incomplete, [""]);
    let read = tools.read("secret3.txt");
    assert!(!format!("{read:?}").contains(SENTINEL), "{read:?}");
    assert!(
        read.expect_err("secret3")
            .0
            .contains("git cannot read the ignore rules for this folder")
    );
    assert!(tools.read("a.txt").is_err(), "the whole folder");
}

/// WP8b's cost: the index part of the incomplete-rules check (`rev-parse`,
/// the `ls-files -t -s --sparse` listing, `cat-file`, `config`) is read at
/// attach and kept while the index, `HEAD` and the configuration are
/// unchanged, so repeated reads start no `ls-files --sparse` (counted in
/// `GIT_TRACE`, the runner's seam); a change to the index reads it once more.
/// Mutants: the cache never consulted (every read lists the index again);
/// the index file left out of the key (a staged change is never seen).
#[test]
fn wp8b_the_index_part_is_read_once_per_index() {
    let scratch = Scratch::new("read-wp8b-cache");
    let repo = scratch.repo("cached");
    let mut tools = Tools::new(&scratch, &repo, &NoOverlay);
    let trace = scratch.path().join("cache.trace");
    tools.runner.trace = Some(trace.clone());
    let index_listings = || {
        std::fs::read_to_string(&trace)
            .unwrap_or_default()
            .lines()
            .filter(|line| line.contains("ls-files") && line.contains("--sparse"))
            .count()
    };
    for _ in 0..3 {
        assert!(tools.read("a.txt").unwrap().contains('a'));
    }
    let _ = tools.list(None, Some(1)).unwrap();
    assert!(trace.exists(), "the trace was written by the reads' git");
    assert_eq!(
        index_listings(),
        0,
        "attach read the index part; reads kept it"
    );
    std::fs::write(repo.join("b.txt"), b"b\n").unwrap();
    scratch.git(&repo, &["add", "b.txt"]);
    assert!(tools.read("b.txt").unwrap().contains('b'));
    assert_eq!(index_listings(), 1, "the index changed: read once more");
    assert!(tools.read("a.txt").is_ok());
    assert_eq!(index_listings(), 1, "and kept again");
}
