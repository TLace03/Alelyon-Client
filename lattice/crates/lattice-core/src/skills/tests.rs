//! Skills: the reader's own and a trusted folder's, their names and
//! descriptions, the list the leading item gets, `use_skill`'s answer and the
//! reader's own skill files.

use lattice_protocol::conversation::TrustState;

use super::*;
use crate::git::tests::Scratch;
use crate::workspace::attach::attach_path;

fn state(scratch: &Scratch) -> StateRoot {
    StateRoot::at(scratch.path().join("state"))
}

fn write(path: &Path, text: impl AsRef<[u8]>) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, text).unwrap();
}

#[test]
fn the_readers_own_skills_are_named_described_and_kept_apart() {
    let scratch = Scratch::new("skills-user");
    let state = state(&scratch);
    let dir = user_skills_dir(&state);
    write(
        &dir.join("review-pr").join("SKILL.md"),
        "---\nname: review-pr\ndescription: Review a pull request the user names.\n---\nRead the diff first.\n",
    );
    write(
        &dir.join("notes").join("SKILL.md"),
        "\n# Note taking\n\nKeep notes short.\n",
    );
    write(&dir.join("empty").join("README.md"), "not a skill");
    write(
        &dir.join("zz-other").join("SKILL.md"),
        "---\nname: Review-PR\n---\nA second skill of that name.\n",
    );
    write(
        &dir.join("spaced").join("SKILL.md"),
        "---\nname: bad name!\n---\nx\n",
    );
    write(
        &dir.join("big").join("SKILL.md"),
        "y".repeat(MAX_SKILL as usize + 1),
    );
    let found = load(&state, None);
    let named: Vec<(&str, &str)> = found
        .skills
        .iter()
        .map(|skill| (skill.name.as_str(), skill.description.as_str()))
        .collect();
    assert_eq!(
        named,
        [
            ("notes", "Note taking"),
            ("review-pr", "Review a pull request the user names."),
        ]
    );
    assert_eq!(
        found.find("REVIEW-PR").unwrap().body,
        "Read the diff first.\n"
    );
    assert_eq!(found.skills[0].path, "(yours) notes/SKILL.md");
    let notices = found.notices.join("\n");
    assert!(
        notices.contains("big/SKILL.md is larger than 64 KiB"),
        "{notices}"
    );
    assert!(
        notices.contains("a skill named Review-PR is listed already"),
        "{notices}"
    );
    assert!(
        notices.contains("spaced/SKILL.md was left out: a skill's name"),
        "{notices}"
    );
    assert!(
        !notices.contains("empty"),
        "a folder without SKILL.md is no skill: {notices}"
    );
    let lead = found.lead_text().unwrap();
    assert!(lead.starts_with(HEADER));
    assert!(lead.contains("\n- notes (yours): Note taking"));
    assert!(lead.contains("\n- review-pr (yours): Review a pull request the user names."));
}

#[test]
fn the_list_names_at_most_fifty_skills() {
    let scratch = Scratch::new("skills-many");
    let state = state(&scratch);
    let dir = user_skills_dir(&state);
    for n in 0..MAX_LISTED + 2 {
        write(&dir.join(format!("s{n:02}")).join("SKILL.md"), "Do it.");
    }
    let found = load(&state, None);
    assert_eq!(found.skills.len(), MAX_LISTED + 2);
    let lead = found.lead_text().unwrap();
    assert_eq!(lead.matches("\n- ").count(), MAX_LISTED);
    assert!(lead.ends_with("(2 more are not listed.)"));
    assert!(Skills::default().lead_text().is_none());
}

#[test]
fn a_folders_skills_are_read_only_when_it_is_trusted() {
    let scratch = Scratch::new("skills-folder");
    let state = state(&scratch);
    write(
        &user_skills_dir(&state).join("deploy").join("SKILL.md"),
        "Deploy the user's way.",
    );
    let folder = scratch.path().join("work");
    write(
        &folder
            .join(".claude")
            .join("skills")
            .join("alelyon")
            .join("SKILL.md"),
        "---\nname: alelyon\ndescription: Keep results tied to evidence.\n---\nSee references/verify.md.\n",
    );
    write(
        &folder
            .join(".claude")
            .join("skills")
            .join("alelyon")
            .join("references")
            .join("verify.md"),
        "How to verify.",
    );
    write(
        &folder
            .join(".lattice")
            .join("skills")
            .join("deploy")
            .join("SKILL.md"),
        "The folder's deploy.",
    );
    let runner = scratch.runner();
    let workspace = attach_path(&folder, &scratch.env(), &state, &runner).unwrap();
    let load_as = |trust| {
        workspace
            .with_rules(&runner, |rules| load(&state, Some((rules, trust))))
            .unwrap()
    };
    let untrusted = load_as(TrustState::Untrusted);
    assert_eq!(untrusted.skills.len(), 1, "only the reader's own");
    let found = load_as(TrustState::Trusted);
    let named: Vec<&str> = found
        .skills
        .iter()
        .map(|skill| skill.name.as_str())
        .collect();
    assert_eq!(
        named,
        ["deploy", "alelyon"],
        "the reader's own deploy comes first"
    );
    assert!(
        found
            .notices
            .join("\n")
            .contains("a skill named deploy is listed already")
    );
    let alelyon = found.find("alelyon").unwrap();
    assert_eq!(
        alelyon.place,
        Place::Folder(".claude/skills/alelyon".into())
    );
    assert_eq!(alelyon.path, ".claude/skills/alelyon/SKILL.md");
    assert!(
        found
            .lead_text()
            .unwrap()
            .contains("\n- alelyon (.claude/skills/alelyon): Keep results tied to evidence.")
    );
    let used = use_text(alelyon);
    assert!(used.starts_with(
        "Skill alelyon (.claude/skills/alelyon/SKILL.md):\n\nSee references/verify.md."
    ));
    assert!(
        used.ends_with("in .claude/skills/alelyon/: read them with read_file or read_skill_file.")
    );
}

#[test]
fn the_readers_own_skill_files_are_named_and_read_only_inside_its_folder() {
    let scratch = Scratch::new("skills-files");
    let state = state(&scratch);
    let dir = user_skills_dir(&state).join("guide");
    write(&dir.join("SKILL.md"), "Follow references/steps.md.");
    write(&dir.join("references").join("steps.md"), "Step one.");
    write(&dir.join("scripts").join("run.ps1"), "Write-Output hi");
    write(&dir.join(".hidden"), "secret");
    write(&dir.join("logo.png"), [137u8, 80, 78, 71, 0, 1]);
    write(&dir.join("big.txt"), "z".repeat(MAX_FILE as usize + 1));
    let outside = scratch.path().join("outside");
    write(&outside.join("keys.env"), "TOKEN=1");
    std::fs::create_dir(dir.join("away")).unwrap();
    lattice_sys::fs::seam::create_junction(&dir.join("away"), &outside).unwrap();
    let found = load(&state, None);
    let guide = found.find("guide").unwrap();
    let used = use_text(guide);
    assert!(
        used.ends_with(
            "Its other files (read one with read_skill_file): big.txt, logo.png, references/steps.md, scripts/run.ps1"
        ),
        "no SKILL.md, dot file or link: {used}"
    );
    assert_eq!(
        read_user_file(&dir, "references/steps.md").unwrap(),
        "Step one."
    );
    assert_eq!(
        read_user_file(&dir, r"references\steps.md").unwrap(),
        "Step one."
    );
    for bad in [
        "../guide/SKILL.md",
        "/etc/passwd",
        r"C:\Windows\win.ini",
        "",
    ] {
        assert!(read_user_file(&dir, bad).is_err(), "{bad}");
    }
    let refused = read_user_file(&dir, "away/keys.env").unwrap_err();
    assert!(
        refused.contains("leads out of the skill's folder"),
        "{refused}"
    );
    assert!(
        read_user_file(&dir, "logo.png")
            .unwrap_err()
            .contains("not text")
    );
    assert!(
        read_user_file(&dir, "big.txt")
            .unwrap_err()
            .contains("larger than 256 KiB")
    );
    assert!(
        read_user_file(&dir, "scripts")
            .unwrap_err()
            .contains("is a folder")
    );
}
