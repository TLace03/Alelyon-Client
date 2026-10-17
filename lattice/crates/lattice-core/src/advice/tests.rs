//! The stack advisor's catalog, ranking, project scan, review and step detection.

use super::*;
use crate::testkit::TempDir;

#[test]
fn the_catalog_is_whole_and_every_reference_in_it_resolves() {
    let c = catalog();
    assert!(c.areas.len() >= 15, "{}", c.areas.len());
    let groups: BTreeSet<&str> = c.areas.iter().map(|a| a.group.as_str()).collect();
    assert_eq!(
        groups,
        ["Databases and storage", "Frameworks and architecture", "Hosted services", "Languages and runtimes"].into_iter().collect()
    );
    let mut ids = BTreeSet::new();
    for area in &c.areas {
        assert!(ids.insert(area.id.as_str()), "area {} twice", area.id);
        assert!(!area.triggers.is_empty() && !area.needs.is_empty() && area.options.len() >= 3, "{}", area.id);
        let mut options = BTreeSet::new();
        for o in &area.options {
            assert!(options.insert(o.id.as_str()), "{} twice in {}", o.id, area.id);
            for need in o.best_for.iter().chain(&o.poor_for) {
                assert!(area.needs.contains_key(need), "{}/{} names the unknown need {need}", area.id, o.id);
            }
            assert!(o.pros.len() >= 2 && o.cons.len() >= 2 && !o.guide.is_empty(), "{}/{}", area.id, o.id);
            for check in &o.checks {
                assert!(!check.suggest.is_empty() && !check.why.is_empty(), "{}/{}/{}", area.id, o.id, check.id);
            }
        }
    }
}

#[test]
fn ranking_follows_the_needs_and_prefers_what_the_project_uses() {
    let db = area("database").unwrap();
    let none = Project::default();
    let embedded = rank(db, &["embedded".into(), "single-user".into()], &none);
    assert_eq!(embedded[0].choice.id, "sqlite");
    let relational = rank(db, &["relational".into(), "transactions".into(), "managed".into()], &none);
    assert_eq!(relational[0].choice.id, "postgres");
    // A project on MySQL keeps MySQL ahead for the same needs.
    let mut on_mysql = Project::default();
    on_mysql.text.push_str("\"mysql2\": \"^3\"");
    let kept = rank(db, &["relational".into(), "transactions".into(), "managed".into()], &on_mysql);
    assert_eq!((kept[0].choice.id.as_str(), kept[0].used), ("mysql", true));
}

#[test]
fn a_scan_reads_names_and_manifests_and_never_an_env_file() {
    let dir = TempDir::new("advice-scan");
    let root = dir.path();
    std::fs::write(root.join("package.json"), r#"{"dependencies": {"express": "^4", "pg": "^8"}}"#).unwrap();
    std::fs::write(root.join(".env"), "DATABASE_URL=postgres://secret").unwrap();
    std::fs::write(root.join("Dockerfile"), "FROM node:22").unwrap();
    std::fs::create_dir_all(root.join("web")).unwrap();
    std::fs::write(root.join("web").join("astro.config.mjs"), "import astro from 'astro'").unwrap();
    std::fs::create_dir_all(root.join("node_modules").join("x")).unwrap();
    std::fs::write(root.join("node_modules").join("x").join("package.json"), "{\"sentry\": 1}").unwrap();
    let project = scan(root);
    assert!(!project.text.contains("secret"), "a .env file was read");
    assert!(!project.text.contains("sentry"), "node_modules was read");
    let used: Vec<&str> = detected(&project).iter().map(|(_, c)| c.id.as_str()).collect();
    for id in ["typescript", "postgres", "container-platform", "fastify"] {
        assert!(used.contains(&id), "{id} not detected in {used:?}");
    }
    // What the catalog suggests: a lock file, a pool, migrations, helmet, a .dockerignore.
    let checks: Vec<String> = review(&project).into_iter().map(|s| s.check).collect();
    for id in ["node-lock", "pg-pool", "pg-migrations", "express-helmet", "dockerignore"] {
        assert!(checks.contains(&id.to_string()), "{id} not suggested in {checks:?}");
    }
    // With them in place, they are not.
    std::fs::write(root.join("package-lock.json"), "{}").unwrap();
    std::fs::write(root.join(".dockerignore"), "node_modules").unwrap();
    std::fs::write(root.join("package.json"), r#"{"dependencies": {"express": "^4", "pg": "^8", "pg-pool": "1", "helmet": "7", "knex": "3"}}"#).unwrap();
    let checks: Vec<String> = review(&scan(root)).into_iter().map(|s| s.check).collect();
    assert!(checks.is_empty(), "{checks:?}");
}

#[test]
fn a_step_names_the_area_it_needs_a_decision_in() {
    assert_eq!(area_for_step("Set up the database schema").map(|a| a.id.as_str()), Some("database"));
    assert_eq!(area_for_step("Add sign in with Google").map(|a| a.id.as_str()), Some("auth"));
    assert_eq!(area_for_step("Deploy to production").map(|a| a.id.as_str()), Some("hosting"));
    assert!(area_for_step("Fix the typo in the README").is_none());
    // Work on what is there is not a decision: a bug in the login, the database's tests.
    assert!(area_for_step("Fix the login bug").is_none());
    assert!(area_for_step("Test the database queries").is_none());
    // A word inside another word is not a trigger: "uses" is not "ses", "apiary" is not "api".
    assert!(!found("it uses the cache", "ses") && found("send with ses today", "ses"));
}
