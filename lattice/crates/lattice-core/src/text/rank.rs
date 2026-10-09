//! Fuzzy path ranking for the file picker (the chat core's spec §4.3
//! `files`, row D6). A port of the shipping app's quick open, `rankPaths` in
//! the web Lattice's `src/code/tree.ts`, held to a table
//! ported from its tests (`src/code/code.test.ts`, "quick open").
//!
//! The rule, as the shipping app has it: the query, with its whitespace
//! removed and lower-cased, must be a subsequence of the path. It is looked
//! for in the file name first, then in the whole path. A match in the file
//! name scores 10; each matched character scores 1, 5 more when it follows the
//! previous match, 8 more at the start of the path or after one of `/-_. `,
//! and 3 more inside the file name. Best first; then the shorter path; then
//! the paths in a natural order.
//!
//! Deviations, each invisible on ASCII paths: positions count characters
//! (Unicode scalar values), not UTF-16 code units; lower-casing is per
//! character, keeping a character whose lower case is longer than one
//! character as it is, so positions stay aligned with the path; and the last
//! tie-break is a natural order without case (digit runs compared as numbers),
//! not the browser's `Intl.Collator`, whose order of punctuation depends on
//! its locale data.

use std::cmp::Ordering;

use lattice_protocol::conversation::RankedPath;

/// The shipping app's default number of results.
pub const DEFAULT_LIMIT: usize = 50;

const SEPARATORS: [char; 5] = ['/', '-', '_', '.', ' '];

fn lower(c: char) -> char {
    let mut lowered = c.to_lowercase();
    match (lowered.next(), lowered.next()) {
        (Some(one), None) => one,
        _ => c,
    }
}

/// Where `needle` occurs in order in `hay`, starting at `from`.
fn subsequence(needle: &[char], hay: &[char], from: usize) -> Option<Vec<usize>> {
    let mut positions = Vec::with_capacity(needle.len());
    let mut at = from;
    for character in needle {
        let found = at + hay.get(at..)?.iter().position(|c| c == character)?;
        positions.push(found);
        at = found + 1;
    }
    Some(positions)
}

fn score(needle: &[char], path: &str) -> Option<RankedPath> {
    let chars: Vec<char> = path.chars().collect();
    let hay: Vec<char> = chars.iter().copied().map(lower).collect();
    let name_start = chars.iter().rposition(|c| *c == '/').map_or(0, |at| at + 1);
    let in_name = subsequence(needle, &hay, name_start);
    let positions = match &in_name {
        Some(positions) => positions.clone(),
        None => subsequence(needle, &hay, 0)?,
    };
    let mut total: u32 = if in_name.is_some() { 10 } else { 0 };
    for (index, &at) in positions.iter().enumerate() {
        total += 1;
        if index > 0 && positions[index - 1] + 1 == at {
            total += 5;
        }
        if at == 0 || SEPARATORS.contains(&chars[at - 1]) {
            total += 8;
        }
        if at >= name_start {
            total += 3;
        }
    }
    Some(RankedPath {
        path: path.to_owned(),
        score: total,
        positions: positions.into_iter().map(|at| at as u32).collect(),
    })
}

/// A natural order without case: digit runs compared as numbers.
pub fn natural_cmp(a: &str, b: &str) -> Ordering {
    let mut a = a.chars().map(lower).peekable();
    let mut b = b.chars().map(lower).peekable();
    loop {
        match (a.peek().copied(), b.peek().copied()) {
            (None, None) => return Ordering::Equal,
            (None, Some(_)) => return Ordering::Less,
            (Some(_), None) => return Ordering::Greater,
            (Some(x), Some(y)) if x.is_ascii_digit() && y.is_ascii_digit() => {
                let mut left = String::new();
                while let Some(c) = a.peek().copied().filter(char::is_ascii_digit) {
                    left.push(c);
                    a.next();
                }
                let mut right = String::new();
                while let Some(c) = b.peek().copied().filter(char::is_ascii_digit) {
                    right.push(c);
                    b.next();
                }
                let left = left.trim_start_matches('0');
                let right = right.trim_start_matches('0');
                let order = left.len().cmp(&right.len()).then_with(|| left.cmp(right));
                if order != Ordering::Equal {
                    return order;
                }
            }
            (Some(x), Some(y)) => {
                if x != y {
                    return x.cmp(&y);
                }
                a.next();
                b.next();
            }
        }
    }
}

/// `rankPaths(query, paths, limit)`: the paths matching `query`, best first,
/// at most `limit`.
pub fn rank_paths<'a>(
    query: &str,
    paths: impl IntoIterator<Item = &'a str>,
    limit: usize,
) -> Vec<RankedPath> {
    let needle: Vec<char> = query
        .chars()
        .filter(|c| !c.is_whitespace())
        .map(lower)
        .collect();
    if needle.is_empty() {
        return Vec::new();
    }
    let mut ranked: Vec<RankedPath> = paths
        .into_iter()
        .filter_map(|path| score(&needle, path))
        .collect();
    ranked.sort_by(|a, b| {
        b.score
            .cmp(&a.score)
            .then_with(|| a.path.chars().count().cmp(&b.path.chars().count()))
            .then_with(|| natural_cmp(&a.path, &b.path))
    });
    ranked.truncate(limit);
    ranked
}

#[cfg(test)]
mod tests {
    use super::*;

    fn paths() -> Vec<&'static str> {
        vec![
            "alelyon/service/lattice_service/app.py",
            "alelyon/desktop/lattice_desktop/src/App.tsx",
            "tests/products/test_lattice_service.py",
            "docs/apps.md",
        ]
    }

    fn ranked(query: &str, paths: &[&str]) -> Vec<String> {
        rank_paths(query, paths.iter().copied(), DEFAULT_LIMIT)
            .into_iter()
            .map(|row| row.path)
            .collect()
    }

    /// code.test.ts: "puts a match in the file name ahead of one spread
    /// through directories", with the shipping test's own assertions. They
    /// hold even without the file-name bonus of 10 (the per-character bonus
    /// in the name carries them); that bonus is pinned by the scores test.
    #[test]
    fn a_match_in_the_file_name_comes_first() {
        let mut more = paths();
        more.extend(["alelyon/lattice/service/app.py", "lsa.md"]);
        let lsa = ranked("lsa", &more);
        assert_eq!(lsa[0], "lsa.md");
        assert!(lsa.contains(&"alelyon/lattice/service/app.py".to_owned()));
        let named = ranked("app", &paths());
        let mut top: Vec<String> = named[..3].to_vec();
        top.sort();
        assert_eq!(
            top,
            [
                "alelyon/desktop/lattice_desktop/src/App.tsx",
                "alelyon/service/lattice_service/app.py",
                "docs/apps.md",
            ]
        );
        assert!(!named.contains(&"tests/products/test_lattice_service.py".to_owned()));
    }

    /// code.test.ts: "matches a subsequence and reports where, and nothing for
    /// no match".
    #[test]
    fn a_subsequence_reports_where_and_no_match_is_nothing() {
        let path = "tests/lattice_service.py";
        let best = rank_paths("tls", [path], DEFAULT_LIMIT);
        let chars: Vec<char> = path.chars().collect();
        assert_eq!(
            best[0]
                .positions
                .iter()
                .map(|at| chars[*at as usize])
                .collect::<String>(),
            "tls"
        );
        assert!(rank_paths("zzz", paths(), DEFAULT_LIMIT).is_empty());
        assert!(rank_paths("   ", paths(), DEFAULT_LIMIT).is_empty());
    }

    /// code.test.ts: "stops at the limit".
    #[test]
    fn it_stops_at_the_limit() {
        let many: Vec<String> = (0..80).map(|index| format!("src/file{index}.ts")).collect();
        assert_eq!(
            rank_paths("file", many.iter().map(String::as_str), 25).len(),
            25
        );
    }

    /// The arithmetic, by hand from tree.ts's `score`, and the order of ties.
    /// Mutants: no file-name bonus; no adjacency bonus; the shorter path not
    /// preferred.
    #[test]
    fn the_scores_and_the_order_are_the_shipping_apps() {
        // "lsa" in "lsa.md": name match 10; l 1+8+3, s 1+5+3, a 1+5+3.
        assert_eq!(rank_paths("lsa", ["lsa.md"], 50)[0].score, 40);
        // "app" in "docs/apps.md" (name starts at 5): 10; a 1+8+3, p 1+5+3,
        // p 1+5+3.
        let apps = &rank_paths("app", ["docs/apps.md"], 50)[0];
        assert_eq!((apps.score, apps.positions.clone()), (40, vec![5, 6, 7]));
        // Not in the name: "dap" over "docs/apps.md": d 1+8, a 1+8+3, p 1+5+3.
        let spread = &rank_paths("dap", ["docs/apps.md"], 50)[0];
        assert_eq!(
            (spread.score, spread.positions.clone()),
            (30, vec![0, 5, 6])
        );
        // Equal scores: the shorter path first, then the natural order.
        let order = ranked("a", &["b/a10.txt", "b/a2.txt", "b/a.txt", "c/a.txt"]);
        assert_eq!(order, ["b/a.txt", "c/a.txt", "b/a2.txt", "b/a10.txt"]);
        assert_eq!(natural_cmp("file2", "FILE10"), Ordering::Less);
        assert_eq!(natural_cmp("a", "A"), Ordering::Equal);
    }

    #[test]
    fn whitespace_in_the_query_is_ignored_and_case_does_not_matter() {
        assert_eq!(
            ranked("L S A", &["LSA.md", "x/y"]),
            ["LSA.md"],
            "the query's spaces are removed"
        );
    }
}
