//! Find and replace for a person: the IDE's find bar and its search across
//! the folder ([`crate::tools::read::search`]). One matcher for what is
//! typed, as literal text or as a regular expression, with or without case,
//! whole words or not. A line is matched on its own, so a match never runs
//! across a line end, as an editor's find does. Not a port.
//!
//! Regular expressions are the `regex` crate's (as the agent's `grep`
//! compiles them): no back-references or look-around, compiled to at most
//! [`MAX_PATTERN_BYTES`]. Literal text is escaped, so `a.b` finds `a.b` and
//! nothing else. Without case, case folds as Unicode folds it. An empty match
//! is never a match: a pattern such as `x*` finds only its runs of `x`.

use std::ops::Range;

use regex::{Regex, RegexBuilder};

/// The most bytes a compiled pattern may take (the agent's `grep` allows as
/// many).
pub const MAX_PATTERN_BYTES: usize = 1024 * 1024;
/// The most matches one line yields to [`Matcher::find_in`]: a pattern that
/// matches every character of a long line stops there.
pub const MAX_PER_LINE: usize = 1000;

/// What is looked for.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Query {
    pub text: String,
    /// `text` is a regular expression, not literal text.
    pub regex: bool,
    pub case_sensitive: bool,
    /// Only whole words: a match starts and ends at a word boundary.
    pub whole_word: bool,
}

/// A query, compiled.
#[derive(Clone, Debug)]
pub struct Matcher {
    regex: Regex,
    /// A regular expression's replacement fills in `$1` and `${name}`; literal
    /// text's is taken as it is.
    expand: bool,
}

impl Matcher {
    /// The query compiled, or one sentence saying why it cannot be: it is
    /// empty, not a valid regular expression, or too large.
    pub fn new(query: &Query) -> Result<Matcher, String> {
        if query.text.is_empty() {
            return Err("Type what to look for.".to_owned());
        }
        let body = if query.regex {
            query.text.clone()
        } else {
            regex::escape(&query.text)
        };
        let pattern = if query.whole_word {
            format!(r"\b(?:{body})\b")
        } else {
            body
        };
        let regex = RegexBuilder::new(&pattern)
            .case_insensitive(!query.case_sensitive)
            .size_limit(MAX_PATTERN_BYTES)
            .build()
            .map_err(|error| match error {
                regex::Error::CompiledTooBig(_) => {
                    "That pattern is too large; Lattice compiles patterns of at most 1 MiB.".to_owned()
                }
                _ => "That is not a valid regular expression (no back-references or look-around)."
                    .to_owned(),
            })?;
        Ok(Matcher {
            regex,
            expand: query.regex,
        })
    }

    /// Where `line` matches, in order, as byte ranges: at most
    /// [`MAX_PER_LINE`], never an empty one.
    pub fn find_in(&self, line: &str) -> Vec<Range<usize>> {
        self.regex
            .find_iter(line)
            .filter(|found| !found.is_empty())
            .take(MAX_PER_LINE)
            .map(|found| found.range())
            .collect()
    }

    /// What replaces the match at `range` of `line`: `with` as it is, or, for a
    /// regular expression, `with` with `$1` and `${name}` filled in from that
    /// match (`$$` is a dollar sign).
    pub fn replacement(&self, line: &str, range: Range<usize>, with: &str) -> String {
        if !self.expand {
            return with.to_owned();
        }
        match self.regex.captures_at(line, range.start) {
            Some(captures) if captures.get(0).is_some_and(|whole| whole.range() == range) => {
                let mut out = String::new();
                captures.expand(with, &mut out);
                out
            }
            _ => with.to_owned(),
        }
    }

    /// `line` with every match replaced as [`Matcher::replacement`] replaces
    /// one, and how many were.
    pub fn replace_line(&self, line: &str, with: &str) -> (String, usize) {
        let mut out = String::with_capacity(line.len());
        let mut last = 0;
        let mut count = 0;
        for captures in self.regex.captures_iter(line) {
            let Some(whole) = captures.get(0) else {
                continue;
            };
            if whole.is_empty() {
                continue;
            }
            out.push_str(&line[last..whole.start()]);
            if self.expand {
                captures.expand(with, &mut out);
            } else {
                out.push_str(with);
            }
            last = whole.end();
            count += 1;
        }
        out.push_str(&line[last..]);
        (out, count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn query(text: &str) -> Query {
        Query {
            text: text.to_owned(),
            ..Query::default()
        }
    }

    fn found(query: &Query, line: &str) -> Vec<String> {
        Matcher::new(query)
            .unwrap()
            .find_in(line)
            .into_iter()
            .map(|range| line[range].to_owned())
            .collect()
    }

    #[test]
    fn literal_text_is_escaped_and_case_folds_unless_asked() {
        assert_eq!(found(&query("a.b"), "a.b axb A.B"), ["a.b", "A.B"]);
        let exact = Query {
            case_sensitive: true,
            ..query("a.b")
        };
        assert_eq!(found(&exact, "a.b axb A.B"), ["a.b"]);
        assert_eq!(found(&query("ÉTÉ"), "un été"), ["été"], "case folds as Unicode folds it");
        assert_eq!(found(&query("(x)"), "f(x) + (y)"), ["(x)"]);
    }

    #[test]
    fn whole_words_and_regular_expressions_and_no_empty_match() {
        let word = Query {
            whole_word: true,
            ..query("is")
        };
        assert_eq!(found(&word, "this is his island, is it"), ["is", "is"]);
        let regex = Query {
            regex: true,
            ..query(r"fn (\w+)\(")
        };
        assert_eq!(found(&regex, "pub fn main() { fn helper(x) }"), ["fn main(", "fn helper("]);
        let runs = Query {
            regex: true,
            ..query("x*")
        };
        assert_eq!(found(&runs, "abc"), Vec::<String>::new());
        assert_eq!(found(&runs, "axxbx"), ["xx", "x"]);
    }

    #[test]
    fn a_bad_or_empty_query_says_why() {
        assert_eq!(Matcher::new(&query("")).err().as_deref(), Some("Type what to look for."));
        let bad = Query {
            regex: true,
            ..query(r"(a)\1")
        };
        assert_eq!(
            Matcher::new(&bad).err().as_deref(),
            Some("That is not a valid regular expression (no back-references or look-around).")
        );
        let unbalanced = Query {
            regex: true,
            ..query("(")
        };
        assert!(Matcher::new(&unbalanced).is_err());
        assert!(Matcher::new(&query("(")).is_ok(), "as literal text it is fine");
        let huge = Query {
            regex: true,
            ..query(r"\w{2000}\w{2000}")
        };
        assert_eq!(
            Matcher::new(&huge).err().as_deref(),
            Some("That pattern is too large; Lattice compiles patterns of at most 1 MiB.")
        );
    }

    #[test]
    fn a_replacement_fills_in_groups_only_for_a_regular_expression() {
        let regex = Matcher::new(&Query {
            regex: true,
            ..query(r"(\w+)@(\w+)")
        })
        .unwrap();
        let line = "mail ann@home and bob@work";
        let second = regex.find_in(line)[1].clone();
        assert_eq!(regex.replacement(line, second, "$2:$1"), "work:bob");
        assert_eq!(regex.replace_line(line, "<$1>"), ("mail <ann> and <bob>".to_owned(), 2));
        assert_eq!(regex.replace_line(line, "$$"), ("mail $ and $".to_owned(), 2));
        let literal = Matcher::new(&query("ann")).unwrap();
        assert_eq!(literal.replacement("ann", 0..3, "$1"), "$1", "literal text is not expanded");
        assert_eq!(literal.replace_line("Ann ann", "x"), ("x x".to_owned(), 2));
        let runs = Matcher::new(&Query {
            regex: true,
            ..query("x*")
        })
        .unwrap();
        assert_eq!(runs.replace_line("axxb", "-"), ("a-b".to_owned(), 1), "empty matches are not replaced");
    }

    #[test]
    fn a_line_yields_at_most_its_cap() {
        let every = Matcher::new(&Query {
            regex: true,
            ..query(".")
        })
        .unwrap();
        assert_eq!(every.find_in(&"z".repeat(MAX_PER_LINE + 50)).len(), MAX_PER_LINE);
    }
}
