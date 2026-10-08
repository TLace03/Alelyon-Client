# Contributing to Alelyon-Client

Thank you for reading the code, and for any issue or change you send.

## How changes land

This repository is a generated mirror. Its files are exported from Alelyon's private source repository, which stays
the single source of truth, so a pull request here is not merged as is:

1. Open an issue, or a pull request with the change. Small, focused changes with a test are easiest to take.
2. A maintainer reviews it. An accepted change is ported into the private source, with you credited
   (`Co-authored-by:`) in the commit.
3. The next export carries it back here, and the pull request is closed with a link to that export.

Files that are not exported (anything outside the export's list) are overwritten or removed by the next export, so
please do not rely on adding new top-level files here.

## What a change should carry

- `cargo test --locked` passing in the crate you changed, on Rust 1.97 or later.
- `cargo fmt` (`max_width = 140`, `use_small_heuristics = "Max"`) and no new `cargo clippy` warnings.
- No new dependency unless it is needed: each one is reviewed for licence, platforms and supply chain, and the
  lockfile is committed.
- A test for a fixed defect where one is practical.
- For anything touching tokens, the vault or the sign-in flow: say what the change protects against, and what it
  does not.

Behaviour of the identity service itself (rate limits, what a route returns) is the service's, documented in
`crates/alelyon-identity-client/docs/`. Report a disagreement between that contract and the service as an issue.

## Licence of contributions

By contributing you agree that your contribution is licensed under the Apache License 2.0, as the rest of the
repository is (see section 5 of the [LICENSE](https://github.com/TLace03/Alelyon-Client/blob/main/LICENSE)).

## Security issues

Not here: see [SECURITY.md](SECURITY.md).
