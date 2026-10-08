# Alelyon-Client

The open pieces of the Alelyon desktop client: source that users, developers and anyone evaluating Alelyon can read,
build, test and build on.

## What is here

| path | what it is |
|---|---|
| [`crates/alelyon-identity-client`](https://github.com/TLace03/Alelyon-Client/tree/main/crates/alelyon-identity-client) | The client of Alelyon's identity service, without a user interface: native sign-in (password, providers through the system browser with PKCE and a loopback redirect, QR pairing), "stay signed in" sealed with Windows' DPAPI, and friends, presence and chat. Its HTTP contracts are in its `docs/`. |

More of the client will be opened piece by piece. Each piece arrives as a crate that builds and tests on its own.

## Building and testing

Each crate is its own Cargo workspace with a committed lockfile. With Rust 1.97 or later:

```
cd crates/alelyon-identity-client
cargo test --locked
cargo run --example status
```

The example prints the identity service's status and needs no account.

## Where this source comes from

This repository is generated. The source of truth is Alelyon's private repository, and an export tool copies an
explicit list of files here, with `UPSTREAM.json` naming the source commit and a SHA-256 digest of each exported
file. That manifest is declared traceability, not a signature.

So a change is not merged here directly. Open an issue or a pull request: a maintainer ports an accepted change into
the private source, and the next export brings it back here, credited in the commit. See
[CONTRIBUTING.md](CONTRIBUTING.md).

## Security

Please do not report a vulnerability in a public issue. See [SECURITY.md](SECURITY.md): report it privately through
GitHub's security advisories for this repository.

## Licence

Apache-2.0: see [LICENSE](https://github.com/TLace03/Alelyon-Client/blob/main/LICENSE).
