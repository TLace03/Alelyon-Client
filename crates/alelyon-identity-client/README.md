# alelyon-identity-client

The client of Alelyon's identity service, as the Alelyon desktop app uses it, without a user interface. It covers:

- **native sign-in**: password sign-in, refresh (which rotates the token), sign-out, and the service's status
  (notices, which providers are on);
- **provider sign-in** (GitHub, Google, Microsoft, Apple, ORCID, Hugging Face, LinkedIn) through the system browser,
  answered on a loopback port (RFC 8252) with PKCE S256 (RFC 7636);
- **QR pairing**: a code shown on the desktop and approved on a phone that is already signed in;
- **stay signed in**: the refresh token sealed with Windows' DPAPI for the signed-in Windows user;
- **friends**: profile and username, exact-username search, friend requests, presence, and one-to-one chat.

An application draws its own screens over these calls. The HTTP contracts are in [`docs/`](docs/):
[native sign-in](docs/native-sign-in-contract.md) and [social](docs/social-contract.md).

## Quick start

```toml
[dependencies]
alelyon-identity-client = { path = "crates/alelyon-identity-client" }
```

```rust,no_run
use alelyon_identity_client::{Client, client, vault};

async fn sign_in(login: &str, password: &str) -> Result<(), String> {
    let base = client::base_url().ok_or("signing in is switched off (ALELYON_IDENTITY_URL=off)")?;
    let c = Client::new(base)?;
    let status = c.status().await.map_err(|f| f.words())?;
    if status.in_maintenance() {
        return Err("the service is down for maintenance".into());
    }
    let session = c.sign_in(login, password, true, None).await.map_err(|f| f.words())?;
    println!("signed in as {}", session.account.name()); // `{session:?}` would not show the token either
    if let Some(at) = vault::path() {
        vault::keep(&at, &session.refresh_token)?; // sealed with DPAPI; never written in the clear
    }
    Ok(())
}
```

The calls are `async` and run on any executor that can drive [reqwest](https://docs.rs/reqwest) (tokio). The
loopback listener's `wait` blocks its thread: run it off the UI thread.

Print the service's status (no account, no credential):

```
cargo run --example status
```

`ALELYON_IDENTITY_URL` names another service (for example a local stand-in, `http://127.0.0.1:8765`), or `off` for
none. `ALELYON_SESSION_FILE` names another file for the sealed session than `~/.alelyon/alelyon/session.sealed`
(`vault::path_or` lets an application use its own setting's name).

## Building and testing

The crate is its own Cargo workspace with a committed `Cargo.lock`:

```
cargo test --locked
```

Rust 1.97 or later (edition 2024). The DPAPI test runs on Windows only; elsewhere the vault refuses to keep anything.

## Threat model, in brief

- **The refresh token is the one secret.** With "stay signed in" it is sealed with DPAPI under the signed-in Windows
  user and this crate's own entropy, written whole to a new file and moved over the old one. Another Windows user, or
  the file copied to another PC, cannot open it. Code running as the same Windows user can: DPAPI separates users
  and machines, not programs. Without DPAPI (not Windows) nothing is kept. Without "stay signed in" an application
  should keep nothing; the service then issues a short-lived token.
- **Provider sign-in never passes through the app.** The system browser talks to the provider; the app opens the
  service's start page with a fresh PKCE S256 challenge and a random state, listens once on
  `http://127.0.0.1:<free port>/callback`, refuses an answer whose state does not match, and trades the one-time code
  with the verifier only it holds. A provider email that already has an account is never linked silently: the
  answer is a link ticket, honoured only by a password sign-in to that account.
- **No token in logs.** `Session`'s `Debug` omits the token, transport errors are reduced to a few words (never the
  URL, which can carry a code), and errors carry the service's words, not the request.
- **The password** is sent once, over TLS (the system's, through native-tls), and never kept.
- **The server enforces security.** Rate limits, lockouts, token rotation and revocation, consent before a provider
  creates an account, and every social rule (exact-match search, friends-only chat, request limits) are the
  service's. This client's checks are for a clear message, not a boundary: a modified client gains nothing the
  service does not grant.
- **Chat is not end-to-end encrypted** in social contract v1: messages are stored on the server in plain text.

## Licence

Apache-2.0.
