//! The client of Alelyon's identity service, without a user interface.
//!
//! - [`client`]: the service's native routes (`/identity/status`, `/identity/native/*`): password sign-in, refresh
//!   (which rotates the token), sign-out, the provider sign-in's code exchange, and QR pairing.
//! - [`loopback`]: a provider sign-in through the system browser, as OAuth 2.0 for native apps describes it
//!   (RFC 8252, loopback redirect) with PKCE (RFC 7636, S256).
//! - [`vault`]: "stay signed in": the refresh token sealed with Windows' DPAPI for the signed-in Windows user.
//! - [`social`]: friends, presence and one-to-one chat (`/identity/native/social/*`).
//!
//! The two contracts these follow are in `docs/` beside this crate. Nothing here logs a token: `Session`'s `Debug`
//! leaves it out, and a transport error is reported in a few words, never with the request or its URL.

#![deny(unsafe_code)]

pub mod client;
pub mod loopback;
pub mod social;
// Seals the kept session with Windows' DPAPI, so it may use unsafe code; nothing else in the crate may.
#[allow(unsafe_code)]
pub mod vault;

pub use client::{Account, Client, Failure, Notice, Pairing, Provider, Session, Status};
