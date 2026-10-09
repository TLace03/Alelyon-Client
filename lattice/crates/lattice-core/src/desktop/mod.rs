//! Auto mode: the agent on the whole desktop (as chosen on
//! 2026-10-08). Not a port.
//!
//! The default route for a connection is the agent's own browser
//! (`crate::browser`). "If users want to enable auto mode with bypass
//! permissions then whole desktop should be the route": switched on only
//! through the core's own dialog (`ConfirmRequest::AutoMode`), auto mode gives
//! Agent-mode turns the desktop's tools, with a choice of what still
//! asks, "Money and accounts ask": a purchase or a payment, and a sign-in,
//! security, consent or account change, wait for the reader; posting,
//! messaging, editing and deleting go ahead. A global hotkey (Ctrl+Alt+End)
//! stops every running agent turn, and the agent never acts on Alelyon itself,
//! a password manager or Windows' own sign-in and permission prompts.
//!
//! - [`policy`]: the protected windows, what still asks, the picture's size.
//! - [`image`]: the screen scaled and encoded for the model.
//! - [`session`]: [`Desktop`], its [`Driver`] and the setting ([`prefs`]).

pub mod image;
pub mod policy;
pub mod session;
#[cfg(test)]
pub(crate) mod tests;

pub use session::{Desktop, Driver, Look, STOP_KEYS, SystemDriver, prefs};
