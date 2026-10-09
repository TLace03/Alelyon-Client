//! A credential that cannot leak by being printed.
//!
//! Not a port: openai-agents-python passes keys around as plain `str`, so a
//! stray `repr()` or log line can carry one. This type makes the mistake
//! impossible to write by accident.
//!
//! Invariants: `Debug` and `Display` print `***` for every value; the type has
//! no `Serialize` impl, so it cannot reach a JSON value, a span or a run event
//! through serde; the only way to read the text is [`SecretString::expose`],
//! which the model client calls once, when it builds the `Authorization`
//! header.

use std::fmt;

#[derive(Clone)]
pub struct SecretString(String);

impl SecretString {
    pub fn new(secret: impl Into<String>) -> Self {
        Self(secret.into())
    }

    /// The secret text. Call sites are the places a credential is allowed to
    /// leave this type; there is exactly one in this crate (the request header).
    pub fn expose(&self) -> &str {
        &self.0
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl fmt::Debug for SecretString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("***")
    }
}

impl fmt::Display for SecretString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("***")
    }
}

impl From<String> for SecretString {
    fn from(secret: String) -> Self {
        Self(secret)
    }
}

impl From<&str> for SecretString {
    fn from(secret: &str) -> Self {
        Self(secret.to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_secret_prints_as_stars() {
        let secret = SecretString::new("sk-live-1234");
        assert_eq!(format!("{secret:?}"), "***");
        assert_eq!(format!("{secret}"), "***");
        assert_eq!(format!("{:?}", Some(&secret)), "Some(***)");
        assert_eq!(secret.expose(), "sk-live-1234");
    }
}
