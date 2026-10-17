//! The Alelyon app on this computer, through its agent bridge.
//!
//! When the person switches on Settings, Agents, "Let my agent drive Alelyon", the app listens on this computer
//! only and writes its address and a new secret to `%USERPROFILE%\.alelyon\app\agent-bridge.json`, a file only their
//! Windows user can read. This reads that file and calls the app: `POST <url>/v1/<command>` with the command's
//! fields and the secret as a bearer token. The app answers reads at once; it asks the person before the first action
//! of a session, and shows every action in its window.

use std::path::PathBuf;
use std::time::Duration;

use serde_json::{Value, json};

use crate::api::{Failure, Pages};

/// Where the app writes its bridge's address and secret.
pub fn handoff_path() -> Option<PathBuf> {
    let home = std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME"))?;
    Some(PathBuf::from(home).join(".alelyon").join("app").join("agent-bridge.json"))
}

fn off() -> Failure {
    Failure {
        status: 0,
        code: "app-bridge-off".into(),
        message: "The Alelyon app is not open, or its agent bridge is off. Open Alelyon, then Settings, Agents, and \
                  switch \"Let my agent drive Alelyon\" on."
            .into(),
    }
}

/// The app's bridge, read from its handoff file at each call (the app writes a new one each time it starts).
pub struct App {
    pub handoff: Option<PathBuf>,
}

impl App {
    fn endpoint(&self) -> Result<(String, String), Failure> {
        let path = self.handoff.as_ref().ok_or_else(off)?;
        let text = std::fs::read_to_string(path).map_err(|_| off())?;
        let v: Value = serde_json::from_str(&text).map_err(|_| off())?;
        let url = v.get("url").and_then(Value::as_str).unwrap_or("");
        let token = v.get("token").and_then(Value::as_str).unwrap_or("");
        let local = url.starts_with("http://127.0.0.1:") && url[17..].bytes().all(|b| b.is_ascii_digit());
        if !local || token.len() < 32 || !token.bytes().all(|b| b.is_ascii_alphanumeric()) {
            return Err(off());
        }
        Ok((url.to_string(), token.to_string()))
    }
}

impl Pages for App {
    fn call(&self, command: &str, body: &Value) -> Result<Value, Failure> {
        let (url, token) = self.endpoint()?;
        let http = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(40))
            .redirect(reqwest::redirect::Policy::none())
            .no_proxy()
            .build()
            .map_err(|_| off())?;
        let sent = http
            .post(format!("{url}/v1/{command}"))
            .header("Authorization", format!("Bearer {token}"))
            .header("Content-Type", "application/json")
            .body(serde_json::to_vec(body).unwrap_or_default())
            .send()
            .map_err(|_| off())?;
        let status = sent.status().as_u16();
        let parsed: Value = sent.text().ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or(json!({}));
        if status == 200 {
            return Ok(parsed);
        }
        Err(Failure {
            status,
            code: parsed.get("error").and_then(Value::as_str).unwrap_or("error").to_string(),
            message: parsed.get("message").and_then(Value::as_str).unwrap_or("The app refused the request.").to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn without_the_bridge_the_answer_says_how_to_switch_it_on() {
        let app = App { handoff: Some(std::env::temp_dir().join("alelyon-cli-no-such-bridge.json")) };
        let failure = app.call("status", &json!({})).unwrap_err();
        assert_eq!(failure.code, "app-bridge-off");
        assert!(failure.message.contains("Settings, Agents"));
    }

    #[test]
    fn a_handoff_that_points_anywhere_but_this_computer_is_refused() {
        let path = std::env::temp_dir().join(format!("alelyon-cli-bridge-{}.json", std::process::id()));
        std::fs::write(&path, json!({"url": "http://example.com:80", "token": "a".repeat(64)}).to_string()).unwrap();
        let failure = App { handoff: Some(path.clone()) }.call("status", &json!({})).unwrap_err();
        assert_eq!(failure.code, "app-bridge-off");
        let _ = std::fs::remove_file(path);
    }
}
