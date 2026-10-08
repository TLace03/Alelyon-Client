//! Signing in with a provider (GitHub, Google, ...): the system browser does it, and the answer comes back to the
//! application on loopback, as OAuth for native apps asks (RFC 8252 with PKCE, RFC 7636 S256). The application never
//! sees the provider's password page or its tokens: it opens the identity service's start page with a one-time
//! challenge, listens on `http://127.0.0.1:<a free port>/callback` for one answer carrying a one-time code and the
//! state it sent, and trades the code (with the secret verifier only it holds) for a session.
//!
//! The flow, with a [`crate::Client`] `c`:
//!
//! ```no_run
//! # async fn flow(c: &alelyon_identity_client::Client) -> Result<(), String> {
//! use alelyon_identity_client::loopback::{Answer, Listener, Pkce, open_in_browser};
//! let pkce = Pkce::new()?;
//! let listener = Listener::bind()?;
//! let redirect = listener.redirect_uri.clone();
//! open_in_browser(&c.provider_start_url("github", &redirect, &pkce.state, &pkce.challenge, false))?;
//! // `wait` blocks: run it off the UI thread.
//! match listener.wait(&pkce.state, std::time::Duration::from_secs(300))? {
//!     Answer::Code(code) => { let _session = c.exchange(&redirect, &code, &pkce.verifier).await; }
//!     Answer::Link(_ticket) => { /* ask for the account's password; sign_in(.., Some(ticket)) links it */ }
//! }
//! # Ok(()) }
//! ```

use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::time::{Duration, Instant};

use base64::Engine as _;
use sha2::Digest as _;

/// A provider sign-in's secrets, made fresh for each attempt.
pub struct Pkce {
    pub verifier: String,
    pub challenge: String,
    pub state: String,
}

fn random_text(bytes: usize) -> Result<String, String> {
    let mut buf = vec![0u8; bytes];
    getrandom::getrandom(&mut buf).map_err(|e| format!("no randomness from the system: {e}"))?;
    Ok(base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(buf))
}

impl Pkce {
    pub fn new() -> Result<Pkce, String> {
        let verifier = random_text(32)?;
        let challenge = challenge_of(&verifier);
        Ok(Pkce { verifier, challenge, state: random_text(16)? })
    }
}

/// The S256 challenge of a verifier: base64url(SHA-256(verifier)), no padding.
pub fn challenge_of(verifier: &str) -> String {
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(sha2::Sha256::digest(verifier.as_bytes()))
}

/// A listener on a free loopback port, waiting for one callback.
pub struct Listener {
    listener: TcpListener,
    pub redirect_uri: String,
}

impl Listener {
    pub fn bind() -> Result<Listener, String> {
        let listener = TcpListener::bind(("127.0.0.1", 0)).map_err(|e| format!("could not listen on this PC: {e}"))?;
        let port = listener.local_addr().map_err(|e| e.to_string())?.port();
        Ok(Listener { listener, redirect_uri: format!("http://127.0.0.1:{port}/callback") })
    }

    /// Wait (up to `limit`) for the browser to come back; answer it with a page saying to return to Alelyon. Requests
    /// that are not the callback (a browser asking for a favicon) are answered 404 and the wait goes on.
    pub fn wait(self, expected_state: &str, limit: Duration) -> Result<Answer, String> {
        let deadline = Instant::now() + limit;
        self.listener.set_nonblocking(true).map_err(|e| e.to_string())?;
        loop {
            if Instant::now() >= deadline {
                return Err("No answer came back from the browser in time. Try again.".into());
            }
            match self.listener.accept() {
                Ok((stream, _)) => {
                    if let Some(result) = answer(stream, expected_state) {
                        return result;
                    }
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => std::thread::sleep(Duration::from_millis(100)),
                Err(e) => return Err(format!("the browser's answer could not be read: {e}")),
            }
        }
    }
}

/// What the browser brought back.
#[derive(Clone, Debug, PartialEq)]
pub enum Answer {
    /// A one-time code to trade for a session.
    Code(String),
    /// The provider's verified email already has an account: a ticket that links the provider to it once the person
    /// signs in to that account with its password (the service never links silently).
    Link(String),
}

/// What a callback line says: the code (or a link ticket), or the provider's refusal; None when it is not the callback.
pub fn read_callback(request_line: &str, expected_state: &str) -> Option<Result<Answer, String>> {
    // "GET /callback?code=..&state=.. HTTP/1.1"
    let target = request_line.split_whitespace().nth(1)?;
    let query = target.strip_prefix("/callback")?.strip_prefix('?').unwrap_or("");
    let mut code = None;
    let mut state = None;
    let mut error = None;
    let mut ticket = None;
    for pair in query.split('&') {
        let (k, v) = pair.split_once('=').unwrap_or((pair, ""));
        let v = percent_encoding::percent_decode_str(&v.replace('+', " ")).decode_utf8_lossy().into_owned();
        match k {
            "code" => code = Some(v),
            "state" => state = Some(v),
            "error" => error = Some(v),
            "link_ticket" => ticket = Some(v),
            _ => {}
        }
    }
    Some(if state.as_deref() != Some(expected_state) {
        Err("The browser's answer did not match this sign-in. Try again.".into())
    } else if let (Some("account-exists"), Some(t)) = (error.as_deref(), ticket.filter(|t| !t.is_empty())) {
        Ok(Answer::Link(t))
    } else if let Some(e) = error {
        Err(match e.as_str() {
            "access_denied" | "access-denied" => "You cancelled the sign-in.".to_string(),
            "provider-disabled" => "That way of signing in is not switched on yet.".to_string(),
            "provider-failed" => "The other service did not complete the sign-in. Try again in a moment.".to_string(),
            "account-unavailable" => "That account cannot be signed in to right now.".to_string(),
            "maintenance" => "Alelyon's accounts are down for scheduled maintenance. Try again later, or use Alelyon offline.".to_string(),
            other => format!("The sign-in did not finish ({other})."),
        })
    } else {
        code.filter(|c| !c.is_empty()).map(Answer::Code).ok_or_else(|| "The browser's answer carried no code.".to_string())
    })
}

fn answer(stream: TcpStream, expected_state: &str) -> Option<Result<Answer, String>> {
    let _ = stream.set_read_timeout(Some(Duration::from_secs(5)));
    let mut line = String::new();
    BufReader::new(&stream).read_line(&mut line).ok()?;
    let result = read_callback(&line, expected_state);
    let (status, page) = match &result {
        None => ("404 Not Found", "Not found."),
        Some(Ok(_)) => (
            "200 OK",
            "<!doctype html><title>Alelyon</title><body style=\"font-family:sans-serif;background:#111;color:#eee;text-align:center;padding-top:20vh\"><h2>You are signed in.</h2><p>You can close this tab and return to Alelyon.</p></body>",
        ),
        Some(Err(_)) => (
            "200 OK",
            "<!doctype html><title>Alelyon</title><body style=\"font-family:sans-serif;background:#111;color:#eee;text-align:center;padding-top:20vh\"><h2>The sign-in did not finish.</h2><p>Return to Alelyon to see why.</p></body>",
        ),
    };
    let mut s = &stream;
    let _ = write!(
        s,
        "HTTP/1.1 {status}\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\nCache-Control: no-store\r\n\r\n{page}",
        page.len()
    );
    result
}

/// Open an address in the person's default browser, windowless. Windows only (it asks the shell's URL handler);
/// elsewhere the spawn fails and says so.
pub fn open_in_browser(url: &str) -> Result<(), String> {
    let mut command = std::process::Command::new("rundll32.exe");
    command.args(["url.dll,FileProtocolHandler", url]);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x0800_0000);
    }
    command.spawn().map(|_| ()).map_err(|e| format!("the browser could not be opened: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_challenge_is_rfc_7636s_s256() {
        // RFC 7636 appendix B
        assert_eq!(challenge_of("dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"), "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM");
        let a = Pkce::new().unwrap();
        let b = Pkce::new().unwrap();
        assert_ne!(a.verifier, b.verifier, "fresh each time");
        assert_eq!(a.verifier.len(), 43);
        assert_eq!(challenge_of(&a.verifier), a.challenge);
    }

    #[test]
    fn the_callback_gives_its_code_only_with_this_sign_ins_state() {
        assert_eq!(read_callback("GET /callback?code=abc%2B1&state=s1 HTTP/1.1", "s1"), Some(Ok(Answer::Code("abc+1".into()))));
        assert_eq!(
            read_callback("GET /callback?error=account-exists&link_ticket=t1&state=s1 HTTP/1.1", "s1"),
            Some(Ok(Answer::Link("t1".into()))),
            "an existing account: a ticket to link it, never a session"
        );
        assert!(matches!(read_callback("GET /callback?error=account-exists&state=s1 HTTP/1.1", "s1"), Some(Err(_))), "no ticket, no link");
        assert!(matches!(read_callback("GET /callback?code=abc&state=other HTTP/1.1", "s1"), Some(Err(w)) if w.contains("did not match")));
        assert!(
            matches!(read_callback("GET /callback?error=access_denied&state=s1 HTTP/1.1", "s1"), Some(Err(w)) if w.contains("cancelled"))
        );
        assert_eq!(read_callback("GET /favicon.ico HTTP/1.1", "s1"), None);
        assert!(matches!(read_callback("GET /callback?state=s1 HTTP/1.1", "s1"), Some(Err(_))));
    }

    #[test]
    fn a_real_loopback_answer_reaches_the_listener() {
        let l = Listener::bind().unwrap();
        let uri = l.redirect_uri.clone();
        assert!(uri.starts_with("http://127.0.0.1:") && uri.ends_with("/callback"));
        let port: u16 = uri.trim_start_matches("http://127.0.0.1:").trim_end_matches("/callback").parse().unwrap();
        let browser = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(100));
            let mut s = TcpStream::connect(("127.0.0.1", port)).unwrap();
            write!(s, "GET /callback?code=c0de&state=st HTTP/1.1\r\nHost: x\r\n\r\n").unwrap();
            let mut page = String::new();
            let _ = std::io::Read::read_to_string(&mut s, &mut page);
            page
        });
        assert_eq!(l.wait("st", Duration::from_secs(5)), Ok(Answer::Code("c0de".into())));
        assert!(browser.join().unwrap().contains("You are signed in"));
    }
}
