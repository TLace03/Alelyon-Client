//! The cloud ears' transport: Windows' own HTTP stack (WinHTTP), so HTTPS needs no TLS library here.
//!
//! One synchronous request per call: no redirect is followed, the system's proxy is used for a provider (none
//! for an address on this machine), and every stage is bounded by the call's timeout. The reply's status, the
//! headers the ears read (`retry-after`, `x-ratelimit-*`, `content-type`) and a bounded body come back; an
//! error is one sentence without the address's query, the key or the reply.

use std::time::Duration;

use windows::core::{HSTRING, PCWSTR};
use windows::Win32::Networking::WinHttp::{
    WinHttpCloseHandle, WinHttpConnect, WinHttpOpen, WinHttpOpenRequest, WinHttpQueryHeaders, WinHttpReadData,
    WinHttpReceiveResponse, WinHttpSendRequest, WinHttpSetOption, WinHttpSetTimeouts, WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY,
    WINHTTP_ACCESS_TYPE_NO_PROXY, WINHTTP_DISABLE_REDIRECTS, WINHTTP_FLAG_SECURE, WINHTTP_OPEN_REQUEST_FLAGS,
    WINHTTP_OPTION_DISABLE_FEATURE, WINHTTP_QUERY_RAW_HEADERS_CRLF,
};

use crate::cloud::{Reply, Transport};

/// The most of a reply read.
const MAX_BODY: usize = 16 * 1024 * 1024;

/// WinHTTP, as the ears use it.
#[derive(Clone, Copy, Debug, Default)]
pub struct WinHttp;

/// A handle closed when dropped.
struct Handle(*mut core::ffi::c_void);

impl Drop for Handle {
    fn drop(&mut self) {
        if !self.0.is_null() {
            // SAFETY: a handle WinHTTP returned, closed once, here.
            unsafe {
                let _ = WinHttpCloseHandle(self.0);
            }
        }
    }
}

/// An address's parts: secure, host, port, path with query.
fn split(url: &str) -> Result<(bool, String, u16, String), String> {
    let (secure, rest) = if let Some(r) = url.strip_prefix("https://") {
        (true, r)
    } else if let Some(r) = url.strip_prefix("http://") {
        (false, r)
    } else {
        return Err("the provider's address is not http or https".into());
    };
    let (authority, path) = match rest.find('/') {
        Some(i) => (&rest[..i], &rest[i..]),
        None => (rest, "/"),
    };
    if authority.contains('@') {
        return Err("the provider's address may not carry credentials".into());
    }
    let (host, port) = match authority.rsplit_once(':') {
        Some((h, p)) if !h.ends_with(']') || authority.starts_with('[') => {
            (h.to_string(), p.parse::<u16>().map_err(|_| "the provider's address has a bad port".to_string())?)
        }
        _ => (authority.to_string(), if secure { 443 } else { 80 }),
    };
    let host = host.trim_start_matches('[').trim_end_matches(']').to_string();
    if host.is_empty() {
        return Err("the provider's address has no host".into());
    }
    Ok((secure, host, port, path.to_string()))
}

/// Status line and headers (lower-case names, only those the ears read) from WinHTTP's raw CRLF block.
fn parse_head(raw: &str) -> (u16, Vec<(String, String)>) {
    let mut lines = raw.split("\r\n");
    let status = lines.next().and_then(|l| l.split_whitespace().nth(1)).and_then(|s| s.parse().ok()).unwrap_or(0);
    let headers = lines
        .filter_map(|l| l.split_once(':'))
        .map(|(k, v)| (k.trim().to_ascii_lowercase(), v.trim().to_string()))
        .filter(|(k, _)| k == "retry-after" || k == "content-type" || k.starts_with("x-ratelimit-"))
        .collect();
    (status, headers)
}

fn ms(d: Duration) -> i32 {
    d.as_millis().min(i32::MAX as u128) as i32
}

impl Transport for WinHttp {
    fn post(&self, url: &str, headers: &[(String, String)], body: &[u8], timeout: Duration) -> Result<Reply, String> {
        let (secure, host, port, path) = split(url)?;
        let loopback = matches!(host.as_str(), "127.0.0.1" | "localhost" | "::1");
        if !secure && !loopback {
            return Err("the ears send to a provider over HTTPS only".into());
        }
        let unreachable = |_| "the provider could not be reached".to_string();
        // SAFETY: every handle is checked for null before use and closed by `Handle`'s drop, children before
        // parents (they are dropped in reverse order of creation); every buffer outlives the call it is passed to.
        unsafe {
            let access = if loopback { WINHTTP_ACCESS_TYPE_NO_PROXY } else { WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY };
            let session = Handle(WinHttpOpen(&HSTRING::from("angel-ears"), access, PCWSTR::null(), PCWSTR::null(), 0));
            if session.0.is_null() {
                return Err("the system's HTTP stack could not start".into());
            }
            let t = ms(timeout);
            WinHttpSetTimeouts(session.0, t, t, t, t).map_err(unreachable)?;
            let connect = Handle(WinHttpConnect(session.0, &HSTRING::from(host.as_str()), port, 0));
            if connect.0.is_null() {
                return Err("the provider could not be reached".into());
            }
            let flags = if secure { WINHTTP_FLAG_SECURE } else { WINHTTP_OPEN_REQUEST_FLAGS(0) };
            let request = Handle(WinHttpOpenRequest(
                connect.0,
                &HSTRING::from("POST"),
                &HSTRING::from(path.as_str()),
                PCWSTR::null(),
                PCWSTR::null(),
                std::ptr::null(),
                flags,
            ));
            if request.0.is_null() {
                return Err("the request could not be made".into());
            }
            let no_redirects = WINHTTP_DISABLE_REDIRECTS.to_ne_bytes();
            WinHttpSetOption(Some(request.0), WINHTTP_OPTION_DISABLE_FEATURE, Some(&no_redirects)).map_err(unreachable)?;
            let head: String = headers.iter().map(|(k, v)| format!("{k}: {v}\r\n")).collect();
            let head: Vec<u16> = head.encode_utf16().collect();
            let length = u32::try_from(body.len()).map_err(|_| "the audio is too large to send".to_string())?;
            WinHttpSendRequest(request.0, Some(&head), Some(body.as_ptr().cast()), length, length, 0).map_err(|e| {
                if e.code().0 as u32 & 0xFFFF == 12002 {
                    "the provider did not answer in time".to_string()
                } else {
                    "the provider could not be reached".to_string()
                }
            })?;
            WinHttpReceiveResponse(request.0, std::ptr::null_mut()).map_err(|_| "the provider did not answer".to_string())?;
            let mut size = 0u32;
            let _ = WinHttpQueryHeaders(request.0, WINHTTP_QUERY_RAW_HEADERS_CRLF, PCWSTR::null(), None, &mut size, std::ptr::null_mut());
            let mut raw = vec![0u16; (size as usize / 2) + 1];
            let mut size_bytes = (raw.len() * 2) as u32;
            WinHttpQueryHeaders(
                request.0,
                WINHTTP_QUERY_RAW_HEADERS_CRLF,
                PCWSTR::null(),
                Some(raw.as_mut_ptr().cast()),
                &mut size_bytes,
                std::ptr::null_mut(),
            )
            .map_err(|_| "the provider's answer could not be read".to_string())?;
            let raw = String::from_utf16_lossy(&raw[..(size_bytes as usize / 2).min(raw.len())]);
            let (status, headers) = parse_head(&raw);
            let mut body = Vec::new();
            let mut chunk = vec![0u8; 64 * 1024];
            loop {
                let mut read = 0u32;
                WinHttpReadData(request.0, chunk.as_mut_ptr().cast(), chunk.len() as u32, &mut read)
                    .map_err(|_| "the provider's answer was cut off".to_string())?;
                if read == 0 {
                    break;
                }
                if body.len() + read as usize > MAX_BODY {
                    return Err("the provider's answer was too large".into());
                }
                body.extend_from_slice(&chunk[..read as usize]);
            }
            drop(request);
            drop(connect);
            drop(session);
            Ok(Reply { status, headers, body })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;

    #[test]
    fn addresses_are_split_and_only_https_leaves_this_machine() {
        assert_eq!(split("https://api.groq.com/openai/v1/audio/transcriptions").unwrap(), (true, "api.groq.com".into(), 443, "/openai/v1/audio/transcriptions".into()));
        assert_eq!(split("http://127.0.0.1:8080/x?y=1").unwrap(), (false, "127.0.0.1".into(), 8080, "/x?y=1".into()));
        assert!(split("ftp://x").is_err());
        assert!(split("https://user@host/").is_err());
        let refused = WinHttp.post("http://example.com/x", &[], b"", Duration::from_secs(1)).unwrap_err();
        assert!(refused.contains("HTTPS only"), "{refused}");
        let (status, headers) = parse_head("HTTP/1.1 429 Too Many\r\nRetry-After: 30\r\nX-RateLimit-Remaining-Requests: 0\r\nSet-Cookie: no\r\n\r\n");
        assert_eq!(status, 429);
        assert_eq!(headers, vec![("retry-after".to_string(), "30".to_string()), ("x-ratelimit-remaining-requests".to_string(), "0".to_string())]);
    }

    #[test]
    fn a_request_reaches_a_stub_on_this_machine_with_its_headers_and_body_and_the_reply_comes_back() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            let (mut s, _) = listener.accept().unwrap();
            let mut got = Vec::new();
            let mut buf = [0u8; 4096];
            loop {
                let n = s.read(&mut buf).unwrap();
                got.extend_from_slice(&buf[..n]);
                if let Some(i) = got.windows(4).position(|w| w == b"\r\n\r\n") {
                    let head = String::from_utf8_lossy(&got[..i]).to_lowercase();
                    let length: usize = head.lines().find_map(|l| l.strip_prefix("content-length:")).map_or(0, |v| v.trim().parse().unwrap());
                    if got.len() >= i + 4 + length {
                        break;
                    }
                }
            }
            let body = b"{\"text\":\"hi\"}";
            let _ = s.write_all(format!("HTTP/1.1 429 Too Many\r\nRetry-After: 7\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n", body.len()).as_bytes());
            let _ = s.write_all(body);
            String::from_utf8_lossy(&got).to_string()
        });
        let reply = WinHttp
            .post(&format!("http://{addr}/openai/v1/audio/transcriptions"), &[("Authorization".into(), "Bearer stub".into())], b"payload", Duration::from_secs(10))
            .unwrap();
        assert_eq!(reply.status, 429);
        assert_eq!(reply.header("retry-after"), Some("7"));
        assert_eq!(reply.body, b"{\"text\":\"hi\"}");
        let seen = server.join().unwrap();
        assert!(seen.starts_with("POST /openai/v1/audio/transcriptions HTTP/1.1\r\n"), "{seen}");
        assert!(seen.contains("Authorization: Bearer stub\r\n"));
        assert!(seen.ends_with("payload"));
    }
}
