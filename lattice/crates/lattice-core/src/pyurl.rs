//! Python's `urlsplit` and `ipaddress`, for the rules that decide what an endpoint's address means.
//!
//! An endpoint's `base_url` decides two things that must not differ between the
//! runtimes: whether the row is acceptable at all (`ModelEndpoint.__post_init__`
//! in `model_config.py`) and whether the model is on this machine
//! (`is_local_url`), which is the boundary that keeps a task from leaving the
//! machine unannounced. Both are defined by what Python's `urlsplit` and
//! `ipaddress.ip_address` make of the text, quirks included (a `[` only in the
//! user-info part is an error; `::ffff:127.0.0.1` is loopback; a zone id is
//! accepted). This module ports that behaviour from CPython 3.12.10's
//! `urllib/parse.py` and `ipaddress.py`; `tests/parity/registry/is_local_url.json`
//! and the registry goldens check the port against the real functions.
//!
//! Deliberate difference: `urlsplit` rejects a host whose Unicode NFKC form
//! contains `/ ? # @ :`. Reproducing that needs the NFKC tables, which would be a
//! new dependency; instead a non-ASCII authority is refused outright. That is
//! the safe direction: an internationalised host is written in its `xn--` form,
//! and `is_local_url` was already `False` for any non-ASCII host.
//!
//! Invariant: no DNS lookup, no panic on any input.

/// An IP address as Python's `ipaddress.ip_address` reads it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Ip {
    V4(u32),
    V6(u128),
}

impl Ip {
    /// `ip_address(text).is_loopback`: `127.0.0.0/8`, `::1`, and an IPv4-mapped
    /// IPv6 address whose IPv4 part is loopback.
    pub(crate) fn is_loopback(self) -> bool {
        match self {
            Ip::V4(ip) => ip >> 24 == 127,
            Ip::V6(ip) if ip >> 32 == 0xffff => Ip::V4((ip & 0xffff_ffff) as u32).is_loopback(),
            Ip::V6(ip) => ip == 1,
        }
    }
}

/// `IPv4Address(text)`: exactly four decimal octets, no leading zeros.
fn parse_v4(text: &str) -> Option<u32> {
    if text.is_empty() || text.contains('/') {
        return None;
    }
    let octets: Vec<&str> = text.split('.').collect();
    if octets.len() != 4 {
        return None;
    }
    let mut value = 0u32;
    for octet in octets {
        if octet.is_empty()
            || !octet.bytes().all(|b| b.is_ascii_digit())
            || octet.len() > 3
            || (octet != "0" && octet.starts_with('0'))
        {
            return None;
        }
        let number: u32 = octet.parse().ok()?;
        if number > 255 {
            return None;
        }
        value = value << 8 | number;
    }
    Some(value)
}

fn parse_hextet(text: &str) -> Option<u128> {
    if text.is_empty() || text.len() > 4 || !text.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    u128::from_str_radix(text, 16).ok()
}

/// `IPv6Address(text)`, including an RFC 4007 zone id after `%`.
fn parse_v6(text: &str) -> Option<u128> {
    if text.contains('/') {
        return None;
    }
    let text = match text.split_once('%') {
        None => text,
        Some((address, zone)) => {
            if zone.is_empty() || zone.contains('%') {
                return None;
            }
            address
        }
    };
    if text.is_empty() {
        return None;
    }
    let mut parts: Vec<String> = text.split(':').map(str::to_owned).collect();
    if parts.len() < 3 {
        return None;
    }
    if parts.last().is_some_and(|last| last.contains('.')) {
        let v4 = parse_v4(&parts.pop()?)?;
        parts.push(format!("{:x}", (v4 >> 16) & 0xffff));
        parts.push(format!("{:x}", v4 & 0xffff));
    }
    const HEXTETS: usize = 8;
    if parts.len() > HEXTETS + 1 {
        return None;
    }
    let mut skip_index = None;
    for (i, part) in parts.iter().enumerate().skip(1).take(parts.len() - 2) {
        if part.is_empty() {
            if skip_index.is_some() {
                return None;
            }
            skip_index = Some(i);
        }
    }
    let (parts_hi, parts_lo, parts_skipped);
    if let Some(skip) = skip_index {
        let mut hi = skip;
        let mut lo = parts.len() - skip - 1;
        if parts[0].is_empty() {
            hi -= 1;
            if hi != 0 {
                return None;
            }
        }
        if parts[parts.len() - 1].is_empty() {
            lo -= 1;
            if lo != 0 {
                return None;
            }
        }
        let skipped = HEXTETS as isize - (hi + lo) as isize;
        if skipped < 1 {
            return None;
        }
        parts_hi = hi;
        parts_lo = lo;
        parts_skipped = skipped as usize;
    } else {
        if parts.len() != HEXTETS || parts[0].is_empty() || parts[parts.len() - 1].is_empty() {
            return None;
        }
        parts_hi = parts.len();
        parts_lo = 0;
        parts_skipped = 0;
    }
    let mut value = 0u128;
    for part in &parts[..parts_hi] {
        value = value << 16 | parse_hextet(part)?;
    }
    // Eight skipped hextets are the whole address: nothing left to shift.
    value = value.checked_shl(16 * parts_skipped as u32).unwrap_or(0);
    for part in &parts[parts.len() - parts_lo..] {
        value = value << 16 | parse_hextet(part)?;
    }
    Some(value)
}

/// `ipaddress.ip_address(text)`: IPv4 first, then IPv6.
pub(crate) fn parse_ip(text: &str) -> Option<Ip> {
    parse_v4(text)
        .map(Ip::V4)
        .or_else(|| parse_v6(text).map(Ip::V6))
}

/// The parts of `urlsplit(url)` that the endpoint rules read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Split {
    /// Lower-cased, empty when the URL has none.
    pub scheme: String,
    pub netloc: String,
}

const SCHEME_CHARS: &str = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-.";

/// `urlsplit(url)`, or `Err(())` where Python raises `ValueError`.
pub(crate) fn urlsplit(url: &str) -> Result<Split, ()> {
    let url = url.trim_start_matches(|c: char| c <= ' ');
    let cleaned: String = url
        .chars()
        .filter(|c| !matches!(c, '\t' | '\r' | '\n'))
        .collect();
    let mut rest = cleaned.as_str();
    let mut scheme = String::new();
    if let Some(i) = rest.find(':')
        && i > 0
        && rest.chars().next().is_some_and(|c| c.is_ascii_alphabetic())
        && rest[..i].chars().all(|c| SCHEME_CHARS.contains(c))
    {
        scheme = rest[..i].to_ascii_lowercase();
        rest = &rest[i + 1..];
    }
    let mut netloc = "";
    if let Some(after) = rest.strip_prefix("//") {
        let end = after.find(['/', '?', '#']).unwrap_or(after.len());
        netloc = &after[..end];
        let (open, close) = (netloc.contains('['), netloc.contains(']'));
        if open != close {
            return Err(());
        }
        if open && close {
            check_bracketed_netloc(netloc)?;
        }
    }
    if !netloc.is_ascii() {
        return Err(());
    }
    Ok(Split {
        scheme,
        netloc: netloc.to_owned(),
    })
}

/// `_check_bracketed_netloc`: the host inside brackets must be an IPv6 address
/// (or an IPvFuture literal), with nothing but a port after the bracket.
fn check_bracketed_netloc(netloc: &str) -> Result<(), ()> {
    let host_and_port = netloc.rsplit_once('@').map_or(netloc, |(_, after)| after);
    let hostname = match host_and_port.split_once('[') {
        Some((before, bracketed)) => {
            if !before.is_empty() {
                return Err(());
            }
            let (hostname, port) = bracketed.split_once(']').unwrap_or((bracketed, ""));
            if !port.is_empty() && !port.starts_with(':') {
                return Err(());
            }
            hostname
        }
        None => host_and_port
            .split_once(':')
            .map_or(host_and_port, |(host, _)| host),
    };
    check_bracketed_host(hostname)
}

/// `_check_bracketed_host`.
fn check_bracketed_host(hostname: &str) -> Result<(), ()> {
    if let Some(future) = hostname.strip_prefix('v') {
        // `\Av[a-fA-F0-9]+\..+\Z`
        let digits = future.bytes().take_while(u8::is_ascii_hexdigit).count();
        let after = &future[digits..];
        return if digits > 0 && after.starts_with('.') && after.len() > 1 {
            Ok(())
        } else {
            Err(())
        };
    }
    match parse_ip(hostname) {
        Some(Ip::V6(_)) => Ok(()),
        _ => Err(()),
    }
}

impl Split {
    /// Whether the authority has user-info (`username` or `password` is not `None`).
    pub(crate) fn has_userinfo(&self) -> bool {
        self.netloc.contains('@')
    }

    /// `(hostname, port)` as `_hostinfo` cuts them, before validation.
    fn hostinfo(&self) -> (&str, Option<&str>) {
        let hostinfo = self
            .netloc
            .rsplit_once('@')
            .map_or(self.netloc.as_str(), |(_, after)| after);
        let (hostname, port) = match hostinfo.split_once('[') {
            Some((_, bracketed)) => {
                let (hostname, after) = bracketed.split_once(']').unwrap_or((bracketed, ""));
                (hostname, after.split_once(':').map_or("", |(_, port)| port))
            }
            None => hostinfo.split_once(':').unwrap_or((hostinfo, "")),
        };
        (hostname, (!port.is_empty()).then_some(port))
    }

    /// `.hostname`: lower-cased up to a `%` zone, `None` when empty.
    pub(crate) fn hostname(&self) -> Option<String> {
        let (hostname, _) = self.hostinfo();
        if hostname.is_empty() {
            return None;
        }
        Some(match hostname.split_once('%') {
            Some((address, zone)) => format!("{}%{zone}", address.to_ascii_lowercase()),
            None => hostname.to_ascii_lowercase(),
        })
    }

    /// `.port` as validation: `Err(())` where Python raises `ValueError`.
    pub(crate) fn check_port(&self) -> Result<(), ()> {
        let (_, port) = self.hostinfo();
        let Some(port) = port else { return Ok(()) };
        if !port.bytes().all(|b| b.is_ascii_digit()) {
            return Err(());
        }
        match port.parse::<u32>() {
            Ok(number) if number <= 65535 => Ok(()),
            _ => Err(()),
        }
    }
}

/// `is_local_url`: does this URL address THIS machine? Loopback only: the host
/// is `localhost` or `localhost.localdomain` (any case), or an IP literal that
/// is loopback. Anything unparseable, and any other name, is not local; no DNS.
pub(crate) fn is_local_url(url: &str) -> bool {
    let Ok(split) = urlsplit(crate::py::strip(url)) else {
        return false;
    };
    let hostname = split.hostname().unwrap_or_default();
    let host = hostname.trim_matches(['[', ']']);
    if host.is_empty() {
        return false;
    }
    if matches!(
        host.to_lowercase().as_str(),
        "localhost" | "localhost.localdomain"
    ) {
        return true;
    }
    parse_ip(host).is_some_and(Ip::is_loopback)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ipv4_is_four_plain_octets() {
        assert_eq!(parse_v4("127.0.0.1"), Some(0x7f00_0001));
        assert_eq!(parse_v4("0.0.0.0"), Some(0));
        for bad in [
            "",
            "1.2.3",
            "1.2.3.4.5",
            "01.2.3.4",
            "1.2.3.256",
            "1.2.3.-1",
            "1.2.3.a",
            "1..2.3",
            "1.2.3.4/8",
            "1.2.3.\u{661}",
            "1.2.3.0000",
        ] {
            assert_eq!(parse_v4(bad), None, "{bad:?}");
        }
    }

    #[test]
    fn ipv6_follows_pythons_parser() {
        assert_eq!(parse_v6("::1"), Some(1));
        assert_eq!(parse_v6("::"), Some(0));
        assert_eq!(parse_v6("1::"), Some(1 << 112));
        assert_eq!(parse_v6("::ffff:127.0.0.1"), Some(0xffff_7f00_0001));
        assert_eq!(
            parse_v6("fe80::1%eth0"),
            Some(0xfe80 << 112 | 1),
            "a zone id is accepted"
        );
        assert_eq!(
            parse_v6("2001:db8:0:0:0:0:0:1"),
            Some(0x2001_0db8 << 96 | 1)
        );
        for bad in [
            "",
            ":",
            "1:2",
            "1:::2",
            "1::2::3",
            ":1::",
            "1::2:",
            ":1:2:3:4:5:6:7",
            "1:2:3:4:5:6:7:8:9",
            "1:2:3:4:5:6:7",
            "12345::",
            "g::1",
            "::1%",
            "::1%a%b",
            "::1/128",
            "::ffff:1.2.3",
            "::1.2.3.4.5",
            "1:2:3:4:5:6:7::8",
        ] {
            assert_eq!(parse_v6(bad), None, "{bad:?}");
        }
        assert_eq!(
            parse_v6("1:2:3:4:5:6:7::"),
            Some(0x0001_0002_0003_0004_0005_0006_0007_0000)
        );
    }

    #[test]
    fn loopback_includes_mapped_ipv4() {
        for (text, loopback) in [
            ("127.0.0.1", true),
            ("127.255.255.254", true),
            ("128.0.0.1", false),
            ("0.0.0.0", false),
            ("::1", true),
            ("::", false),
            ("::2", false),
            ("::ffff:127.0.0.1", true),
            ("::ffff:8.8.8.8", false),
            ("192.168.1.1", false),
            ("::1%lo", true),
        ] {
            assert_eq!(
                parse_ip(text).is_some_and(Ip::is_loopback),
                loopback,
                "{text}"
            );
        }
        assert_eq!(parse_ip("localhost"), None);
    }

    #[test]
    fn urlsplit_cuts_scheme_and_authority() {
        let split = urlsplit("HTTP://User:pw@Example.COM:8080/v1?x=1#f").unwrap();
        assert_eq!(split.scheme, "http");
        assert_eq!(split.netloc, "User:pw@Example.COM:8080");
        assert!(split.has_userinfo());
        assert_eq!(split.hostname().as_deref(), Some("example.com"));
        assert!(split.check_port().is_ok());

        let bare = urlsplit("localhost:8000/v1").unwrap();
        assert_eq!(
            bare.scheme, "localhost",
            "Python reads `localhost:` as a scheme"
        );
        assert_eq!(bare.netloc, "");
    }

    #[test]
    fn urlsplit_removes_control_characters_the_way_python_does() {
        // Leading C0 controls and spaces go; tabs and newlines inside go.
        let split = urlsplit("\u{1f}  http://exa\tmp\nle.com/x").unwrap();
        assert_eq!(split.scheme, "http");
        assert_eq!(split.netloc, "example.com");
    }

    #[test]
    fn brackets_must_hold_an_ipv6_address() {
        assert!(urlsplit("http://[::1]:8000/v1").is_ok());
        assert!(urlsplit("http://[fe80::1%25eth0]/v1").is_ok());
        assert!(urlsplit("http://[v1.fe80]/v1").is_ok());
        for bad in [
            "http://[::1/v1",
            "http://::1]/v1",
            "http://[127.0.0.1]/v1",
            "http://[localhost]/v1",
            "http://x[::1]/v1",
            "http://[::1]x/v1",
            "http://[v.x]/v1",
            "http://[vg.x]/v1",
            "http://[v1.]/v1",
            "http://user[1]@host/v1",
            "http://\u{ff0f}host/v1",
        ] {
            assert!(urlsplit(bad).is_err(), "{bad}");
        }
        // The deliberate difference: Python accepts this one (its NFKC form holds
        // no delimiter); here a non-ASCII authority is refused outright.
        assert!(urlsplit("http://m\u{fc}nchen.example/v1").is_err());
    }

    #[test]
    fn hostname_and_port_are_cut_like_python_cuts_them() {
        let host = |url: &str| urlsplit(url).unwrap().hostname();
        assert_eq!(host("http://[::1]:8000/x").as_deref(), Some("::1"));
        assert_eq!(
            host("http://[FE80::1%Eth0]/x").as_deref(),
            Some("fe80::1%Eth0"),
            "the zone keeps its case"
        );
        assert_eq!(host("http:///x"), None);
        assert_eq!(host("http://:80/x"), None);
        assert_eq!(
            host("http://a@b@c/x").as_deref(),
            Some("c"),
            "the last @ ends the user-info"
        );
        for (url, ok) in [
            ("http://h:8000", true),
            ("http://h:0", true),
            ("http://h:65535", true),
            ("http://h:65536", false),
            ("http://h:abc", false),
            ("http://h:-1", false),
            ("http://h:1_0", false),
            ("http://h:99999999999999999999999", false),
            ("http://h:", true),
            ("http://[::1]:x", false),
        ] {
            assert_eq!(urlsplit(url).unwrap().check_port().is_ok(), ok, "{url}");
        }
    }

    #[test]
    fn local_urls_are_loopback_only() {
        for (url, local) in [
            ("http://localhost:11434", true),
            ("HTTP://LOCALHOST/v1", true),
            ("http://localhost.localdomain/v1", true),
            ("http://127.0.0.1:8000/v1", true),
            ("http://127.9.9.9/v1", true),
            ("http://[::1]:8000/v1", true),
            ("http://[::ffff:127.0.0.1]/v1", true),
            ("  http://localhost/v1  ", true),
            ("localhost:8000", false),
            ("http://localhost./v1", false),
            ("http://192.168.1.10:8000/v1", false),
            ("http://10.0.0.1/v1", false),
            ("http://0.0.0.0/v1", false),
            ("https://api.openai.com/v1", false),
            ("http://localhost.evil.example/v1", false),
            ("http://127.0.0.1.evil.example/v1", false),
            ("http://user@localhost/v1", true),
            ("http://evil.example@127.0.0.1/v1", true),
            ("http://127.0.0.1@evil.example/v1", false),
            ("", false),
            ("http://", false),
            ("http://[::1", false),
            ("http://localhost:notaport/v1", true),
            ("http://\u{ff4c}ocalhost/v1", false),
        ] {
            assert_eq!(is_local_url(url), local, "{url:?}");
        }
    }
}
