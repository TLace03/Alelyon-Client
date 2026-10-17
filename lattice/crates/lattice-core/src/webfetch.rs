//! `web_fetch`: a web page as text, without the browser. Not a port.
//!
//! - **What it may fetch:** the same addresses the agent's browser opens
//!   ([`crate::browser::policy::check_url`]): `http` and `https` on the public
//!   web. Before each request the name is resolved and refused when any of its
//!   addresses is this PC or a private network, and a redirect is followed only
//!   after the same checks, at most [`MAX_REDIRECTS`] times. No cookie, key or
//!   sign-in is sent.
//! - **What it reads:** at most [`MAX_BODY`] bytes. HTML becomes its readable
//!   text ([`html_text`]: no scripts or styles, headings, paragraphs and list
//!   items on lines of their own); plain text, Markdown, JSON, XML and CSV as
//!   they are; anything else (a PDF, an image, a download) is not read and the
//!   answer says to open it in the browser. The text is redacted before the
//!   model sees it.

use std::net::IpAddr;
use std::time::Duration;

use crate::browser::policy;
use crate::browser::session::PageText;
use crate::secrets;

/// The most bytes read of one answer.
pub const MAX_BODY: usize = 5 * 1024 * 1024;
/// The most redirects followed.
pub const MAX_REDIRECTS: usize = 5;
const CONNECT_WAIT: Duration = Duration::from_secs(15);
const READ_WAIT: Duration = Duration::from_secs(30);

/// Whether `host` (a name or an address, `port` its port) names only the
/// public web once resolved.
async fn public(host: &str, port: u16) -> Result<(), String> {
    let host = host.trim_start_matches('[').trim_end_matches(']');
    let addresses: Vec<IpAddr> = match host.parse::<IpAddr>() {
        Ok(ip) => vec![ip],
        Err(_) => tokio::net::lookup_host((host, port))
            .await
            .map_err(|_| format!("{host} could not be found."))?
            .map(|a| a.ip())
            .collect(),
    };
    if addresses.is_empty() {
        return Err(format!("{host} could not be found."));
    }
    let private = |ip: &IpAddr| match ip {
        IpAddr::V4(v4) => policy::private_v4(*v4),
        IpAddr::V6(v6) => policy::private_v6(*v6),
    };
    if addresses.iter().any(private) {
        return Err(format!("{host} leads to this PC or a private network, which web_fetch never reaches."));
    }
    Ok(())
}

/// Fetch `address` and read it as text.
pub async fn fetch(address: &str) -> Result<PageText, String> {
    fetch_from(address, false).await
}

/// [`fetch`], or for a test, from this PC's own loopback server too
/// (`local`): every other address still passes the public web's checks.
pub(crate) async fn fetch_from(address: &str, local: bool) -> Result<PageText, String> {
    let check = |text: &str| -> Result<String, String> {
        if local && policy::local_port(text).is_some() {
            url::Url::parse(text).map(|u| u.to_string()).map_err(|_| "That is not a web address.".to_owned())
        } else {
            policy::check_url(text)
        }
    };
    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .connect_timeout(CONNECT_WAIT)
        .read_timeout(READ_WAIT)
        .user_agent("Lattice (web_fetch)")
        .build()
        .map_err(|_| "Lattice could not make a web client.".to_owned())?;
    let mut url = check(address)?;
    for _ in 0..=MAX_REDIRECTS {
        let parsed = url::Url::parse(&url).map_err(|_| "That is not a web address.".to_owned())?;
        let host = parsed.host_str().unwrap_or("").to_owned();
        if !(local && policy::local_port(parsed.as_str()).is_some()) {
            public(&host, parsed.port_or_known_default().unwrap_or(443)).await?;
        }
        let response = client
            .get(parsed.clone())
            .header("Accept", "text/html,text/plain,text/markdown,application/json,application/xml;q=0.9,*/*;q=0.1")
            .send()
            .await
            .map_err(|why| format!("The page could not be fetched: {}", if why.is_timeout() { "it took too long" } else { "the connection failed" }))?;
        let status = response.status();
        if status.is_redirection() {
            let next = response
                .headers()
                .get(reqwest::header::LOCATION)
                .and_then(|v| v.to_str().ok())
                .and_then(|to| parsed.join(to).ok())
                .ok_or_else(|| "The page redirected without saying where.".to_owned())?;
            url = check(next.as_str())?;
            continue;
        }
        if !status.is_success() {
            return Err(format!("The page answered {} ({}).", status.as_u16(), status.canonical_reason().unwrap_or("an error")));
        }
        let kind = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_ascii_lowercase();
        let readable = kind.is_empty()
            || kind.starts_with("text/")
            || kind.contains("json")
            || kind.contains("xml")
            || kind.contains("javascript");
        if !readable {
            let what = kind.split(';').next().unwrap_or("").trim().to_owned();
            return Err(format!(
                "That address serves {what}, not text. Open it with browser_open to see it (a PDF shows in the browser's viewer)."
            ));
        }
        let mut body: Vec<u8> = Vec::new();
        let mut response = response;
        while let Some(chunk) = response.chunk().await.map_err(|_| "The page stopped part way.".to_owned())? {
            body.extend_from_slice(&chunk);
            if body.len() > MAX_BODY {
                body.truncate(MAX_BODY);
                break;
            }
        }
        let raw = String::from_utf8_lossy(&body).into_owned();
        let (title, text) = if kind.contains("html") || (kind.is_empty() && raw.trim_start().starts_with('<')) {
            html_text(&raw)
        } else {
            (String::new(), raw)
        };
        return Ok(PageText { url: parsed.to_string(), title: secrets::redact(&title), text: secrets::redact(&text) });
    }
    Err(format!("The page redirected more than {MAX_REDIRECTS} times."))
}

/// The readable text of an HTML page, and its title: scripts, styles and
/// other unseen parts left out; headings, paragraphs, rows and list items on
/// lines of their own; the common character references decoded.
pub fn html_text(html: &str) -> (String, String) {
    const SKIP: [&str; 5] = ["script", "style", "noscript", "svg", "template"];
    const BLOCK: [&str; 22] = [
        "p", "div", "br", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr", "section", "article", "header", "footer",
        "pre", "blockquote", "table", "ul", "ol", "main", "nav",
    ];
    let mut text = String::new();
    let mut title = String::new();
    let mut skipping: Option<String> = None;
    let mut in_title = false;
    let mut rest = html;
    while let Some(open) = rest.find('<') {
        let words = &rest[..open];
        if skipping.is_none() {
            if in_title {
                title.push_str(words);
            } else {
                text.push_str(words);
            }
        }
        let after = &rest[open + 1..];
        // A comment ends at -->.
        if let Some(comment) = after.strip_prefix("!--") {
            rest = comment.find("-->").map_or("", |end| &comment[end + 3..]);
            continue;
        }
        let Some(close) = after.find('>') else {
            rest = "";
            break;
        };
        let tag = &after[..close];
        rest = &after[close + 1..];
        let closing = tag.starts_with('/');
        let name: String = tag
            .trim_start_matches('/')
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric())
            .collect::<String>()
            .to_ascii_lowercase();
        if name == "title" {
            in_title = !closing;
            continue;
        }
        if let Some(skipped) = &skipping {
            if closing && *skipped == name {
                skipping = None;
            }
            continue;
        }
        if !closing && SKIP.contains(&name.as_str()) && !tag.ends_with('/') {
            skipping = Some(name);
            continue;
        }
        // A block starts on a line of its own; the end of a list or a table
        // ends its lines, so a list is not spread out.
        if BLOCK.contains(&name.as_str()) && (!closing || matches!(name.as_str(), "ul" | "ol" | "table" | "pre" | "blockquote")) {
            text.push('\n');
            if name == "li" && !closing {
                text.push_str("- ");
            }
            if name.starts_with('h') && name.len() == 2 && !closing {
                text.push_str(&"#".repeat(name[1..].parse::<usize>().unwrap_or(1)));
                text.push(' ');
            }
        }
    }
    if skipping.is_none() {
        text.push_str(rest);
    }
    let decode = |s: &str| -> String {
        let mut out = String::with_capacity(s.len());
        let mut rest = s;
        while let Some(at) = rest.find('&') {
            out.push_str(&rest[..at]);
            let tail = &rest[at..];
            let end = tail.find(';').filter(|e| *e <= 10);
            let entity = end.map(|e| &tail[1..e]);
            let decoded = match entity {
                Some("amp") => Some('&'),
                Some("lt") => Some('<'),
                Some("gt") => Some('>'),
                Some("quot") => Some('"'),
                Some("apos") | Some("#39") => Some('\''),
                Some("nbsp") => Some(' '),
                Some(e) if e.starts_with("#x") || e.starts_with("#X") => u32::from_str_radix(&e[2..], 16).ok().and_then(char::from_u32),
                Some(e) if e.starts_with('#') => e[1..].parse::<u32>().ok().and_then(char::from_u32),
                _ => None,
            };
            match (decoded, end) {
                (Some(c), Some(e)) => {
                    out.push(c);
                    rest = &tail[e + 1..];
                }
                _ => {
                    out.push('&');
                    rest = &tail[1..];
                }
            }
        }
        out.push_str(rest);
        out
    };
    // Lines of their own, each with its spaces collapsed; no run of blank lines.
    let mut lines: Vec<String> = Vec::new();
    for line in decode(&text).lines() {
        let squeezed = line.split_whitespace().collect::<Vec<_>>().join(" ");
        if squeezed.is_empty() && lines.last().is_none_or(|l| l.is_empty()) {
            continue;
        }
        lines.push(squeezed);
    }
    while lines.last().is_some_and(|l| l.is_empty()) {
        lines.pop();
    }
    let title = decode(&title).split_whitespace().collect::<Vec<_>>().join(" ");
    (title, lines.join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mutants: a script's or a style's words kept; a heading or a list item
    /// run into the line before; an entity left encoded.
    #[test]
    fn html_becomes_its_readable_text() {
        let (title, text) = html_text(
            "<html><head><title>The &amp; page</title><style>p{color:red}</style><script>var x = '<p>no</p>';</script></head>\
             <body><h1>Heading</h1><p>One&nbsp;line &lt;here&gt; &#169; &#x263A;</p><!-- hidden -->\
             <ul><li>first</li><li>second</li></ul><noscript>no script</noscript><p>  spaced   out  </p></body></html>",
        );
        assert_eq!(title, "The & page");
        assert_eq!(text, "# Heading\nOne line <here> \u{a9} \u{263a}\n\n- first\n- second\n\nspaced out");
        assert!(!text.contains("color") && !text.contains("var x") && !text.contains("hidden") && !text.contains("no script"));
    }

    /// Over the test's own server: a page, its text and title; JSON as it is;
    /// a redirect followed; an image refused; and this PC refused unless a
    /// test allows it.
    /// Mutants: the PC reachable; a redirect not followed; an image read.
    #[test]
    fn a_page_is_fetched_as_text_and_this_pc_is_refused() {
        let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        let port = crate::browser::tests::serve();
        let base = format!("http://127.0.0.1:{port}");
        let refused = runtime.block_on(fetch(&format!("{base}/two"))).unwrap_err();
        assert!(refused.contains("this PC"), "{refused}");
        let page = runtime.block_on(fetch_from(&format!("{base}/two"), true)).unwrap();
        assert_eq!((page.title.as_str(), page.text.as_str()), ("Page two", "two"));
        let json = runtime.block_on(fetch_from(&format!("{base}/api"), true)).unwrap();
        assert!(json.text.contains("\"items\": [1, 2, 3]"), "{}", json.text);
        let moved = runtime.block_on(fetch_from(&format!("{base}/moved"), true)).unwrap();
        assert!(moved.url.ends_with("/two") && moved.text == "two", "{moved:?}");
        let image = runtime.block_on(fetch_from(&format!("{base}/image"), true)).unwrap_err();
        assert!(image.contains("image/png") && image.contains("browser_open"), "{image}");
        // A redirect to a name of the local network is refused by the address rules themselves, before any lookup.
        let away = runtime.block_on(fetch_from(&format!("{base}/away"), true)).unwrap_err();
        assert!(away.contains("That name is on this PC or its local network"), "{away}");
    }

    /// A name is resolved and refused when it leads to this PC or a private
    /// network; a public address passes.
    /// Mutant: the resolved addresses not checked.
    #[test]
    fn an_address_on_this_pc_or_a_private_network_is_never_fetched() {
        let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        for private in ["127.0.0.1", "10.1.2.3", "192.168.0.10", "[::1]", "169.254.1.1", "100.64.0.1"] {
            assert!(runtime.block_on(public(private, 80)).unwrap_err().contains("private network"), "{private}");
        }
        assert!(runtime.block_on(public("8.8.8.8", 443)).is_ok());
    }
}
