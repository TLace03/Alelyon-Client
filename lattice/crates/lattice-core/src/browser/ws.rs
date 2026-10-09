//! A WebSocket client for one loopback connection: the DevTools endpoint of a
//! browser started with `--remote-debugging-port` (RFC 6455, the parts a
//! client needs). The browser Lattice starts listens on no port and speaks
//! over its pipes ([`super::pipe`]); this client is the other transport
//! [`super::cdp`] takes. Not a port.
//!
//! - **The handshake** sends the upgrade with a random key and checks the
//!   answer's `101` and its `Sec-WebSocket-Accept` (SHA-1 of the key and the
//!   RFC's GUID, [`sha1`] here: no crate is added for it).
//! - **Frames** the client sends are masked (RFC 6455 §5.3); the server's are
//!   read unmasked, text and binary, fragmented or not, at most
//!   [`MAX_MESSAGE`] bytes a message; a ping is answered with a pong, and a
//!   close ends the connection.
//! - **Threads**, as the MCP client's: [`connect`] gives a [`Socket`] whose
//!   reader thread hands each whole text message to a callback and whose
//!   writer thread sends what [`Socket::send`] queues, so neither side waits on
//!   the other. Both block on the socket or their channel: an idle connection
//!   costs no wake-up.
//!
//! The connection is plain TCP to `127.0.0.1`: there is no TLS, and nothing
//! else is accepted.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::sync::mpsc;
use std::time::Duration;

/// The largest message read.
pub const MAX_MESSAGE: usize = 64 * 1024 * 1024;
/// RFC 6455's GUID for the accept key.
const GUID: &str = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";

/// SHA-1 (FIPS 180-1), for the handshake's accept key only.
pub fn sha1(data: &[u8]) -> [u8; 20] {
    let mut h: [u32; 5] = [
        0x6745_2301,
        0xEFCD_AB89,
        0x98BA_DCFE,
        0x1032_5476,
        0xC3D2_E1F0,
    ];
    let mut message = data.to_vec();
    let bits = (data.len() as u64).wrapping_mul(8);
    message.push(0x80);
    while message.len() % 64 != 56 {
        message.push(0);
    }
    message.extend_from_slice(&bits.to_be_bytes());
    for block in message.chunks(64) {
        let mut w = [0u32; 80];
        for (i, word) in block.chunks(4).enumerate() {
            w[i] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
        }
        for i in 16..80 {
            w[i] = (w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16]).rotate_left(1);
        }
        let [mut a, mut b, mut c, mut d, mut e] = h;
        for (i, word) in w.iter().enumerate() {
            let (f, k) = match i {
                0..=19 => ((b & c) | ((!b) & d), 0x5A82_7999),
                20..=39 => (b ^ c ^ d, 0x6ED9_EBA1),
                40..=59 => ((b & c) | (b & d) | (c & d), 0x8F1B_BCDC),
                _ => (b ^ c ^ d, 0xCA62_C1D6),
            };
            let next = a
                .rotate_left(5)
                .wrapping_add(f)
                .wrapping_add(e)
                .wrapping_add(k)
                .wrapping_add(*word);
            e = d;
            d = c;
            c = b.rotate_left(30);
            b = a;
            a = next;
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
    }
    let mut out = [0u8; 20];
    for (i, word) in h.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }
    out
}

const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Standard base64 with padding.
pub fn base64(data: &[u8]) -> String {
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let n = match chunk.len() {
            3 => (u32::from(chunk[0]) << 16) | (u32::from(chunk[1]) << 8) | u32::from(chunk[2]),
            2 => (u32::from(chunk[0]) << 16) | (u32::from(chunk[1]) << 8),
            _ => u32::from(chunk[0]) << 16,
        };
        for i in 0..4 {
            if i <= chunk.len() {
                out.push(B64[((n >> (18 - 6 * i)) & 63) as usize] as char);
            } else {
                out.push('=');
            }
        }
    }
    out
}

/// The `Sec-WebSocket-Accept` a server must answer for `key`.
pub fn accept_for(key: &str) -> String {
    base64(&sha1(format!("{key}{GUID}").as_bytes()))
}

/// 16 random bytes, from the process's own random source (a UUID v4's).
fn random_bytes() -> [u8; 16] {
    *uuid::Uuid::new_v4().as_bytes()
}

/// One frame to send: FIN set, masked.
pub fn frame(opcode: u8, payload: &[u8], mask: [u8; 4]) -> Vec<u8> {
    let mut out = Vec::with_capacity(payload.len() + 14);
    out.push(0x80 | (opcode & 0x0f));
    let len = payload.len();
    if len < 126 {
        out.push(0x80 | len as u8);
    } else if len <= 0xffff {
        out.push(0x80 | 126);
        out.extend_from_slice(&(len as u16).to_be_bytes());
    } else {
        out.push(0x80 | 127);
        out.extend_from_slice(&(len as u64).to_be_bytes());
    }
    out.extend_from_slice(&mask);
    out.extend(payload.iter().enumerate().map(|(i, b)| b ^ mask[i % 4]));
    out
}

/// What the reader thread read.
#[derive(Debug, PartialEq, Eq)]
pub enum Incoming {
    Text(String),
    Ping(Vec<u8>),
    Close,
}

/// Read one whole message (continuations joined), or why the connection
/// ended.
pub fn read_message(reader: &mut impl Read) -> Result<Incoming, String> {
    let mut message: Vec<u8> = Vec::new();
    let mut opcode_of_message: Option<u8> = None;
    loop {
        let mut head = [0u8; 2];
        reader
            .read_exact(&mut head)
            .map_err(|_| "The browser closed its connection.".to_owned())?;
        let fin = head[0] & 0x80 != 0;
        let opcode = head[0] & 0x0f;
        let masked = head[1] & 0x80 != 0;
        let mut len = u64::from(head[1] & 0x7f);
        if len == 126 {
            let mut bytes = [0u8; 2];
            reader.read_exact(&mut bytes).map_err(|e| e.to_string())?;
            len = u64::from(u16::from_be_bytes(bytes));
        } else if len == 127 {
            let mut bytes = [0u8; 8];
            reader.read_exact(&mut bytes).map_err(|e| e.to_string())?;
            len = u64::from_be_bytes(bytes);
        }
        if len as usize > MAX_MESSAGE || message.len() + len as usize > MAX_MESSAGE {
            return Err("The browser sent a message larger than 64 MiB.".to_owned());
        }
        let mut mask = [0u8; 4];
        if masked {
            reader.read_exact(&mut mask).map_err(|e| e.to_string())?;
        }
        let mut payload = vec![0u8; len as usize];
        reader.read_exact(&mut payload).map_err(|e| e.to_string())?;
        if masked {
            for (i, byte) in payload.iter_mut().enumerate() {
                *byte ^= mask[i % 4];
            }
        }
        match opcode {
            0x8 => return Ok(Incoming::Close),
            0x9 => return Ok(Incoming::Ping(payload)),
            0xA => continue,
            0x1 | 0x2 => {
                opcode_of_message = Some(opcode);
                message = payload;
            }
            0x0 if opcode_of_message.is_some() => message.extend_from_slice(&payload),
            _ => return Err("The browser sent a frame this client does not read.".to_owned()),
        }
        if fin {
            return String::from_utf8(message)
                .map(Incoming::Text)
                .map_err(|_| "The browser sent a message that is not UTF-8.".to_owned());
        }
    }
}

enum Out {
    Text(String),
    Pong(Vec<u8>),
    Close,
}

/// One open connection.
pub struct Socket {
    out: mpsc::Sender<Out>,
}

impl Socket {
    /// Queue a text message.
    pub fn send(&self, text: String) -> bool {
        self.out.send(Out::Text(text)).is_ok()
    }

    /// Close the connection (the writer sends a close frame and stops).
    pub fn close(&self) {
        let _ = self.out.send(Out::Close);
    }
}

impl Drop for Socket {
    fn drop(&mut self) {
        self.close();
    }
}

/// Connect to `ws://127.0.0.1:<port><path>`. `on_text` gets each whole text
/// message on the reader thread; `on_end` the reason, once, when it ends.
pub fn connect(
    port: u16,
    path: &str,
    on_text: Box<dyn Fn(String) + Send>,
    on_end: Box<dyn FnOnce(String) + Send>,
) -> Result<Socket, String> {
    if !path.starts_with('/') || path.contains(['\r', '\n', ' ']) {
        return Err("The browser's DevTools path is not one this client opens.".to_owned());
    }
    let address = SocketAddr::from(([127, 0, 0, 1], port));
    let stream = TcpStream::connect_timeout(&address, Duration::from_secs(5))
        .map_err(|_| "The browser's DevTools port did not answer.".to_owned())?;
    stream.set_nodelay(true).ok();
    let key = base64(&random_bytes());
    let request = format!(
        "GET {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n"
    );
    let mut writer = stream
        .try_clone()
        .map_err(|_| "The DevTools connection could not be opened.".to_owned())?;
    writer
        .write_all(request.as_bytes())
        .map_err(|_| "The DevTools handshake could not be sent.".to_owned())?;
    // The handshake's answer, read with a deadline (the only timer here).
    stream.set_read_timeout(Some(Duration::from_secs(5))).ok();
    let mut reader = BufReader::new(
        stream
            .try_clone()
            .map_err(|_| "The DevTools connection could not be opened.".to_owned())?,
    );
    let mut status = String::new();
    reader
        .read_line(&mut status)
        .map_err(|_| "The browser did not answer the DevTools handshake.".to_owned())?;
    if !status.starts_with("HTTP/1.1 101") {
        return Err("The browser refused the DevTools connection.".to_owned());
    }
    let mut accepted = false;
    loop {
        let mut line = String::new();
        reader
            .read_line(&mut line)
            .map_err(|_| "The DevTools handshake ended early.".to_owned())?;
        let line = line.trim_end();
        if line.is_empty() {
            break;
        }
        if let Some((name, value)) = line.split_once(':')
            && name.trim().eq_ignore_ascii_case("sec-websocket-accept")
        {
            accepted = value.trim() == accept_for(&key);
        }
    }
    if !accepted {
        return Err("The DevTools endpoint did not answer as a WebSocket server.".to_owned());
    }
    stream.set_read_timeout(None).ok();
    let (out, lines) = mpsc::channel::<Out>();
    let pongs = out.clone();
    std::thread::Builder::new()
        .name("lattice-browser-ws-out".to_owned())
        .spawn(move || {
            for message in lines {
                let mask: [u8; 4] = {
                    let bytes = random_bytes();
                    [bytes[0], bytes[1], bytes[2], bytes[3]]
                };
                let bytes = match &message {
                    Out::Text(text) => frame(0x1, text.as_bytes(), mask),
                    Out::Pong(payload) => frame(0xA, payload, mask),
                    Out::Close => frame(0x8, &[], mask),
                };
                if writer.write_all(&bytes).is_err() || matches!(message, Out::Close) {
                    let _ = writer.shutdown(std::net::Shutdown::Both);
                    break;
                }
            }
        })
        .map_err(|_| "The DevTools connection's writer could not start.".to_owned())?;
    std::thread::Builder::new()
        .name("lattice-browser-ws-in".to_owned())
        .spawn(move || {
            let why = loop {
                match read_message(&mut reader) {
                    Ok(Incoming::Text(text)) => on_text(text),
                    Ok(Incoming::Ping(payload)) => {
                        let _ = pongs.send(Out::Pong(payload));
                    }
                    Ok(Incoming::Close) => break "The browser closed its connection.".to_owned(),
                    Err(why) => break why,
                }
            };
            let _ = pongs.send(Out::Close);
            on_end(why);
        })
        .map_err(|_| "The DevTools connection's reader could not start.".to_owned())?;
    Ok(Socket { out })
}
