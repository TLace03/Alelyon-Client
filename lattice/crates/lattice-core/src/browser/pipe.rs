//! The browser's DevTools over two pipes: Chromium's `--remote-debugging-pipe`,
//! which reads requests on its C runtime descriptor 3 and writes answers and
//! events on descriptor 4 (`lattice_sys::process::spawn_with_fd_pipes` passes
//! them). No socket is opened: only a process holding one of the pipes'
//! handles can speak to the browser. Not a port.
//!
//! - **Frames**, each way: one JSON message, then a NUL byte (Chromium's
//!   `content/browser/devtools/devtools_pipe_handler.cc`, its default JSON
//!   mode). A message read is at most [`MAX_MESSAGE`] bytes, the WebSocket
//!   client's cap, and must be UTF-8; a message sent may not hold a NUL
//!   (serde_json writes one inside a string as `\u0000`, so none does).
//! - **Threads**, as the WebSocket client's: [`connect`] gives a [`Pipe`] whose
//!   reader thread hands each whole message to a callback and whose writer
//!   thread sends what [`Pipe::send`] queues. Both block on their pipe or their
//!   channel: an idle connection costs no wake-up.
//! - **The end**: closing (or dropping) the [`Pipe`] closes the browser's
//!   descriptor 3, which ends its DevTools, and Chromium then closes the
//!   browser. The reader ends when descriptor 4 is closed: when the browser
//!   has ended, or closed it.

use std::io::{BufRead, BufReader, Read, Write};
use std::sync::mpsc;

/// The largest message read: the WebSocket client's.
pub const MAX_MESSAGE: usize = super::ws::MAX_MESSAGE;
/// The reader's buffer: Chromium writes in pieces of 64 KiB.
const READ_BUFFER: usize = 64 * 1024;
/// Why the reader ended when the browser closed its end between messages.
pub const CLOSED: &str = "The browser closed its connection.";

/// `text` as one frame: its bytes and a NUL.
pub fn frame(text: &str) -> Result<Vec<u8>, String> {
    if text.as_bytes().contains(&0) {
        return Err("A DevTools message cannot hold a NUL character.".to_owned());
    }
    let mut bytes = Vec::with_capacity(text.len() + 1);
    bytes.extend_from_slice(text.as_bytes());
    bytes.push(0);
    Ok(bytes)
}

/// Read one message: `Ok(Some(text))`, `Ok(None)` when the stream ends
/// between messages, or why it cannot be read (it ended inside a message, the
/// message passed [`MAX_MESSAGE`], or it is not UTF-8).
pub fn read_message(reader: &mut impl BufRead) -> Result<Option<String>, String> {
    let mut message: Vec<u8> = Vec::new();
    loop {
        let available = match reader.fill_buf() {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(_) => return Err("The browser's connection failed.".to_owned()),
        };
        if available.is_empty() {
            return if message.is_empty() {
                Ok(None)
            } else {
                Err("The browser's connection ended inside a message.".to_owned())
            };
        }
        let end = available.iter().position(|byte| *byte == 0);
        let taken = end.unwrap_or(available.len());
        if message.len() + taken > MAX_MESSAGE {
            return Err("The browser sent a message larger than 64 MiB.".to_owned());
        }
        message.extend_from_slice(&available[..taken]);
        reader.consume(taken + usize::from(end.is_some()));
        if end.is_some() {
            return String::from_utf8(message)
                .map(Some)
                .map_err(|_| "The browser sent a message that is not UTF-8.".to_owned());
        }
    }
}

enum Out {
    Text(String),
    Close,
}

/// One open connection.
pub struct Pipe {
    out: mpsc::Sender<Out>,
}

impl Pipe {
    /// Queue a message. One that cannot be framed (it holds a NUL) is
    /// refused, and the connection stays open.
    pub fn send(&self, text: String) -> bool {
        !text.contains('\0') && self.out.send(Out::Text(text)).is_ok()
    }

    /// Close the connection: the writer stops and closes the browser's
    /// descriptor 3.
    pub fn close(&self) {
        let _ = self.out.send(Out::Close);
    }
}

impl Drop for Pipe {
    fn drop(&mut self) {
        self.close();
    }
}

/// Speak DevTools over `to_browser` (the browser's descriptor 3) and
/// `from_browser` (its descriptor 4). `on_text` gets each whole message on the
/// reader thread; `on_end` the reason, once, when it ends.
pub fn connect<W, R>(
    to_browser: W,
    from_browser: R,
    on_text: Box<dyn Fn(String) + Send>,
    on_end: Box<dyn FnOnce(String) + Send>,
) -> Result<Pipe, String>
where
    W: Write + Send + 'static,
    R: Read + Send + 'static,
{
    let (out, lines) = mpsc::channel::<Out>();
    let closer = out.clone();
    std::thread::Builder::new()
        .name("lattice-browser-pipe-out".to_owned())
        .spawn(move || {
            let mut writer = to_browser;
            for message in lines {
                let Out::Text(text) = message else { break };
                // `send` queues only what frames.
                let Ok(bytes) = frame(&text) else { continue };
                if writer
                    .write_all(&bytes)
                    .and_then(|()| writer.flush())
                    .is_err()
                {
                    break;
                }
            }
            // Dropping the writer here closes the browser's descriptor 3.
        })
        .map_err(|_| "The DevTools connection's writer could not start.".to_owned())?;
    std::thread::Builder::new()
        .name("lattice-browser-pipe-in".to_owned())
        .spawn(move || {
            let mut reader = BufReader::with_capacity(READ_BUFFER, from_browser);
            let why = loop {
                match read_message(&mut reader) {
                    Ok(Some(text)) => on_text(text),
                    Ok(None) => break CLOSED.to_owned(),
                    Err(why) => break why,
                }
            };
            let _ = closer.send(Out::Close);
            on_end(why);
        })
        .map_err(|_| "The DevTools connection's reader could not start.".to_owned())?;
    Ok(Pipe { out })
}
