//! Model files fetched from Hugging Face's public file store, only once the person agreed to them.
//!
//! A [`Plan`] names each file to fetch: its address on the Hub (`https://huggingface.co/<repo>/resolve/<revision>/
//! <path>`), the name it is kept under in the destination folder, its size and its SHA-256, both as the Hub's own file
//! listing gives them ([`tree_url`], [`parse_tree`]). A file the listing gives no SHA-256 for is not planned: nothing
//! is fetched that cannot be checked. Before anything is fetched the plan's total is held against the free disk
//! ([`Plan::check_disk`]); a free disk that is not known refuses.
//!
//! [`download`] then fetches the files one by one into a partial file beside where each is kept, reports progress,
//! and stops between pieces when its [`Control`] is paused (the partial file stays, and the next download resumes it
//! with an HTTP `Range`) or cancelled (the partial files are removed). A finished file's SHA-256 is checked before it is
//! moved into place, never over a file that is there; a file that does not match is removed and the plan fails. A file
//! already in place with the planned checksum is kept as it is; one with other contents is never replaced.
//!
//! A partial file is one of Lattice's own temporaries (`.<name>.lattice-<pid>-<n>.tmp`, [`crate::fsx::temporary_for`]),
//! so the never-delete rule's one removal ([`crate::fsx::remove_own_temporary`]) is the only way one is taken away, and a
//! later download finds and resumes one a previous run left (by its `.<name>.lattice-` prefix).
//!
//! **Addresses.** Requests go to the Hub's origin only, and follow a redirect only to an `https://` address on
//! `huggingface.co` or `hf.co` (where the Hub keeps large files), at most five times. No key or cookie is sent: these
//! are public files. A debug build or a test may name a stub on this machine instead ([`Source::stub`]).

use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU8, Ordering};
use std::time::Duration;

use serde_json::Value;

/// The Hub's origin.
pub const HUB: &str = "https://huggingface.co";
/// Disk kept free beyond a plan's total: 1 GiB.
pub const DISK_MARGIN: u64 = 1 << 30;
const CONNECT_WAIT: Duration = Duration::from_secs(15);
const READ_WAIT: Duration = Duration::from_secs(60);

/// Where files are fetched from: the Hub, or a stub on this machine (tests, debug builds).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Source {
    origin: String,
    stub: bool,
}

impl Source {
    pub fn hub() -> Source {
        Source { origin: HUB.to_owned(), stub: false }
    }

    /// A stub on this machine (an `http://` loopback origin with no path), in a debug build only.
    #[cfg(any(test, debug_assertions))]
    pub fn stub(origin: &str) -> Option<Source> {
        crate::hosted::parts::loopback_origin(origin).map(|origin| Source { origin, stub: true })
    }

    pub fn origin(&self) -> &str {
        &self.origin
    }

    pub fn is_stub(&self) -> bool {
        self.stub
    }
}

fn segment_ok(s: &str) -> bool {
    !s.is_empty()
        && s != "."
        && s != ".."
        && s.len() <= 96
        && s.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
}

/// A repository id (`owner/name`), checked: two plain segments.
pub fn repo_ok(repo: &str) -> bool {
    let mut parts = repo.split('/');
    matches!((parts.next(), parts.next(), parts.next()), (Some(a), Some(b), None) if segment_ok(a) && segment_ok(b))
}

/// A path inside a repository, checked: plain segments separated by `/`, never `..`.
pub fn path_ok(path: &str) -> bool {
    !path.is_empty() && path.len() <= 512 && path.split('/').all(segment_ok)
}

/// A file name to keep a file under: one plain segment.
pub fn name_ok(name: &str) -> bool {
    segment_ok(name)
}

/// The address of a file of `repo` at `revision`.
pub fn file_url(source: &Source, repo: &str, revision: &str, path: &str) -> Result<String, String> {
    if !repo_ok(repo) || !segment_ok(revision) || !path_ok(path) {
        return Err("That file is not named as a Hugging Face file is.".to_owned());
    }
    Ok(format!("{}/{repo}/resolve/{revision}/{path}", source.origin))
}

/// The address of the Hub's listing of `repo`'s files at `revision` (every folder, with each file's size and, for a
/// large file, its SHA-256).
pub fn tree_url(source: &Source, repo: &str, revision: &str) -> Result<String, String> {
    if !repo_ok(repo) || !segment_ok(revision) {
        return Err("That repository is not named as a Hugging Face repository is.".to_owned());
    }
    Ok(format!("{}/api/models/{repo}/tree/{revision}?recursive=true", source.origin))
}

/// A file as the Hub lists it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Listed {
    pub path: String,
    pub size: u64,
    /// The SHA-256 the Hub keeps for a large file (`lfs.oid`); None for a small one, which this module does not fetch.
    pub sha256: Option<String>,
}

/// The files of a listing (`[{type, path, size, lfs: {oid, size}}]`); folders and unreadable rows are left out.
pub fn parse_tree(value: &Value) -> Vec<Listed> {
    value
        .as_array()
        .into_iter()
        .flatten()
        .filter(|row| row.get("type").and_then(Value::as_str) == Some("file"))
        .filter_map(|row| {
            let path = row.get("path")?.as_str()?.to_owned();
            let lfs = row.get("lfs");
            let size = lfs.and_then(|l| l.get("size")).or_else(|| row.get("size")).and_then(Value::as_u64)?;
            let sha256 = lfs
                .and_then(|l| l.get("oid").or_else(|| l.get("sha256")))
                .and_then(Value::as_str)
                .map(|s| s.trim().to_ascii_lowercase())
                .filter(|s| crate::sha::is_sha256_hex(s));
            Some(Listed { path, size, sha256 })
        })
        .collect()
}

/// One file to fetch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlannedFile {
    pub url: String,
    /// The name it is kept under in the plan's folder.
    pub name: String,
    pub size: u64,
    pub sha256: String,
}

/// The files to fetch into one folder.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Plan {
    pub dir: PathBuf,
    pub files: Vec<PlannedFile>,
}

/// Plan `wanted` (paths of `repo` at `revision`, each with the name it is kept under) from the Hub's `listing`: each
/// must be listed with its size and SHA-256.
pub fn plan(source: &Source, repo: &str, revision: &str, wanted: &[(String, String)], listing: &[Listed], dir: &Path) -> Result<Plan, String> {
    let mut files = Vec::new();
    for (path, name) in wanted {
        if !name_ok(name) {
            return Err(format!("{name} is not a plain file name, so nothing is planned."));
        }
        let found = listing
            .iter()
            .find(|l| &l.path == path)
            .ok_or_else(|| format!("{repo} does not list {path} at {revision}, so nothing is planned."))?;
        let sha256 = found
            .sha256
            .clone()
            .ok_or_else(|| format!("{repo} gives no checksum for {path}, so it is not fetched."))?;
        if found.size == 0 {
            return Err(format!("{repo} lists {path} as empty, so it is not fetched."));
        }
        files.push(PlannedFile { url: file_url(source, repo, revision, path)?, name: name.clone(), size: found.size, sha256 });
    }
    Ok(Plan { dir: dir.to_path_buf(), files })
}

impl Plan {
    /// Every file's size, added up.
    pub fn total(&self) -> u64 {
        self.files.iter().map(|f| f.size).sum()
    }

    /// What is left to fetch: the files not yet in place, less what their partial files already hold.
    pub fn remaining(&self) -> u64 {
        self.files
            .iter()
            .map(|f| {
                if fs::metadata(self.dir.join(&f.name)).is_ok_and(|m| m.len() == f.size) {
                    return 0;
                }
                let partial = existing_partial(&self.dir.join(&f.name)).and_then(|p| fs::metadata(p).ok()).map(|m| m.len()).unwrap_or(0);
                f.size.saturating_sub(partial)
            })
            .sum()
    }

    /// Whether the files fit on the disk, with [`DISK_MARGIN`] to spare; `free` None (not known) refuses.
    pub fn check_disk(&self, free: Option<u64>) -> Result<(), String> {
        let need = self.remaining();
        match free {
            None => Err("The free disk space is not known, so nothing was downloaded.".to_owned()),
            Some(free) if need.saturating_add(DISK_MARGIN) > free => Err(format!(
                "The download needs {} and 1 GiB to spare, and the disk has {} free, so nothing was downloaded.",
                words(need),
                words(free)
            )),
            Some(_) => Ok(()),
        }
    }
}

/// A size in words: GB or MB, as the Hub writes them (1 GB = 10^9 bytes).
pub fn words(bytes: u64) -> String {
    if bytes >= 1_000_000_000 {
        format!("{:.1} GB", bytes as f64 / 1e9)
    } else {
        format!("{:.0} MB", (bytes as f64 / 1e6).max(1.0))
    }
}

/// The partial files of `target` a download left beside it: Lattice's own temporaries named for it.
fn partials_of(target: &Path) -> Vec<PathBuf> {
    let (Some(dir), Some(name)) = (target.parent(), target.file_name().and_then(|n| n.to_str())) else { return Vec::new() };
    let prefix = format!(".{name}.lattice-");
    let Ok(entries) = fs::read_dir(dir) else { return Vec::new() };
    entries
        .flatten()
        .filter(|e| {
            let n = e.file_name().to_string_lossy().into_owned();
            n.starts_with(&prefix) && crate::fsx::is_own_temporary_name(&n) && e.file_type().is_ok_and(|t| t.is_file())
        })
        .map(|e| e.path())
        .collect()
}

/// The partial file to resume for `target`: the largest a download left, if any.
fn existing_partial(target: &Path) -> Option<PathBuf> {
    partials_of(target).into_iter().max_by_key(|p| fs::metadata(p).map(|m| m.len()).unwrap_or(0))
}

/// The partial file for `target`: one to resume, else a new temporary name beside it.
fn partial_of(target: &Path) -> io::Result<PathBuf> {
    match existing_partial(target) {
        Some(p) => Ok(p),
        None => crate::fsx::temporary_for(target),
    }
}

/// Whether a redirect may be followed to `url`: an `https://` address on the Hub's hosts, or the stub's own origin.
pub fn redirect_allowed(url: &url::Url, source: &Source) -> bool {
    if source.stub {
        return url.origin().ascii_serialization() == source.origin;
    }
    let host = url.host_str().unwrap_or("").to_ascii_lowercase();
    url.scheme() == "https"
        && url.username().is_empty()
        && url.password().is_none()
        && (host == "huggingface.co" || host.ends_with(".huggingface.co") || host == "hf.co" || host.ends_with(".hf.co"))
}

/// The client a plan is fetched with: no key, bounded waits, redirects only as [`redirect_allowed`] says, the
/// system's proxy (none for a stub on this machine).
pub fn client(source: &Source) -> Result<reqwest::Client, String> {
    let allowed = source.clone();
    let policy = reqwest::redirect::Policy::custom(move |attempt| {
        if attempt.previous().len() >= 5 {
            attempt.error("too many redirects")
        } else if redirect_allowed(attempt.url(), &allowed) {
            attempt.follow()
        } else {
            attempt.stop()
        }
    });
    let mut builder = reqwest::Client::builder()
        .redirect(policy)
        .connect_timeout(CONNECT_WAIT)
        .read_timeout(READ_WAIT)
        .user_agent(concat!("Alelyon/", env!("CARGO_PKG_VERSION")));
    if source.stub {
        builder = builder.no_proxy();
    }
    builder.build().map_err(|_| "Alelyon could not set up its connection to Hugging Face.".to_owned())
}

/// Read the Hub's listing of `repo` at `revision` (public: no key is sent).
pub async fn list(client: &reqwest::Client, source: &Source, repo: &str, revision: &str) -> Result<Vec<Listed>, String> {
    let url = tree_url(source, repo, revision)?;
    let response = client
        .get(url)
        .header("Accept", "application/json")
        .send()
        .await
        .map_err(|_| "Hugging Face could not be reached.".to_owned())?;
    if !response.status().is_success() {
        return Err(format!("Hugging Face did not list {repo} ({}).", response.status().as_u16()));
    }
    let bytes = response.bytes().await.map_err(|_| "Hugging Face's listing was cut off.".to_owned())?;
    if bytes.len() > crate::hosted::MAX_BODY {
        return Err("Hugging Face's listing was too large.".to_owned());
    }
    let value: Value = serde_json::from_slice(&bytes).map_err(|_| "Hugging Face's listing could not be read.".to_owned())?;
    Ok(parse_tree(&value))
}

/// Pause or cancel a download from another thread.
#[derive(Debug, Default)]
pub struct Control {
    state: AtomicU8,
}

const RUN: u8 = 0;
const PAUSE: u8 = 1;
const CANCEL: u8 = 2;

impl Control {
    pub fn new() -> Control {
        Control::default()
    }

    /// Stop between pieces, keeping what was fetched for the next download to resume.
    pub fn pause(&self) {
        let _ = self.state.compare_exchange(RUN, PAUSE, Ordering::SeqCst, Ordering::SeqCst);
    }

    /// Stop between pieces and remove the partial files.
    pub fn cancel(&self) {
        self.state.store(CANCEL, Ordering::SeqCst);
    }

    /// Clear a pause, for the next download.
    pub fn resume(&self) {
        let _ = self.state.compare_exchange(PAUSE, RUN, Ordering::SeqCst, Ordering::SeqCst);
    }

    fn now(&self) -> u8 {
        self.state.load(Ordering::SeqCst)
    }
}

/// How a download ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// Every file is in place and matched its checksum.
    Done,
    /// Paused: the partial files stay for the next download.
    Paused,
    /// Cancelled: the partial files were removed (files already in place stay).
    Cancelled,
}

/// Where a download is.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Progress {
    /// The file being fetched, by its kept name.
    pub name: String,
    /// The file's bytes so far and its size.
    pub done: u64,
    pub size: u64,
    /// The plan's bytes so far (files in place included) and its total.
    pub all_done: u64,
    pub all_total: u64,
    /// The file is being checked against its SHA-256.
    pub checking: bool,
}

/// Take away every partial file `plan`'s files have (Lattice's own temporaries only); the files in place stay.
pub fn discard_partials(plan: &Plan) {
    for f in &plan.files {
        for p in partials_of(&plan.dir.join(&f.name)) {
            let _ = crate::fsx::remove_own_temporary(&p);
        }
    }
}

/// Fetch `plan`'s files, as the module's note says. `progress` is told after every piece.
pub async fn download(client: &reqwest::Client, plan: &Plan, control: &Control, progress: &mut (dyn FnMut(Progress) + Send)) -> Result<Outcome, String> {
    fs::create_dir_all(&plan.dir).map_err(|_| format!("{} could not be made.", plan.dir.display()))?;
    let all_total = plan.total();
    let mut before = 0u64;
    for f in &plan.files {
        if !name_ok(&f.name) {
            return Err(format!("{} is not a plain file name, so nothing was fetched.", f.name));
        }
        let target = plan.dir.join(&f.name);
        let partial = partial_of(&target).map_err(|_| format!("{} could not be named.", f.name))?;
        // a file already in place: kept when it is the planned one, never replaced when it is another
        if target.exists() {
            let (sha, len) = crate::sha::sha256_file(&target).map_err(|_| format!("{} could not be read.", f.name))?;
            if sha == f.sha256 && len == f.size {
                before += f.size;
                progress(Progress { name: f.name.clone(), done: f.size, size: f.size, all_done: before, all_total, checking: false });
                continue;
            }
            return Err(format!("A different file named {} is already in {}; it was not replaced.", f.name, plan.dir.display()));
        }
        let mut have = fs::metadata(&partial).map(|m| m.len()).unwrap_or(0);
        if have > f.size {
            let _ = crate::fsx::remove_own_temporary(&partial);
            have = 0;
        }
        if have < f.size {
            match control.now() {
                CANCEL => {
                    discard_partials(plan);
                    return Ok(Outcome::Cancelled);
                }
                PAUSE => return Ok(Outcome::Paused),
                _ => {}
            }
            let mut request = client.get(&f.url);
            if have > 0 {
                request = request.header("Range", format!("bytes={have}-"));
            }
            let mut response = request.send().await.map_err(|_| "Hugging Face could not be reached.".to_owned())?;
            let status = response.status().as_u16();
            let resumed = status == 206
                && response
                    .headers()
                    .get("content-range")
                    .and_then(|v| v.to_str().ok())
                    .is_some_and(|v| v.trim().starts_with(&format!("bytes {have}-")));
            let mut out: File = match (status, resumed) {
                (206, true) => OpenOptions::new().append(true).open(&partial),
                (200, _) => {
                    have = 0;
                    File::create(&partial)
                }
                _ => return Err(format!("Hugging Face answered {status} for {}, so it was not fetched.", f.name)),
            }
            .map_err(|_| format!("{} could not be written.", partial.display()))?;
            loop {
                match control.now() {
                    CANCEL => {
                        drop(out);
                        discard_partials(plan);
                        return Ok(Outcome::Cancelled);
                    }
                    PAUSE => return Ok(Outcome::Paused),
                    _ => {}
                }
                let Some(chunk) = response.chunk().await.map_err(|_| format!("The download of {} was cut off; it resumes from where it stopped.", f.name))? else {
                    break;
                };
                if have + chunk.len() as u64 > f.size {
                    drop(out);
                    let _ = crate::fsx::remove_own_temporary(&partial);
                    return Err(format!("{} came larger than Hugging Face listed it, so it was removed.", f.name));
                }
                out.write_all(&chunk).map_err(|_| format!("{} could not be written.", partial.display()))?;
                have += chunk.len() as u64;
                progress(Progress { name: f.name.clone(), done: have, size: f.size, all_done: before + have, all_total, checking: false });
            }
            out.flush().map_err(|_| format!("{} could not be written.", partial.display()))?;
            drop(out);
            if have != f.size {
                return Err(format!("The download of {} ended early; it resumes from where it stopped.", f.name));
            }
        }
        progress(Progress { name: f.name.clone(), done: have, size: f.size, all_done: before + have, all_total, checking: true });
        let (sha, _) = crate::sha::sha256_file(&partial).map_err(|_| format!("{} could not be read back.", f.name))?;
        if sha != f.sha256 {
            let _ = crate::fsx::remove_own_temporary(&partial);
            return Err(format!("{} did not match its checksum, so it was removed.", f.name));
        }
        // never over a file that is there (one that appeared while this one downloaded stays)
        crate::fsx::move_new(&partial, &target).map_err(|_| format!("{} could not be put in place: another file of that name may be there.", f.name))?;
        before += f.size;
    }
    Ok(Outcome::Done)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::TempDir;
    use serde_json::json;
    use std::io::Read;
    use std::net::TcpListener;
    use std::sync::{Arc, Mutex};

    /// A file server on this machine: one blob at `/o/r/resolve/main/<path>`, served with `Range`, its body in two
    /// halves a moment apart; it records each request's `Range`.
    struct Server {
        origin: String,
        ranges: Arc<Mutex<Vec<Option<String>>>>,
    }

    fn serve(blob: Vec<u8>) -> Server {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let origin = format!("http://{}", listener.local_addr().unwrap());
        let ranges = Arc::new(Mutex::new(Vec::new()));
        let log = ranges.clone();
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { continue };
                let mut head = Vec::new();
                let mut byte = [0u8; 1];
                while !head.ends_with(b"\r\n\r\n") && stream.read(&mut byte).unwrap_or(0) == 1 {
                    head.push(byte[0]);
                }
                let head = String::from_utf8_lossy(&head).to_string();
                let range = head.lines().find_map(|l| l.strip_prefix("Range: ").or_else(|| l.strip_prefix("range: ")).map(str::to_owned));
                log.lock().unwrap().push(range.clone());
                let from = range.as_deref().and_then(|r| r.strip_prefix("bytes=")?.trim_end_matches('-').parse::<usize>().ok()).unwrap_or(0);
                let body = &blob[from.min(blob.len())..];
                let status = if from > 0 { format!("206 Partial\r\nContent-Range: bytes {from}-{}/{}", blob.len() - 1, blob.len()) } else { "200 OK".to_owned() };
                let _ = stream.write_all(format!("HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len()).as_bytes());
                let half = body.len() / 2;
                let _ = stream.write_all(&body[..half]);
                let _ = stream.flush();
                std::thread::sleep(Duration::from_millis(150));
                let _ = stream.write_all(&body[half..]);
            }
        });
        Server { origin, ranges }
    }

    fn run<T>(work: impl std::future::Future<Output = T>) -> T {
        tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap().block_on(work)
    }

    fn blob() -> Vec<u8> {
        (0..200_000u32).map(|i| (i % 251) as u8).collect()
    }

    fn plan_for(server: &Server, dir: &Path, sha: String, size: u64) -> (Source, Plan) {
        let source = Source::stub(&server.origin).unwrap();
        let listing = vec![Listed { path: "m/model-Q4_K_M.gguf".into(), size, sha256: Some(sha) }];
        let plan = plan(&source, "o/r", "main", &[("m/model-Q4_K_M.gguf".into(), "model-Q4_K_M.gguf".into())], &listing, dir).unwrap();
        (source, plan)
    }

    #[test]
    fn names_and_addresses_are_checked() {
        let hub = Source::hub();
        assert_eq!(file_url(&hub, "Qwen/Qwen3-8B-GGUF", "main", "Qwen3-8B-Q4_K_M.gguf").unwrap(), "https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf");
        for (repo, path) in [("../x/y", "a"), ("a", "b"), ("a/b/c", "d"), ("a/b", "../../etc"), ("a/b", "x\\y"), ("a/b", "x?y"), ("a/b", "")] {
            assert!(file_url(&hub, repo, "main", path).is_err(), "{repo} {path}");
        }
        assert!(tree_url(&hub, "a/b", "main").unwrap().ends_with("/api/models/a/b/tree/main?recursive=true"));
        assert!(!name_ok("a/b") && !name_ok("..") && name_ok("ggml-small.bin"));
        let ok = |u: &str| redirect_allowed(&url::Url::parse(u).unwrap(), &hub);
        assert!(ok("https://cdn-lfs.hf.co/x") && ok("https://cas-bridge.xethub.hf.co/x") && ok("https://huggingface.co/x"));
        assert!(!ok("http://cdn-lfs.hf.co/x") && !ok("https://evil.example/x") && !ok("https://hf.co.evil.example/x") && !ok("https://u:p@cdn-lfs.hf.co/x"));
        assert!(Source::stub("https://example.com").is_none(), "a stub is on this machine only");
    }

    #[test]
    fn a_listing_gives_sizes_and_checksums_and_a_file_without_one_is_not_planned() {
        let sha = "a".repeat(64);
        let listing = parse_tree(&json!([
            {"type": "directory", "path": "m"},
            {"type": "file", "path": "m/big.gguf", "size": 1234, "lfs": {"oid": sha, "size": 99_000}},
            {"type": "file", "path": "README.md", "size": 10},
            {"type": "file", "path": "bad.gguf", "lfs": {"oid": "zz"}, "size": 5},
        ]));
        assert_eq!(listing.len(), 3);
        assert_eq!(listing[0], Listed { path: "m/big.gguf".into(), size: 99_000, sha256: Some(sha.clone()) }, "the large file's own size");
        assert_eq!(listing[2].sha256, None, "an unreadable checksum is none");
        let dir = TempDir::new("fetch-plan");
        let hub = Source::hub();
        let p = plan(&hub, "o/r", "main", &[("m/big.gguf".into(), "big.gguf".into())], &listing, dir.path()).unwrap();
        assert_eq!(p.total(), 99_000);
        assert!(plan(&hub, "o/r", "main", &[("README.md".into(), "README.md".into())], &listing, dir.path()).unwrap_err().contains("no checksum"));
        assert!(plan(&hub, "o/r", "main", &[("gone.gguf".into(), "gone.gguf".into())], &listing, dir.path()).unwrap_err().contains("does not list"));
        assert!(plan(&hub, "o/r", "main", &[("m/big.gguf".into(), "../big.gguf".into())], &listing, dir.path()).is_err());
        // the disk is checked first: unknown or too little free refuses
        assert!(p.check_disk(None).unwrap_err().contains("not known"));
        assert!(p.check_disk(Some(DISK_MARGIN)).unwrap_err().contains("nothing was downloaded"));
        assert!(p.check_disk(Some(DISK_MARGIN + 99_000)).is_ok());
    }

    #[test]
    fn a_download_is_checked_against_its_checksum_and_put_in_place() {
        let blob = blob();
        let server = serve(blob.clone());
        let dir = TempDir::new("fetch-ok");
        let (source, plan) = plan_for(&server, dir.path(), crate::sha::sha256_hex(&blob), blob.len() as u64);
        let client = client(&source).unwrap();
        let mut seen = Vec::new();
        let outcome = run(download(&client, &plan, &Control::new(), &mut |p: Progress| seen.push(p))).unwrap();
        assert_eq!(outcome, Outcome::Done);
        assert_eq!(fs::read(dir.path().join("model-Q4_K_M.gguf")).unwrap(), blob);
        assert!(partials_of(&dir.path().join("model-Q4_K_M.gguf")).is_empty(), "no partial file is left");
        assert!(seen.iter().any(|p| p.checking), "the checksum is checked");
        assert_eq!(seen.last().unwrap().all_done, blob.len() as u64);
        // a second run keeps the file in place without fetching it again
        let asked = server.ranges.lock().unwrap().len();
        assert_eq!(run(download(&client, &plan, &Control::new(), &mut |_| {})).unwrap(), Outcome::Done);
        assert_eq!(server.ranges.lock().unwrap().len(), asked, "nothing fetched again");
    }

    #[test]
    fn a_file_that_does_not_match_its_checksum_is_removed() {
        let blob = blob();
        let server = serve(blob.clone());
        let dir = TempDir::new("fetch-bad");
        let (source, plan) = plan_for(&server, dir.path(), "0".repeat(64), blob.len() as u64);
        let err = run(download(&client(&source).unwrap(), &plan, &Control::new(), &mut |_| {})).unwrap_err();
        assert!(err.contains("did not match its checksum"), "{err}");
        assert!(!dir.path().join("model-Q4_K_M.gguf").exists() && partials_of(&dir.path().join("model-Q4_K_M.gguf")).is_empty());
        // a file larger than listed is removed too
        let (source, plan) = plan_for(&server, dir.path(), crate::sha::sha256_hex(&blob), 1000);
        let err = run(download(&client(&source).unwrap(), &plan, &Control::new(), &mut |_| {})).unwrap_err();
        assert!(err.contains("larger than"), "{err}");
        assert!(partials_of(&dir.path().join("model-Q4_K_M.gguf")).is_empty());
    }

    #[test]
    fn a_paused_download_resumes_with_a_range_and_a_cancelled_one_leaves_nothing() {
        let blob = blob();
        let server = serve(blob.clone());
        let dir = TempDir::new("fetch-pause");
        let (source, plan) = plan_for(&server, dir.path(), crate::sha::sha256_hex(&blob), blob.len() as u64);
        let client = client(&source).unwrap();
        let control = Control::new();
        let outcome = run(download(&client, &plan, &control, &mut |_| control.pause())).unwrap();
        assert_eq!(outcome, Outcome::Paused);
        let partial = existing_partial(&dir.path().join("model-Q4_K_M.gguf")).expect("a partial file stays");
        assert!(crate::fsx::is_own_temporary_name(&partial.file_name().unwrap().to_string_lossy()));
        let kept = fs::metadata(&partial).unwrap().len();
        assert!(kept > 0 && kept < blob.len() as u64, "part of it is kept: {kept}");
        assert!(plan.remaining() < blob.len() as u64, "the disk check counts what is left");
        control.resume();
        assert_eq!(run(download(&client, &plan, &control, &mut |_| {})).unwrap(), Outcome::Done);
        assert_eq!(fs::read(dir.path().join("model-Q4_K_M.gguf")).unwrap(), blob, "resumed whole");
        assert_eq!(server.ranges.lock().unwrap().last().unwrap().as_deref(), Some(format!("bytes={kept}-").as_str()));

        let dir = TempDir::new("fetch-cancel");
        let (_, plan) = plan_for(&server, dir.path(), crate::sha::sha256_hex(&blob), blob.len() as u64);
        let control = Control::new();
        let outcome = run(download(&client, &plan, &control, &mut |_| control.cancel())).unwrap();
        assert_eq!(outcome, Outcome::Cancelled);
        assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 0, "nothing is left behind");
    }

    #[test]
    fn another_file_of_the_same_name_is_never_replaced() {
        let blob = blob();
        let server = serve(blob.clone());
        let dir = TempDir::new("fetch-other");
        fs::write(dir.path().join("model-Q4_K_M.gguf"), b"the person's own file").unwrap();
        let (source, plan) = plan_for(&server, dir.path(), crate::sha::sha256_hex(&blob), blob.len() as u64);
        let err = run(download(&client(&source).unwrap(), &plan, &Control::new(), &mut |_| {})).unwrap_err();
        assert!(err.contains("was not replaced"), "{err}");
        assert_eq!(fs::read(dir.path().join("model-Q4_K_M.gguf")).unwrap(), b"the person's own file");
        assert!(server.ranges.lock().unwrap().is_empty(), "nothing was fetched");
    }
}
