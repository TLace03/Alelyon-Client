//! `lattice_sys::process::spawn` against real children (spec §7.6 X5, X7, X8;
//! CF3, CF8, CF9, CF14).
//!
//! The children are this test binary itself, run with `--ignored --exact
//! <role>`: each `role_*` test below is ignored in a normal run, and acts only
//! when its working folder holds `lattice-sys-role.txt`, which the parent test
//! writes into a temporary folder of its own. A role reads its parameters from
//! files there and writes what it saw back, because the child's environment is
//! the explicit block under test. No child reaches the network or the GPU; a
//! test ends every process it started (closing a Child closes its Job).

#![cfg(windows)]

use std::ffi::OsString;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use lattice_sys::process::{
    Child, Exit, FdPipes, JobLimits, SpawnRequest, spawn, spawn_with_fd_pipes, spawn_with_input,
};

const MARKER: &str = "lattice-sys-role.txt";

// ------------------------------------------------------------------ the roles

fn role_dir() -> Option<PathBuf> {
    let cwd = std::env::current_dir().ok()?;
    cwd.join(MARKER).is_file().then_some(cwd)
}

#[test]
#[ignore = "runs only as a child of this file's tests"]
fn role_args() {
    let Some(dir) = role_dir() else { return };
    let args: Vec<String> = std::env::args_os()
        .map(|arg| arg.to_string_lossy().into_owned())
        .collect();
    std::fs::write(dir.join("args.json"), serde_json_like(&args)).unwrap();
}

#[test]
#[ignore = "runs only as a child of this file's tests"]
fn role_env() {
    let Some(dir) = role_dir() else { return };
    let mut lines: Vec<String> = std::env::vars_os()
        .map(|(name, value)| format!("{}={}", name.to_string_lossy(), value.to_string_lossy()))
        .collect();
    lines.sort();
    std::fs::write(dir.join("env.txt"), lines.join("\n")).unwrap();
}

#[test]
#[ignore = "runs only as a child of this file's tests"]
fn role_stdin() {
    let Some(dir) = role_dir() else { return };
    let mut bytes = Vec::new();
    let read = std::io::stdin()
        .read_to_end(&mut bytes)
        .map(|_| bytes.len());
    std::fs::write(dir.join("stdin.txt"), format!("{read:?}")).unwrap();
}

#[test]
#[ignore = "runs only as a child of this file's tests"]
fn role_sleep() {
    let Some(dir) = role_dir() else { return };
    let millis = std::fs::read_to_string(dir.join("sleep_ms.txt"))
        .ok()
        .and_then(|text| text.trim().parse::<u64>().ok())
        .unwrap_or(30_000)
        .min(60_000);
    std::thread::sleep(Duration::from_millis(millis));
}

#[test]
#[ignore = "runs only as a child of this file's tests"]
fn role_hello() {
    if role_dir().is_none() {
        return;
    }
    println!("hello");
}

/// Descriptors 3 and 4 as Chromium's `--remote-debugging-pipe` finds them: the
/// C runtime's own table, filled from the inherited-handle block as the
/// program started. Writes which descriptors 0 to 6 are open, reads 3 to its
/// end, and writes back to 4 what it read after `got:`.
#[test]
#[ignore = "runs only as a child of this file's tests"]
fn role_fd_pipes() {
    use std::io::Write;
    let Some(dir) = role_dir() else { return };
    let open: Vec<String> = (0..7)
        .filter(|fd| crt::os_handle(*fd).is_some())
        .map(|fd| fd.to_string())
        .collect();
    std::fs::write(dir.join("fds.txt"), open.join(" ")).unwrap();
    let (Some(read), Some(write)) = (crt::os_handle(3), crt::os_handle(4)) else {
        return;
    };
    let mut bytes = b"got:".to_vec();
    crt::borrowed_file(read).read_to_end(&mut bytes).unwrap();
    crt::borrowed_file(write).write_all(&bytes).unwrap();
}

#[test]
#[ignore = "runs only as a child of this file's tests"]
fn role_grandchild() {
    let Some(dir) = role_dir() else { return };
    // The grandchild is started the ordinary way; the Job holds it anyway.
    let mut grandchild = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--ignored", "--exact", "role_sleep"])
        .current_dir(&dir)
        .spawn()
        .unwrap();
    // Written aside and renamed into place, so the file appears with its id.
    std::fs::write(dir.join("grandchild.tmp"), grandchild.id().to_string()).unwrap();
    std::fs::rename(dir.join("grandchild.tmp"), dir.join("grandchild.txt")).unwrap();
    std::thread::sleep(Duration::from_secs(30));
    // Not reached in these tests: the Job ends both first.
    let _ = grandchild.wait();
}

// --------------------------------------------------------------- the harness

/// A folder of the test's own; removed (with what the children wrote) when the
/// test ends. Nothing outside it is touched.
struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "lattice-sys-spawn-{tag}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&path).unwrap();
        std::fs::write(path.join(MARKER), "a role may act here").unwrap();
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        // Only the test's own temporary folder.
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// A JSON array of strings, without a JSON dependency in this test.
fn serde_json_like(items: &[String]) -> String {
    let escaped: Vec<String> = items
        .iter()
        .map(|item| {
            let mut out = String::from("\"");
            for c in item.chars() {
                match c {
                    '"' => out.push_str("\\\""),
                    '\\' => out.push_str("\\\\"),
                    '\n' => out.push_str("\\n"),
                    '\t' => out.push_str("\\t"),
                    c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
                    c => out.push(c),
                }
            }
            out.push('"');
            out
        })
        .collect();
    format!("[{}]", escaped.join(","))
}

/// The minimal environment a child needs to run at all.
fn base_env() -> Vec<(OsString, OsString)> {
    let root = std::env::var_os("SystemRoot").expect("SystemRoot is set on Windows");
    let mut system32 = PathBuf::from(&root);
    system32.push("System32");
    vec![
        (OsString::from("SystemRoot"), root),
        (OsString::from("PATH"), system32.into_os_string()),
    ]
}

fn exe() -> PathBuf {
    std::env::current_exe().unwrap()
}

/// Start this binary as `role`, with `extra` after `--`.
fn start_role(dir: &Path, role: &str, extra: &[&str], env: &[(OsString, OsString)]) -> Child {
    let exe = exe();
    let mut argv: Vec<OsString> = vec![
        exe.clone().into_os_string(),
        "--ignored".into(),
        "--exact".into(),
        role.into(),
    ];
    if !extra.is_empty() {
        argv.push("--".into());
        argv.extend(extra.iter().map(OsString::from));
    }
    spawn(&SpawnRequest {
        program: &exe,
        argv: &argv,
        cwd: dir,
        env,
        limits: JobLimits::default(),
    })
    .unwrap()
}

fn finished(child: &Child) -> u32 {
    match child.wait_or_kill(Duration::from_secs(60)).unwrap() {
        Exit::Exited(code) => code,
        Exit::TimedOut => panic!("a role did not finish within 60 s"),
    }
}

fn wait_for(what: &str, seconds: u64, mut condition: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(seconds);
    while !condition() {
        assert!(
            Instant::now() < deadline,
            "{what} did not happen within {seconds} s"
        );
        std::thread::sleep(Duration::from_millis(20));
    }
}

/// The grandchild's process id, once `role_grandchild` has written it. The file
/// can exist before its contents do, so this waits for a number, not a file.
fn grandchild_pid(dir: &Path) -> u32 {
    let mut pid = None;
    wait_for("the grandchild", 30, || {
        pid = std::fs::read_to_string(dir.join("grandchild.txt"))
            .ok()
            .and_then(|text| text.trim().parse().ok());
        pid.is_some()
    });
    pid.unwrap()
}

// ------------------------------------------------------------------ the tests

/// CF3: the argv the child receives is exactly the argv handed to spawn.
#[test]
fn the_spawned_argv_is_exactly_the_matched_argv() {
    let scratch = Scratch::new("argv");
    let corpus = [
        "two words",
        "",
        "a\"b",
        "trailing\\",
        "C:\\path with space\\",
        "\\\\server\\share",
        "--%",
        "x & y | z",
        "$env:PATH",
        "Ünïcödé",
        "\"quoted\"",
        "a\\\\\"b",
    ];
    let child = start_role(scratch.path(), "role_args", &corpus, &base_env());
    assert_eq!(finished(&child), 0);
    let seen = std::fs::read_to_string(scratch.path().join("args.json")).unwrap();
    let mut expected: Vec<String> = vec![
        exe().to_string_lossy().into_owned(),
        "--ignored".into(),
        "--exact".into(),
        "role_args".into(),
        "--".into(),
    ];
    expected.extend(corpus.iter().map(|item| (*item).to_owned()));
    assert_eq!(seen, serde_json_like(&expected));
}

#[test]
fn stdin_is_nul_and_stdout_is_a_pipe() {
    let scratch = Scratch::new("stdin");
    let mut child = start_role(scratch.path(), "role_stdin", &[], &base_env());
    let mut stdout = String::new();
    child
        .take_stdout()
        .unwrap()
        .read_to_string(&mut stdout)
        .unwrap();
    assert_eq!(finished(&child), 0);
    assert_eq!(
        std::fs::read_to_string(scratch.path().join("stdin.txt")).unwrap(),
        "Ok(0)",
        "stdin reads as empty at once"
    );
    assert!(stdout.contains("running 1 test"), "{stdout}");
    assert!(child.take_stdin().is_none(), "spawn keeps no stdin end");
}

/// `spawn_with_input`: the child reads what this process writes, and its
/// stdin ends when this process drops the write end.
#[test]
fn spawn_with_input_gives_the_child_a_stdin_pipe_that_ends_on_drop() {
    use std::io::Write;
    let scratch = Scratch::new("stdin-pipe");
    let exe = exe();
    let argv: Vec<OsString> = vec![
        exe.clone().into_os_string(),
        "--ignored".into(),
        "--exact".into(),
        "role_stdin".into(),
    ];
    let mut child = spawn_with_input(&SpawnRequest {
        program: &exe,
        argv: &argv,
        cwd: scratch.path(),
        env: &base_env(),
        limits: JobLimits::default(),
    })
    .unwrap();
    let mut stdin = child.take_stdin().expect("a stdin end");
    assert!(child.take_stdin().is_none(), "taken once");
    stdin.write_all(b"{\"jsonrpc\":\"2.0\"}\n").unwrap();
    drop(stdin);
    let mut stdout = String::new();
    child
        .take_stdout()
        .unwrap()
        .read_to_string(&mut stdout)
        .unwrap();
    assert_eq!(finished(&child), 0);
    assert_eq!(
        std::fs::read_to_string(scratch.path().join("stdin.txt")).unwrap(),
        "Ok(18)",
        "the child read the 18 bytes written, then the end"
    );
}

/// Read `file` to its end on a thread, waiting at most `seconds`.
fn read_all_within(file: std::fs::File, seconds: u64) -> Result<Vec<u8>, String> {
    let (sender, receiver) = mpsc::channel();
    std::thread::spawn(move || {
        let mut file = file;
        let mut bytes = Vec::new();
        let _ = sender.send(file.read_to_end(&mut bytes).map(|_| bytes));
    });
    match receiver.recv_timeout(Duration::from_secs(seconds)) {
        Ok(read) => read.map_err(|error| error.to_string()),
        Err(_) => Err(format!("no end of file within {seconds} s")),
    }
}

/// `spawn_with_fd_pipes`, as Chromium's `--remote-debugging-pipe` takes it: the
/// child's C runtime has descriptors 0 to 4 and nothing past them; it reads on
/// descriptor 3 what this process writes, to its end once this process closes
/// it; what it writes on descriptor 4 reaches this process, which reads to the
/// end once the child has ended (so neither side keeps a stray copy of the
/// other's end). An inheritable handle of this process does not reach it.
#[test]
fn spawn_with_fd_pipes_gives_the_child_descriptors_3_and_4_and_nothing_else() {
    use std::io::Write;
    let scratch = Scratch::new("fd-pipes");
    let exe = exe();
    let argv: Vec<OsString> = vec![
        exe.clone().into_os_string(),
        "--ignored".into(),
        "--exact".into(),
        "role_fd_pipes".into(),
    ];
    let request = SpawnRequest {
        program: &exe,
        argv: &argv,
        cwd: scratch.path(),
        env: &base_env(),
        limits: JobLimits::default(),
    };
    // What another spawn's ends look like at the moment of a concurrent start.
    let (decoy_read, decoy_write) = win::inheritable_pipe();
    let mut child = spawn_with_fd_pipes(&request).unwrap();
    drop(decoy_write);
    let FdPipes {
        mut to_child,
        from_child,
    } = child.take_fd_pipes().expect("the two pipes");
    assert!(child.take_fd_pipes().is_none(), "taken once");
    assert!(child.take_stdin().is_none(), "stdin stays NUL");
    // The child is waiting on descriptor 3; the decoy ends at once.
    assert_eq!(
        read_all_within(decoy_read, 5),
        Ok(Vec::new()),
        "the child holds the decoy's write end"
    );
    let message = b"{\"id\":1,\"method\":\"Browser.getVersion\"}\0";
    to_child.write_all(message).unwrap();
    drop(to_child);
    let reply = read_all_within(from_child, 30).expect("descriptor 4 reads to its end");
    assert_eq!(reply, [&b"got:"[..], message].concat());
    assert_eq!(finished(&child), 0);
    assert_eq!(
        std::fs::read_to_string(scratch.path().join("fds.txt")).unwrap(),
        "0 1 2 3 4",
        "the descriptors the child's C runtime has"
    );
    // The same checks as spawn: a relative program is refused before any call.
    let relative = SpawnRequest {
        program: Path::new("msedge.exe"),
        ..request
    };
    assert_eq!(
        spawn_with_fd_pipes(&relative).unwrap_err().kind(),
        std::io::ErrorKind::InvalidInput
    );
}

/// A child started by `spawn` has no descriptor past the standard three, and
/// no pipes to take: the block is only for `spawn_with_fd_pipes`.
#[test]
fn spawn_passes_no_descriptor_past_the_standard_three() {
    let scratch = Scratch::new("no-fd-pipes");
    let mut child = start_role(scratch.path(), "role_fd_pipes", &[], &base_env());
    assert!(child.take_fd_pipes().is_none());
    assert_eq!(finished(&child), 0);
    assert_eq!(
        std::fs::read_to_string(scratch.path().join("fds.txt")).unwrap(),
        "0 1 2"
    );
}

/// The Job's process list names the child and its grandchild while they run,
/// and nobody once the tree has ended.
#[test]
fn the_jobs_process_list_names_the_child_and_its_grandchild() {
    let scratch = Scratch::new("ids");
    // The grandchild sleeps a minute, far past the checks below.
    std::fs::write(scratch.path().join("sleep_ms.txt"), "60000").unwrap();
    let child = start_role(scratch.path(), "role_grandchild", &[], &base_env());
    let grandchild = grandchild_pid(scratch.path());
    let ids = child.process_ids().unwrap();
    assert!(
        ids.contains(&child.pid()) && ids.contains(&grandchild),
        "{ids:?} (child {}, grandchild {grandchild})",
        child.pid()
    );
    assert_eq!(ids.len() as u32, child.active_processes().unwrap());
    child.kill_tree().unwrap();
    wait_for("the tree to end", 5, || {
        child.process_ids().unwrap().is_empty()
    });
}

/// §12's `.cmd` shim, for real: `cmd.exe` runs the batch file with its
/// arguments as written (quoted where they must be), in the folder given, with
/// a stdin pipe, and ends with it.
#[test]
fn a_batch_file_starts_through_cmd_with_its_arguments_as_written() {
    use lattice_sys::process::{BatchRequest, spawn_batch_with_input};
    let scratch = Scratch::new("batch");
    let script = scratch.path().join("shim.cmd");
    std::fs::write(
        &script,
        "@echo off\r\necho [%*]> \"%~dp0args.txt\"\r\ncd> \"%~dp0cwd.txt\"\r\n",
    )
    .unwrap();
    let cmd = PathBuf::from(std::env::var_os("SystemRoot").unwrap()).join(r"System32\cmd.exe");
    let args: Vec<OsString> = ["-y", "@scope/server", "two words", "k=v"]
        .iter()
        .map(OsString::from)
        .collect();
    let mut child = spawn_batch_with_input(&BatchRequest {
        cmd: &cmd,
        script: &script,
        args: &args,
        cwd: scratch.path(),
        env: &base_env(),
        limits: JobLimits::default(),
    })
    .unwrap();
    assert!(child.take_stdin().is_some(), "a stdin pipe");
    assert_eq!(finished(&child), 0);
    let seen = std::fs::read_to_string(scratch.path().join("args.txt")).unwrap();
    assert_eq!(seen.trim_end(), r#"[-y @scope/server "two words" "k=v"]"#);
    let cwd = std::fs::read_to_string(scratch.path().join("cwd.txt")).unwrap();
    assert!(
        cwd.trim_end()
            .eq_ignore_ascii_case(&scratch.path().display().to_string()),
        "{cwd}"
    );
}

/// The child's environment is the block and nothing else.
#[test]
fn the_environment_is_exactly_the_block() {
    let scratch = Scratch::new("env");
    let mut env = base_env();
    env.push((
        OsString::from("LATTICE_SYS_PROBE"),
        OsString::from("visible"),
    ));
    let child = start_role(scratch.path(), "role_env", &[], &env);
    assert_eq!(finished(&child), 0);
    let seen = std::fs::read_to_string(scratch.path().join("env.txt")).unwrap();
    let names: Vec<String> = seen
        .lines()
        .map(|line| line.split('=').next().unwrap().to_uppercase())
        .collect();
    assert_eq!(names, ["LATTICE_SYS_PROBE", "PATH", "SYSTEMROOT"], "{seen}");
    assert!(seen.contains("LATTICE_SYS_PROBE=visible"));
}

/// CF8: stopping ends a grandchild too.
#[test]
fn stop_ends_the_grandchild_as_well() {
    let scratch = Scratch::new("stop");
    // The grandchild sleeps a minute, far past the checks below.
    std::fs::write(scratch.path().join("sleep_ms.txt"), "60000").unwrap();
    let child = start_role(scratch.path(), "role_grandchild", &[], &base_env());
    wait_for("the grandchild", 30, || {
        scratch.path().join("grandchild.txt").is_file()
    });
    // The child and the grandchild, and the console host Windows may start for
    // the grandchild (the child has no console): all in the Job.
    let alive = child.active_processes().unwrap();
    assert!(alive >= 2, "{alive} processes in the Job");
    child.kill_tree().unwrap();
    // Promptly: the grandchild would otherwise sleep for a minute.
    wait_for("the tree to end", 5, || {
        child.active_processes().unwrap() == 0
    });
    assert!(child.wait(Some(Duration::from_secs(5))).unwrap().is_some());
}

/// CF9: a timeout ends the whole tree.
#[test]
fn a_timeout_ends_the_tree() {
    let scratch = Scratch::new("timeout");
    // The grandchild sleeps a minute, far past the checks below.
    std::fs::write(scratch.path().join("sleep_ms.txt"), "60000").unwrap();
    let child = start_role(scratch.path(), "role_grandchild", &[], &base_env());
    wait_for("the grandchild", 30, || {
        scratch.path().join("grandchild.txt").is_file()
    });
    let started = Instant::now();
    assert_eq!(
        child.wait_or_kill(Duration::from_millis(300)).unwrap(),
        Exit::TimedOut
    );
    assert!(started.elapsed() >= Duration::from_millis(250));
    // Promptly: the grandchild would otherwise sleep for a minute.
    wait_for("the tree to end", 5, || {
        child.active_processes().unwrap() == 0
    });
}

/// The Job carries its limits, and closing it ends a grandchild.
#[test]
fn the_job_has_its_limits_and_closing_it_ends_the_tree() {
    let scratch = Scratch::new("close");
    // The grandchild sleeps a minute, far past the checks below.
    std::fs::write(scratch.path().join("sleep_ms.txt"), "60000").unwrap();
    let child = start_role(scratch.path(), "role_grandchild", &[], &base_env());
    assert_eq!(child.job_limits().unwrap(), JobLimits::default());
    let pid = grandchild_pid(scratch.path());
    let grandchild = win::open_for_wait(pid);
    drop(child);
    assert!(
        win::exited_within(&grandchild, Duration::from_secs(5)),
        "closing the Job ends the grandchild"
    );
}

/// CF14: an inheritable handle of this process does not reach the child.
#[test]
fn the_child_inherits_only_its_own_standard_handles() {
    let scratch = Scratch::new("handles");
    std::fs::write(scratch.path().join("sleep_ms.txt"), "3000").unwrap();
    // A pipe whose write end is inheritable: what another spawn's pipe ends
    // look like at the moment of a concurrent CreateProcess.
    let (decoy_read, decoy_write) = win::inheritable_pipe();
    let child = start_role(scratch.path(), "role_sleep", &[], &base_env());
    drop(decoy_write);
    let (sender, receiver) = mpsc::channel();
    std::thread::spawn(move || {
        let mut decoy_read = decoy_read;
        let mut buffer = [0u8; 16];
        let _ = sender.send(decoy_read.read(&mut buffer).map_err(|e| e.kind()));
    });
    // Read 0 (end of file) at once: no other process holds the write end.
    let read = receiver.recv_timeout(Duration::from_secs(1));
    assert_eq!(read, Ok(Ok(0)), "the child holds the decoy's write end");
    assert!(
        child.wait(Some(Duration::ZERO)).unwrap().is_none(),
        "the child still runs"
    );
}

/// CF14: two children started at the same moment keep their own pipes.
#[test]
fn two_children_spawned_at_once_keep_their_own_pipes() {
    for round in 0..6 {
        let quick = Scratch::new("quick");
        let slow = Scratch::new("slow");
        std::fs::write(slow.path().join("sleep_ms.txt"), "5000").unwrap();
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
        let quick_path = quick.path().to_path_buf();
        let slow_path = slow.path().to_path_buf();
        let quick_barrier = barrier.clone();
        let quick_thread = std::thread::spawn(move || {
            quick_barrier.wait();
            start_role(&quick_path, "role_hello", &[], &base_env())
        });
        let slow_thread = std::thread::spawn(move || {
            barrier.wait();
            start_role(&slow_path, "role_sleep", &[], &base_env())
        });
        let mut quick_child = quick_thread.join().unwrap();
        let slow_child = slow_thread.join().unwrap();
        let mut stdout = quick_child.take_stdout().unwrap();
        let (sender, receiver) = mpsc::channel();
        std::thread::spawn(move || {
            let mut text = String::new();
            let _ = sender.send(stdout.read_to_string(&mut text).map(|_| text));
        });
        // The quick child's stdout ends when it does, not when the slow one does.
        let text = receiver
            .recv_timeout(Duration::from_secs(3))
            .unwrap_or_else(|_| {
                panic!("round {round}: the slow child holds the quick child's stdout")
            })
            .unwrap();
        assert!(text.contains("hello"), "{text}");
        assert_eq!(finished(&quick_child), 0);
        drop(slow_child);
    }
}

/// The C runtime's view of a descriptor, in this test binary's UCRT (linked
/// dynamically; Chromium links its own statically): `_get_osfhandle`, with
/// this thread's invalid-parameter handler set to return rather than end the
/// process on a closed descriptor, as Chromium's `devtools_pipe` does around
/// the same call.
mod crt {
    use std::fs::File;
    use std::mem::ManuallyDrop;
    use std::os::windows::io::{FromRawHandle, RawHandle};

    type Handler = Option<unsafe extern "C" fn(*const u16, *const u16, *const u16, u32, usize)>;

    unsafe extern "C" {
        fn _get_osfhandle(fd: i32) -> isize;
        fn _set_thread_local_invalid_parameter_handler(handler: Handler) -> Handler;
    }

    extern "C" fn carry_on(_: *const u16, _: *const u16, _: *const u16, _: u32, _: usize) {}

    /// The handle behind descriptor `fd`, if the C runtime has it open.
    pub fn os_handle(fd: i32) -> Option<isize> {
        // SAFETY: setting this thread's handler has no precondition; the
        // previous one is put back below.
        let previous = unsafe { _set_thread_local_invalid_parameter_handler(Some(carry_on)) };
        // SAFETY: a lookup in the C runtime's table; with the handler above, a
        // descriptor it does not have returns -1 instead of ending the process.
        let handle = unsafe { _get_osfhandle(fd) };
        // SAFETY: restoring the handler read above.
        unsafe { _set_thread_local_invalid_parameter_handler(previous) };
        // -1 is no handle, -2 a standard descriptor with no console behind it.
        (handle != -1 && handle != -2).then_some(handle)
    }

    /// A file over a descriptor's handle that never closes it: the C runtime
    /// owns the handle.
    pub fn borrowed_file(handle: isize) -> ManuallyDrop<File> {
        // SAFETY: the handle is open (`os_handle` found it) and stays open for
        // the process's life; the File is never dropped, so never closes it.
        ManuallyDrop::new(unsafe { File::from_raw_handle(handle as RawHandle) })
    }
}

/// Small Win32 helpers for the tests above (this crate's own tests may use
/// `unsafe`; each block says why it is sound).
mod win {
    use std::fs::File;
    use std::os::windows::io::{FromRawHandle, OwnedHandle, RawHandle};
    use std::ptr;
    use std::time::Duration;

    use windows_sys::Win32::Foundation::{
        HANDLE, HANDLE_FLAG_INHERIT, SetHandleInformation, WAIT_OBJECT_0,
    };
    use windows_sys::Win32::Security::SECURITY_ATTRIBUTES;
    use windows_sys::Win32::System::Pipes::CreatePipe;
    use windows_sys::Win32::System::Threading::{
        OpenProcess, PROCESS_SYNCHRONIZE, WaitForSingleObject,
    };

    /// (read end, write end); the write end inheritable, the read end not.
    pub fn inheritable_pipe() -> (File, File) {
        let attributes = SECURITY_ATTRIBUTES {
            nLength: std::mem::size_of::<SECURITY_ATTRIBUTES>() as u32,
            lpSecurityDescriptor: ptr::null_mut(),
            bInheritHandle: 1,
        };
        let mut read: HANDLE = ptr::null_mut();
        let mut write: HANDLE = ptr::null_mut();
        // SAFETY: the out-pointers and the attributes are valid for the call.
        let ok = unsafe { CreatePipe(&mut read, &mut write, &attributes, 0) };
        assert_ne!(ok, 0, "CreatePipe");
        // SAFETY: `read` is an open handle; only its inherit flag changes.
        let ok = unsafe { SetHandleInformation(read, HANDLE_FLAG_INHERIT, 0) };
        assert_ne!(ok, 0, "SetHandleInformation");
        // SAFETY: both handles were just created and are owned by nothing else.
        unsafe {
            (
                File::from_raw_handle(read as RawHandle),
                File::from_raw_handle(write as RawHandle),
            )
        }
    }

    pub fn open_for_wait(pid: u32) -> OwnedHandle {
        // SAFETY: OpenProcess has no memory preconditions; a null result is
        // checked below.
        let handle = unsafe { OpenProcess(PROCESS_SYNCHRONIZE, 0, pid) };
        assert!(!handle.is_null(), "the grandchild {pid} is running");
        // SAFETY: a handle OpenProcess just returned, owned by nothing else.
        unsafe { OwnedHandle::from_raw_handle(handle as RawHandle) }
    }

    pub fn exited_within(process: &OwnedHandle, timeout: Duration) -> bool {
        use std::os::windows::io::AsRawHandle;
        // SAFETY: the handle is open and owned by `process`.
        let result =
            unsafe { WaitForSingleObject(process.as_raw_handle(), timeout.as_millis() as u32) };
        result == WAIT_OBJECT_0
    }
}
