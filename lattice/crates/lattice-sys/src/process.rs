//! Starting a program with nothing implicit: an explicit program path, argv,
//! working folder and environment block, a `NUL` stdin, pipes for stdout and
//! stderr, an explicit list of the handles it inherits, and a Job Object that
//! holds the program and everything it starts (the chat core's spec §7.6
//! X5, X7, X8). Not a port.
//!
//! How [`spawn`] starts a child:
//! 1. a Job Object with `KILL_ON_JOB_CLOSE`, no breakaway, an active-process
//!    limit and a job memory limit;
//! 2. `CreateProcessW` with `lpApplicationName` set to the absolute program
//!    (no search of any folder), `lpCommandLine` the argv quoted by the
//!    MSVCRT rules ([`quote_argv`]), the given environment block only
//!    (`CREATE_UNICODE_ENVIRONMENT`), `CREATE_SUSPENDED`, `CREATE_NO_WINDOW`,
//!    and a `STARTUPINFOEXW` whose `PROC_THREAD_ATTRIBUTE_HANDLE_LIST` names
//!    exactly the child's three standard handles, so a child started at the
//!    same moment by another thread cannot inherit this one's pipe ends;
//! 3. the suspended process is assigned to its Job, **then** resumed, so no
//!    grandchild can start outside the Job.
//!
//! The child's ends are made inheritable just before `CreateProcessW` and
//! closed in this process right after it. Closing the [`Child`] (dropping it)
//! closes the Job, which ends the whole tree.
//!
//! [`spawn_with_input`] is the same start with one difference: the child's
//! stdin is a pipe whose write end this process keeps ([`Child::take_stdin`]),
//! for a program Lattice talks to, an MCP server (spec §12). Commands and git
//! keep `NUL`.
//!
//! [`spawn_with_fd_pipes`] is [`spawn`] with two more pipes, the child's C
//! runtime descriptors 3 and 4 ([`FdPipes`]), for a program that takes a
//! channel there: Chromium's `--remote-debugging-pipe` reads DevTools
//! requests from descriptor 3 and writes its answers to 4
//! (`content/public/browser/devtools_agent_host.h`, `kReadFD` and `kWriteFD`;
//! on Windows it finds them with `_get_osfhandle`). A descriptor past the
//! standard three reaches a Windows child only through the C runtime's
//! inherited-handle block (`STARTUPINFOW.lpReserved2`, laid out by
//! [`crt_descriptor_block`]), which the child's C runtime reads as it starts
//! (the UCRT's `lowio/ioinit.cpp`); Node's libuv passes Puppeteer's and
//! Playwright's browsers the same two pipes that way. The handle list still
//! names exactly the handles the child gets: five here.

use std::ffi::{OsStr, OsString};
use std::fs::File;
use std::io;
use std::path::Path;
use std::time::Duration;

/// The Job's limits (spec X8: defaults to tune).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JobLimits {
    /// Processes alive at once in the tree.
    pub active_processes: u32,
    /// Committed memory of the whole tree, in bytes.
    pub job_memory: u64,
}

impl Default for JobLimits {
    fn default() -> Self {
        Self {
            active_processes: 64,
            job_memory: 4 * 1024 * 1024 * 1024,
        }
    }
}

/// What to start.
#[derive(Clone, Copy, Debug)]
pub struct SpawnRequest<'a> {
    /// An absolute path to the program; nothing is searched.
    pub program: &'a Path,
    /// The child's argv, its own name first.
    pub argv: &'a [OsString],
    pub cwd: &'a Path,
    /// The whole environment: nothing of this process's is passed on.
    pub env: &'a [(OsString, OsString)],
    pub limits: JobLimits,
}

/// How a wait ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Exit {
    /// The program exited with this code.
    Exited(u32),
    /// The time ran out; the whole tree was ended.
    TimedOut,
}

/// This process's ends of the two pipes [`spawn_with_fd_pipes`] gives a child
/// as its C runtime descriptors 3 and 4.
#[derive(Debug)]
pub struct FdPipes {
    /// The write end of the pipe the child reads as descriptor 3. Dropping it
    /// ends what the child reads there.
    pub to_child: File,
    /// The read end of the pipe the child writes as descriptor 4. It reads to
    /// its end once the child (and whatever the child passed it to) has
    /// closed it.
    pub from_child: File,
}

/// A running program, its Job, and the read ends of its pipes (and, for
/// [`spawn_with_input`], the write end of its stdin; for
/// [`spawn_with_fd_pipes`], this process's ends of descriptors 3 and 4).
#[derive(Debug)]
pub struct Child {
    pid: u32,
    stdin: Option<File>,
    stdout: Option<File>,
    stderr: Option<File>,
    fd_pipes: Option<FdPipes>,
    inner: imp::Handles,
}

impl Child {
    pub fn pid(&self) -> u32 {
        self.pid
    }

    /// The write end of the child's stdin (once): only a child started by
    /// [`spawn_with_input`] has one. Dropping it closes the child's stdin.
    pub fn take_stdin(&mut self) -> Option<File> {
        self.stdin.take()
    }

    /// This process's ends of the child's descriptors 3 and 4 (once): only a
    /// child started by [`spawn_with_fd_pipes`] has them.
    pub fn take_fd_pipes(&mut self) -> Option<FdPipes> {
        self.fd_pipes.take()
    }

    /// The read end of the child's stdout (once).
    pub fn take_stdout(&mut self) -> Option<File> {
        self.stdout.take()
    }

    /// The read end of the child's stderr (once).
    pub fn take_stderr(&mut self) -> Option<File> {
        self.stderr.take()
    }

    /// Wait for the program (not its descendants) to exit: `None` when
    /// `timeout` passes first. `None` timeout waits as long as it takes.
    pub fn wait(&self, timeout: Option<Duration>) -> io::Result<Option<u32>> {
        imp::wait(&self.inner, timeout)
    }

    /// Wait at most `timeout`; when it passes, end the whole tree.
    pub fn wait_or_kill(&self, timeout: Duration) -> io::Result<Exit> {
        match self.wait(Some(timeout))? {
            Some(code) => Ok(Exit::Exited(code)),
            None => {
                self.kill_tree()?;
                Ok(Exit::TimedOut)
            }
        }
    }

    /// End every process in the Job: the program and all it started.
    pub fn kill_tree(&self) -> io::Result<()> {
        imp::kill_tree(&self.inner)
    }

    /// How many processes of the tree are alive.
    pub fn active_processes(&self) -> io::Result<u32> {
        imp::active_processes(&self.inner)
    }

    /// The ids of the tree's live processes: the program and all it started
    /// that still run (the Job's own list).
    pub fn process_ids(&self) -> io::Result<Vec<u32>> {
        imp::process_ids(&self.inner)
    }

    /// The limits the Job enforces, read back from it.
    pub fn job_limits(&self) -> io::Result<JobLimits> {
        imp::job_limits(&self.inner)
    }
}

/// Would `CreateProcessW` run `program` through `cmd.exe`? True when its name,
/// with trailing dots and spaces trimmed (Windows drops them), ends in `.bat`
/// or `.cmd` in any case: the rule Rust's std adopted for CVE-2024-24576
/// (spec Â§22.6 X2b).
pub fn is_batch_file(program: &Path) -> bool {
    let Some(name) = program.file_name() else {
        return false;
    };
    let name = name.to_string_lossy();
    let trimmed = name.trim_end_matches(['.', ' ']).to_ascii_lowercase();
    trimmed.ends_with(".bat") || trimmed.ends_with(".cmd")
}

/// The refusal of a program that is an app execution alias, or a reparse
/// point of a kind that is not a link.
pub const NOT_A_PROGRAM_FILE: &str =
    "a program that is an app execution alias or an unknown reparse point is never started";
/// The refusal of a batch file.
pub const A_BATCH_FILE: &str = "a batch file is never started: cmd.exe would run it";

/// X2b, LR3a, X2c in depth: the program's own file, read before anything
/// starts it. Its last component must be a plain file, a symlink or a
/// junction, or a file stored differently but not redirected (WOF, dedup);
/// an app execution alias (`CreateProcessW` would start whatever its data
/// names: a `.cmd` through `cmd.exe`, or any other path) or an unknown
/// reparse point is refused. A link is followed by the system's own open,
/// and its final target must be neither a batch file nor such a reparse
/// point (an alias at the end of a chain cannot be opened as a file at all,
/// which refuses it too). The callers' own resolution
/// (`find_binary`, `resolve_program`) already refuses these; this is the
/// last check, so no caller's gap can start one. A name swapped after this
/// check needs write access to the program's folder.
#[cfg(windows)]
fn check_program_file(program: &Path) -> io::Result<()> {
    use std::os::windows::fs::OpenOptionsExt;

    use crate::fs::{Access, LinkKind, final_path, is_storage_only_tag, open_no_follow};

    // FILE_READ_ATTRIBUTES (winnt.h).
    const READ_ATTRIBUTES: u32 = 0x80;
    let allowed = |kind: LinkKind, tag: u32| match kind {
        LinkKind::Symlink | LinkKind::Junction => true,
        LinkKind::Other => is_storage_only_tag(tag),
        LinkKind::AppExecLink => false,
    };
    let opened = open_no_follow(program, Access::Attributes)?;
    let Some(link) = opened.link else {
        return Ok(());
    };
    if !allowed(link.kind, link.tag) {
        return Err(invalid(NOT_A_PROGRAM_FILE));
    }
    if !matches!(link.kind, LinkKind::Symlink | LinkKind::Junction) {
        return Ok(());
    }
    let followed = std::fs::OpenOptions::new()
        .access_mode(READ_ATTRIBUTES)
        .open(program)?;
    let end = final_path(&followed)?;
    if is_batch_file(&end) {
        return Err(invalid(A_BATCH_FILE));
    }
    // The final path has every link resolved, so only a storage-only
    // reparse point may remain at its end.
    match open_no_follow(&end, Access::Attributes)?.link {
        Some(link) if link.kind != LinkKind::Other || !is_storage_only_tag(link.tag) => {
            Err(invalid(NOT_A_PROGRAM_FILE))
        }
        _ => Ok(()),
    }
}

/// Start a program as the module header says. A batch file is refused
/// ([`is_batch_file`]): `cmd.exe` would run it, and the MSVCRT quoting of
/// [`quote_argv`] leaves `cmd.exe`'s metacharacters (`&`, `|`, `%`, `^`) live
/// in any argument without a space, so an argument could start another
/// program (X2b). So is a program whose file is an app execution alias or an
/// unknown reparse point, or a link whose final target is either
/// (`check_program_file`).
pub fn spawn(request: &SpawnRequest<'_>) -> io::Result<Child> {
    checked_spawn(request, Input::Null, Extra::Nothing)
}

/// [`spawn`] with a pipe for stdin instead of `NUL` ([`Child::take_stdin`]),
/// under the same checks: for a program Lattice talks to on its stdin, an MCP
/// server (spec §12).
pub fn spawn_with_input(request: &SpawnRequest<'_>) -> io::Result<Child> {
    checked_spawn(request, Input::Pipe, Extra::Nothing)
}

/// [`spawn`] with two more pipes ([`Child::take_fd_pipes`]): the child's C
/// runtime descriptors 3, which it reads, and 4, which it writes, passed in
/// the C runtime's inherited-handle block ([`crt_descriptor_block`]: `NUL`,
/// stdout, stderr and the two pipe ends as descriptors 0 to 4) and named in
/// the handle list, so the child inherits those five handles and nothing
/// else. For Chromium's `--remote-debugging-pipe`; under the same checks as
/// [`spawn`].
pub fn spawn_with_fd_pipes(request: &SpawnRequest<'_>) -> io::Result<Child> {
    checked_spawn(request, Input::Null, Extra::FdPipes)
}

/// What the child's stdin is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Input {
    Null,
    Pipe,
}

/// What the child gets besides its three standard handles.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Extra {
    Nothing,
    /// Descriptors 3 and 4, two pipes ([`spawn_with_fd_pipes`]).
    FdPipes,
}

/// A C runtime descriptor's flags in the inherited-handle block (its `osfile`
/// bits, as the UCRT's `inc/corecrt_internal_lowio.h` and libuv's
/// `src/win/process-stdio.c` define them): the descriptor is open,
pub const CRT_FOPEN: u8 = 0x01;
/// it is a pipe (the child's C runtime then never calls `GetFileType` on it),
pub const CRT_FPIPE: u8 = 0x08;
/// or it is a character device, such as `NUL`.
pub const CRT_FDEV: u8 = 0x40;

/// The C runtime's inherited-handle block, for `STARTUPINFOW.lpReserved2`
/// (its length goes in `cbReserved2`): the number of descriptors as a 32-bit
/// integer, one flag byte per descriptor, then one pointer-sized handle per
/// descriptor, in this machine's byte order and packed with no padding. Entry
/// `n` is descriptor `n`. The child's UCRT reads it as the program starts
/// (`initialize_inherited_file_handles_nolock` in `lowio/ioinit.cpp`),
/// skipping an entry without [`CRT_FOPEN`]. A block longer than
/// `cbReserved2` can state is refused.
pub fn crt_descriptor_block(entries: &[(u8, usize)]) -> io::Result<Vec<u8>> {
    let count = i32::try_from(entries.len())
        .map_err(|_| invalid("too many descriptors for the C runtime's block"))?;
    let mut block = Vec::with_capacity(4 + entries.len() * (1 + std::mem::size_of::<usize>()));
    block.extend_from_slice(&count.to_ne_bytes());
    block.extend(entries.iter().map(|(flags, _)| *flags));
    for (_, handle) in entries {
        block.extend_from_slice(&handle.to_ne_bytes());
    }
    if u16::try_from(block.len()).is_err() {
        return Err(invalid(
            "the C runtime's descriptor block is longer than cbReserved2 can state",
        ));
    }
    Ok(block)
}

fn checked_spawn(request: &SpawnRequest<'_>, input: Input, extra: Extra) -> io::Result<Child> {
    if !request.program.is_absolute() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "the program must be an absolute path",
        ));
    }
    if is_batch_file(request.program) {
        return Err(invalid(A_BATCH_FILE));
    }
    if request.argv.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "the argv must hold the program's name",
        ));
    }
    #[cfg(windows)]
    check_program_file(request.program)?;
    let mut command_line = quote_argv(request.argv)?;
    command_line.push(0);
    imp::spawn(
        request.program,
        command_line,
        request.cwd,
        request.env,
        request.limits,
        input,
        extra,
    )
}

// ------------------------------------------------------------ batch files

/// What a batch file's line may not hold anywhere ([`batch_command_line`]):
/// `cmd.exe` expands `%` and `!` even inside quotes, and a `"` would end one.
pub const BATCH_REFUSED: [char; 3] = ['"', '%', '!'];
/// What makes an argument quoted in a batch file's line: inside quotes,
/// `cmd.exe` gives none of these a meaning.
pub const BATCH_QUOTED: [char; 12] = [' ', '\t', '&', '|', '<', '>', '^', '(', ')', ',', ';', '='];
/// The refusal of a batch file's line that `cmd.exe` could read otherwise.
pub const A_BATCH_LINE: &str = "a batch file's line may not hold a double quote, a percent sign, an exclamation mark, a control character or text that is not Unicode";

/// A batch file (a `.cmd` shim such as `npx`) to start through `cmd.exe`,
/// with a stdin pipe, for the reader's own MCP configuration (spec §12): the
/// one way a batch file starts here.
#[derive(Clone, Copy, Debug)]
pub struct BatchRequest<'a> {
    /// `cmd.exe`, absolute (`%SystemRoot%\System32\cmd.exe`).
    pub cmd: &'a Path,
    /// The batch file, absolute.
    pub script: &'a Path,
    /// Its arguments.
    pub args: &'a [OsString],
    pub cwd: &'a Path,
    /// The whole environment, as for [`spawn`].
    pub env: &'a [(OsString, OsString)],
    pub limits: JobLimits,
}

/// `"<cmd>" /d /v:off /s /c ""<script>" <args>"`, or why not. `/d` runs no
/// AutoRun command, `/v:off` expands no `!`, and `/s` takes away exactly the
/// outer quotes. No part may hold one of [`BATCH_REFUSED`] or a control
/// character; an argument that is empty or holds one of [`BATCH_QUOTED`] is
/// quoted (a run of backslashes before its closing quote doubled, so the C
/// runtime of the program the batch file starts reads them as written), and
/// the script always is. With nothing in the line that `cmd.exe` reads as an
/// operator, a variable or an escape, the batch file gets its arguments as
/// written.
pub fn batch_command_line(cmd: &Path, script: &Path, args: &[OsString]) -> io::Result<String> {
    let text = |part: &OsStr| -> io::Result<String> {
        let part = part.to_str().ok_or_else(|| invalid(A_BATCH_LINE))?;
        if part
            .chars()
            .any(|c| (c.is_control() && c != '\t') || BATCH_REFUSED.contains(&c))
        {
            return Err(invalid(A_BATCH_LINE));
        }
        Ok(part.to_owned())
    };
    let cmd = text(cmd.as_os_str())?;
    let script = text(script.as_os_str())?;
    if cmd.contains('\t') || script.contains('\t') {
        return Err(invalid(A_BATCH_LINE));
    }
    let mut line = format!("\"{cmd}\" /d /v:off /s /c \"\"{script}\"");
    for arg in args {
        let arg = text(arg)?;
        line.push(' ');
        if arg.is_empty() || arg.contains(BATCH_QUOTED) {
            let trailing = arg.len() - arg.trim_end_matches('\\').len();
            line.push('"');
            line.push_str(&arg);
            line.push_str(&"\\".repeat(trailing));
            line.push('"');
        } else {
            line.push_str(&arg);
        }
    }
    line.push('"');
    Ok(line)
}

/// Start a batch file through `cmd.exe` with a stdin pipe
/// ([`batch_command_line`]), in a Job Object as [`spawn`] starts a program.
/// `cmd` must be named `cmd.exe`, `script` must be a batch file, both
/// absolute; neither may be an app execution alias or a link of any kind.
pub fn spawn_batch_with_input(request: &BatchRequest<'_>) -> io::Result<Child> {
    if !request.cmd.is_absolute() || !request.script.is_absolute() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "the program must be an absolute path",
        ));
    }
    let is_cmd = request
        .cmd
        .file_name()
        .and_then(OsStr::to_str)
        .is_some_and(|name| name.eq_ignore_ascii_case("cmd.exe"));
    if !is_cmd {
        return Err(invalid("a batch file starts only through cmd.exe"));
    }
    if !is_batch_file(request.script) {
        return Err(invalid("that is not a batch file"));
    }
    let line = batch_command_line(request.cmd, request.script, request.args)?;
    #[cfg(windows)]
    {
        check_program_file(request.cmd)?;
        check_plain_file(request.script)?;
    }
    let mut command_line: Vec<u16> = line.encode_utf16().collect();
    command_line.push(0);
    imp::spawn(
        request.cmd,
        command_line,
        request.cwd,
        request.env,
        request.limits,
        Input::Pipe,
        Extra::Nothing,
    )
}

/// A batch file's own file: no reparse point but a storage-only one (WOF,
/// dedup), so what runs is the file named.
#[cfg(windows)]
fn check_plain_file(path: &Path) -> io::Result<()> {
    use crate::fs::{Access, LinkKind, is_storage_only_tag, open_no_follow};
    match open_no_follow(path, Access::Attributes)?.link {
        None => Ok(()),
        Some(link) if link.kind == LinkKind::Other && is_storage_only_tag(link.tag) => Ok(()),
        Some(_) => Err(invalid(NOT_A_PROGRAM_FILE)),
    }
}

fn invalid(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
}

/// `argv` as one command line, quoted so that `CommandLineToArgvW` (and the
/// Microsoft C runtime, and Rust's own parser) give back exactly `argv`.
///
/// The first element is the program's name, which those parsers read without
/// backslash escapes: it may not hold a `"`, and is quoted when it holds a
/// space or a tab or is empty. Every other element is quoted when it is empty
/// or holds a space, a tab, a newline, a vertical tab or a `"`; inside quotes,
/// backslashes are doubled before a `"` and at the end, and a `"` is escaped.
/// A NUL anywhere is refused. UTF-16, without a terminator.
pub fn quote_argv(argv: &[OsString]) -> io::Result<Vec<u16>> {
    let mut line: Vec<u16> = Vec::new();
    let quote = u16::from(b'"');
    let backslash = u16::from(b'\\');
    for (index, argument) in argv.iter().enumerate() {
        let units = imp::wide_units(argument);
        if units.contains(&0) {
            return Err(invalid("an argument holds a NUL character"));
        }
        if index > 0 {
            line.push(u16::from(b' '));
        }
        let blank = |unit: &u16| matches!(*unit, 0x20 | 0x09 | 0x0a | 0x0b);
        if index == 0 {
            if units.contains(&quote) {
                return Err(invalid("a program name cannot hold a quote"));
            }
            let wrap = units.is_empty() || units.iter().any(|unit| matches!(*unit, 0x20 | 0x09));
            if wrap {
                line.push(quote);
            }
            line.extend_from_slice(&units);
            if wrap {
                line.push(quote);
            }
            continue;
        }
        if !units.is_empty() && !units.iter().any(|unit| blank(unit) || *unit == quote) {
            line.extend_from_slice(&units);
            continue;
        }
        line.push(quote);
        let mut backslashes = 0usize;
        for &unit in &units {
            if unit == backslash {
                backslashes += 1;
                continue;
            }
            if unit == quote {
                // 2n + 1 backslashes, then the quote.
                line.extend(std::iter::repeat_n(backslash, backslashes * 2 + 1));
            } else {
                line.extend(std::iter::repeat_n(backslash, backslashes));
            }
            backslashes = 0;
            line.push(unit);
        }
        // Before the closing quote, every backslash is doubled.
        line.extend(std::iter::repeat_n(backslash, backslashes * 2));
        line.push(quote);
    }
    Ok(line)
}

/// How `CommandLineToArgvW` splits `line`: the inverse [`quote_argv`] is held
/// to.
pub fn command_line_to_argv(line: &OsStr) -> io::Result<Vec<OsString>> {
    imp::command_line_to_argv(line)
}

/// An environment block for `CreateProcessW` with `CREATE_UNICODE_ENVIRONMENT`:
/// `NAME=value` entries, each NUL-terminated, sorted by name without regard to
/// case, and a final NUL. A name that is empty, holds `=` or a NUL, a value that
/// holds a NUL, or a name given twice (in any case) is refused.
pub fn environment_block(env: &[(OsString, OsString)]) -> io::Result<Vec<u16>> {
    let mut entries: Vec<(Vec<u16>, Vec<u16>, Vec<u16>)> = Vec::with_capacity(env.len());
    for (name, value) in env {
        let name_units = imp::wide_units(name);
        let value_units = imp::wide_units(value);
        if name_units.is_empty() || name_units.contains(&u16::from(b'=')) {
            return Err(invalid("an environment name is empty or holds '='"));
        }
        if name_units.contains(&0) || value_units.contains(&0) {
            return Err(invalid("an environment entry holds a NUL character"));
        }
        let folded: Vec<u16> = name
            .to_string_lossy()
            .to_uppercase()
            .encode_utf16()
            .collect();
        entries.push((folded, name_units, value_units));
    }
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    if entries.windows(2).any(|pair| pair[0].0 == pair[1].0) {
        return Err(invalid("an environment name is given twice"));
    }
    let mut block = Vec::new();
    for (_, name, value) in entries {
        block.extend_from_slice(&name);
        block.push(u16::from(b'='));
        block.extend_from_slice(&value);
        block.push(0);
    }
    if block.is_empty() {
        block.push(0);
    }
    block.push(0);
    Ok(block)
}

#[cfg(windows)]
mod imp {
    use std::ffi::{OsStr, OsString, c_void};
    use std::fs::File;
    use std::io;
    use std::mem::{MaybeUninit, size_of, size_of_val};
    use std::os::windows::ffi::OsStrExt;
    use std::os::windows::io::{AsRawHandle, FromRawHandle, OwnedHandle, RawHandle};
    use std::ptr;
    use std::time::Duration;

    use windows_sys::Win32::Foundation::{
        ERROR_MORE_DATA, GENERIC_READ, HANDLE, HANDLE_FLAG_INHERIT, INVALID_HANDLE_VALUE,
        LocalFree, SetHandleInformation, WAIT_OBJECT_0, WAIT_TIMEOUT,
    };
    use windows_sys::Win32::Storage::FileSystem::{
        CreateFileW, FILE_SHARE_READ, FILE_SHARE_WRITE, OPEN_EXISTING,
    };
    use windows_sys::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, JOB_OBJECT_LIMIT_ACTIVE_PROCESS,
        JOB_OBJECT_LIMIT_JOB_MEMORY, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
        JOBOBJECT_BASIC_ACCOUNTING_INFORMATION, JOBOBJECT_BASIC_PROCESS_ID_LIST,
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JobObjectBasicAccountingInformation,
        JobObjectBasicProcessIdList, JobObjectExtendedLimitInformation, QueryInformationJobObject,
        SetInformationJobObject, TerminateJobObject,
    };
    use windows_sys::Win32::System::Pipes::CreatePipe;
    use windows_sys::Win32::System::Threading::{
        CREATE_NO_WINDOW, CREATE_SUSPENDED, CREATE_UNICODE_ENVIRONMENT, CreateProcessW,
        DeleteProcThreadAttributeList, EXTENDED_STARTUPINFO_PRESENT, GetExitCodeProcess, INFINITE,
        InitializeProcThreadAttributeList, PROC_THREAD_ATTRIBUTE_HANDLE_LIST, PROCESS_INFORMATION,
        ResumeThread, STARTF_USESTDHANDLES, STARTUPINFOEXW, TerminateProcess,
        UpdateProcThreadAttribute, WaitForSingleObject,
    };
    use windows_sys::Win32::UI::Shell::CommandLineToArgvW;

    use std::path::Path;

    use super::{
        CRT_FDEV, CRT_FOPEN, CRT_FPIPE, Child, Extra, FdPipes, Input, JobLimits,
        crt_descriptor_block, environment_block,
    };
    use crate::wide::{from_wide, to_wide};

    #[derive(Debug)]
    pub(super) struct Handles {
        process: OwnedHandle,
        job: OwnedHandle,
    }

    pub(super) fn wide_units(text: &OsStr) -> Vec<u16> {
        text.encode_wide().collect()
    }

    fn owned(handle: HANDLE) -> io::Result<OwnedHandle> {
        if handle.is_null() || handle == INVALID_HANDLE_VALUE {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: `handle` was just returned by a successful Win32 call and is
        // owned by nothing else; the OwnedHandle closes it once.
        Ok(unsafe { OwnedHandle::from_raw_handle(handle as RawHandle) })
    }

    fn job(limits: JobLimits) -> io::Result<OwnedHandle> {
        // SAFETY: null security attributes and a null name make an unnamed job
        // with default security.
        let job = owned(unsafe { CreateJobObjectW(ptr::null(), ptr::null()) })?;
        let mut info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            | JOB_OBJECT_LIMIT_ACTIVE_PROCESS
            | JOB_OBJECT_LIMIT_JOB_MEMORY;
        info.BasicLimitInformation.ActiveProcessLimit = limits.active_processes;
        info.JobMemoryLimit = usize::try_from(limits.job_memory).unwrap_or(usize::MAX);
        // SAFETY: the job handle is open; `info` is the structure this
        // information class takes, of exactly the size passed.
        let ok = unsafe {
            SetInformationJobObject(
                job.as_raw_handle(),
                JobObjectExtendedLimitInformation,
                (&info as *const JOBOBJECT_EXTENDED_LIMIT_INFORMATION).cast::<c_void>(),
                size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
            )
        };
        if ok == 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(job)
    }

    /// An anonymous pipe: (read end, write end), neither inheritable.
    fn pipe() -> io::Result<(OwnedHandle, OwnedHandle)> {
        let mut read: HANDLE = ptr::null_mut();
        let mut write: HANDLE = ptr::null_mut();
        // SAFETY: both out-pointers are valid; null attributes make both ends
        // non-inheritable; 0 asks for the default buffer size.
        let ok = unsafe { CreatePipe(&mut read, &mut write, ptr::null(), 0) };
        if ok == 0 {
            return Err(io::Error::last_os_error());
        }
        Ok((owned(read)?, owned(write)?))
    }

    /// `NUL`, opened for reading, not inheritable.
    fn null_input() -> io::Result<OwnedHandle> {
        let name = to_wide(OsStr::new("NUL"))?;
        // SAFETY: `name` is NUL-terminated and outlives the call; null
        // attributes and template are allowed.
        owned(unsafe {
            CreateFileW(
                name.as_ptr(),
                GENERIC_READ,
                FILE_SHARE_READ | FILE_SHARE_WRITE,
                ptr::null(),
                OPEN_EXISTING,
                0,
                ptr::null_mut(),
            )
        })
    }

    fn set_inheritable(handle: &OwnedHandle, inheritable: bool) -> io::Result<()> {
        let flags = if inheritable { HANDLE_FLAG_INHERIT } else { 0 };
        // SAFETY: the handle is open; only its inherit flag changes.
        let ok =
            unsafe { SetHandleInformation(handle.as_raw_handle(), HANDLE_FLAG_INHERIT, flags) };
        if ok == 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }

    /// A `PROC_THREAD_ATTRIBUTE_HANDLE_LIST` over `handles`, freed on drop.
    struct AttributeList {
        buffer: Vec<u8>,
        // Kept alive for as long as the list points at it.
        handles: Box<[HANDLE]>,
    }

    impl AttributeList {
        fn handle_list(handles: Box<[HANDLE]>) -> io::Result<Self> {
            let mut size = 0usize;
            // SAFETY: a null list with a size pointer asks for the size needed;
            // that call fails by design (ERROR_INSUFFICIENT_BUFFER).
            unsafe { InitializeProcThreadAttributeList(ptr::null_mut(), 1, 0, &mut size) };
            let mut buffer = vec![0u8; size];
            // SAFETY: `buffer` has the size just asked for, for one attribute.
            let ok = unsafe {
                InitializeProcThreadAttributeList(buffer.as_mut_ptr().cast(), 1, 0, &mut size)
            };
            if ok == 0 {
                return Err(io::Error::last_os_error());
            }
            let mut list = Self { buffer, handles };
            // SAFETY: the list was initialised for one attribute; the value
            // points at the boxed handle array, which the list keeps alive, and
            // the size is that array's size in bytes.
            let ok = unsafe {
                UpdateProcThreadAttribute(
                    list.buffer.as_mut_ptr().cast(),
                    0,
                    PROC_THREAD_ATTRIBUTE_HANDLE_LIST as usize,
                    list.handles.as_ptr().cast::<c_void>(),
                    size_of_val(&*list.handles),
                    ptr::null_mut(),
                    ptr::null(),
                )
            };
            if ok == 0 {
                return Err(io::Error::last_os_error());
            }
            Ok(list)
        }

        fn as_ptr(&mut self) -> *mut c_void {
            self.buffer.as_mut_ptr().cast()
        }
    }

    impl Drop for AttributeList {
        fn drop(&mut self) {
            // SAFETY: the list was initialised by InitializeProcThreadAttributeList
            // (a list only exists once that succeeded) and is deleted once.
            unsafe { DeleteProcThreadAttributeList(self.buffer.as_mut_ptr().cast()) };
        }
    }

    /// Start `program` with `command_line` (UTF-16, NUL-terminated) as the
    /// module header says.
    pub(super) fn spawn(
        program: &Path,
        mut command_line: Vec<u16>,
        cwd: &Path,
        env: &[(OsString, OsString)],
        limits: JobLimits,
        input: Input,
        extra: Extra,
    ) -> io::Result<Child> {
        let program = to_wide(program.as_os_str())?;
        let cwd = to_wide(cwd.as_os_str())?;
        let environment = environment_block(env)?;
        let job = job(limits)?;
        let (stdout_read, stdout_write) = pipe()?;
        let (stderr_read, stderr_write) = pipe()?;
        // The child's stdin: `NUL`, or the read end of a pipe whose write end
        // this process keeps (never inheritable, so no child holds it).
        let (stdin, stdin_write) = match input {
            Input::Null => (null_input()?, None),
            Input::Pipe => {
                let (read, write) = pipe()?;
                (read, Some(write))
            }
        };
        // Descriptors 3 and 4: the read end of one pipe and the write end of
        // another, whose other ends this process keeps (never inheritable).
        let (fd_ends, fd_pipes) = match extra {
            Extra::Nothing => (Vec::new(), None),
            Extra::FdPipes => {
                let (child_reads, to_child) = pipe()?;
                let (from_child, child_writes) = pipe()?;
                let pipes = FdPipes {
                    to_child: File::from(to_child),
                    from_child: File::from(from_child),
                };
                (vec![child_reads, child_writes], Some(pipes))
            }
        };
        // The child's ends become inheritable for this one call, and the
        // handle list keeps the child from inheriting anything else.
        let mut child_ends = vec![&stdin, &stdout_write, &stderr_write];
        child_ends.extend(&fd_ends);
        for handle in &child_ends {
            set_inheritable(handle, true)?;
        }
        let raw: Vec<HANDLE> = child_ends
            .iter()
            .map(|handle| handle.as_raw_handle() as HANDLE)
            .collect();
        let mut list = AttributeList::handle_list(raw.clone().into_boxed_slice())?;
        // Descriptors past the standard three travel in the C runtime's block,
        // with the standard three before them as libuv writes it: `NUL` a
        // device, a pipe a pipe.
        let mut crt_block = if fd_ends.is_empty() {
            None
        } else {
            let entries: Vec<(u8, usize)> = raw
                .iter()
                .enumerate()
                .map(|(descriptor, handle)| {
                    let flags = match (descriptor, input) {
                        (0, Input::Null) => CRT_FOPEN | CRT_FDEV,
                        _ => CRT_FOPEN | CRT_FPIPE,
                    };
                    (flags, *handle as usize)
                })
                .collect();
            Some(crt_descriptor_block(&entries)?)
        };
        let mut startup = STARTUPINFOEXW::default();
        startup.StartupInfo.cb = size_of::<STARTUPINFOEXW>() as u32;
        startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
        startup.StartupInfo.hStdInput = raw[0];
        startup.StartupInfo.hStdOutput = raw[1];
        startup.StartupInfo.hStdError = raw[2];
        if let Some(block) = crt_block.as_mut() {
            // crt_descriptor_block refuses a block whose length a u16 cannot hold.
            startup.StartupInfo.cbReserved2 = block.len() as u16;
            startup.StartupInfo.lpReserved2 = block.as_mut_ptr();
        }
        startup.lpAttributeList = list.as_ptr();
        let mut information = MaybeUninit::<PROCESS_INFORMATION>::zeroed();
        // SAFETY: every string is NUL-terminated and outlives the call; the
        // command line is a mutable buffer of our own, as the API requires; the
        // environment is a complete UTF-16 block (CREATE_UNICODE_ENVIRONMENT);
        // the STARTUPINFOEXW, its attribute list and the C runtime's block
        // (when there is one, `cbReserved2` bytes long) are initialised and
        // alive; the handles they name are open; `information` is writable.
        let ok = unsafe {
            CreateProcessW(
                program.as_ptr(),
                command_line.as_mut_ptr(),
                ptr::null(),
                ptr::null(),
                1,
                CREATE_SUSPENDED
                    | CREATE_NO_WINDOW
                    | CREATE_UNICODE_ENVIRONMENT
                    | EXTENDED_STARTUPINFO_PRESENT,
                environment.as_ptr().cast::<c_void>(),
                cwd.as_ptr(),
                &startup.StartupInfo,
                information.as_mut_ptr(),
            )
        };
        let spawn_error = (ok == 0).then(io::Error::last_os_error);
        // The child's ends close here whatever happened; this process keeps
        // only its own ends.
        drop(list);
        drop(crt_block);
        drop((stdin, stdout_write, stderr_write, fd_ends));
        if let Some(error) = spawn_error {
            return Err(error);
        }
        // SAFETY: CreateProcessW succeeded, so it filled the structure.
        let information = unsafe { information.assume_init() };
        let process = owned(information.hProcess)?;
        let thread = owned(information.hThread)?;
        // SAFETY: both handles are open; the process is suspended and has run
        // no instruction yet.
        let assigned =
            unsafe { AssignProcessToJobObject(job.as_raw_handle(), process.as_raw_handle()) };
        if assigned == 0 {
            let error = io::Error::last_os_error();
            // SAFETY: the process handle is open; ending a suspended process
            // that never ran is the only cleanup there is.
            unsafe { TerminateProcess(process.as_raw_handle(), 1) };
            return Err(error);
        }
        // SAFETY: the thread handle is open; the thread was created suspended.
        if unsafe { ResumeThread(thread.as_raw_handle()) } == u32::MAX {
            let error = io::Error::last_os_error();
            // SAFETY: the job handle is open.
            unsafe { TerminateJobObject(job.as_raw_handle(), 1) };
            return Err(error);
        }
        drop(thread);
        Ok(Child {
            pid: information.dwProcessId,
            stdin: stdin_write.map(File::from),
            stdout: Some(File::from(stdout_read)),
            stderr: Some(File::from(stderr_read)),
            fd_pipes,
            inner: Handles { process, job },
        })
    }

    pub(super) fn wait(handles: &Handles, timeout: Option<Duration>) -> io::Result<Option<u32>> {
        let milliseconds = match timeout {
            None => INFINITE,
            Some(duration) => u32::try_from(duration.as_millis())
                .unwrap_or(INFINITE - 1)
                .min(INFINITE - 1),
        };
        // SAFETY: the process handle is open and owned by `handles`.
        match unsafe { WaitForSingleObject(handles.process.as_raw_handle(), milliseconds) } {
            WAIT_OBJECT_0 => {
                let mut code = 0u32;
                // SAFETY: the handle is open; `code` is writable.
                let ok = unsafe { GetExitCodeProcess(handles.process.as_raw_handle(), &mut code) };
                if ok == 0 {
                    return Err(io::Error::last_os_error());
                }
                Ok(Some(code))
            }
            WAIT_TIMEOUT => Ok(None),
            _ => Err(io::Error::last_os_error()),
        }
    }

    pub(super) fn kill_tree(handles: &Handles) -> io::Result<()> {
        // SAFETY: the job handle is open and owned by `handles`.
        let ok = unsafe { TerminateJobObject(handles.job.as_raw_handle(), 1) };
        if ok == 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }

    pub(super) fn active_processes(handles: &Handles) -> io::Result<u32> {
        let mut info = JOBOBJECT_BASIC_ACCOUNTING_INFORMATION::default();
        // SAFETY: the job handle is open; `info` is the structure this class
        // fills, of exactly the size passed.
        let ok = unsafe {
            QueryInformationJobObject(
                handles.job.as_raw_handle(),
                JobObjectBasicAccountingInformation,
                (&mut info as *mut JOBOBJECT_BASIC_ACCOUNTING_INFORMATION).cast::<c_void>(),
                size_of::<JOBOBJECT_BASIC_ACCOUNTING_INFORMATION>() as u32,
                ptr::null_mut(),
            )
        };
        if ok == 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(info.ActiveProcesses)
    }

    pub(super) fn process_ids(handles: &Handles) -> io::Result<Vec<u32>> {
        // The list: two 32-bit counts, then one pointer-sized id per process.
        // A buffer of usize keeps the ids aligned as the structure has them.
        let header = std::mem::offset_of!(JOBOBJECT_BASIC_PROCESS_ID_LIST, ProcessIdList)
            / size_of::<usize>();
        let mut room = 64usize;
        // The tree can grow between two calls; a few tries cover that.
        for _ in 0..8 {
            let mut buffer = vec![0usize; header + room];
            let bytes = u32::try_from(size_of_val(buffer.as_slice()))
                .map_err(|_| io::Error::other("the job's process list is too long"))?;
            // SAFETY: the job handle is open; the buffer is writable for
            // `bytes` bytes and aligned as JOBOBJECT_BASIC_PROCESS_ID_LIST.
            let ok = unsafe {
                QueryInformationJobObject(
                    handles.job.as_raw_handle(),
                    JobObjectBasicProcessIdList,
                    buffer.as_mut_ptr().cast::<c_void>(),
                    bytes,
                    ptr::null_mut(),
                )
            };
            let error = (ok == 0).then(io::Error::last_os_error);
            let counts: Vec<u8> = buffer[..header]
                .iter()
                .flat_map(|word| word.to_ne_bytes())
                .collect();
            let count = |at: usize| {
                u32::from_ne_bytes([counts[at], counts[at + 1], counts[at + 2], counts[at + 3]])
                    as usize
            };
            let (assigned, listed) = (count(0), count(4));
            match error {
                None => {
                    return Ok(buffer[header..header + listed.min(room)]
                        .iter()
                        .map(|id| *id as u32)
                        .collect());
                }
                Some(error) if error.raw_os_error() == Some(ERROR_MORE_DATA as i32) => {
                    room = (room * 2).max(assigned + 16);
                }
                Some(error) => return Err(error),
            }
        }
        Err(io::Error::other("the job's process list kept growing"))
    }

    pub(super) fn job_limits(handles: &Handles) -> io::Result<JobLimits> {
        let mut info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
        // SAFETY: the job handle is open; `info` is the structure this class
        // fills, of exactly the size passed.
        let ok = unsafe {
            QueryInformationJobObject(
                handles.job.as_raw_handle(),
                JobObjectExtendedLimitInformation,
                (&mut info as *mut JOBOBJECT_EXTENDED_LIMIT_INFORMATION).cast::<c_void>(),
                size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
                ptr::null_mut(),
            )
        };
        if ok == 0 {
            return Err(io::Error::last_os_error());
        }
        let flags = info.BasicLimitInformation.LimitFlags;
        let wanted = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            | JOB_OBJECT_LIMIT_ACTIVE_PROCESS
            | JOB_OBJECT_LIMIT_JOB_MEMORY;
        if flags & wanted != wanted {
            return Err(io::Error::other("the job lost one of its limits"));
        }
        Ok(JobLimits {
            active_processes: info.BasicLimitInformation.ActiveProcessLimit,
            job_memory: info.JobMemoryLimit as u64,
        })
    }

    pub(super) fn command_line_to_argv(line: &OsStr) -> io::Result<Vec<OsString>> {
        let wide = to_wide(line)?;
        let mut count = 0i32;
        // SAFETY: `wide` is NUL-terminated and outlives the call; `count` is
        // writable. The result is one LocalAlloc block, freed below.
        let array = unsafe { CommandLineToArgvW(wide.as_ptr(), &mut count) };
        if array.is_null() {
            return Err(io::Error::last_os_error());
        }
        let mut out = Vec::with_capacity(count.max(0) as usize);
        for index in 0..count.max(0) as usize {
            // SAFETY: the array holds `count` pointers to NUL-terminated
            // strings inside the same block, alive until LocalFree.
            let argument = unsafe { *array.add(index) };
            let mut length = 0usize;
            // SAFETY: as above; the string ends at its NUL.
            while unsafe { *argument.add(length) } != 0 {
                length += 1;
            }
            // SAFETY: `length` units were just read as part of this string.
            let units = unsafe { std::slice::from_raw_parts(argument, length) };
            out.push(from_wide(units));
        }
        // SAFETY: the block came from CommandLineToArgvW, which asks for
        // LocalFree; it is freed once and not used after.
        unsafe { LocalFree(array.cast()) };
        Ok(out)
    }
}

/// Portable stand-ins, used only by tests on other targets.
#[cfg(not(windows))]
mod imp {
    use std::ffi::{OsStr, OsString};
    use std::io;
    use std::time::Duration;

    use std::path::Path;

    use super::{Child, Extra, Input, JobLimits};

    #[derive(Debug)]
    pub(super) struct Handles;

    fn unsupported() -> io::Error {
        io::Error::new(
            io::ErrorKind::Unsupported,
            "Job Objects exist only on Windows",
        )
    }

    pub(super) fn wide_units(text: &OsStr) -> Vec<u16> {
        text.to_string_lossy().encode_utf16().collect()
    }

    pub(super) fn spawn(
        _program: &Path,
        _command_line: Vec<u16>,
        _cwd: &Path,
        _env: &[(OsString, OsString)],
        _limits: JobLimits,
        _input: Input,
        _extra: Extra,
    ) -> io::Result<Child> {
        Err(unsupported())
    }

    pub(super) fn wait(_: &Handles, _: Option<Duration>) -> io::Result<Option<u32>> {
        Err(unsupported())
    }

    pub(super) fn kill_tree(_: &Handles) -> io::Result<()> {
        Err(unsupported())
    }

    pub(super) fn active_processes(_: &Handles) -> io::Result<u32> {
        Err(unsupported())
    }

    pub(super) fn process_ids(_: &Handles) -> io::Result<Vec<u32>> {
        Err(unsupported())
    }

    pub(super) fn job_limits(_: &Handles) -> io::Result<JobLimits> {
        Err(unsupported())
    }

    pub(super) fn command_line_to_argv(_: &OsStr) -> io::Result<Vec<OsString>> {
        Err(unsupported())
    }
}

#[cfg(all(test, windows))]
mod tests {
    use super::*;

    fn os(items: &[&str]) -> Vec<OsString> {
        items.iter().map(OsString::from).collect()
    }

    fn round_trip(argv: &[OsString]) -> Vec<OsString> {
        let line = quote_argv(argv).unwrap();
        let line = OsString::from(String::from_utf16(&line).unwrap());
        command_line_to_argv(&line).unwrap()
    }

    /// CF3: `CommandLineToArgvW(quote(argv)) == argv` over a corpus.
    #[test]
    fn quoting_round_trips_through_command_line_to_argv() {
        let program = r"C:\Program Files\tool dir\tool.exe";
        let corpus: Vec<&str> = vec![
            "plain",
            "",
            " ",
            "two words",
            "tab\there",
            "new\nline",
            "vertical\x0btab",
            "\"",
            "\"\"",
            "a\"b",
            "\\",
            "\\\\",
            "a\\",
            "a\\\\",
            "a\\\"b",
            "a\\\\\"b",
            "\\\"",
            "end with space \\",
            "trailing backslashes \\\\\\",
            "C:\\path with space\\",
            "C:\\path\\",
            "--flag=value with spaces",
            "--%",
            "%PATH%",
            "$env:USERPROFILE",
            "x & y | z",
            "<>^()",
            "'single'",
            "Ünïcödé — dash",
            "emoji \u{1F600}",
            "\"quoted\" word \\\"escaped\\\"",
            "back\\slash\\in\\middle",
            "ends with quote\"",
            "\\\\server\\share",
            "a\"\"b",
        ];
        let mut argv = vec![OsString::from(program)];
        argv.extend(corpus.iter().map(OsString::from));
        assert_eq!(round_trip(&argv), argv);
        // Each argument alone, too.
        for argument in &corpus {
            let argv = vec![OsString::from("tool.exe"), OsString::from(argument)];
            assert_eq!(round_trip(&argv), argv, "{argument:?}");
        }
        // A program name with no space is not quoted; one with a space is.
        let line = quote_argv(&os(&["git", "status"])).unwrap();
        assert_eq!(String::from_utf16(&line).unwrap(), "git status");
        let line = quote_argv(&os(&["a b", "c"])).unwrap();
        assert_eq!(String::from_utf16(&line).unwrap(), "\"a b\" c");
    }

    #[test]
    fn a_joined_argv_does_not_round_trip() {
        // The mutant CF3 names: `argv.join(" ")`.
        let argv = os(&["tool.exe", "two words", "a\"b"]);
        let joined = OsString::from(
            argv.iter()
                .map(|a| a.to_string_lossy().into_owned())
                .collect::<Vec<_>>()
                .join(" "),
        );
        assert_ne!(command_line_to_argv(&joined).unwrap(), argv);
        assert_eq!(round_trip(&argv), argv);
    }

    #[test]
    fn quoting_refuses_what_cannot_be_represented() {
        assert!(quote_argv(&os(&["a\"b.exe"])).is_err());
        assert!(quote_argv(&os(&["tool.exe", "a\0b"])).is_err());
        assert_eq!(
            String::from_utf16(&quote_argv(&os(&["", "x"])).unwrap()).unwrap(),
            "\"\" x"
        );
    }

    #[test]
    fn the_environment_block_is_sorted_terminated_and_checked() {
        let block = environment_block(&[
            (OsString::from("b"), OsString::from("2")),
            (OsString::from("A"), OsString::from("1")),
            (OsString::from("C"), OsString::from("")),
        ])
        .unwrap();
        assert_eq!(String::from_utf16(&block).unwrap(), "A=1\0b=2\0C=\0\0");
        assert_eq!(
            String::from_utf16(&environment_block(&[]).unwrap()).unwrap(),
            "\0\0"
        );
        for bad in [
            vec![(OsString::from(""), OsString::from("x"))],
            vec![(OsString::from("A=B"), OsString::from("x"))],
            vec![(OsString::from("A"), OsString::from("x\0y"))],
            vec![
                (OsString::from("Path"), OsString::from("1")),
                (OsString::from("PATH"), OsString::from("2")),
            ],
        ] {
            assert_eq!(
                environment_block(&bad).unwrap_err().kind(),
                io::ErrorKind::InvalidInput
            );
        }
    }

    /// The C runtime's block as the UCRT's `ioinit.cpp` reads it: the count
    /// as an `int`, the flag bytes, then the handles, unaligned, in order.
    #[test]
    fn the_c_runtimes_block_is_a_count_the_flags_then_the_handles_packed() {
        let word = std::mem::size_of::<usize>();
        let block = crt_descriptor_block(&[
            (CRT_FOPEN | CRT_FDEV, 0x1234),
            (CRT_FOPEN | CRT_FPIPE, 0xABCD),
            (0, usize::MAX),
        ])
        .unwrap();
        let mut expected = vec![3, 0, 0, 0, 0x41, 0x09, 0x00];
        for handle in [0x1234usize, 0xABCD, usize::MAX] {
            expected.extend_from_slice(&handle.to_ne_bytes());
        }
        assert_eq!(block, expected);
        assert_eq!(block.len(), 4 + 3 + 3 * word);
        // The handles begin right after the flags, wherever that falls.
        assert_eq!(
            usize::from_ne_bytes(block[7..7 + word].try_into().unwrap()),
            0x1234
        );
        // Five descriptors, as spawn_with_fd_pipes passes them.
        let five = crt_descriptor_block(&[(CRT_FOPEN | CRT_FPIPE, 1); 5]).unwrap();
        assert_eq!(five.len(), 4 + 5 + 5 * word);
        assert_eq!(&five[..4], &5i32.to_ne_bytes());
        // The longest block cbReserved2 (a u16) can state is made; one entry
        // more is refused.
        let most = (usize::from(u16::MAX) - 4) / (1 + word);
        let longest = crt_descriptor_block(&vec![(CRT_FOPEN, 0usize); most]).unwrap();
        assert!(longest.len() <= usize::from(u16::MAX), "{}", longest.len());
        assert_eq!(
            crt_descriptor_block(&vec![(CRT_FOPEN, 0usize); most + 1])
                .unwrap_err()
                .kind(),
            io::ErrorKind::InvalidInput
        );
        assert_eq!(crt_descriptor_block(&[]).unwrap(), 0i32.to_ne_bytes());
    }

    #[test]
    fn a_relative_program_or_an_empty_argv_is_refused_before_any_call() {
        let env: Vec<(OsString, OsString)> = Vec::new();
        let request = SpawnRequest {
            program: Path::new("cmd.exe"),
            argv: &os(&["cmd.exe"]),
            cwd: Path::new("."),
            env: &env,
            limits: JobLimits::default(),
        };
        assert_eq!(
            spawn(&request).unwrap_err().kind(),
            io::ErrorKind::InvalidInput
        );
        let request = SpawnRequest {
            program: Path::new(r"C:\Windows\System32\cmd.exe"),
            argv: &[],
            ..request
        };
        assert_eq!(
            spawn(&request).unwrap_err().kind(),
            io::ErrorKind::InvalidInput
        );
    }

    /// X2b (spec §22.6): which names `CreateProcessW` would hand to `cmd.exe`.
    #[test]
    fn batch_files_are_known_by_their_trimmed_extension() {
        for name in [
            r"C:\w\run.bat",
            r"C:\w\run.cmd",
            r"C:\w\RUN.CMD",
            r"C:\w\run.Bat",
            r"C:\w\run.cmd.",
            r"C:\w\run.cmd .. ",
            r"C:\w\run.bat ",
        ] {
            assert!(is_batch_file(Path::new(name)), "{name:?}");
        }
        for name in [
            r"C:\w\run.exe",
            r"C:\w\run.cmd.exe",
            r"C:\w\run.com",
            r"C:\w\runcmd",
            r"C:\w\run.bat.txt",
            r"C:\w\.cmd\run.exe",
            r"C:\",
        ] {
            assert!(!is_batch_file(Path::new(name)), "{name:?}");
        }
    }

    /// §12's `.cmd` shim: the line `cmd.exe` runs, quoted only where it must
    /// be, and refused when `cmd.exe` could read it as anything but the
    /// batch file and its arguments.
    #[test]
    fn a_batch_files_line_quotes_what_cmd_would_read_and_refuses_what_it_would_expand() {
        let os = |items: &[&str]| -> Vec<OsString> { items.iter().map(OsString::from).collect() };
        let cmd = Path::new(r"C:\Windows\System32\cmd.exe");
        let npx = Path::new(r"C:\Program Files (x86)\nodejs\npx.cmd");
        let line = batch_command_line(
            cmd,
            npx,
            &os(&[
                "-y",
                "@scope/server-files",
                r"C:\My Docs\",
                "",
                "a&b",
                "k=v",
            ]),
        )
        .unwrap();
        assert_eq!(
            line,
            r#""C:\Windows\System32\cmd.exe" /d /v:off /s /c ""C:\Program Files (x86)\nodejs\npx.cmd" -y @scope/server-files "C:\My Docs\\" "" "a&b" "k=v"""#
        );
        assert_eq!(
            batch_command_line(cmd, npx, &[]).unwrap(),
            r#""C:\Windows\System32\cmd.exe" /d /v:off /s /c ""C:\Program Files (x86)\nodejs\npx.cmd"""#
        );
        for bad in ["%PATH%", "!x!", "say \"hi\"", "a\nb", "a\rb", "\u{7}"] {
            assert_eq!(
                batch_command_line(cmd, npx, &os(&[bad]))
                    .unwrap_err()
                    .kind(),
                io::ErrorKind::InvalidInput,
                "{bad:?}"
            );
        }
        assert!(batch_command_line(cmd, Path::new(r"C:\100%\x.cmd"), &[]).is_err());
        assert!(batch_command_line(cmd, Path::new("C:\\a\tb\\x.cmd"), &[]).is_err());
        // Not named cmd.exe, not a batch file, or relative: refused before
        // anything starts.
        let env: Vec<(OsString, OsString)> = Vec::new();
        let request = BatchRequest {
            cmd: Path::new(r"C:\Windows\System32\powershell.exe"),
            script: npx,
            args: &[],
            cwd: Path::new(r"C:\"),
            env: &env,
            limits: JobLimits::default(),
        };
        assert!(spawn_batch_with_input(&request).is_err());
        let not_batch = BatchRequest {
            cmd,
            script: Path::new(r"C:\w\server.exe"),
            ..request
        };
        assert!(spawn_batch_with_input(&not_batch).is_err());
        let relative = BatchRequest {
            cmd,
            script: Path::new(r"npx.cmd"),
            ..request
        };
        assert!(spawn_batch_with_input(&relative).is_err());
    }

    /// X2b: a batch file is never started, whatever case or trailing dots and
    /// spaces its name has, even when it exists and would run.
    /// Mutant: the batch-file refusal in `spawn` dropped.
    #[test]
    fn a_batch_file_is_never_started() {
        let dir = std::env::temp_dir().join(format!("lattice-sys-batch-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let script = dir.join("wrapper.cmd");
        std::fs::write(&script, "@exit /b 0\r\n").unwrap();
        // What cmd.exe needs to run at all, so only the refusal can stop it.
        let env: Vec<(OsString, OsString)> = ["SystemRoot", "windir", "ComSpec", "PATH"]
            .into_iter()
            .filter_map(|name| Some((OsString::from(name), std::env::var_os(name)?)))
            .collect();
        let spelled = [
            script.clone(),
            dir.join("WRAPPER.CMD"),
            dir.join("wrapper.cmd."),
            dir.join("wrapper.cmd . "),
        ];
        let mut outcomes = Vec::new();
        for program in &spelled {
            let argv = vec![program.clone().into_os_string(), OsString::from("a&b")];
            let outcome = spawn(&SpawnRequest {
                program,
                argv: &argv,
                cwd: &dir,
                env: &env,
                limits: JobLimits::default(),
            });
            println!("{}: {outcome:?}", program.display());
            outcomes.push(
                outcome
                    .map(|child| {
                        let _ = child.wait(Some(std::time::Duration::from_secs(5)));
                        child.pid()
                    })
                    .map_err(|error| error.to_string()),
            );
        }
        // The test's own temporary folder, and only it.
        let _ = std::fs::remove_dir_all(&dir);
        let refused = Err("a batch file is never started: cmd.exe would run it".to_owned());
        assert!(
            outcomes.iter().all(|outcome| *outcome == refused),
            "{outcomes:?}"
        );
    }

    /// X2b, LR3a (the verifier's probe, phaseHA/logs/VERIFY-probe-appexeclink):
    /// an app execution alias named `llama-server.exe` whose data names a
    /// `.cmd` is never started, nor is a symlink to that alias or a symlink
    /// named `.exe` whose target is a `.cmd`; a symlink to a real program
    /// still starts it (the positive control). Each refused start would run
    /// `cmd.exe` with `&echo.INJECTED>inj.txt` live, which the test looks for.
    /// Mutant: the `check_program_file` call in `spawn` dropped.
    #[test]
    fn an_alias_or_a_link_to_a_batch_file_is_never_started() {
        use crate::fs::seam::create_app_exec_link;
        let dir = std::env::temp_dir().join(format!("lattice-sys-alias-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let script = dir.join("wrapper.cmd");
        std::fs::write(
            &script,
            "@echo off
echo %*
",
        )
        .unwrap();
        let alias = dir.join("llama-server.exe");
        create_app_exec_link(&alias, &script).unwrap();
        let to_alias = dir.join("to-alias.exe");
        std::os::windows::fs::symlink_file(&alias, &to_alias).unwrap();
        let to_script = dir.join("to-script.exe");
        std::os::windows::fs::symlink_file(&script, &to_script).unwrap();
        let env: Vec<(OsString, OsString)> = ["SystemRoot", "windir", "ComSpec", "PATH"]
            .into_iter()
            .filter_map(|name| Some((OsString::from(name), std::env::var_os(name)?)))
            .collect();
        let start = |program: &Path, argv: Vec<OsString>| {
            spawn(&SpawnRequest {
                program,
                argv: &argv,
                cwd: &dir,
                env: &env,
                limits: JobLimits::default(),
            })
            .map(|child| {
                let _ = child.wait(Some(std::time::Duration::from_secs(5)));
                child.pid()
            })
            .map_err(|error| error.to_string())
        };
        let mut outcomes = Vec::new();
        for program in [&alias, &to_alias, &to_script] {
            let argv = vec![
                program.clone().into_os_string(),
                OsString::from("--alias"),
                OsString::from("qwen&echo.INJECTED>inj.txt"),
            ];
            outcomes.push((program.clone(), start(program, argv)));
        }
        let injected = dir.join("inj.txt").exists();
        // The positive control: a symlink to a real program starts it.
        let real = std::path::PathBuf::from(std::env::var_os("SystemRoot").unwrap())
            .join("System32")
            .join("findstr.exe");
        let to_real = dir.join("to-real.exe");
        std::os::windows::fs::symlink_file(&real, &to_real).unwrap();
        let control = start(
            &to_real,
            vec![to_real.clone().into_os_string(), OsString::from("/?")],
        );
        // The test's own temporary folder, and only it.
        let _ = std::fs::remove_dir_all(&dir);
        println!("{outcomes:?} injected={injected} control={control:?}");
        assert!(!injected, "cmd.exe ran an argument as a command");
        assert_eq!(
            outcomes[0].1,
            Err(NOT_A_PROGRAM_FILE.to_owned()),
            "the alias itself"
        );
        assert!(outcomes[1].1.is_err(), "a symlink to the alias");
        assert_eq!(
            outcomes[2].1,
            Err(A_BATCH_FILE.to_owned()),
            "a symlink to a batch file"
        );
        assert!(control.is_ok(), "{control:?}");
    }
}
