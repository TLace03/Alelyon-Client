//! The ears' own whisper-server: started with the speech model (`setup`), tied to this process, watched.
//!
//! The child is put in a Windows job object that kills it when the job's last handle closes, which happens
//! when the ears exit for any reason, a crash included. So the ears never leave a recogniser holding the
//! card's memory behind them. Its output goes to a log file, and it counts as ready once `GET /` answers.

use std::fs::File;
use std::net::SocketAddr;
use std::os::windows::io::AsRawHandle;
use std::os::windows::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use windows::core::PCWSTR;
use windows::Win32::Foundation::{CloseHandle, HANDLE};
use windows::Win32::System::JobObjects::{
    AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation, SetInformationJobObject,
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
};

use crate::whisper::WhisperServer;

const CREATE_NO_WINDOW: u32 = 0x0800_0000;

pub struct Spec {
    pub exe: PathBuf,
    pub model: PathBuf,
    pub port: u16,
    /// `-ng`: run on the processor only.
    pub cpu_only: bool,
    pub threads: Option<u32>,
    pub log: PathBuf,
}

pub struct RecognizerProcess {
    child: Child,
    job: HANDLE,
    pub addr: SocketAddr,
}

// SAFETY: the job handle is only closed in Drop, once.
unsafe impl Send for RecognizerProcess {}

impl RecognizerProcess {
    /// Start whisper-server and wait up to `patience` for it to answer.
    pub fn start(spec: &Spec, patience: Duration) -> Result<RecognizerProcess, String> {
        for (what, path) in [("whisper-server", &spec.exe), ("model", &spec.model)] {
            if !path.is_file() {
                return Err(format!("the {what} is missing: {}", path.display()));
            }
        }
        let addr: SocketAddr = format!("127.0.0.1:{}", spec.port).parse().map_err(|e| format!("{e}"))?;
        if WhisperServer::new(addr).alive() {
            return Err(format!("something already answers at {addr}; pass --attach {addr} to use it, or another --whisper-port"));
        }
        if let Some(dir) = spec.log.parent() {
            std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        }
        let log = File::create(&spec.log).map_err(|e| format!("{}: {e}", spec.log.display()))?;
        let log2 = log.try_clone().map_err(|e| e.to_string())?;
        let mut args: Vec<String> = vec![
            "-m".into(),
            spec.model.display().to_string(),
            "--host".into(),
            "127.0.0.1".into(),
            "--port".into(),
            spec.port.to_string(),
        ];
        if spec.cpu_only {
            args.push("-ng".into());
        }
        if let Some(t) = spec.threads {
            args.push("-t".into());
            args.push(t.to_string());
        }
        let job = make_job()?;
        let child = Command::new(&spec.exe)
            .args(&args)
            .current_dir(spec.exe.parent().unwrap_or(Path::new(".")))
            .stdin(Stdio::null())
            .stdout(Stdio::from(log))
            .stderr(Stdio::from(log2))
            .creation_flags(CREATE_NO_WINDOW)
            .spawn();
        let child = match child {
            Ok(c) => c,
            Err(e) => {
                // SAFETY: the job was created above and holds no process.
                unsafe {
                    let _ = CloseHandle(job);
                }
                return Err(format!("cannot start {}: {e}", spec.exe.display()));
            }
        };
        // SAFETY: both handles are live: the job just created, the child just started.
        let assigned = unsafe { AssignProcessToJobObject(job, HANDLE(child.as_raw_handle())) };
        let mut process = RecognizerProcess { child, job, addr };
        if let Err(e) = assigned {
            let _ = process.child.kill();
            return Err(format!("cannot tie whisper-server to the ears: {e}"));
        }
        let deadline = Instant::now() + patience;
        let probe = WhisperServer::new(addr);
        loop {
            if probe.alive() {
                return Ok(process);
            }
            if let Ok(Some(status)) = process.child.try_wait() {
                return Err(format!("whisper-server exited during start-up ({status}); see {}", spec.log.display()));
            }
            if Instant::now() >= deadline {
                return Err(format!("whisper-server did not answer within {patience:?}; see {}", spec.log.display()));
            }
            std::thread::sleep(Duration::from_millis(250));
        }
    }

    /// Still running?
    pub fn running(&mut self) -> bool {
        matches!(self.child.try_wait(), Ok(None))
    }
}

impl Drop for RecognizerProcess {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        // SAFETY: closed exactly once, here.
        unsafe {
            let _ = CloseHandle(self.job);
        }
    }
}

fn make_job() -> Result<HANDLE, String> {
    // SAFETY: a fresh unnamed job; the info struct outlives the call.
    unsafe {
        let job = CreateJobObjectW(None, PCWSTR::null()).map_err(|e| format!("no job object: {e}"))?;
        let mut info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        if let Err(e) = SetInformationJobObject(
            job,
            JobObjectExtendedLimitInformation,
            &info as *const _ as *const core::ffi::c_void,
            std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
        ) {
            let _ = CloseHandle(job);
            return Err(format!("cannot set the job's kill-on-close: {e}"));
        }
        Ok(job)
    }
}
