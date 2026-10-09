//! The native Lattice's operating-system calls: the only crate in the workspace
//! allowed `unsafe` (the chat core's spec §2.1).
//!
//! The chat core (`lattice-core`) keeps `#![deny(unsafe_code)]`. What it needs
//! from Windows that `std` does not offer is here, each as a small safe function
//! over one or two Win32 calls:
//!
//! - [`fs::file_identity`]: the 64-bit volume serial and 128-bit file id of an
//!   open file (`FILE_ID_INFO`), from which `WorkspaceId` is made;
//! - [`fs::file_times`]: an open file's creation, access, last-write and
//!   change times (`FILE_BASIC_INFO`), for the checkpoint's racily-clean cache,
//!   and [`fs::file_attributes`], its attributes, for the writer lease's
//!   checkout identity (Python's `st_mode`);
//! - [`fs::final_path`]: the path an open handle really names
//!   (`GetFinalPathNameByHandleW`);
//! - [`fs::open_no_follow`]: open a path without following a link at its last
//!   component, and read that link's target first (`FILE_FLAG_OPEN_REPARSE_POINT`,
//!   `FSCTL_GET_REPARSE_POINT`), so a link to a network share is refused before
//!   anything connects to it;
//! - [`fs::drive_type`]: whether a path's drive is remote (`GetDriveTypeW`);
//! - [`fs::replace_file`]: `ReplaceFileW` with a backup name, reporting its two
//!   partial failures (1176 and 1177) as their own errors;
//! - [`fs::move_no_replace`]: `MoveFileExW` without `MOVEFILE_REPLACE_EXISTING`
//!   (std's `rename` replaces an existing target);
//! - [`fs::create_private_dir`] and [`fs::create_private_file`]: a new folder
//!   or file with an explicit, protected DACL of SYSTEM, Administrators and
//!   the current user (`CreateDirectoryW`/`CreateFileW` with a security
//!   descriptor from SDDL), inheriting nothing; [`fs::read_dacl`] reads a
//!   DACL back (`GetNamedSecurityInfoW`);
//! - [`net::tcp_listeners_v4`] and [`net::tcp_listeners_v6`]: every TCP
//!   listener and its owning process id (`GetExtendedTcpTable`,
//!   `TCP_TABLE_OWNER_PID_LISTENER`), so the chat core sends nothing to a port
//!   its own child does not hold;
//! - [`desktop`]: the whole desktop for the agent's auto mode: the primary
//!   screen's pixels, the window at a point or in front and its executable,
//!   whether the focus is a classic password box, input as a person gives it
//!   (`SendInput`), and one global hotkey on a thread of its own;
//! - [`process::spawn`]: a program started with an explicit environment block,
//!   a `NUL` stdin, pipes, an explicit handle list and a Job Object that ends
//!   its whole tree (and [`process::Child::process_ids`], the tree's processes);
//!   [`process::spawn_with_fd_pipes`], the same with two more pipes as the
//!   child's C runtime descriptors 3 and 4 (Chromium's
//!   `--remote-debugging-pipe`), passed in the C runtime's inherited-handle
//!   block ([`process::crt_descriptor_block`]); [`process::quote_argv`] and
//!   [`process::command_line_to_argv`], the quoting `CreateProcessW` needs and
//!   its inverse.
//!
//! This crate deletes nothing: no function here removes a file or a directory
//! (the never-delete rule, ND1–ND4; the guard in `lattice-core` scans this
//! crate's sources too).
//!
//! It is not a port of anything; it ports no Python and no SDK code. Every
//! `unsafe` block carries a `SAFETY:` comment (`clippy::undocumented_unsafe_blocks`
//! is denied), and `tests/unsafe_guard.rs` holds every other crate of the
//! workspace to no `unsafe` at all, apart from one existing, listed site.
//!
//! On targets other than Windows each function has a portable fallback, used
//! only by tests; the chat core ships on Windows.

#![deny(unsafe_op_in_unsafe_fn)]
#![deny(clippy::undocumented_unsafe_blocks)]

pub mod cred;
pub mod desktop;
pub mod fs;
pub mod net;
pub mod process;

#[cfg(windows)]
mod wide;
