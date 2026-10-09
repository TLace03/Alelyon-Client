//! Who listens on a TCP port (the chat core's spec §22.6 LR7a). Not a
//! port.
//!
//! [`tcp_listeners_v4`] reads the system's IPv4 listener table with its owners
//! (`GetExtendedTcpTable` with `TCP_TABLE_OWNER_PID_LISTENER`). The managed
//! llama.cpp server counts as ready only when the `127.0.0.1:<port>` listener
//! it answers on belongs to the process the core started; a listener owned by
//! anything else is never sent a request. [`tcp_listeners_v6`] reads the IPv6
//! table the same way (the agent's browser's tests show its processes hold no
//! listener in either). Reading a table opens no socket and sends nothing.

use std::io;

/// One listening IPv4 TCP socket and the process that owns it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Listener {
    /// The local address, in the order it is written (`[127, 0, 0, 1]`).
    pub address: [u8; 4],
    pub port: u16,
    pub pid: u32,
}

/// One listening IPv6 TCP socket and the process that owns it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Listener6 {
    /// The local address, in the order it is written (`::1` ends in 1).
    pub address: [u8; 16],
    pub port: u16,
    pub pid: u32,
}

/// Every IPv4 TCP listener on this machine, with its owning process id.
pub fn tcp_listeners_v4() -> io::Result<Vec<Listener>> {
    imp::tcp_listeners_v4()
}

/// Every IPv6 TCP listener on this machine, with its owning process id.
pub fn tcp_listeners_v6() -> io::Result<Vec<Listener6>> {
    imp::tcp_listeners_v6()
}

/// A row's port: the low 16 bits of `dwLocalPort`, in network byte order.
#[cfg_attr(not(windows), allow(dead_code))]
fn row_port(local_port: u32) -> u16 {
    u16::from_be((local_port & 0xFFFF) as u16)
}

/// A row's address: `dwLocalAddr` as it lies in memory (network byte order).
#[cfg_attr(not(windows), allow(dead_code))]
fn row_address(local_address: u32) -> [u8; 4] {
    local_address.to_ne_bytes()
}

#[cfg(windows)]
mod imp {
    use std::ffi::c_void;
    use std::io;
    use std::mem::size_of;

    use windows_sys::Win32::NetworkManagement::IpHelper::{
        GetExtendedTcpTable, MIB_TCP6ROW_OWNER_PID, MIB_TCPROW_OWNER_PID,
        TCP_TABLE_OWNER_PID_LISTENER,
    };

    use super::{Listener, Listener6, row_address, row_port};

    /// `AF_INET` and `AF_INET6` (ws2def.h); their constants live in a
    /// windows-sys feature this crate does not take for two numbers.
    const AF_INET: u32 = 2;
    const AF_INET6: u32 = 23;
    /// `ERROR_INSUFFICIENT_BUFFER`.
    const INSUFFICIENT_BUFFER: u32 = 122;

    /// The listener table of `family` (`AF_INET` or `AF_INET6`) as rows of
    /// u32 units: the count first, then `row_units` units per row.
    fn listener_rows(family: u32, row_units: usize) -> io::Result<Vec<Vec<u32>>> {
        let mut size = 0u32;
        // The table can grow between the two calls; a few tries cover that.
        for _ in 0..8 {
            // u32 units keep the buffer aligned for the table's rows.
            let mut buffer = vec![0u32; (size as usize).div_ceil(4).max(1)];
            let mut length = (buffer.len() * 4) as u32;
            // SAFETY: the buffer is writable for `length` bytes and aligned
            // for MIB_TCPTABLE_OWNER_PID and MIB_TCP6TABLE_OWNER_PID (u32
            // fields and byte arrays); the size pointer is writable; the table
            // class and address family are ones the API documents.
            let status = unsafe {
                GetExtendedTcpTable(
                    buffer.as_mut_ptr().cast::<c_void>(),
                    &mut length,
                    0,
                    family,
                    TCP_TABLE_OWNER_PID_LISTENER,
                    0,
                )
            };
            if status == INSUFFICIENT_BUFFER {
                size = length;
                continue;
            }
            if status != 0 {
                return Err(io::Error::from_raw_os_error(status as i32));
            }
            let count = buffer[0] as usize;
            let rows = buffer.get(1..1 + count * row_units).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "the listener table is shorter than its count",
                )
            })?;
            return Ok(rows.chunks_exact(row_units).map(<[u32]>::to_vec).collect());
        }
        Err(io::Error::other("the listener table kept growing"))
    }

    pub(super) fn tcp_listeners_v4() -> io::Result<Vec<Listener>> {
        let row_units = size_of::<MIB_TCPROW_OWNER_PID>() / 4;
        // Each row: state, local address, local port, remote address,
        // remote port, owning process id.
        Ok(listener_rows(AF_INET, row_units)?
            .into_iter()
            .map(|row| Listener {
                address: row_address(row[1]),
                port: row_port(row[2]),
                pid: row[5],
            })
            .collect())
    }

    pub(super) fn tcp_listeners_v6() -> io::Result<Vec<Listener6>> {
        let row_units = size_of::<MIB_TCP6ROW_OWNER_PID>() / 4;
        // Each row: local address (4 units), its scope, local port, remote
        // address (4), its scope, remote port, state, owning process id.
        Ok(listener_rows(AF_INET6, row_units)?
            .into_iter()
            .map(|row| {
                let mut address = [0u8; 16];
                for (unit, bytes) in row[..4].iter().zip(address.chunks_exact_mut(4)) {
                    bytes.copy_from_slice(&unit.to_ne_bytes());
                }
                Listener6 {
                    address,
                    port: row_port(row[5]),
                    pid: row[13],
                }
            })
            .collect())
    }
}

/// Portable stand-in, used only by tests on other targets.
#[cfg(not(windows))]
mod imp {
    use std::io;

    use super::{Listener, Listener6};

    pub(super) fn tcp_listeners_v4() -> io::Result<Vec<Listener>> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "the owning-process table is read only on Windows",
        ))
    }

    pub(super) fn tcp_listeners_v6() -> io::Result<Vec<Listener6>> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "the owning-process table is read only on Windows",
        ))
    }
}

#[cfg(all(test, windows))]
mod tests {
    use super::*;

    #[test]
    fn a_row_reads_its_port_and_address_in_network_order() {
        // Port 8080 (0x1F90) as the table stores it on a little-endian host.
        assert_eq!(row_port(0x0000_901F), 8080);
        assert_eq!(
            row_address(u32::from_ne_bytes([127, 0, 0, 1])),
            [127, 0, 0, 1]
        );
    }

    /// LR7a: a listener this process opens on 127.0.0.1 is in the table,
    /// owned by this process; once it is closed it is gone.
    #[test]
    fn a_listener_of_this_process_is_listed_with_its_owner() {
        let listener = std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let port = listener.local_addr().unwrap().port();
        let me = std::process::id();
        let found: Vec<Listener> = tcp_listeners_v4()
            .unwrap()
            .into_iter()
            .filter(|row| row.port == port)
            .collect();
        assert_eq!(
            found,
            [Listener {
                address: [127, 0, 0, 1],
                port,
                pid: me
            }]
        );
        drop(listener);
        assert!(
            !tcp_listeners_v4()
                .unwrap()
                .iter()
                .any(|row| row.port == port && row.pid == me)
        );
    }

    /// The same for IPv6: a listener this process opens on `[::1]` is in the
    /// IPv6 table, owned by this process, and not in the IPv4 one; once it is
    /// closed it is gone.
    #[test]
    fn an_ipv6_listener_of_this_process_is_listed_with_its_owner() {
        let listener = std::net::TcpListener::bind(("::1", 0)).unwrap();
        let port = listener.local_addr().unwrap().port();
        let me = std::process::id();
        let found: Vec<Listener6> = tcp_listeners_v6()
            .unwrap()
            .into_iter()
            .filter(|row| row.port == port)
            .collect();
        let mut loopback = [0u8; 16];
        loopback[15] = 1;
        assert_eq!(
            found,
            [Listener6 {
                address: loopback,
                port,
                pid: me
            }]
        );
        assert!(
            !tcp_listeners_v4()
                .unwrap()
                .iter()
                .any(|row| row.port == port && row.pid == me)
        );
        drop(listener);
        assert!(
            !tcp_listeners_v6()
                .unwrap()
                .iter()
                .any(|row| row.port == port && row.pid == me)
        );
    }
}
