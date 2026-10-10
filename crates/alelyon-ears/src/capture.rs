//! Live audio from Windows: the microphone, or what the computer is playing (WASAPI loopback).
//!
//! Shared mode, so the ears never take a device away from anything else that is using it; the device's own
//! mix format, converted to f32 here and to 16 kHz by the caller. Capture runs on its own thread, wakes on
//! WASAPI's event (or every 20 ms, which also covers loopback streams whose event does not fire), and sends
//! each packet down a channel. Dropping the `Capture` stops it.
//!
//! A loopback stream delivers nothing while nothing plays. Silence is then supplied at the device's rate, so
//! time keeps moving for the speech detector and an utterance cut off by a paused video still ends.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Sender};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use windows::core::{Interface, PCWSTR};
use windows::Win32::Devices::FunctionDiscovery::PKEY_Device_FriendlyName;
use windows::Win32::Foundation::CloseHandle;
use windows::Win32::Media::Audio::{
    eAll, eCapture, eConsole, eRender, IAudioCaptureClient, IAudioClient, IMMDevice, IMMDeviceEnumerator, IMMEndpoint,
    MMDeviceEnumerator, AUDCLNT_BUFFERFLAGS_SILENT, AUDCLNT_SHAREMODE_SHARED, AUDCLNT_STREAMFLAGS_EVENTCALLBACK,
    AUDCLNT_STREAMFLAGS_LOOPBACK, DEVICE_STATE_ACTIVE, WAVEFORMATEX, WAVEFORMATEXTENSIBLE,
};
use windows::Win32::Media::KernelStreaming::KSDATAFORMAT_SUBTYPE_PCM;
use windows::Win32::Media::Multimedia::KSDATAFORMAT_SUBTYPE_IEEE_FLOAT;
use windows::Win32::System::Com::{CoCreateInstance, CoInitializeEx, CoTaskMemFree, CoUninitialize, CLSCTX_ALL, COINIT_MULTITHREADED, STGM_READ};
use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};

const WAVE_FORMAT_PCM: u16 = 1;
const WAVE_FORMAT_IEEE_FLOAT: u16 = 3;
const WAVE_FORMAT_EXTENSIBLE: u16 = 0xFFFE;
/// The shared-mode buffer asked for: 100 ms, in 100 ns units.
const BUFFER_HNS: i64 = 1_000_000;
/// Longest wait between looks at the device.
const WAKE_MS: u32 = 20;
/// A loopback stream silent for this long is treated as playing silence.
const LOOPBACK_IDLE: Duration = Duration::from_millis(100);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Endpoint {
    /// The default recording device.
    Microphone,
    /// The default playback device, heard from the inside.
    Loopback,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    F32,
    I16,
    I24,
    I32,
}

type Ready = mpsc::SyncSender<Result<(String, u32, usize), String>>;

pub struct Capture {
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
    name: String,
    rate: u32,
    channels: usize,
}

impl Capture {
    /// Open `endpoint` and start sending interleaved f32 blocks to `tx`. Fails, without waiting forever,
    /// when there is no such device or it will not open.
    pub fn start(endpoint: Endpoint, tx: Sender<Vec<f32>>) -> Result<Capture, String> {
        Self::start_on(endpoint, None, tx)
    }

    /// As `start`, on the active device whose name contains `device` (any case) rather than the default.
    pub fn start_on(endpoint: Endpoint, device: Option<&str>, tx: Sender<Vec<f32>>) -> Result<Capture, String> {
        let device = device.map(str::to_lowercase);
        let stop = Arc::new(AtomicBool::new(false));
        let (ready_tx, ready_rx) = mpsc::sync_channel::<Result<(String, u32, usize), String>>(1);
        let flag = stop.clone();
        let thread = std::thread::Builder::new()
            .name("ears-capture".into())
            .spawn(move || {
                // SAFETY: COM and WASAPI calls on this thread only, with the objects they return.
                let result = unsafe { run(endpoint, device.as_deref(), &tx, &flag, &ready_tx) };
                if let Err(e) = result {
                    let _ = ready_tx.try_send(Err(e));
                }
            })
            .map_err(|e| format!("cannot start the capture thread: {e}"))?;
        match ready_rx.recv_timeout(Duration::from_secs(10)) {
            Ok(Ok((name, rate, channels))) => Ok(Capture { stop, thread: Some(thread), name, rate, channels }),
            Ok(Err(e)) => {
                let _ = thread.join();
                Err(e)
            }
            Err(_) => {
                stop.store(true, Ordering::SeqCst);
                Err("the audio device did not open within 10 s".into())
            }
        }
    }

    pub fn device_name(&self) -> &str {
        &self.name
    }

    pub fn rate(&self) -> u32 {
        self.rate
    }

    pub fn channels(&self) -> usize {
        self.channels
    }
}

impl Drop for Capture {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

fn hr(context: &str, e: windows::core::Error) -> String {
    format!("{context}: {e}")
}

unsafe fn run(endpoint: Endpoint, named: Option<&str>, tx: &Sender<Vec<f32>>, stop: &AtomicBool, ready: &Ready) -> Result<(), String> {
    CoInitializeEx(None, COINIT_MULTITHREADED).ok().map_err(|e| hr("COM would not start", e))?;
    let result = capture_loop(endpoint, named, tx, stop, ready);
    CoUninitialize();
    result
}

unsafe fn capture_loop(endpoint: Endpoint, named: Option<&str>, tx: &Sender<Vec<f32>>, stop: &AtomicBool, ready: &Ready) -> Result<(), String> {
    let enumerator: IMMDeviceEnumerator =
        CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL).map_err(|e| hr("no audio device list", e))?;
    let (flow, what) = match endpoint {
        Endpoint::Microphone => (eCapture, "no microphone is set as the default recording device"),
        Endpoint::Loopback => (eRender, "no speakers are set as the default playback device"),
    };
    let device = match named {
        None => enumerator.GetDefaultAudioEndpoint(flow, eConsole).map_err(|e| hr(what, e))?,
        Some(part) => {
            let all = enumerator.EnumAudioEndpoints(flow, DEVICE_STATE_ACTIVE).map_err(|e| hr("no device list", e))?;
            let mut found = None;
            let mut names = Vec::new();
            for i in 0..all.GetCount().unwrap_or(0) {
                let Ok(d) = all.Item(i) else { continue };
                let name = device_name(&d);
                if found.is_none() && name.to_lowercase().contains(part) {
                    found = Some(d);
                }
                names.push(name);
            }
            found.ok_or_else(|| format!("no active device named like {part:?}; the active ones are: {}", names.join("; ")))?
        }
    };
    let name = device_name(&device);
    let client: IAudioClient = device.Activate(CLSCTX_ALL, None).map_err(|e| hr("the device would not open", e))?;
    let mix = client.GetMixFormat().map_err(|e| hr("the device has no mix format", e))?;
    let parsed = parse_format(mix);
    let mut flags = AUDCLNT_STREAMFLAGS_EVENTCALLBACK;
    if endpoint == Endpoint::Loopback {
        flags |= AUDCLNT_STREAMFLAGS_LOOPBACK;
    }
    let init = client.Initialize(AUDCLNT_SHAREMODE_SHARED, flags, BUFFER_HNS, 0, mix, None);
    CoTaskMemFree(Some(mix as *const _));
    let (kind, channels, rate) = parsed?;
    init.map_err(|e| hr("the device would not start a shared stream", e))?;
    let event = CreateEventW(None, false, false, PCWSTR::null()).map_err(|e| hr("no wake event", e))?;
    let outcome = (|| -> Result<(), String> {
        client.SetEventHandle(event).map_err(|e| hr("no wake event", e))?;
        let capture: IAudioCaptureClient = client.GetService().map_err(|e| hr("no capture service", e))?;
        client.Start().map_err(|e| hr("the stream would not start", e))?;
        let _ = ready.try_send(Ok((name, rate, channels)));
        let mut last = Instant::now();
        while !stop.load(Ordering::SeqCst) {
            WaitForSingleObject(event, WAKE_MS);
            let mut got = false;
            loop {
                let size = capture.GetNextPacketSize().map_err(|e| hr("the device stopped", e))?;
                if size == 0 {
                    break;
                }
                let mut data = std::ptr::null_mut();
                let mut frames = 0u32;
                let mut bflags = 0u32;
                capture
                    .GetBuffer(&mut data, &mut frames, &mut bflags, None, None)
                    .map_err(|e| hr("the device stopped", e))?;
                let count = frames as usize * channels;
                let block = if bflags & AUDCLNT_BUFFERFLAGS_SILENT.0 as u32 != 0 || data.is_null() {
                    vec![0.0; count]
                } else {
                    convert(data, count, kind)
                };
                capture.ReleaseBuffer(frames).map_err(|e| hr("the device stopped", e))?;
                if tx.send(block).is_err() {
                    return Ok(()); // nobody is listening any more
                }
                got = true;
            }
            if got {
                last = Instant::now();
            } else if endpoint == Endpoint::Loopback && last.elapsed() >= LOOPBACK_IDLE {
                let frames = (last.elapsed().as_secs_f64() * f64::from(rate)) as usize;
                last = Instant::now();
                if tx.send(vec![0.0; frames * channels]).is_err() {
                    return Ok(());
                }
            }
        }
        Ok(())
    })();
    let _ = client.Stop();
    let _ = CloseHandle(event);
    outcome
}

/// The mix format as (sample kind, channels, rate); refused by name if it is not one handled here.
unsafe fn parse_format(mix: *mut WAVEFORMATEX) -> Result<(Kind, usize, u32), String> {
    let base = std::ptr::read_unaligned(mix);
    let tag = base.wFormatTag;
    let bits = base.wBitsPerSample;
    let channels = usize::from(base.nChannels);
    let rate = base.nSamplesPerSec;
    let float = match tag {
        WAVE_FORMAT_IEEE_FLOAT => true,
        WAVE_FORMAT_PCM => false,
        WAVE_FORMAT_EXTENSIBLE => {
            let ext = std::ptr::read_unaligned(mix as *const WAVEFORMATEXTENSIBLE);
            let sub = ext.SubFormat;
            if sub == KSDATAFORMAT_SUBTYPE_IEEE_FLOAT {
                true
            } else if sub == KSDATAFORMAT_SUBTYPE_PCM {
                false
            } else {
                return Err(format!("the device mixes in an unknown sub-format {sub:?}"));
            }
        }
        other => return Err(format!("the device mixes in format tag {other:#06x}, which is not handled")),
    };
    let kind = match (float, bits) {
        (true, 32) => Kind::F32,
        (false, 16) => Kind::I16,
        (false, 24) => Kind::I24,
        (false, 32) => Kind::I32,
        (f, b) => {
            return Err(format!("the device mixes {} at {b} bits, which is not handled", if f { "float" } else { "integer PCM" }))
        }
    };
    if channels == 0 || rate == 0 {
        return Err(format!("the device reports {channels} channels at {rate} Hz"));
    }
    Ok((kind, channels, rate))
}

unsafe fn convert(data: *const u8, count: usize, kind: Kind) -> Vec<f32> {
    match kind {
        Kind::F32 => (0..count).map(|i| std::ptr::read_unaligned((data as *const f32).add(i))).collect(),
        Kind::I16 => (0..count).map(|i| f32::from(std::ptr::read_unaligned((data as *const i16).add(i))) / 32768.0).collect(),
        Kind::I24 => (0..count)
            .map(|i| {
                let p = data.add(i * 3);
                let v = i32::from_le_bytes([0, *p, *p.add(1), *p.add(2)]) >> 8;
                v as f32 / 8_388_608.0
            })
            .collect(),
        Kind::I32 => (0..count)
            .map(|i| (f64::from(std::ptr::read_unaligned((data as *const i32).add(i))) / 2_147_483_648.0) as f32)
            .collect(),
    }
}

unsafe fn device_name(device: &IMMDevice) -> String {
    device
        .OpenPropertyStore(STGM_READ)
        .and_then(|store| store.GetValue(&PKEY_Device_FriendlyName))
        .map(|v| v.to_string())
        .unwrap_or_else(|_| "an unnamed device".to_string())
}

unsafe fn id_of(device: &IMMDevice) -> String {
    match device.GetId() {
        Ok(p) => {
            let s = p.to_string().unwrap_or_default();
            CoTaskMemFree(Some(p.0 as *const _));
            s
        }
        Err(_) => String::new(),
    }
}

/// Every active recording and playback device, with the defaults marked: `angel-ears devices`.
pub fn print_devices() -> Result<(), String> {
    // SAFETY: COM calls on this thread, released before it returns.
    unsafe {
        CoInitializeEx(None, COINIT_MULTITHREADED).ok().map_err(|e| hr("COM would not start", e))?;
        let result = (|| -> Result<(), String> {
            let enumerator: IMMDeviceEnumerator =
                CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL).map_err(|e| hr("no audio device list", e))?;
            let default_in = enumerator.GetDefaultAudioEndpoint(eCapture, eConsole).ok().map(|d| id_of(&d));
            let default_out = enumerator.GetDefaultAudioEndpoint(eRender, eConsole).ok().map(|d| id_of(&d));
            let all = enumerator.EnumAudioEndpoints(eAll, DEVICE_STATE_ACTIVE).map_err(|e| hr("no device list", e))?;
            for i in 0..all.GetCount().unwrap_or(0) {
                let Ok(device) = all.Item(i) else { continue };
                let id = id_of(&device);
                let flow = device.cast::<IMMEndpoint>().and_then(|e| e.GetDataFlow()).ok();
                let (role, default) = if flow == Some(eCapture) { ("mic", &default_in) } else { ("out", &default_out) };
                let mark = if default.as_deref() == Some(id.as_str()) { "   (default)" } else { "" };
                println!("{role}  {}{mark}", device_name(&device));
            }
            Ok(())
        })();
        CoUninitialize();
        result
    }
}
