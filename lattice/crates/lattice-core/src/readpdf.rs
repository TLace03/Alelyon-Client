//! `read_pdf`: a page of a PDF of the folder, rendered by Windows' own PDF
//! renderer (`Windows.Data.Pdf`, part of Windows 10 and later) and shown to
//! the model as an image, as a screenshot is. Not a port.
//!
//! No crate is added and nothing leaves the PC: the page is rendered by
//! Windows PowerShell (`exec::run::powershell_path`, never searched for) with
//! a fixed script given as `-EncodedCommand`, the file's path written into
//! it as a literal, and its picture comes back as base64 on its output. A
//! render that takes longer than [`RENDER_WAIT`] is ended.

use std::ffi::OsString;
use std::path::Path;
use std::time::Duration;

/// The longest a page's render may take.
pub const RENDER_WAIT: Duration = Duration::from_secs(60);
/// The largest PDF read.
pub const MAX_PDF: u64 = 50 * 1024 * 1024;
/// The width a page is rendered at, in pixels.
pub const WIDTH: u32 = 1200;

/// The script: load the PDF, say its page count, render the page to PNG at
/// [`WIDTH`] and write it as base64.
fn script(path: &Path, page: u32) -> String {
    let literal = path.to_string_lossy().replace('\'', "''");
    format!(
        r#"$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$null = [Windows.Storage.StorageFile, Windows.Storage, ContentType = WindowsRuntime]
$null = [Windows.Data.Pdf.PdfDocument, Windows.Data.Pdf, ContentType = WindowsRuntime]
$null = [Windows.Storage.Streams.InMemoryRandomAccessStream, Windows.Storage.Streams, ContentType = WindowsRuntime]
$ext = [System.WindowsRuntimeSystemExtensions].GetMethods()
$op = $ext | Where-Object {{ $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' }} | Select-Object -First 1
$act = $ext | Where-Object {{ $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncAction' }} | Select-Object -First 1
function Await($o, [Type]$t) {{ $k = $op.MakeGenericMethod($t).Invoke($null, @($o)); $k.Wait() | Out-Null; $k.Result }}
$file = Await ([Windows.Storage.StorageFile]::GetFileFromPathAsync('{literal}')) ([Windows.Storage.StorageFile])
$doc = Await ([Windows.Data.Pdf.PdfDocument]::LoadFromFileAsync($file)) ([Windows.Data.Pdf.PdfDocument])
"pages=$($doc.PageCount)"
if ({page} -gt $doc.PageCount) {{ exit 0 }}
$p = $doc.GetPage([uint32]({page} - 1))
$o = New-Object Windows.Data.Pdf.PdfPageRenderOptions
$o.DestinationWidth = {WIDTH}
$s = New-Object Windows.Storage.Streams.InMemoryRandomAccessStream
$act.Invoke($null, @($p.RenderToStreamAsync($s, $o))).Wait() | Out-Null
$r = New-Object Windows.Storage.Streams.DataReader($s.GetInputStreamAt(0))
$null = Await ($r.LoadAsync([uint32]$s.Size)) ([uint32])
$b = New-Object byte[] $s.Size
$r.ReadBytes($b)
"png=" + [Convert]::ToBase64String($b)
"#
    )
}

/// A rendered page: the PDF's page count, and the page as a PNG in base64
/// (`None` when the PDF has fewer pages).
#[derive(Debug, PartialEq)]
pub struct Rendered {
    pub pages: u32,
    pub png: Option<String>,
}

/// Render `page` (from 1) of the PDF at `path` with Windows PowerShell at
/// `powershell`.
pub fn render(powershell: &Path, path: &Path, page: u32) -> Result<Rendered, String> {
    let args: Vec<OsString> = ["-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-EncodedCommand"]
        .into_iter()
        .map(OsString::from)
        .chain([OsString::from(crate::exec::run::encoded_command(&script(path, page)))])
        .collect();
    // Its output is read as it comes (a large page does not fill the pipe), within the render's minute.
    let text = crate::exec::run::capture(powershell, &args, RENDER_WAIT)
        .map_err(|why| format!("The page could not be rendered: {why}."))?;
    let pages = text
        .lines()
        .find_map(|l| l.trim().strip_prefix("pages="))
        .and_then(|n| n.trim().parse::<u32>().ok())
        .ok_or_else(|| "Windows could not read that PDF (it may be damaged or protected by a password).".to_owned())?;
    let png = text.lines().find_map(|l| l.trim().strip_prefix("png=")).map(str::to_owned);
    Ok(Rendered { pages, png })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A small PDF of `pages` pages, each saying its number.
    pub(crate) fn sample(pages: u32) -> Vec<u8> {
        let mut objs: Vec<Vec<u8>> = vec![
            b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
            format!(
                "<< /Type /Pages /Kids [{}] /Count {pages} >>",
                (0..pages).map(|i| format!("{} 0 R", 4 + 2 * i)).collect::<Vec<_>>().join(" ")
            )
            .into_bytes(),
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_vec(),
        ];
        for i in 0..pages {
            let content = format!("BT /F1 24 Tf 72 720 Td (Page {}) Tj ET", i + 1);
            objs.push(format!("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {} 0 R /Resources << /Font << /F1 3 0 R >> >> >>", 5 + 2 * i).into_bytes());
            objs.push(format!("<< /Length {} >>\nstream\n{content}\nendstream", content.len()).into_bytes());
        }
        let mut out = b"%PDF-1.4\n".to_vec();
        let mut offsets = Vec::new();
        for (i, o) in objs.iter().enumerate() {
            offsets.push(out.len());
            out.extend(format!("{} 0 obj\n", i + 1).bytes());
            out.extend(o);
            out.extend(b"\nendobj\n");
        }
        let xref = out.len();
        out.extend(format!("xref\n0 {}\n0000000000 65535 f \n", objs.len() + 1).bytes());
        for o in offsets {
            out.extend(format!("{o:010} 00000 n \n").bytes());
        }
        out.extend(format!("trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n", objs.len() + 1).bytes());
        out
    }

    /// Windows renders a page of a real PDF to a PNG, says how many pages
    /// it has, and renders nothing past the last.
    /// Mutants: the page number off by one; the count not read.
    #[test]
    fn windows_renders_a_page_of_a_pdf_to_a_picture() {
        let Some(powershell) = crate::exec::run::powershell_path(&crate::env::ProcessEnv) else {
            eprintln!("no Windows PowerShell: skipped");
            return;
        };
        let dir = crate::testkit::TempDir::new("read-pdf");
        let path = dir.path().join("two pages.pdf");
        std::fs::write(&path, sample(2)).unwrap();
        let second = render(&powershell, &path, 2).unwrap();
        assert_eq!(second.pages, 2);
        let png = second.png.expect("page 2 rendered");
        assert!(png.starts_with("iVBORw0KGgo"), "a PNG");
        let first = render(&powershell, &path, 1).unwrap().png.unwrap();
        assert_ne!(first, png, "each page its own picture");
        assert_eq!(render(&powershell, &path, 3).unwrap(), Rendered { pages: 2, png: None });
        std::fs::write(dir.path().join("broken.pdf"), b"%PDF-1.4 not really").unwrap();
        assert!(render(&powershell, &dir.path().join("broken.pdf"), 1).unwrap_err().contains("could not read"));
    }
}
