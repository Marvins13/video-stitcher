//! Live dashboard sink for `reco camera --dashboard-dir <DIR>`.
//!
//! Attached to the session as an extra encoder, so it receives every
//! stitched NV12 frame on its own background thread. Roughly once per
//! second it writes two files into `<DIR>` (atomically, via
//! write-to-temp + rename):
//!
//! - `preview.jpg`: the panorama downscaled to about 960 px wide.
//! - `status.json`: frame count, elapsed time, average and recent fps.
//!
//! When the recording ends normally `status.json` is rewritten with
//! `"state": "finished"`. A web dashboard (see `scripts/dashboard/`)
//! polls these files. Write failures are logged once and never abort
//! the recording: the dashboard is best-effort, the video file is not.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use reco_core::encoder::{EncodeError, Encoder, OutputFrame, PixelFormat};

/// Target width of the preview JPEG (actual width is `width / step`).
const PREVIEW_TARGET_WIDTH: u32 = 960;
/// How often preview and status are refreshed.
const WRITE_INTERVAL: Duration = Duration::from_secs(1);
const JPEG_QUALITY: u8 = 70;

pub struct DashboardSink {
    dir: PathBuf,
    output: String,
    frames: u64,
    start: Option<Instant>,
    last_write: Option<Instant>,
    /// (frame count, time) at the previous status write, for the recent fps.
    last_checkpoint: Option<(u64, Instant)>,
    fps_recent: f64,
    /// Output dimensions of the most recent frame (0x0 before the first).
    dims: (u32, u32),
    warned: bool,
}

impl DashboardSink {
    /// Create the sink, creating `dir` if needed and removing stale
    /// files from a previous run so the dashboard never shows them.
    pub fn new(dir: &Path, output: &str) -> anyhow::Result<Self> {
        std::fs::create_dir_all(dir)?;
        for name in ["preview.jpg", "status.json"] {
            let _ = std::fs::remove_file(dir.join(name));
        }
        Ok(Self {
            dir: dir.to_path_buf(),
            output: output.to_string(),
            frames: 0,
            start: None,
            last_write: None,
            last_checkpoint: None,
            fps_recent: 0.0,
            dims: (0, 0),
            warned: false,
        })
    }

    fn warn_once(&mut self, what: &str, err: &dyn std::fmt::Display) {
        if !self.warned {
            log::warn!("dashboard: {what} failed: {err} (further errors suppressed)");
            self.warned = true;
        }
    }

    fn write_status(&mut self, state: &str, now: Instant) {
        let (width, height) = self.dims;
        let elapsed = self.start.map_or(0.0, |s| (now - s).as_secs_f64());
        let fps_avg = if elapsed > 0.0 {
            self.frames as f64 / elapsed
        } else {
            0.0
        };
        let updated_unix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0.0, |d| d.as_secs_f64());
        let status = serde_json::json!({
            "state": state,
            "output": self.output,
            "width": width,
            "height": height,
            "frames": self.frames,
            "elapsed_s": round1(elapsed),
            "fps_avg": round1(fps_avg),
            "fps_recent": round1(self.fps_recent),
            "updated_unix": updated_unix,
        });
        let path = self.dir.join("status.json");
        if let Err(e) = write_atomic(&path, status.to_string().as_bytes()) {
            self.warn_once("writing status.json", &e);
        }
    }

    fn write_preview(&mut self, frame: &OutputFrame<'_>) {
        if frame.format != PixelFormat::Nv12 {
            return;
        }
        let Some((rgb, w, h)) = nv12_downscale_to_rgb(frame.data, frame.width, frame.height) else {
            self.warn_once("preview conversion", &"unexpected NV12 buffer size");
            return;
        };
        let mut jpeg = Vec::with_capacity(64 * 1024);
        let encoded = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut jpeg, JPEG_QUALITY)
            .encode(&rgb, w, h, image::ExtendedColorType::Rgb8);
        if let Err(e) = encoded {
            self.warn_once("JPEG encoding", &e);
            return;
        }
        if let Err(e) = write_atomic(&self.dir.join("preview.jpg"), &jpeg) {
            self.warn_once("writing preview.jpg", &e);
        }
    }
}

impl Encoder for DashboardSink {
    fn submit(&mut self, frame: OutputFrame<'_>) -> Result<(), EncodeError> {
        let now = Instant::now();
        self.start.get_or_insert(now);
        self.frames += 1;
        self.dims = (frame.width, frame.height);

        let due = self
            .last_write
            .is_none_or(|t| now.duration_since(t) >= WRITE_INTERVAL);
        if !due {
            return Ok(());
        }
        if let Some((prev_frames, prev_t)) = self.last_checkpoint {
            let dt = now.duration_since(prev_t).as_secs_f64();
            if dt > 0.0 {
                self.fps_recent = (self.frames - prev_frames) as f64 / dt;
            }
        }
        self.last_checkpoint = Some((self.frames, now));
        self.last_write = Some(now);

        self.write_preview(&frame);
        self.write_status("recording", now);
        Ok(())
    }

    fn finish(&mut self) -> Result<(), EncodeError> {
        self.write_status("finished", Instant::now());
        Ok(())
    }
}

fn round1(x: f64) -> f64 {
    (x * 10.0).round() / 10.0
}

/// Write `bytes` to `path` via a hidden temp file + rename, so readers
/// never see a half-written file.
fn write_atomic(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let tmp = path.with_file_name(format!(".{name}.tmp"));
    {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(bytes)?;
    }
    std::fs::rename(&tmp, path)
}

/// Downscale a tightly packed NV12 frame (BT.709 limited range, as
/// produced by reco's RGBA->NV12 shader) by an integer factor and
/// convert it to RGB8. Point sampling is good enough for a preview.
/// Returns `None` if the buffer is smaller than `width * height * 3/2`.
fn nv12_downscale_to_rgb(data: &[u8], width: u32, height: u32) -> Option<(Vec<u8>, u32, u32)> {
    let (w, h) = (width as usize, height as usize);
    if w == 0 || h == 0 || data.len() < w * h + w * h.div_ceil(2) {
        return None;
    }
    let step = (width / PREVIEW_TARGET_WIDTH).max(1) as usize;
    let (out_w, out_h) = (w / step, h / step);
    let (y_plane, uv_plane) = data.split_at(w * h);
    let mut rgb = Vec::with_capacity(out_w * out_h * 3);
    for oy in 0..out_h {
        let sy = oy * step;
        let y_row = &y_plane[sy * w..];
        let uv_row = &uv_plane[(sy / 2) * w..];
        for ox in 0..out_w {
            let sx = ox * step;
            let y = 1.1644 * (f32::from(y_row[sx]) - 16.0);
            let uv = (sx / 2) * 2;
            let cb = f32::from(uv_row[uv]) - 128.0;
            let cr = f32::from(uv_row[uv + 1]) - 128.0;
            let r = y + 1.7927 * cr;
            let g = y - 0.2132 * cb - 0.5329 * cr;
            let b = y + 2.1124 * cb;
            rgb.extend([clamp_u8(r), clamp_u8(g), clamp_u8(b)]);
        }
    }
    Some((rgb, out_w as u32, out_h as u32))
}

fn clamp_u8(x: f32) -> u8 {
    x.round().clamp(0.0, 255.0) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nv12(width: usize, height: usize, y: u8, cb: u8, cr: u8) -> Vec<u8> {
        let mut v = vec![y; width * height];
        for _ in 0..(width * height / 4) {
            v.extend([cb, cr]);
        }
        v
    }

    #[test]
    fn downscales_by_integer_factor() {
        let data = nv12(3840, 1080, 128, 128, 128);
        let (rgb, w, h) = nv12_downscale_to_rgb(&data, 3840, 1080).unwrap();
        assert_eq!((w, h), (960, 270));
        assert_eq!(rgb.len(), 960 * 270 * 3);
    }

    #[test]
    fn limited_range_black_white_and_red() {
        let (rgb, ..) = nv12_downscale_to_rgb(&nv12(4, 2, 16, 128, 128), 4, 2).unwrap();
        assert_eq!(&rgb[..3], &[0, 0, 0]);
        let (rgb, ..) = nv12_downscale_to_rgb(&nv12(4, 2, 235, 128, 128), 4, 2).unwrap();
        assert_eq!(&rgb[..3], &[255, 255, 255]);
        // BT.709 limited-range pure red: Y=63, Cb=102, Cr=240.
        let (rgb, ..) = nv12_downscale_to_rgb(&nv12(4, 2, 63, 102, 240), 4, 2).unwrap();
        assert!(rgb[0] > 250 && rgb[1] < 5 && rgb[2] < 5, "{:?}", &rgb[..3]);
    }

    #[test]
    fn rejects_short_buffer() {
        assert!(nv12_downscale_to_rgb(&[0; 10], 4, 4).is_none());
    }

    #[test]
    fn writes_status_and_preview_then_finished() {
        let dir = std::env::temp_dir().join(format!("reco-dash-test-{}", std::process::id()));
        let mut sink = DashboardSink::new(&dir, "out.mkv").unwrap();
        let data = nv12(64, 16, 128, 128, 128);
        sink.submit(OutputFrame {
            data: &data,
            width: 64,
            height: 16,
            format: PixelFormat::Nv12,
            pts_us: 0,
        })
        .unwrap();
        assert!(dir.join("preview.jpg").exists());
        let s: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("status.json")).unwrap()).unwrap();
        assert_eq!(s["state"], "recording");
        assert_eq!(s["frames"], 1);
        sink.finish().unwrap();
        let s: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("status.json")).unwrap()).unwrap();
        assert_eq!(s["state"], "finished");
        assert_eq!(s["width"], 64);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
