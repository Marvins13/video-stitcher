//! Live stereo camera capture via GStreamer.
//!
//! Builds a GStreamer pipeline per camera, pulls frames from
//! `appsink` in dedicated threads, and pairs them into stereo
//! frame pairs via a bounded channel.
//!
//! Supports two capture modes:
//! - **I420** (YUV420P): three separate planes, works everywhere
//! - **NV12**: Y + interleaved UV, native NVIDIA / Qualcomm ISP output -
//!   skips format conversion on Jetson and Qualcomm boards for lower latency
//!
//! Supported capture elements: `nvarguscamerasrc` (Jetson), `qtiqmmfsrc`
//! (Qualcomm camera service, e.g. QCS8300 / IQ-8275 boards), `avfvideosrc`
//! (macOS), `mfvideosrc` (Windows) and `v4l2src` (generic Linux).
//!
//! Left and right frames are paired by capture timestamp, so a camera that
//! starts late or drops a frame does not shift the pairing.

use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_app as gst_app;
use gstreamer_video as gst_video;
use reco_core::source::{
    FramePair, Nv12Data, Nv12FramePair, SourceError, SourceInfo, StereoFrame, YuvData,
};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock, mpsc};

/// Camera capture configuration.
#[derive(Debug, Clone)]
pub struct CameraConfig {
    /// Capture width in pixels.
    pub width: u32,
    /// Capture height in pixels.
    pub height: u32,
    /// Capture frame rate.
    pub fps: u32,
    /// Left camera device or sensor ID.
    ///
    /// - Jetson: sensor index as string ("0", "1")
    /// - Linux: V4L2 device path ("/dev/video0")
    pub left_device: String,
    /// Right camera device or sensor ID.
    pub right_device: String,
}

/// Capture pixel format for the GStreamer pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureFormat {
    /// YUV420P: three separate planes (Y, U, V). Works everywhere.
    I420,
    /// NV12: Y + interleaved UV. Native NVIDIA ISP output - avoids
    /// a format conversion step on Jetson.
    Nv12,
}

/// Validate a device string before interpolating it into a GStreamer pipeline.
///
/// Accepted formats per platform:
/// - Jetson (nvarguscamerasrc): numeric sensor ID only, e.g. `"0"`, `"1"`
/// - Qualcomm (qtiqmmfsrc): numeric camera ID only, e.g. `"0"`, `"1"`
/// - macOS (avfvideosrc): numeric device index only
/// - Windows (mfvideosrc): numeric device index only
/// - Linux V4L2: path matching `/dev/video<digits>`, e.g. `/dev/video0`
///
/// Returns `Err` with a descriptive message if the device string does not
/// match the expected pattern, preventing injection of arbitrary GStreamer
/// elements or shell metacharacters into the pipeline description.
fn validate_device_string(device: &str) -> Result<(), String> {
    if is_tegra()
        || is_qcom_camera_available()
        || cfg!(target_os = "macos")
        || cfg!(target_os = "windows")
    {
        validate_numeric_index(device)
    } else {
        validate_v4l2_path(device)
    }
}

/// Numeric index only (one or more digits, nothing else).
fn validate_numeric_index(device: &str) -> Result<(), String> {
    if device.chars().all(|c| c.is_ascii_digit()) && !device.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "invalid device string {device:?}: expected a numeric index (e.g. \"0\")"
        ))
    }
}

/// Linux V4L2: must be exactly `/dev/video<digits>`.
fn validate_v4l2_path(device: &str) -> Result<(), String> {
    let valid = device
        .strip_prefix("/dev/video")
        .is_some_and(|n| !n.is_empty() && n.chars().all(|c| c.is_ascii_digit()));
    if valid {
        Ok(())
    } else {
        Err(format!(
            "invalid device string {device:?}: expected a V4L2 path like \"/dev/video0\""
        ))
    }
}

/// Build the platform-appropriate GStreamer pipeline string.
///
/// Returns an error if `device` fails [`validate_device_string`].
fn build_pipeline_string(
    device: &str,
    width: u32,
    height: u32,
    fps: u32,
    format: CaptureFormat,
) -> Result<String, String> {
    validate_device_string(device)?;

    let fmt_str = match format {
        CaptureFormat::I420 => "I420",
        CaptureFormat::Nv12 => "NV12",
    };

    let pipeline = if is_tegra() {
        // Jetson: nvarguscamerasrc runs the full NVIDIA ISP
        // (debayer, AWB, AE, denoise). Output is NV12 in NVMM;
        // nvvidconv copies to system memory (and converts format if needed).
        format!(
            "nvarguscamerasrc sensor-id={device} ! \
             video/x-raw(memory:NVMM),width={width},height={height},format=NV12,framerate={fps}/1 ! \
             nvvidconv ! \
             video/x-raw,format={fmt_str} ! \
             appsink name=sink emit-signals=false sync=false"
        )
    } else if is_qcom_camera_available() {
        // Qualcomm: qtiqmmfsrc talks to the camera service (cam-server),
        // which runs the ISP and emits NV12 (with padded scanlines, see
        // `PlaneLayout`). I420 needs a CPU conversion; prefer NV12.
        let convert = match format {
            CaptureFormat::Nv12 => String::new(),
            CaptureFormat::I420 => "videoconvert ! video/x-raw,format=I420 ! ".to_string(),
        };
        format!(
            "qtiqmmfsrc camera={device} ! \
             video/x-raw,format=NV12,width={width},height={height},framerate={fps}/1 ! \
             {convert}appsink name=sink emit-signals=false sync=false"
        )
    } else if cfg!(target_os = "macos") {
        format!(
            "avfvideosrc device-index={device} ! \
             video/x-raw,width={width},height={height},framerate={fps}/1 ! \
             videoconvert ! \
             video/x-raw,format={fmt_str} ! \
             appsink name=sink emit-signals=false sync=false"
        )
    } else if cfg!(target_os = "windows") {
        format!(
            "mfvideosrc device-index={device} ! \
             video/x-raw,width={width},height={height},framerate={fps}/1 ! \
             videoconvert ! \
             video/x-raw,format={fmt_str} ! \
             appsink name=sink emit-signals=false sync=false"
        )
    } else {
        // Linux: generic V4L2
        format!(
            "v4l2src device={device} ! \
             video/x-raw,width={width},height={height},framerate={fps}/1 ! \
             videoconvert ! \
             video/x-raw,format={fmt_str} ! \
             appsink name=sink emit-signals=false sync=false"
        )
    };

    Ok(pipeline)
}

/// Detect if we're running on NVIDIA Tegra (Jetson platform).
fn is_tegra() -> bool {
    Path::new("/etc/nv_tegra_release").exists()
        || std::fs::read_to_string("/proc/device-tree/compatible")
            .unwrap_or_default()
            .contains("nvidia,tegra")
}

/// Detect a Qualcomm camera stack (`qtiqmmfsrc` from gstreamer1.0-plugins-qcom).
///
/// The result is cached: the plugin registry does not change at runtime.
pub fn is_qcom_camera_available() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        !is_tegra() && gst::init().is_ok() && gst::ElementFactory::find("qtiqmmfsrc").is_some()
    })
}

/// Byte layout of the planes inside a raw video buffer.
///
/// Many capture stacks pad rows (stride > width) or planes (e.g. Qualcomm
/// aligns NV12 scanlines to 32 rows, so 2160p has its UV plane at
/// `3840 * 2176`). The layout comes from the buffer's `GstVideoMeta`,
/// falling back to the negotiated caps, then to tight packing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PlaneLayout {
    /// Byte offset of each plane from the start of the buffer.
    offsets: [usize; 3],
    /// Bytes per row of each plane.
    strides: [usize; 3],
}

impl PlaneLayout {
    /// Tightly packed layout (no padding) for `format`.
    fn packed(format: CaptureFormat, width: u32, height: u32) -> Self {
        let (w, h) = (width as usize, height as usize);
        match format {
            CaptureFormat::I420 => Self {
                offsets: [0, w * h, w * h + (w / 2) * (h / 2)],
                strides: [w, w / 2, w / 2],
            },
            CaptureFormat::Nv12 => Self {
                offsets: [0, w * h, 0],
                strides: [w, w, 0],
            },
        }
    }

    /// Resolve the layout of `buffer`: `GstVideoMeta`, then caps, then packed.
    fn resolve(
        buffer: &gst::BufferRef,
        caps: Option<&gst::CapsRef>,
        format: CaptureFormat,
        width: u32,
        height: u32,
    ) -> Result<Self, SourceError> {
        let n_planes = match format {
            CaptureFormat::I420 => 3,
            CaptureFormat::Nv12 => 2,
        };
        let from_parts = |offsets: &[usize], strides: &[i32]| -> Result<Self, SourceError> {
            if offsets.len() < n_planes || strides.len() < n_planes {
                return Err(SourceError::Read {
                    reason: format!(
                        "video layout has {} planes, expected {n_planes}",
                        offsets.len().min(strides.len())
                    ),
                });
            }
            let mut layout = Self {
                offsets: [0; 3],
                strides: [0; 3],
            };
            for i in 0..n_planes {
                layout.offsets[i] = offsets[i];
                layout.strides[i] = usize::try_from(strides[i]).map_err(|_| SourceError::Read {
                    reason: format!("negative stride {} on plane {i}", strides[i]),
                })?;
            }
            Ok(layout)
        };

        if let Some(meta) = buffer.meta::<gst_video::VideoMeta>() {
            return from_parts(meta.offset(), meta.stride());
        }
        if let Some(info) = caps.and_then(|c| gst_video::VideoInfo::from_caps(c).ok()) {
            return from_parts(info.offset(), info.stride());
        }
        Ok(Self::packed(format, width, height))
    }
}

/// Copy `rows` rows of `row_bytes` bytes from a plane at `offset` with
/// `stride` bytes per row into a tightly packed `Vec`.
fn copy_plane(
    data: &[u8],
    offset: usize,
    stride: usize,
    row_bytes: usize,
    rows: usize,
    plane: &str,
) -> Result<Vec<u8>, SourceError> {
    if rows == 0 || row_bytes == 0 {
        return Ok(Vec::new());
    }
    if stride < row_bytes {
        return Err(SourceError::Read {
            reason: format!("{plane} plane stride {stride} < row size {row_bytes}"),
        });
    }
    let end = offset + stride * (rows - 1) + row_bytes;
    if data.len() < end {
        return Err(SourceError::Read {
            reason: format!("buffer too small for {plane} plane: {} < {end}", data.len()),
        });
    }
    if stride == row_bytes {
        return Ok(data[offset..end].to_vec());
    }
    let mut out = Vec::with_capacity(row_bytes * rows);
    for row in 0..rows {
        let start = offset + row * stride;
        out.extend_from_slice(&data[start..start + row_bytes]);
    }
    Ok(out)
}

/// Extract I420 planes from a GStreamer buffer.
fn extract_i420(
    data: &[u8],
    width: u32,
    height: u32,
    layout: &PlaneLayout,
) -> Result<YuvData, SourceError> {
    let (w, h) = (width as usize, height as usize);
    let (cw, ch) = (w / 2, h / 2);
    Ok(YuvData {
        y: copy_plane(data, layout.offsets[0], layout.strides[0], w, h, "Y")?,
        u: copy_plane(data, layout.offsets[1], layout.strides[1], cw, ch, "U")?,
        v: copy_plane(data, layout.offsets[2], layout.strides[2], cw, ch, "V")?,
    })
}

/// Extract NV12 planes from a GStreamer buffer.
///
/// NV12 layout: Y plane (width x height), then interleaved UV plane
/// (width x height/2). Output planes are tightly packed.
fn extract_nv12(
    data: &[u8],
    width: u32,
    height: u32,
    layout: &PlaneLayout,
) -> Result<Nv12Data, SourceError> {
    let (w, h) = (width as usize, height as usize);
    Ok(Nv12Data {
        y: copy_plane(data, layout.offsets[0], layout.strides[0], w, h, "Y")?,
        uv: copy_plane(data, layout.offsets[1], layout.strides[1], w, h / 2, "UV")?,
    })
}

/// Log the plane layout of a camera's first buffer (diagnoses padding issues).
fn log_first_buffer(label: &str, buffer: &gst::BufferRef, layout: &PlaneLayout) {
    log::info!(
        "{label}: first buffer {} bytes, video meta: {}, plane offsets {:?}, strides {:?}",
        buffer.size(),
        if buffer.meta::<gst_video::VideoMeta>().is_some() {
            "yes"
        } else {
            "no"
        },
        layout.offsets,
        layout.strides,
    );
}

/// A captured frame plus its capture time on the system clock.
struct Timestamped<T> {
    /// Absolute capture time in nanoseconds (pipeline base time + PTS),
    /// or `None` if the source did not timestamp the buffer.
    time_ns: Option<u64>,
    data: T,
}

/// Absolute capture time of `buffer` on the pipeline clock.
///
/// All capture pipelines are forced onto the shared system clock (see
/// [`build_capture_pipeline`]), so these times are comparable across
/// cameras.
fn capture_time_ns(pipeline: &gst::Pipeline, buffer: &gst::BufferRef) -> Option<u64> {
    let pts = buffer.pts()?;
    let base = pipeline.base_time()?;
    Some((base + pts).nseconds())
}

/// Pair left/right frames by capture timestamp and hand each pair to `emit`.
///
/// Two frames form a pair when their timestamps differ by at most
/// `tolerance_ns` (half a frame period); otherwise the older frame is
/// dropped. Frames without timestamps are paired in arrival order. If no
/// pair is found within `max_consecutive_drops` drops (e.g. the cameras
/// report times on unrelated clocks), pairing falls back to arrival order.
///
/// Returns when either channel closes or `emit` returns `false`.
fn pair_by_timestamp<T>(
    left_rx: &mpsc::Receiver<Timestamped<T>>,
    right_rx: &mpsc::Receiver<Timestamped<T>>,
    tolerance_ns: u64,
    max_consecutive_drops: u32,
    mut emit: impl FnMut(T, T) -> bool,
) {
    let (Ok(mut left), Ok(mut right)) = (left_rx.recv(), right_rx.recv()) else {
        return;
    };
    let mut use_timestamps = true;
    let mut consecutive_drops = 0u32;
    let (mut dropped_left, mut dropped_right) = (0u64, 0u64);
    let mut first_pair = true;

    loop {
        if use_timestamps
            && let (Some(lt), Some(rt)) = (left.time_ns, right.time_ns)
            && lt.abs_diff(rt) > tolerance_ns
        {
            consecutive_drops += 1;
            if consecutive_drops > max_consecutive_drops {
                log::warn!(
                    "camera pairing: no frames within {:.1} ms after {consecutive_drops} \
                     drops (left-right = {:.1} ms); pairing in arrival order instead",
                    tolerance_ns as f64 / 1e6,
                    (lt as f64 - rt as f64) / 1e6,
                );
                use_timestamps = false;
            } else {
                let replaced = if lt < rt {
                    dropped_left += 1;
                    left_rx.recv()
                } else {
                    dropped_right += 1;
                    right_rx.recv()
                };
                let Ok(next) = replaced else { break };
                if lt < rt {
                    left = next;
                } else {
                    right = next;
                }
                continue;
            }
        }

        consecutive_drops = 0;
        if first_pair {
            first_pair = false;
            match (left.time_ns, right.time_ns) {
                (Some(lt), Some(rt)) => log::info!(
                    "camera pairing: first pair, left-right offset {:.1} ms \
                     (dropped {dropped_left} left / {dropped_right} right frames to align)",
                    (lt as f64 - rt as f64) / 1e6
                ),
                _ => log::info!("camera pairing: no capture timestamps, pairing in arrival order"),
            }
        }
        if !emit(left.data, right.data) {
            break;
        }
        match (left_rx.recv(), right_rx.recv()) {
            (Ok(l), Ok(r)) => (left, right) = (l, r),
            _ => break,
        }
    }

    log::info!(
        "camera pairing stopped: dropped {dropped_left} left / {dropped_right} right frames"
    );
}

/// Pairing tolerance: half a frame period at `fps`.
fn pairing_tolerance_ns(fps: u32) -> u64 {
    1_000_000_000 / (2 * u64::from(fps.max(1)))
}

/// Give up on timestamp pairing after this many seconds without a match.
fn pairing_max_drops(fps: u32) -> u32 {
    2 * fps.max(1)
}

/// Build and start a GStreamer pipeline, returning the pipeline and appsink.
fn build_capture_pipeline(
    device: &str,
    label: &'static str,
    width: u32,
    height: u32,
    fps: u32,
    format: CaptureFormat,
) -> Result<(gst::Pipeline, gst_app::AppSink), String> {
    if let Err(e) = gst::init() {
        return Err(format!("{label} GStreamer init failed: {e}"));
    }

    let pipeline_str = build_pipeline_string(device, width, height, fps, format)?;
    log::info!("{label} pipeline: {pipeline_str}");

    // Create and start one camera pipeline at a time. qtiqmmfsrc fails
    // (set_state -> FAILURE, sometimes for both cameras) when two capture
    // threads create their pipelines concurrently; serialising costs only
    // milliseconds and is harmless for the other capture elements.
    static START_LOCK: Mutex<()> = Mutex::new(());
    let _start_guard = START_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    let pipeline = gst::parse::launch(&pipeline_str)
        .map_err(|e| format!("{label} pipeline parse: {e}"))?
        .downcast::<gst::Pipeline>()
        .map_err(|_| format!("{label}: not a pipeline"))?;

    let appsink = pipeline
        .by_name("sink")
        .ok_or_else(|| format!("{label}: appsink not found"))?
        .downcast::<gst_app::AppSink>()
        .map_err(|_| format!("{label}: element is not an AppSink"))?;

    appsink.set_max_buffers(2);
    appsink.set_drop(true);

    // Run every camera pipeline on the shared system clock so that
    // base_time + PTS is comparable between the left and right camera.
    pipeline.use_clock(Some(&gst::SystemClock::obtain()));

    pipeline
        .set_state(gst::State::Playing)
        .map_err(|e| format!("{label} set_state(Playing): {e}"))?;

    Ok((pipeline, appsink))
}

/// Spawn a GStreamer capture thread for one camera (I420 output).
fn spawn_capture_thread(
    device: String,
    label: &'static str,
    width: u32,
    height: u32,
    fps: u32,
) -> mpsc::Receiver<Timestamped<YuvData>> {
    let (tx, rx) = mpsc::sync_channel::<Timestamped<YuvData>>(2);

    std::thread::Builder::new()
        .name(format!("capture_{label}"))
        .spawn(move || {
            let (pipeline, appsink) = match build_capture_pipeline(
                &device,
                label,
                width,
                height,
                fps,
                CaptureFormat::I420,
            ) {
                Ok(p) => p,
                Err(e) => {
                    log::error!("{e}");
                    return;
                }
            };

            let mut layout_logged = false;
            loop {
                let sample = match appsink.pull_sample() {
                    Ok(s) => s,
                    Err(_) => break,
                };

                let Some(buffer) = sample.buffer() else {
                    log::error!("{label}: sample has no buffer");
                    break;
                };

                let layout = match PlaneLayout::resolve(
                    buffer,
                    sample.caps(),
                    CaptureFormat::I420,
                    width,
                    height,
                ) {
                    Ok(l) => l,
                    Err(e) => {
                        log::error!("{label}: {e}");
                        break;
                    }
                };
                if !layout_logged {
                    layout_logged = true;
                    log_first_buffer(label, buffer, &layout);
                }
                let time_ns = capture_time_ns(&pipeline, buffer);

                let Ok(map) = buffer.map_readable() else {
                    log::error!("{label}: buffer map failed");
                    break;
                };

                match extract_i420(map.as_slice(), width, height, &layout) {
                    Ok(data) => {
                        if tx.send(Timestamped { time_ns, data }).is_err() {
                            break;
                        }
                    }
                    Err(e) => {
                        log::error!("{label}: {e}");
                        break;
                    }
                }
            }

            let _ = pipeline.set_state(gst::State::Null);
        })
        .expect("spawn capture thread");

    rx
}

/// Spawn a GStreamer capture thread for one camera (NV12 output).
///
/// The stop signal allows graceful shutdown: send EOS, wait for pipeline
/// to reach Null state, then exit. This prevents Argus teardown crashes.
fn spawn_nv12_capture_thread(
    device: String,
    label: &'static str,
    width: u32,
    height: u32,
    fps: u32,
    stop: Arc<AtomicBool>,
) -> mpsc::Receiver<Timestamped<Nv12Data>> {
    let (tx, rx) = mpsc::sync_channel::<Timestamped<Nv12Data>>(2);

    std::thread::Builder::new()
        .name(format!("capture_{label}"))
        .spawn(move || {
            let (pipeline, appsink) = match build_capture_pipeline(
                &device,
                label,
                width,
                height,
                fps,
                CaptureFormat::Nv12,
            ) {
                Ok(p) => p,
                Err(e) => {
                    log::error!("{e}");
                    return;
                }
            };

            let mut layout_logged = false;
            loop {
                if stop.load(Ordering::Relaxed) {
                    break;
                }

                let sample = match appsink.pull_sample() {
                    Ok(s) => s,
                    Err(_) => break,
                };

                let Some(buffer) = sample.buffer() else {
                    log::error!("{label}: sample has no buffer");
                    break;
                };

                let layout = match PlaneLayout::resolve(
                    buffer,
                    sample.caps(),
                    CaptureFormat::Nv12,
                    width,
                    height,
                ) {
                    Ok(l) => l,
                    Err(e) => {
                        log::error!("{label}: {e}");
                        break;
                    }
                };
                if !layout_logged {
                    layout_logged = true;
                    log_first_buffer(label, buffer, &layout);
                }
                let time_ns = capture_time_ns(&pipeline, buffer);

                let Ok(map) = buffer.map_readable() else {
                    log::error!("{label}: buffer map failed");
                    break;
                };

                match extract_nv12(map.as_slice(), width, height, &layout) {
                    Ok(data) => {
                        if tx.send(Timestamped { time_ns, data }).is_err() {
                            break;
                        }
                    }
                    Err(e) => {
                        log::error!("{label}: {e}");
                        break;
                    }
                }
            }

            // Graceful shutdown: send EOS, then transition to Null
            log::info!("{label}: sending EOS for graceful shutdown");
            pipeline.send_event(gst::event::Eos::new());
            let _ = pipeline.set_state(gst::State::Null);
            // Wait for state change to complete
            let _ = pipeline.state(gst::ClockTime::from_seconds(2));
            log::info!("{label}: pipeline stopped");
        })
        .expect("spawn capture thread");

    rx
}

/// Stereo camera source using GStreamer (I420 output).
///
/// Each camera runs in its own thread pulling frames from appsink.
/// A pairing thread zips left+right into `FramePair`s and sends
/// them through a bounded channel.
pub struct GstreamerCameraSource {
    rx: mpsc::Receiver<FramePair>,
    info: SourceInfo,
}

impl GstreamerCameraSource {
    /// Open a stereo camera source with threaded capture.
    pub fn open(config: &CameraConfig) -> Result<Self, SourceError> {
        gst::init().map_err(|e| SourceError::Init {
            path: format!("{} + {}", config.left_device, config.right_device),
            reason: format!("GStreamer init: {e}"),
        })?;

        let left_rx = spawn_capture_thread(
            config.left_device.clone(),
            "left",
            config.width,
            config.height,
            config.fps,
        );
        let right_rx = spawn_capture_thread(
            config.right_device.clone(),
            "right",
            config.width,
            config.height,
            config.fps,
        );

        let (tx, rx) = mpsc::sync_channel::<FramePair>(2);
        let tolerance_ns = pairing_tolerance_ns(config.fps);
        let max_drops = pairing_max_drops(config.fps);

        std::thread::Builder::new()
            .name("capture_pair".into())
            .spawn(move || {
                pair_by_timestamp(
                    &left_rx,
                    &right_rx,
                    tolerance_ns,
                    max_drops,
                    |left, right| tx.send(FramePair { left, right }).is_ok(),
                );
            })
            .expect("spawn pairing thread");

        let info = SourceInfo {
            width: config.width,
            height: config.height,
            fps: config.fps as f64,
            fps_rational: None,
            total_frames: None,
        };

        log::info!(
            "Camera source ready: {}x{} @ {} fps (I420, threaded)",
            config.width,
            config.height,
            config.fps
        );

        Ok(Self { rx, info })
    }
}

impl reco_core::source::FrameSource for GstreamerCameraSource {
    fn info(&self) -> SourceInfo {
        self.info.clone()
    }

    fn next_frame(&mut self) -> Result<Option<StereoFrame>, SourceError> {
        match self.rx.recv() {
            Ok(pair) => Ok(Some(StereoFrame::Yuv420p(pair))),
            Err(_) => Ok(None),
        }
    }
}

/// Stereo camera source using GStreamer (NV12 output).
///
/// Like `GstreamerCameraSource` but captures in NV12 format, which is
/// the native output of NVIDIA's ISP. This avoids the NV12->I420
/// conversion that nvvidconv would otherwise perform, saving CPU time
/// and memory bandwidth on Jetson.
///
/// Implements graceful shutdown via `Drop` to avoid Argus teardown crashes.
pub struct GstreamerNv12CameraSource {
    rx: mpsc::Receiver<Nv12FramePair>,
    info: SourceInfo,
    stop: Arc<AtomicBool>,
}

impl GstreamerNv12CameraSource {
    /// Open a stereo NV12 camera source with threaded capture.
    pub fn open(config: &CameraConfig) -> Result<Self, SourceError> {
        gst::init().map_err(|e| SourceError::Init {
            path: format!("{} + {}", config.left_device, config.right_device),
            reason: format!("GStreamer init: {e}"),
        })?;

        let stop = Arc::new(AtomicBool::new(false));

        let left_rx = spawn_nv12_capture_thread(
            config.left_device.clone(),
            "left",
            config.width,
            config.height,
            config.fps,
            stop.clone(),
        );
        let right_rx = spawn_nv12_capture_thread(
            config.right_device.clone(),
            "right",
            config.width,
            config.height,
            config.fps,
            stop.clone(),
        );

        let (tx, rx) = mpsc::sync_channel::<Nv12FramePair>(2);
        let tolerance_ns = pairing_tolerance_ns(config.fps);
        let max_drops = pairing_max_drops(config.fps);

        std::thread::Builder::new()
            .name("capture_pair".into())
            .spawn(move || {
                pair_by_timestamp(
                    &left_rx,
                    &right_rx,
                    tolerance_ns,
                    max_drops,
                    |left, right| tx.send(Nv12FramePair { left, right }).is_ok(),
                );
            })
            .expect("spawn pairing thread");

        let info = SourceInfo {
            width: config.width,
            height: config.height,
            fps: config.fps as f64,
            fps_rational: None,
            total_frames: None,
        };

        log::info!(
            "Camera source ready: {}x{} @ {} fps (NV12, threaded)",
            config.width,
            config.height,
            config.fps
        );

        Ok(Self { rx, info, stop })
    }

    /// Signal capture threads to stop.
    pub fn stop(&self) {
        self.stop.store(true, Ordering::Relaxed);
    }

    /// Source metadata.
    pub fn info(&self) -> SourceInfo {
        self.info.clone()
    }

    /// Get the next stereo NV12 frame pair, or `None` if the source is exhausted.
    pub fn next_pair(&mut self) -> Result<Option<Nv12FramePair>, SourceError> {
        match self.rx.recv() {
            Ok(pair) => Ok(Some(pair)),
            Err(_) => Ok(None),
        }
    }
}

impl reco_core::source::FrameSource for GstreamerNv12CameraSource {
    fn info(&self) -> reco_core::source::SourceInfo {
        self.info.clone()
    }

    fn next_frame(
        &mut self,
    ) -> Result<Option<reco_core::source::StereoFrame>, reco_core::source::SourceError> {
        self.next_pair()
            .map(|opt| opt.map(reco_core::source::StereoFrame::Nv12))
    }
}

impl Drop for GstreamerNv12CameraSource {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        // Drain any pending frames to unblock capture threads
        while self.rx.try_recv().is_ok() {}
        // Give capture threads time to send EOS and reach Null state
        std::thread::sleep(std::time::Duration::from_millis(500));
    }
}

// ── NVMM zero-copy camera source (Jetson only) ─────────────────────

#[cfg(target_os = "linux")]
pub use nvmm_source::GstreamerNvmmCameraSource;
#[cfg(target_os = "linux")]
pub use nvmm_source::NvmmFramePair;

#[cfg(target_os = "linux")]
mod nvmm_source {
    use super::*;
    use crate::gstreamer::nvmm::{self, NvmmFrameInfo};

    /// Stereo pair of NVMM frame metadata (DMA-buf fds + surface ptrs).
    pub struct NvmmFramePair {
        pub left: NvmmFrameInfo,
        pub right: NvmmFrameInfo,
    }

    fn build_nvmm_pipeline_string(device: &str, config: &CameraConfig) -> Result<String, String> {
        let width = config.width;
        let height = config.height;
        let fps = config.fps;

        // Test override: RECO_NVMM_LEFT / RECO_NVMM_RIGHT env vars pipe a video
        // file through nvv4l2decoder to produce identical NVMM NV12 output.
        let env_key = if device == "0" {
            "RECO_NVMM_LEFT"
        } else {
            "RECO_NVMM_RIGHT"
        };
        if let Ok(path) = std::env::var(env_key) {
            log::info!("[nvmm] {env_key}={path} (file override, bypassing camera)");
            return Ok(format!(
                "filesrc location={path} ! qtdemux ! h264parse ! nvv4l2decoder ! \
                 video/x-raw(memory:NVMM),format=NV12 ! \
                 appsink name=sink emit-signals=false sync=false max-buffers=4 drop=true"
            ));
        }

        validate_device_string(device)?;
        Ok(format!(
            "nvarguscamerasrc sensor-id={device} ! \
             video/x-raw(memory:NVMM),width={width},height={height},format=NV12,framerate={fps}/1 ! \
             appsink name=sink emit-signals=false sync=false max-buffers=4 drop=true"
        ))
    }

    fn spawn_nvmm_capture_thread(
        device: String,
        label: &'static str,
        config: CameraConfig,
        stop: Arc<AtomicBool>,
    ) -> (mpsc::Receiver<NvmmFrameInfo>, mpsc::SyncSender<()>) {
        let (info_tx, info_rx) = mpsc::sync_channel::<NvmmFrameInfo>(2);
        let (release_tx, release_rx) = mpsc::sync_channel::<()>(4);

        std::thread::Builder::new()
            .name(format!("nvmm_{label}"))
            .spawn(move || {
                let pipeline_str = match build_nvmm_pipeline_string(&device, &config) {
                    Ok(s) => s,
                    Err(e) => {
                        log::error!("[{label}] NVMM pipeline build: {e}");
                        return;
                    }
                };

                log::info!("[{label}] NVMM pipeline: {pipeline_str}");

                let pipeline = gst::parse::launch(&pipeline_str)
                    .expect("NVMM pipeline parse")
                    .downcast::<gst::Pipeline>()
                    .expect("not a pipeline");
                let appsink = pipeline
                    .by_name("sink")
                    .expect("appsink not found")
                    .downcast::<gst_app::AppSink>()
                    .expect("not an AppSink");
                appsink.set_max_buffers(4);
                appsink.set_drop(true);
                pipeline
                    .set_state(gst::State::Playing)
                    .expect("NVMM pipeline start");

                log::info!("[{label}] NVMM capture started (sensor={device})");

                let mut prev_sample: Option<gst::Sample> = None;

                loop {
                    if stop.load(Ordering::Relaxed) {
                        break;
                    }

                    let sample = match appsink.pull_sample() {
                        Ok(s) => s,
                        Err(_) => break,
                    };

                    let buffer = match sample.buffer() {
                        Some(b) => b,
                        None => {
                            log::error!("[{label}] NVMM sample has no buffer");
                            break;
                        }
                    };

                    let map = match buffer.map_readable() {
                        Ok(m) => m,
                        Err(e) => {
                            log::error!("[{label}] NVMM buffer map failed: {e}");
                            break;
                        }
                    };

                    let frame_info =
                        match unsafe { nvmm::extract_nvmm_frame_info(map.as_slice().as_ptr()) } {
                            Ok(info) => info,
                            Err(e) => {
                                log::error!("[{label}] NVMM extract failed: {e}");
                                break;
                            }
                        };

                    drop(map);

                    // Release the previous sample now that the render thread
                    // has had a full frame to use the prior DMA-buf.
                    if let Some(prev) = prev_sample.take() {
                        drop(prev);
                    }

                    if info_tx.send(frame_info).is_err() {
                        break;
                    }

                    prev_sample = Some(sample);

                    // Wait for the render thread to signal it's done importing
                    // this frame's DMA-buf before we let the next pull_sample
                    // potentially recycle the buffer.
                    let _ = release_rx.recv();
                }

                let _ = pipeline.set_state(gst::State::Null);
                log::info!("[{label}] NVMM capture stopped");
            })
            .expect("spawn NVMM capture thread");

        (info_rx, release_tx)
    }

    /// Stereo NVMM camera source for zero-copy Vulkan import + NvBufSurfTransform.
    ///
    /// Each call to `next_pair()` returns DMA-buf fds and NvBufSurface pointers
    /// for both cameras. The caller must call `release_previous()` after GPU
    /// work (rendering + detection) is submitted, before calling `next_pair()`
    /// again - this signals the capture threads to release the old GstSample
    /// and pull the next frame.
    pub struct GstreamerNvmmCameraSource {
        left_rx: mpsc::Receiver<NvmmFrameInfo>,
        right_rx: mpsc::Receiver<NvmmFrameInfo>,
        left_release: mpsc::SyncSender<()>,
        right_release: mpsc::SyncSender<()>,
        info: SourceInfo,
        stop: Arc<AtomicBool>,
    }

    impl GstreamerNvmmCameraSource {
        /// Open a stereo NVMM camera source.
        pub fn open(config: &CameraConfig) -> Result<Self, SourceError> {
            gst::init().map_err(|e| SourceError::Init {
                path: format!("{} + {}", config.left_device, config.right_device),
                reason: format!("GStreamer init: {e}"),
            })?;

            let stop = Arc::new(AtomicBool::new(false));

            let (left_rx, left_release) = spawn_nvmm_capture_thread(
                config.left_device.clone(),
                "left",
                config.clone(),
                stop.clone(),
            );
            let (right_rx, right_release) = spawn_nvmm_capture_thread(
                config.right_device.clone(),
                "right",
                config.clone(),
                stop.clone(),
            );

            let info = SourceInfo {
                width: config.width,
                height: config.height,
                fps: config.fps as f64,
                fps_rational: None,
                total_frames: None,
            };

            log::info!(
                "NVMM stereo source: {}x{} @ {}fps, zero-copy DMA-buf + NvBufSurfTransform",
                config.width,
                config.height,
                config.fps,
            );

            Ok(Self {
                left_rx,
                right_rx,
                left_release,
                right_release,
                info,
                stop,
            })
        }

        /// Get the next stereo NVMM frame pair.
        ///
        /// Blocks until both cameras deliver a frame. Returns `None` on EOS.
        pub fn next_pair(&mut self) -> Result<Option<NvmmFramePair>, SourceError> {
            let left = match self.left_rx.recv() {
                Ok(info) => info,
                Err(_) => return Ok(None),
            };
            let right = match self.right_rx.recv() {
                Ok(info) => info,
                Err(_) => return Ok(None),
            };
            Ok(Some(NvmmFramePair { left, right }))
        }

        /// Signal capture threads that GPU work on the current frame is done.
        ///
        /// Must be called after rendering + detection are submitted for the
        /// frame returned by the most recent `next_pair()`.
        pub fn release_previous(&self) {
            let _ = self.left_release.send(());
            let _ = self.right_release.send(());
        }

        /// Source metadata.
        pub fn info(&self) -> SourceInfo {
            self.info.clone()
        }

        /// Stop capture gracefully.
        pub fn stop(&self) {
            self.stop.store(true, Ordering::Relaxed);
        }
    }

    impl Drop for GstreamerNvmmCameraSource {
        fn drop(&mut self) {
            self.stop.store(true, Ordering::Relaxed);
            // Drain pending frames and send release signals to unblock threads
            while self.left_rx.try_recv().is_ok() {}
            while self.right_rx.try_recv().is_ok() {}
            let _ = self.left_release.send(());
            let _ = self.right_release.send(());
            std::thread::sleep(std::time::Duration::from_millis(500));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The platform dispatch in `validate_device_string` depends on the
    // machine (Tegra, Qualcomm camera plugin, OS), so the two rule sets
    // are tested directly.

    /// Numeric strings are valid indices (Jetson, Qualcomm, macOS, Windows).
    #[test]
    fn numeric_index_accepted() {
        assert!(validate_numeric_index("0").is_ok());
        assert!(validate_numeric_index("1").is_ok());
        assert!(validate_numeric_index("12").is_ok());
    }

    /// Non-numeric strings and injection attempts are rejected as indices.
    #[test]
    fn non_numeric_index_rejected() {
        assert!(validate_numeric_index("").is_err());
        assert!(validate_numeric_index("cam0").is_err());
        assert!(validate_numeric_index("0 ! fakesrc").is_err());
        assert!(validate_numeric_index("/dev/video0").is_err());
    }

    /// Valid V4L2 paths are accepted.
    #[test]
    fn v4l2_path_accepted() {
        assert!(validate_v4l2_path("/dev/video0").is_ok());
        assert!(validate_v4l2_path("/dev/video1").is_ok());
        assert!(validate_v4l2_path("/dev/video10").is_ok());
    }

    /// Injection attempts and malformed paths are rejected as V4L2 paths.
    #[test]
    fn v4l2_injection_rejected() {
        assert!(validate_v4l2_path("").is_err());
        assert!(validate_v4l2_path("/dev/video").is_err()); // no trailing digit
        assert!(validate_v4l2_path("/dev/video0 ! fakesrc").is_err());
        assert!(validate_v4l2_path("0").is_err()); // numeric-only not valid for V4L2
        assert!(validate_v4l2_path("/dev/video0a").is_err());
        assert!(validate_v4l2_path("/dev/../etc/passwd").is_err());
        assert!(validate_v4l2_path("video0").is_err());
    }

    /// Qualcomm pipeline uses qtiqmmfsrc with the numeric camera ID.
    #[test]
    fn qcom_pipeline_string() {
        if !is_qcom_camera_available() {
            return; // not a Qualcomm camera stack
        }
        let p = build_pipeline_string("1", 3840, 2160, 30, CaptureFormat::Nv12).unwrap();
        assert!(p.starts_with("qtiqmmfsrc camera=1 ! "), "{p}");
        assert!(p.contains("format=NV12,width=3840,height=2160,framerate=30/1"));
        assert!(!p.contains("videoconvert"));
        assert!(build_pipeline_string("0 ! fakesrc", 64, 64, 30, CaptureFormat::Nv12).is_err());
    }

    /// Build a synthetic NV12 buffer with padded rows/scanlines. Each
    /// valid byte encodes its plane and position; padding is 0xEE.
    fn padded_nv12(w: usize, h: usize, stride: usize, scanlines: usize) -> Vec<u8> {
        let mut buf = vec![0xEE; stride * scanlines + stride * (h / 2)];
        for r in 0..h {
            for c in 0..w {
                buf[r * stride + c] = ((r * 7 + c) % 200) as u8;
            }
        }
        let uv_off = stride * scanlines;
        for r in 0..h / 2 {
            for c in 0..w {
                buf[uv_off + r * stride + c] = 200 + ((r + c) % 50) as u8;
            }
        }
        buf
    }

    /// Qualcomm-style layout: UV plane after 32-row-aligned scanlines.
    #[test]
    fn nv12_padded_scanlines() {
        let (w, h) = (16usize, 6usize);
        let buf = padded_nv12(w, h, w, 8);
        let layout = PlaneLayout {
            offsets: [0, w * 8, 0],
            strides: [w, w, 0],
        };
        let nv12 = extract_nv12(&buf, w as u32, h as u32, &layout).unwrap();
        assert_eq!(nv12.y.len(), w * h);
        assert_eq!(nv12.uv.len(), w * h / 2);
        assert!(nv12.y.iter().all(|&b| b < 200));
        assert!(nv12.uv.iter().all(|&b| (200..250).contains(&b)));
        assert_eq!(nv12.uv[0], 200);
    }

    /// Row padding (stride > width) is stripped from every row.
    #[test]
    fn nv12_padded_stride() {
        let (w, h, stride) = (10usize, 4usize, 16usize);
        let buf = padded_nv12(w, h, stride, h);
        let layout = PlaneLayout {
            offsets: [0, stride * h, 0],
            strides: [stride, stride, 0],
        };
        let nv12 = extract_nv12(&buf, w as u32, h as u32, &layout).unwrap();
        assert_eq!(nv12.y.len(), w * h);
        assert!(!nv12.y.contains(&0xEE));
        assert!(!nv12.uv.contains(&0xEE));
        assert_eq!(nv12.y[w], 7); // row 1, col 0
    }

    /// Tightly packed layouts round-trip, and short buffers are rejected.
    #[test]
    fn packed_layouts() {
        let (w, h) = (8u32, 4u32);
        let i420 = vec![1u8; 8 * 4 * 3 / 2];
        let yuv =
            extract_i420(&i420, w, h, &PlaneLayout::packed(CaptureFormat::I420, w, h)).unwrap();
        assert_eq!((yuv.y.len(), yuv.u.len(), yuv.v.len()), (32, 8, 8));

        let nv12 = vec![1u8; 8 * 4 * 3 / 2];
        let layout = PlaneLayout::packed(CaptureFormat::Nv12, w, h);
        assert!(extract_nv12(&nv12, w, h, &layout).is_ok());
        assert!(extract_nv12(&nv12[..40], w, h, &layout).is_err());
    }

    /// Layout is read from GstVideoMeta when the buffer carries one.
    #[test]
    fn layout_from_video_meta() {
        gst::init().unwrap();
        let (w, h) = (16u32, 6u32);
        let mut buffer = gst::Buffer::with_size(16 * 8 + 16 * 3).unwrap();
        gst_video::VideoMeta::add_full(
            buffer.get_mut().unwrap(),
            gst_video::VideoFrameFlags::empty(),
            gst_video::VideoFormat::Nv12,
            w,
            h,
            &[0, 16 * 8],
            &[16, 16],
        )
        .unwrap();
        let layout = PlaneLayout::resolve(&buffer, None, CaptureFormat::Nv12, w, h).unwrap();
        assert_eq!(layout.offsets[..2], [0, 128]);
        assert_eq!(layout.strides[..2], [16, 16]);

        // No meta, no caps -> tightly packed.
        let plain = gst::Buffer::with_size(16 * 9).unwrap();
        let layout = PlaneLayout::resolve(&plain, None, CaptureFormat::Nv12, w, h).unwrap();
        assert_eq!(layout, PlaneLayout::packed(CaptureFormat::Nv12, w, h));
    }

    const FRAME_NS: u64 = 33_333_333;

    /// Feed timestamped frames and collect the emitted (left, right) pairs.
    fn run_pairing(
        left: &[Option<u64>],
        right: &[Option<u64>],
        max_drops: u32,
    ) -> Vec<(usize, usize)> {
        let (ltx, lrx) = mpsc::sync_channel(left.len().max(1));
        let (rtx, rrx) = mpsc::sync_channel(right.len().max(1));
        for (i, &t) in left.iter().enumerate() {
            ltx.send(Timestamped {
                time_ns: t,
                data: i,
            })
            .unwrap();
        }
        for (i, &t) in right.iter().enumerate() {
            rtx.send(Timestamped {
                time_ns: t,
                data: i,
            })
            .unwrap();
        }
        drop((ltx, rtx));
        let mut pairs = Vec::new();
        pair_by_timestamp(&lrx, &rrx, FRAME_NS / 2, max_drops, |l, r| {
            pairs.push((l, r));
            true
        });
        pairs
    }

    fn times(start_ns: u64, n: usize) -> Vec<Option<u64>> {
        (0..n)
            .map(|i| Some(start_ns + i as u64 * FRAME_NS))
            .collect()
    }

    /// A constant 20 ms phase offset (measured on the IMX577 pair)
    /// pairs each frame with its nearest partner.
    #[test]
    fn pairs_with_phase_offset() {
        let left = times(1_000_000_000 + 20_000_000, 5);
        let right = times(1_000_000_000, 6);
        // left[0] is 20 ms after right[0] (> 16.7 ms) but 13 ms before right[1].
        assert_eq!(
            run_pairing(&left, &right, 60),
            vec![(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
        );
    }

    /// A camera that starts 2 s late: early frames of the other are dropped.
    #[test]
    fn late_camera_start_is_aligned() {
        let left = times(1_000_000_000, 70); // starts at 1.0 s
        let right = times(1_000_000_000 + 60 * FRAME_NS, 5); // starts 2 s later
        assert_eq!(
            run_pairing(&left, &right, 100),
            vec![(60, 0), (61, 1), (62, 2), (63, 3), (64, 4)]
        );
    }

    /// A dropped right frame drops exactly one left frame.
    #[test]
    fn dropped_frame_realigns() {
        let left = times(0, 5);
        let mut right = times(0, 5);
        right.remove(2); // right skips frame 2
        assert_eq!(
            run_pairing(&left, &right, 60),
            vec![(0, 0), (1, 1), (3, 2), (4, 3)]
        );
    }

    /// Missing timestamps pair in arrival order.
    #[test]
    fn no_timestamps_pairs_in_order() {
        assert_eq!(
            run_pairing(&[None, None, None], &[None, None], 60),
            vec![(0, 0), (1, 1)]
        );
    }

    /// Unrelated clocks (offset never closes) fall back to arrival order.
    #[test]
    fn unrelated_clocks_fall_back() {
        let left = times(0, 10);
        let right = times(1_000_000_000_000, 10);
        let pairs = run_pairing(&left, &right, 3);
        // 3 left frames are dropped before giving up, then in-order pairing.
        assert_eq!(pairs.first(), Some(&(3, 0)));
        assert_eq!(pairs.len(), 7);
    }

    /// Half a frame period.
    #[test]
    fn tolerance_is_half_period() {
        assert_eq!(pairing_tolerance_ns(30), 16_666_666);
        assert_eq!(pairing_tolerance_ns(0), 500_000_000);
    }
}
