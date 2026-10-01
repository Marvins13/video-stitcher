//! Hardware video encode via a GStreamer pipeline.
//!
//! [`GstFileEncoder`] implements [`reco_core::encoder::Encoder`] by
//! pushing the session's NV12 frames into
//!
//! ```text
//! appsrc -> v4l2h264enc -> video/x-h264,profile=high -> h264parse -> <muxer> -> filesink
//! ```
//!
//! It exists for platforms whose V4L2 stateful encoder works through
//! GStreamer's `v4l2h264enc` but not through FFmpeg's `h264_v4l2m2m`
//! wrapper (e.g. Qualcomm `msm_vidc` on the Arduino VENTUNO Q, where
//! FFmpeg gets `POLLERR` on the capture queue).
//!
//! Selected by name (`--encoder gst-v4l2h264`) through
//! [`crate::adapters::create_file_encoder`]; never chosen by
//! auto-detection.
//!
//! # Known limitations
//!
//! - **No audio.** Audio passthrough is implemented only in the FFmpeg
//!   encoder; the output of this backend has a video track only.
//! - **No streaming.** `stream_url` (RTMP tee) and the FLV container are
//!   rejected at construction.
//! - **H.264 only.** `v4l2h265enc` is not wired up yet.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_app as gst_app;
use reco_core::encoder::{EncodeError, Encoder, OutputFrame, PixelFormat};

use crate::adapters::GST_ENCODER_PREFIX;
use crate::output::Format;

/// Stall timeout used by the CLI / [`crate::adapters::create_file_encoder`]:
/// how long [`GstFileEncoder`] waits for the pipeline to make progress
/// before reporting the encoder as stalled.
pub const DEFAULT_STALL_TIMEOUT: Duration = Duration::from_secs(10);

/// Frames the `appsrc` queue may hold before `submit` waits for the
/// encoder to drain. Upstream, `AsyncEncodeThread` already buffers.
const QUEUED_FRAMES: u64 = 2;

/// Granularity of the bus poll while waiting for queue space / EOS.
const POLL_INTERVAL: Duration = Duration::from_millis(2);

/// Encoder elements this backend can drive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum GstVideoEncoder {
    /// V4L2 stateful H.264 encoder (`v4l2h264enc`, gst-plugins-good).
    V4l2H264,
}

impl GstVideoEncoder {
    /// Every encoder this backend supports.
    pub const ALL: &'static [Self] = &[Self::V4l2H264];

    /// Look up an encoder by its reco name (e.g. `"gst-v4l2h264"`).
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.iter().copied().find(|e| e.name() == name)
    }

    /// The reco encoder name, as passed to `--encoder`.
    pub fn name(self) -> &'static str {
        match self {
            Self::V4l2H264 => "gst-v4l2h264",
        }
    }

    /// The GStreamer element factory name.
    pub fn element_name(self) -> &'static str {
        match self {
            Self::V4l2H264 => "v4l2h264enc",
        }
    }

    /// Whether GStreamer initializes and the encoder element is
    /// registered. `v4l2h264enc` is only registered when the kernel
    /// exposes a V4L2 M2M H.264 encoder device.
    pub fn is_available(self) -> bool {
        gst::init().is_ok() && gst::ElementFactory::find(self.element_name()).is_some()
    }
}

/// Configuration for [`GstFileEncoder`]. All fields are explicit; the
/// adapter factory derives them from `EncoderConfig`.
#[derive(Debug, Clone)]
pub struct GstEncoderConfig {
    /// Which encoder element to use.
    pub encoder: GstVideoEncoder,
    /// Output container. `Flv` is rejected.
    pub container: Format,
    /// Target bitrate (VBR), in bits per second.
    pub bitrate_bps: u32,
    /// Peak bitrate (VBR), in bits per second. Must be >= `bitrate_bps`.
    pub peak_bitrate_bps: u32,
    /// Keyframe interval in frames, or `None` for the driver default.
    pub gop_size: Option<u32>,
    /// Maximum time without pipeline progress before `submit` / `finish`
    /// return an error instead of blocking.
    pub stall_timeout: Duration,
}

/// File encoder backed by a GStreamer hardware-encode pipeline.
///
/// Accepts tightly packed NV12 frames (what `AsyncEncodeThread` delivers).
/// Timestamps are derived from the frame count and the configured frame
/// rate (constant frame rate, like the FFmpeg encoder); `pts_us` is
/// ignored. `submit` never blocks longer than the configured stall
/// timeout without the pipeline making progress.
pub struct GstFileEncoder {
    pipeline: gst::Pipeline,
    appsrc: gst_app::AppSrc,
    bus: gst::Bus,
    name: &'static str,
    width: u32,
    height: u32,
    fps: (i32, i32),
    layout: Nv12Layout,
    max_queued_bytes: u64,
    stall_timeout: Duration,
    frame_count: u64,
    finished: bool,
    /// Set once the pipeline reported an error or stalled.
    failed: bool,
    path: PathBuf,
}

impl GstFileEncoder {
    /// Build the pipeline and set it to PLAYING.
    ///
    /// Fails with [`EncodeError::Init`] if an element is missing, the
    /// device cannot be opened, or the configuration is unsupported.
    pub fn new(
        path: &Path,
        width: u32,
        height: u32,
        fps: (i32, i32),
        config: &GstEncoderConfig,
    ) -> Result<Self, EncodeError> {
        validate(width, height, fps, config)?;
        let location = path.to_str().ok_or_else(|| {
            init_err(format!(
                "output path {} is not valid UTF-8 (required by filesink)",
                path.display()
            ))
        })?;

        gst::init().map_err(|e| init_err(format!("GStreamer init failed: {e}")))?;

        let enc_element = config.encoder.element_name();
        let muxer = muxer_for(config.container)?;
        for (factory, package) in [
            ("appsrc", "gstreamer1.0-plugins-base"),
            (
                enc_element,
                "gstreamer1.0-plugins-good (and a V4L2 M2M encoder device)",
            ),
            ("h264parse", "gstreamer1.0-plugins-bad"),
            (muxer.element, "gstreamer1.0-plugins-good"),
            ("filesink", "gstreamer1.0 core"),
        ] {
            if gst::ElementFactory::find(factory).is_none() {
                return Err(init_err(format!(
                    "GStreamer element '{factory}' not found (install {package})"
                )));
            }
        }

        let appsrc = gst_app::AppSrc::builder()
            .name("src")
            .caps(&nv12_caps(width, height, fps))
            .format(gst::Format::Time)
            .is_live(false)
            // Non-blocking push: queue bounding and the stall watchdog are
            // handled in `submit` so it can never hang.
            .block(false)
            .build();
        let encoder = gst::ElementFactory::make(enc_element)
            .name("encoder")
            .property("extra-controls", v4l2_controls(config))
            .build()
            .map_err(|e| init_err(format!("create {enc_element}: {e}")))?;
        // Without this filter caps negotiation fixates the first listed
        // profile (baseline) on msm_vidc.
        let profile = gst::ElementFactory::make("capsfilter")
            .property(
                "caps",
                gst::Caps::builder("video/x-h264")
                    .field("profile", "high")
                    .build(),
            )
            .build()
            .map_err(|e| init_err(format!("create capsfilter: {e}")))?;
        let parse = make("h264parse")?;
        let mux = gst::ElementFactory::make(muxer.element)
            .name("mux")
            .property_if_some("fragment-duration", muxer.fragment_duration_ms)
            .build()
            .map_err(|e| init_err(format!("create {}: {e}", muxer.element)))?;
        let sink = gst::ElementFactory::make("filesink")
            .property("location", location)
            .build()
            .map_err(|e| init_err(format!("create filesink: {e}")))?;

        let pipeline = gst::Pipeline::builder().name("reco-encoder").build();
        let elements = [appsrc.upcast_ref(), &encoder, &profile, &parse, &mux, &sink];
        pipeline
            .add_many(elements)
            .map_err(|e| init_err(format!("add elements: {e}")))?;
        gst::Element::link_many(elements)
            .map_err(|e| init_err(format!("link {enc_element} pipeline: {e}")))?;

        let mut encoder = Self::from_pipeline(
            pipeline,
            appsrc,
            config.encoder.name(),
            path,
            width,
            height,
            fps,
            config.stall_timeout,
        )?;
        encoder.start()?;

        log::info!(
            "Encoder: {}x{} {} (hardware) @ {}/{} fps, VBR {:.1}/{:.1} Mbps, {:?}",
            width,
            height,
            encoder.name,
            fps.0,
            fps.1,
            f64::from(config.bitrate_bps) / 1e6,
            f64::from(config.peak_bitrate_bps) / 1e6,
            config.container,
        );
        log::info!("Encoder frame path: GStreamer appsrc -> {enc_element} (CPU NV12 copy)");
        Ok(encoder)
    }

    /// Wrap an already linked pipeline whose head is `appsrc`.
    #[allow(clippy::too_many_arguments)]
    fn from_pipeline(
        pipeline: gst::Pipeline,
        appsrc: gst_app::AppSrc,
        name: &'static str,
        path: &Path,
        width: u32,
        height: u32,
        fps: (i32, i32),
        stall_timeout: Duration,
    ) -> Result<Self, EncodeError> {
        let bus = pipeline
            .bus()
            .ok_or_else(|| init_err("pipeline has no bus".to_string()))?;
        let layout = Nv12Layout::new(width, height);
        let max_queued_bytes = layout.size as u64 * QUEUED_FRAMES;
        appsrc.set_max_bytes(max_queued_bytes);
        Ok(Self {
            pipeline,
            appsrc,
            bus,
            name,
            width,
            height,
            fps,
            layout,
            max_queued_bytes,
            stall_timeout,
            frame_count: 0,
            finished: false,
            failed: false,
            path: path.to_path_buf(),
        })
    }

    fn start(&mut self) -> Result<(), EncodeError> {
        if let Err(e) = self.pipeline.set_state(gst::State::Playing) {
            let detail = self
                .pending_error()
                .unwrap_or_else(|| "no error message on the bus".to_string());
            let _ = self.pipeline.set_state(gst::State::Null);
            self.finished = true;
            return Err(init_err(format!(
                "{}: failed to start pipeline ({e}): {detail}",
                self.name
            )));
        }
        Ok(())
    }

    /// The reco encoder name (e.g. `"gst-v4l2h264"`).
    pub fn encoder_name(&self) -> &str {
        self.name
    }

    /// Pop a pending error message off the bus without waiting.
    fn pending_error(&self) -> Option<String> {
        while let Some(msg) = self
            .bus
            .pop_filtered(&[gst::MessageType::Error, gst::MessageType::Warning])
        {
            if let Some(err) = handle_message(self.name, &msg) {
                return Some(err);
            }
        }
        None
    }

    /// Wait until the appsrc queue has room for one more frame. Fails
    /// after `stall_timeout` without the queue draining.
    fn wait_for_queue_space(&self) -> Result<(), EncodeError> {
        let mut deadline = Instant::now() + self.stall_timeout;
        let mut last_level = self.appsrc.current_level_bytes();
        while last_level >= self.max_queued_bytes {
            if let Some(msg) = self.bus.timed_pop_filtered(
                gst::ClockTime::from_nseconds(POLL_INTERVAL.as_nanos() as u64),
                &[gst::MessageType::Error, gst::MessageType::Warning],
            ) && let Some(err) = handle_message(self.name, &msg)
            {
                return Err(self.frame_err(err));
            }
            let level = self.appsrc.current_level_bytes();
            if level < last_level {
                deadline = Instant::now() + self.stall_timeout;
            }
            last_level = level;
            if Instant::now() >= deadline {
                return Err(self.frame_err(format!(
                    "encoder stalled: no frame consumed for {:.1}s ({} frames queued)",
                    self.stall_timeout.as_secs_f64(),
                    last_level / self.layout.size as u64,
                )));
            }
        }
        Ok(())
    }

    /// Wait for queue space, then copy `data` into a buffer and push it.
    fn push_frame(&mut self, data: &[u8]) -> Result<(), EncodeError> {
        if let Some(err) = self.pending_error() {
            return Err(self.frame_err(err));
        }
        self.wait_for_queue_space()?;

        let mut buffer = gst::Buffer::with_size(self.layout.size)
            .map_err(|e| self.frame_err(format!("allocate buffer: {e}")))?;
        {
            let buffer = buffer
                .get_mut()
                .expect("newly allocated buffer is writable");
            buffer.set_pts(self.pts(self.frame_count));
            buffer.set_duration(self.pts(self.frame_count + 1) - self.pts(self.frame_count));
            let mut map = buffer
                .map_writable()
                .map_err(|e| self.frame_err(format!("map buffer: {e}")))?;
            self.layout.pack(data, map.as_mut_slice());
        }
        self.appsrc.push_buffer(buffer).map_err(|flow| {
            let detail = self.pending_error().unwrap_or_default();
            self.frame_err(format!("push_buffer: {flow:?} {detail}"))
        })?;
        self.frame_count += 1;
        Ok(())
    }

    fn frame_err(&self, reason: String) -> EncodeError {
        EncodeError::Frame {
            frame_index: Some(self.frame_count),
            reason: format!("{}: {reason}", self.name),
        }
    }

    fn pts(&self, frame: u64) -> gst::ClockTime {
        let (num, den) = (self.fps.0 as u128, self.fps.1 as u128);
        let ns = u128::from(frame) * 1_000_000_000 * den / num;
        gst::ClockTime::from_nseconds(ns as u64)
    }

    /// Send EOS and wait for the muxer to finalize the file. After a
    /// pipeline failure there is no EOS to wait for: tear down directly.
    fn finalize(&mut self) -> Result<(), EncodeError> {
        self.finished = true;
        let result = if self.failed {
            Err(EncodeError::Finalize {
                reason: format!(
                    "{}: pipeline failed earlier; {} is incomplete",
                    self.name,
                    self.path.display()
                ),
            })
        } else {
            self.drain_to_eos()
        };
        let _ = self.pipeline.set_state(gst::State::Null);
        result
    }

    fn drain_to_eos(&self) -> Result<(), EncodeError> {
        let finalize_err = |reason: String| EncodeError::Finalize {
            reason: format!("{} ({}): {reason}", self.name, self.path.display()),
        };
        self.appsrc
            .end_of_stream()
            .map_err(|e| finalize_err(format!("end_of_stream: {e}")))?;

        let mut deadline = Instant::now() + self.stall_timeout;
        let mut last_level = self.appsrc.current_level_bytes();
        loop {
            if let Some(msg) = self.bus.timed_pop_filtered(
                gst::ClockTime::from_nseconds(POLL_INTERVAL.as_nanos() as u64),
                &[
                    gst::MessageType::Eos,
                    gst::MessageType::Error,
                    gst::MessageType::Warning,
                ],
            ) {
                if msg.type_() == gst::MessageType::Eos {
                    return Ok(());
                }
                if let Some(err) = handle_message(self.name, &msg) {
                    return Err(finalize_err(err));
                }
            }
            let level = self.appsrc.current_level_bytes();
            if level < last_level {
                deadline = Instant::now() + self.stall_timeout;
            }
            last_level = level;
            if Instant::now() >= deadline {
                return Err(finalize_err(format!(
                    "no EOS within {:.1}s of the last progress; output may be incomplete",
                    self.stall_timeout.as_secs_f64()
                )));
            }
        }
    }
}

impl Encoder for GstFileEncoder {
    fn submit(&mut self, frame: OutputFrame<'_>) -> Result<(), EncodeError> {
        if self.finished || self.failed {
            return Err(self.frame_err("submit after finish or failure".to_string()));
        }
        if frame.format != PixelFormat::Nv12 {
            return Err(self.frame_err(format!("expected NV12 frames, got {:?}", frame.format)));
        }
        if (frame.width, frame.height) != (self.width, self.height) {
            return Err(self.frame_err(format!(
                "frame is {}x{}, encoder was opened for {}x{}",
                frame.width, frame.height, self.width, self.height
            )));
        }
        let expected = self.layout.packed_size();
        if frame.data.len() != expected {
            return Err(self.frame_err(format!(
                "NV12 frame size mismatch: expected {expected} bytes, got {}",
                frame.data.len()
            )));
        }

        let result = self.push_frame(frame.data);
        if result.is_err() {
            self.failed = true;
        }
        result
    }

    fn finish(&mut self) -> Result<(), EncodeError> {
        if self.finished {
            return Ok(());
        }
        self.finalize()?;
        log::info!(
            "{}: wrote {} frames to {}",
            self.name,
            self.frame_count,
            self.path.display()
        );
        Ok(())
    }
}

impl Drop for GstFileEncoder {
    fn drop(&mut self) {
        if !self.finished {
            log::warn!(
                "GstFileEncoder dropped without calling finish() - output file may be corrupt"
            );
            if let Err(e) = self.finalize() {
                log::warn!("{e}");
            }
        }
    }
}

/// GStreamer's default NV12 memory layout for `width x height`
/// (`GstVideoInfo`: luma stride rounded up to 4 bytes, chroma plane
/// directly after the luma plane). Height must be even.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Nv12Layout {
    width: usize,
    height: usize,
    stride: usize,
    size: usize,
}

impl Nv12Layout {
    fn new(width: u32, height: u32) -> Self {
        let (width, height) = (width as usize, height as usize);
        let stride = width.next_multiple_of(4);
        Self {
            width,
            height,
            stride,
            size: stride * height * 3 / 2,
        }
    }

    /// Size of a tightly packed NV12 frame (stride == width).
    fn packed_size(&self) -> usize {
        self.width * self.height * 3 / 2
    }

    /// Copy a tightly packed NV12 frame into GStreamer's layout.
    fn pack(&self, src: &[u8], dst: &mut [u8]) {
        if self.stride == self.width {
            dst[..src.len()].copy_from_slice(src);
            return;
        }
        let rows = self.height * 3 / 2;
        for (src_row, dst_row) in src
            .chunks_exact(self.width)
            .zip(dst.chunks_exact_mut(self.stride))
            .take(rows)
        {
            dst_row[..self.width].copy_from_slice(src_row);
        }
    }
}

fn validate(
    width: u32,
    height: u32,
    fps: (i32, i32),
    config: &GstEncoderConfig,
) -> Result<(), EncodeError> {
    if width == 0 || height == 0 || !width.is_multiple_of(2) || !height.is_multiple_of(2) {
        return Err(init_err(format!(
            "NV12 output needs even, non-zero dimensions (got {width}x{height})"
        )));
    }
    if fps.0 <= 0 || fps.1 <= 0 {
        return Err(init_err(format!("invalid frame rate {}/{}", fps.0, fps.1)));
    }
    if config.bitrate_bps == 0 || config.peak_bitrate_bps < config.bitrate_bps {
        return Err(init_err(format!(
            "invalid bitrate {} / peak {} bps",
            config.bitrate_bps, config.peak_bitrate_bps
        )));
    }
    if config.stall_timeout.is_zero() {
        return Err(init_err("stall_timeout must be non-zero".to_string()));
    }
    Ok(())
}

/// Muxer element for a container.
#[derive(Debug, PartialEq, Eq)]
struct Muxer {
    element: &'static str,
    /// `mp4mux` `fragment-duration` in ms (fragmented MP4 only).
    fragment_duration_ms: Option<u32>,
}

fn muxer_for(container: Format) -> Result<Muxer, EncodeError> {
    let plain = |element| Muxer {
        element,
        fragment_duration_ms: None,
    };
    match container {
        Format::Mp4 => Ok(plain("mp4mux")),
        // ~1 s fragments, readable mid-write (FFmpeg path: empty_moov +
        // frag_keyframe).
        Format::Mp4Fragmented => Ok(Muxer {
            element: "mp4mux",
            fragment_duration_ms: Some(1000),
        }),
        Format::Mov => Ok(plain("qtmux")),
        Format::Mkv => Ok(plain("matroskamux")),
        Format::Flv => Err(init_err(format!(
            "FLV / RTMP output is not supported by the {GST_ENCODER_PREFIX}* encoders; \
             use an FFmpeg encoder"
        ))),
    }
}

/// V4L2 controls for the encoder. Names are the `msm_vidc` / V4L2 core
/// control names (`v4l2-ctl --list-ctrls`), which `v4l2h264enc` maps
/// to control IDs.
fn v4l2_controls(config: &GstEncoderConfig) -> gst::Structure {
    let mut controls = gst::Structure::builder("controls")
        // 0 = Variable Bitrate. msm_vidc has no constant-QP mode.
        .field("video_bitrate_mode", 0i32)
        .field("video_bitrate", clamp_i32(config.bitrate_bps))
        .field("video_peak_bitrate", clamp_i32(config.peak_bitrate_bps))
        // B-frames add reordering latency; the FFmpeg path sets bf=0 too.
        .field("video_b_frames", 0i32);
    if let Some(gop) = config.gop_size {
        controls = controls.field("video_gop_size", clamp_i32(gop));
    }
    controls.build()
}

fn clamp_i32(v: u32) -> i32 {
    i32::try_from(v).unwrap_or(i32::MAX)
}

/// Caps for the session's NV12 output. `v4l2h264enc` only accepts caps
/// that state interlace mode and colorimetry; `bt709` (limited range)
/// matches reco-core's `rgba_to_nv12.wgsl`.
fn nv12_caps(width: u32, height: u32, fps: (i32, i32)) -> gst::Caps {
    gst::Caps::builder("video/x-raw")
        .field("format", "NV12")
        .field("width", width as i32)
        .field("height", height as i32)
        .field("framerate", gst::Fraction::new(fps.0, fps.1))
        .field("pixel-aspect-ratio", gst::Fraction::new(1, 1))
        .field("interlace-mode", "progressive")
        .field("colorimetry", "bt709")
        .build()
}

fn make(factory: &str) -> Result<gst::Element, EncodeError> {
    gst::ElementFactory::make(factory)
        .build()
        .map_err(|e| init_err(format!("create {factory}: {e}")))
}

/// Log warnings; return a description for errors.
fn handle_message(name: &str, msg: &gst::Message) -> Option<String> {
    let src = msg
        .src()
        .map(|s| s.path_string().to_string())
        .unwrap_or_default();
    match msg.view() {
        gst::MessageView::Error(e) => Some(format!(
            "{src}: {}{}",
            e.error(),
            e.debug().map(|d| format!(" ({d})")).unwrap_or_default()
        )),
        gst::MessageView::Warning(w) => {
            log::warn!("{name}: {src}: {}", w.error());
            None
        }
        _ => None,
    }
}

fn init_err(reason: String) -> EncodeError {
    EncodeError::Init { reason }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> GstEncoderConfig {
        GstEncoderConfig {
            encoder: GstVideoEncoder::V4l2H264,
            container: Format::Mp4,
            bitrate_bps: 12_000_000,
            peak_bitrate_bps: 18_000_000,
            gop_size: Some(60),
            stall_timeout: DEFAULT_STALL_TIMEOUT,
        }
    }

    #[test]
    fn encoder_names_round_trip_and_share_the_prefix() {
        for enc in GstVideoEncoder::ALL {
            assert!(enc.name().starts_with(GST_ENCODER_PREFIX));
            assert_eq!(GstVideoEncoder::from_name(enc.name()), Some(*enc));
        }
        assert_eq!(GstVideoEncoder::from_name("h264_v4l2m2m"), None);
        assert_eq!(GstVideoEncoder::from_name("gst-v4l2h265"), None);
    }

    #[test]
    fn nv12_layout_matches_gstreamer_default_strides() {
        let l = Nv12Layout::new(3840, 1080);
        assert_eq!(
            (l.stride, l.size, l.packed_size()),
            (3840, 6_220_800, 6_220_800)
        );
        let l = Nv12Layout::new(1918, 1080);
        assert_eq!(l.stride, 1920);
        assert_eq!(l.size, 1920 * 1080 * 3 / 2);
        assert_eq!(l.packed_size(), 1918 * 1080 * 3 / 2);
    }

    #[test]
    fn pack_copies_tight_frames_verbatim() {
        let l = Nv12Layout::new(8, 4);
        let src: Vec<u8> = (0..l.packed_size() as u8).collect();
        let mut dst = vec![0u8; l.size];
        l.pack(&src, &mut dst);
        assert_eq!(dst, src);
    }

    #[test]
    fn pack_pads_rows_to_the_gstreamer_stride() {
        let l = Nv12Layout::new(6, 2);
        assert_eq!(l.stride, 8);
        let src: Vec<u8> = (1..=l.packed_size() as u8).collect(); // 3 rows of 6
        let mut dst = vec![0u8; l.size];
        l.pack(&src, &mut dst);
        assert_eq!(
            dst,
            [
                1, 2, 3, 4, 5, 6, 0, 0, 7, 8, 9, 10, 11, 12, 0, 0, 13, 14, 15, 16, 17, 18, 0, 0
            ]
        );
    }

    #[test]
    fn validate_rejects_odd_dims_bad_fps_and_bitrates() {
        let c = config();
        assert!(validate(1920, 1080, (30, 1), &c).is_ok());
        assert!(validate(1921, 1080, (30, 1), &c).is_err());
        assert!(validate(1920, 0, (30, 1), &c).is_err());
        assert!(validate(1920, 1080, (0, 1), &c).is_err());
        let mut c2 = config();
        c2.peak_bitrate_bps = c2.bitrate_bps - 1;
        assert!(validate(1920, 1080, (30, 1), &c2).is_err());
        let mut c3 = config();
        c3.stall_timeout = Duration::ZERO;
        assert!(validate(1920, 1080, (30, 1), &c3).is_err());
    }

    #[test]
    fn muxer_selection_per_container() {
        let element = |f| muxer_for(f).unwrap().element;
        assert_eq!(element(Format::Mp4), "mp4mux");
        assert_eq!(element(Format::Mov), "qtmux");
        assert_eq!(element(Format::Mkv), "matroskamux");
        assert_eq!(muxer_for(Format::Mp4).unwrap().fragment_duration_ms, None);
        assert_eq!(
            muxer_for(Format::Mp4Fragmented).unwrap(),
            Muxer {
                element: "mp4mux",
                fragment_duration_ms: Some(1000),
            }
        );
        assert!(matches!(
            muxer_for(Format::Flv),
            Err(EncodeError::Init { .. })
        ));
    }

    #[test]
    fn controls_use_msm_vidc_names() {
        if gst::init().is_err() {
            return;
        }
        let s = v4l2_controls(&config());
        assert_eq!(s.get::<i32>("video_bitrate_mode").unwrap(), 0);
        assert_eq!(s.get::<i32>("video_bitrate").unwrap(), 12_000_000);
        assert_eq!(s.get::<i32>("video_peak_bitrate").unwrap(), 18_000_000);
        assert_eq!(s.get::<i32>("video_b_frames").unwrap(), 0);
        assert_eq!(s.get::<i32>("video_gop_size").unwrap(), 60);

        let mut no_gop = config();
        no_gop.gop_size = None;
        assert!(!v4l2_controls(&no_gop).has_field("video_gop_size"));
    }

    /// `appsrc ! identity sleep-time=.. ! fakesink`: a pipeline that
    /// consumes far slower than the stall timeout must make `submit`
    /// fail instead of blocking.
    fn slow_pipeline_encoder(stall_timeout: Duration, sleep_us: u32) -> Option<GstFileEncoder> {
        gst::init().ok()?;
        let (w, h, fps) = (64, 32, (30, 1));
        let appsrc = gst_app::AppSrc::builder()
            .caps(&nv12_caps(w, h, fps))
            .format(gst::Format::Time)
            .block(false)
            .build();
        let identity = gst::ElementFactory::make("identity")
            .property("sleep-time", sleep_us)
            .build()
            .ok()?;
        let sink = gst::ElementFactory::make("fakesink")
            .property("sync", false)
            .build()
            .ok()?;
        let pipeline = gst::Pipeline::new();
        let elements = [appsrc.upcast_ref(), &identity, &sink];
        pipeline.add_many(elements).ok()?;
        gst::Element::link_many(elements).ok()?;
        let mut enc = GstFileEncoder::from_pipeline(
            pipeline,
            appsrc,
            "test",
            Path::new("unused"),
            w,
            h,
            fps,
            stall_timeout,
        )
        .ok()?;
        enc.start().ok()?;
        Some(enc)
    }

    fn submit_frame(enc: &mut GstFileEncoder, data: &[u8]) -> Result<(), EncodeError> {
        enc.submit(OutputFrame {
            data,
            width: 64,
            height: 32,
            format: PixelFormat::Nv12,
            pts_us: 0,
        })
    }

    #[test]
    fn submit_reports_a_stalled_pipeline_instead_of_blocking() {
        let timeout = Duration::from_millis(200);
        // 1 s per buffer, 5x the stall timeout.
        let Some(mut enc) = slow_pipeline_encoder(timeout, 1_000_000) else {
            return;
        };
        let frame = vec![128u8; 64 * 32 * 3 / 2];
        let start = Instant::now();
        let mut result = Ok(());
        for _ in 0..10 {
            result = submit_frame(&mut enc, &frame);
            if result.is_err() {
                break;
            }
        }
        let err = result.expect_err("a stalled pipeline must surface as an error");
        assert!(err.to_string().contains("stalled"), "{err}");
        assert!(start.elapsed() < Duration::from_secs(5));
        // A failed encoder refuses further frames and finishes with an
        // error instead of waiting for an EOS that never comes.
        assert!(submit_frame(&mut enc, &frame).is_err());
        let finish_start = Instant::now();
        assert!(matches!(enc.finish(), Err(EncodeError::Finalize { .. })));
        assert!(finish_start.elapsed() < Duration::from_secs(3));
    }

    #[test]
    fn submit_and_finish_succeed_on_a_flowing_pipeline() {
        let Some(mut enc) = slow_pipeline_encoder(Duration::from_secs(5), 0) else {
            return;
        };
        let frame = vec![128u8; 64 * 32 * 3 / 2];
        for _ in 0..30 {
            submit_frame(&mut enc, &frame).unwrap();
        }
        enc.finish().unwrap();
        assert_eq!(enc.frame_count, 30);
        assert!(submit_frame(&mut enc, &frame).is_err());
    }

    #[test]
    fn submit_rejects_wrong_format_and_size() {
        let Some(mut enc) = slow_pipeline_encoder(Duration::from_secs(5), 0) else {
            return;
        };
        let rgba = vec![0u8; 64 * 32 * 4];
        let err = enc
            .submit(OutputFrame {
                data: &rgba,
                width: 64,
                height: 32,
                format: PixelFormat::Rgba8,
                pts_us: 0,
            })
            .unwrap_err();
        assert!(err.to_string().contains("NV12"), "{err}");
        assert!(submit_frame(&mut enc, &[0u8; 10]).is_err());
        enc.finish().unwrap();
    }
}
