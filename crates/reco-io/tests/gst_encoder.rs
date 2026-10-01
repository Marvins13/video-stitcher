//! Integration tests for the GStreamer hardware encoder
//! (`--encoder gst-v4l2h264`), driven through
//! `adapters::create_file_encoder` like the CLI does.
//!
//! Needs a V4L2 M2M H.264 encoder (e.g. msm_vidc on the VENTUNO Q);
//! tests skip when `v4l2h264enc` is not registered. Run with:
//!
//! ```bash
//! cargo test -p reco-io --features gstreamer --test gst_encoder
//! ```

#![cfg(all(feature = "gstreamer", feature = "ffmpeg"))]

use reco_core::encoder::{EncodeError, OutputFrame, PixelFormat};
use reco_io::adapters::create_file_encoder;
use reco_io::ffmpeg::encoder::{Container, EncoderConfig};
use reco_io::gstreamer::encoder::GstVideoEncoder;

const W: u32 = 1920;
const H: u32 = 1080;
const FRAMES: usize = 60;

fn hw_available() -> bool {
    let ok = GstVideoEncoder::V4l2H264.is_available();
    if !ok {
        eprintln!("skipping: v4l2h264enc not available");
    }
    ok
}

fn config(container: Container) -> EncoderConfig {
    EncoderConfig {
        encoder_name: Some("gst-v4l2h264".to_string()),
        container,
        gop_size: Some(30),
        ..Default::default()
    }
}

/// Moving luma gradient with constant chroma, tightly packed NV12.
fn nv12_frame(index: usize) -> Vec<u8> {
    let (w, h) = (W as usize, H as usize);
    let mut data = vec![128u8; w * h * 3 / 2];
    for (y, row) in data[..w * h].chunks_exact_mut(w).enumerate() {
        for (x, px) in row.iter_mut().enumerate() {
            *px = ((x + y + index * 8) % 256) as u8;
        }
    }
    data
}

/// Encode `FRAMES` frames and return (codec, width, height, packets, keyframes).
fn encode_and_probe(
    path: &std::path::Path,
    container: Container,
) -> (String, u32, u32, usize, usize) {
    let (mut encoder, name) = create_file_encoder(path, W, H, (30, 1), &config(container)).unwrap();
    assert_eq!(name, "gst-v4l2h264");
    for i in 0..FRAMES {
        let data = nv12_frame(i);
        encoder
            .submit(OutputFrame {
                data: &data,
                width: W,
                height: H,
                format: PixelFormat::Nv12,
                pts_us: 0,
            })
            .unwrap();
    }
    encoder.finish().unwrap();

    reco_io::init();
    let mut ictx = ffmpeg_next::format::input(path).unwrap();
    let stream = ictx
        .streams()
        .best(ffmpeg_next::media::Type::Video)
        .expect("video stream");
    let index = stream.index();
    let codec = format!("{:?}", stream.parameters().id());
    let video = ffmpeg_next::codec::context::Context::from_parameters(stream.parameters())
        .unwrap()
        .decoder()
        .video()
        .unwrap();
    let (width, height) = (video.width(), video.height());
    let mut packets = 0;
    let mut keyframes = 0;
    for (s, packet) in ictx.packets() {
        if s.index() == index {
            packets += 1;
            if packet.is_key() {
                keyframes += 1;
            }
        }
    }
    (codec, width, height, packets, keyframes)
}

#[test]
fn encodes_h264_mp4_with_every_frame_and_requested_gop() {
    if !hw_available() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("out.mp4");
    let (codec, w, h, packets, keyframes) = encode_and_probe(&path, Container::Mp4);
    assert_eq!(codec, "H264");
    assert_eq!((w, h), (W, H));
    assert_eq!(packets, FRAMES);
    // GOP 30 over 60 frames: keyframes at 0 and 30.
    assert_eq!(keyframes, 2);
}

#[test]
fn encodes_h264_matroska() {
    if !hw_available() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("out.mkv");
    let (codec, w, h, packets, _) = encode_and_probe(&path, Container::Matroska);
    assert_eq!(codec, "H264");
    assert_eq!((w, h), (W, H));
    assert_eq!(packets, FRAMES);
}

#[test]
fn stream_url_is_rejected_with_a_clear_error() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = EncoderConfig {
        stream_url: Some("rtmp://localhost/live".to_string()),
        ..config(Container::Mp4)
    };
    let err = match create_file_encoder(&dir.path().join("x.mp4"), W, H, (30, 1), &cfg) {
        Err(e) => e,
        Ok(_) => panic!("stream_url must be rejected"),
    };
    assert!(matches!(err, EncodeError::Init { .. }));
    assert!(err.to_string().contains("streaming"), "{err}");
}

#[test]
fn unknown_gst_encoder_lists_the_available_ones() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = EncoderConfig {
        encoder_name: Some("gst-nope".to_string()),
        ..Default::default()
    };
    let err = match create_file_encoder(&dir.path().join("x.mp4"), W, H, (30, 1), &cfg) {
        Err(e) => e,
        Ok(_) => panic!("unknown encoder must be rejected"),
    };
    assert!(err.to_string().contains("gst-v4l2h264"), "{err}");
}
