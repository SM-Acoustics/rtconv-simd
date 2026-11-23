//! Zero-configuration convolution from a FIR artifact (as exported by rePhase).
//!
//! Synthesizes a stereo 24-bit PCM WAV FIR in memory (stand-in for
//! `include_bytes!("filter.wav")`), then lets `AutoStereoConvolver` pick the numeric domain,
//! engine and partition size. Run with `cargo run --release --example auto_from_artifact`.

#[cfg(feature = "auto")]
fn main() {
    use rtconv_simd::auto::{AutoStereoConvolver, NativeElem, stereo_workspace_len};
    use rtconv_simd::{FirArtifact, HAS_HARDWARE_FPU};

    const TAPS: usize = 4096;

    // --- Build a stereo 24-bit PCM WAV: linear-phase-like FIR with a centred peak. ----------
    let mut data = Vec::with_capacity(TAPS * 6);
    for k in 0..TAPS {
        let d = (k as f64 - (TAPS / 2) as f64).abs();
        let left = if k == TAPS / 2 {
            0.8
        } else {
            0.02 * (-d / 300.0).exp() * (k as f64 * 0.37).sin()
        };
        let right = 0.5 * left;
        for v in [left, right] {
            let q = (v * 8_388_608.0).round() as i32;
            data.extend_from_slice(&q.to_le_bytes()[..3]);
        }
    }
    let mut wav = Vec::new();
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(36 + data.len() as u32).to_le_bytes());
    wav.extend_from_slice(b"WAVEfmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes()); // WAVE_FORMAT_PCM
    wav.extend_from_slice(&2u16.to_le_bytes()); // stereo
    wav.extend_from_slice(&48_000u32.to_le_bytes());
    wav.extend_from_slice(&(48_000u32 * 6).to_le_bytes());
    wav.extend_from_slice(&6u16.to_le_bytes()); // block align
    wav.extend_from_slice(&24u16.to_le_bytes()); // bits per sample
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&(data.len() as u32).to_le_bytes());
    wav.extend_from_slice(&data);

    // --- Zero-configuration load. --------------------------------------------------------
    let artifact = FirArtifact::parse(&wav).expect("valid WAV");
    println!(
        "artifact: {:?} {:?}, {} ch x {} taps @ {:?} Hz",
        artifact.container(),
        artifact.encoding(),
        artifact.channels(),
        artifact.frames(),
        artifact.sample_rate()
    );

    // On an MCU this would be a `static` `AutoWorkspace<{ stereo_workspace_len(TAPS) }>`.
    let mut workspace: Vec<NativeElem> = vec![Default::default(); stereo_workspace_len(TAPS)];
    let workspace_bytes = core::mem::size_of_val(workspace.as_slice());
    let mut stereo =
        AutoStereoConvolver::from_artifact(&artifact, &mut workspace).expect("prepare");
    println!(
        "hardware FPU: {HAS_HARDWARE_FPU}; left = {:?}; workspace = {} bytes",
        stereo.left().info(),
        workspace_bytes
    );

    // --- Stream interleaved 16-bit frames (typical I2S DMA layout). ------------------------
    let mut frames = vec![0i16; 2 * 256];
    frames[0] = 16384; // +0.5 impulse on the left channel
    frames[1] = 16384; // +0.5 impulse on the right channel
    for _ in 0..(TAPS / 256 + 1) {
        stereo
            .process_interleaved_i16(&mut frames)
            .expect("process");
        frames.fill(0);
    }
    println!(
        "zero latency, saturations L/R = {}/{}",
        stereo.left().saturation_count(),
        stereo.right().saturation_count()
    );
}

#[cfg(not(feature = "auto"))]
fn main() {
    println!("Enable the `auto` feature (on by default) to run this example.");
}
