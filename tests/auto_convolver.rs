//! Zero-configuration `AutoConvolver`: const workspace sizing matches the engines exactly,
//! engine selection follows the cost model, and artifacts decode + convolve correctly in both
//! the `f32` and the Q1.15 domain, mono and stereo, with `f32` and `i16` I/O.
#![cfg(feature = "auto")]

use rtconv_simd::auto::{
    AutoConvolver, AutoEngine, AutoStereoConvolver, NumericDomain, engine_len_f32, engine_len_q15,
    select_f32, select_q15, workspace_len_f32, workspace_len_q15,
};
use rtconv_simd::fixed::{FixedQ15DirectConfig, FixedQ15DirectConvolver};
use rtconv_simd::{
    ClearancePolicy, CostModel, DirectConfig, DirectConvolver, FirArtifact,
    FixedQ15NonUniformConvolver, HybridConfig, HybridConvolver, NonUniformConvolver,
    NonUniformPlan, OptimizerScratchCell, SimdBackend, SimdFixedFftBackend, optimizer_scratch_len,
};

fn auto_max_block_test(taps: usize, ceiling: usize) -> usize {
    let cap = ceiling.min(32usize.max(taps / 4));
    let mut best = 0;
    for b in rtconv_simd::OPTIMIZER_BLOCK_SIZES {
        if b <= cap && ClearancePolicy::Strategy2.min_offset_samples(b, 64) < taps {
            best = b;
        }
    }
    best
}

fn actual_engine_len_f32(taps: usize) -> usize {
    match select_f32(taps) {
        AutoEngine::Hybrid { block_size } => {
            let fft = SimdFixedFftBackend::new(SimdBackend::scalar());
            HybridConvolver::memory_requirements(
                HybridConfig {
                    head_len: block_size,
                    tail_block_size: block_size,
                    max_ir_len: taps,
                },
                &fft,
            )
            .unwrap()
            .total_f32
        }
        AutoEngine::NonUniform => {
            let fft = SimdFixedFftBackend::new(SimdBackend::scalar());
            let max_b = auto_max_block_test(taps, 8192);
            let mut sc = vec![OptimizerScratchCell::ZERO; optimizer_scratch_len(taps, max_b)];
            let (plan, _) = NonUniformPlan::optimize_with_max_block_size(
                taps,
                64,
                max_b,
                &CostModel::for_target_f32(),
                ClearancePolicy::Strategy2,
                &mut sc,
            )
            .unwrap();
            NonUniformConvolver::memory_requirements(&plan, &fft)
                .unwrap()
                .total_f32
        }
        _ => {
            DirectConvolver::memory_requirements(DirectConfig { max_ir_len: taps })
                .unwrap()
                .total_f32
        }
    }
}

fn actual_engine_len_q15(taps: usize) -> usize {
    match select_q15(taps) {
        AutoEngine::FixedDirect => {
            FixedQ15DirectConvolver::memory_requirements(FixedQ15DirectConfig { max_ir_len: taps })
                .unwrap()
                .total_i16
        }
        _ => {
            let max_b = auto_max_block_test(taps, 1024);
            let mut sc = vec![OptimizerScratchCell::ZERO; optimizer_scratch_len(taps, max_b)];
            let (plan, _) = NonUniformPlan::optimize_with_max_block_size(
                taps,
                64,
                max_b,
                &CostModel::esp32_c6_q15_estimate(),
                ClearancePolicy::Strategy2,
                &mut sc,
            )
            .unwrap();
            FixedQ15NonUniformConvolver::memory_requirements(&plan)
                .unwrap()
                .total_i16
        }
    }
}

#[test]
fn const_sizing_matches_engine_requirements_exactly() {
    // Exact equality for Direct / Hybrid / FixedDirect, and mathematically proven upper bound
    // (`>= actual` with tight documented upper-margin ceiling) for cost-optimized NUPC engines.
    // With PR-C shared real input ring + power-of-two `2 * B_s` output rings, the static
    // `const fn` upper bound stays within ~1.4x–2.0x of the exact post-DP workspace requirement.
    for taps in (1..=600).chain([
        700, 1024, 1500, 2048, 3000, 4096, 5120, 5121, 8191, 8192, 9503, 9504, 16383, 16384, 35391,
        35392, 65536, 131072,
    ]) {
        let f32_len = engine_len_f32(taps);
        let f32_actual = actual_engine_len_f32(taps);
        match select_f32(taps) {
            AutoEngine::NonUniform => {
                assert!(
                    f32_len >= f32_actual,
                    "f32 NUPC bound underflow taps={taps}: bound={f32_len} < actual={f32_actual}"
                );
                assert!(
                    f32_len <= f32_actual * 2 + 8192,
                    "f32 NUPC bound unexpectedly loose taps={taps}: bound={f32_len} actual={f32_actual}"
                );
            }
            _ => assert_eq!(f32_len, f32_actual, "f32 taps={taps}"),
        }

        let q15_len = engine_len_q15(taps);
        let q15_actual = actual_engine_len_q15(taps);
        match select_q15(taps) {
            AutoEngine::FixedDirect => assert_eq!(q15_len, q15_actual, "q15 taps={taps}"),
            _ => {
                assert!(
                    q15_len >= q15_actual,
                    "q15 NUPC bound underflow taps={taps}: bound={q15_len} < actual={q15_actual}"
                );
                assert!(
                    q15_len <= q15_actual * 3 + 2048,
                    "q15 NUPC bound unexpectedly loose taps={taps}: bound={q15_len} actual={q15_actual}"
                );
            }
        }
    }

    // Also verify `AutoConvolver::prepare_f32` and `prepare_q15` succeed in exact `engine_len_*`
    // slices across engine transitions including `AutoEngine::NonUniform` and `FixedNonUniform`.
    for taps in [1usize, 64, 128, 129, 300, 700, 2048, 5121, 8192, 16384] {
        let ir_f = vec![0.001f32; taps];
        // Hardware FPU: f32 engines in an f32 workspace.
        #[cfg(not(feature = "native-q15"))]
        {
            let mut ws_f = vec![0.0f32; engine_len_f32(taps)];
            let conv_f = AutoConvolver::prepare_f32(&ir_f, &mut ws_f).unwrap();
            assert_eq!(conv_f.info().engine, select_f32(taps));
        }
        // FPU-less build: f32 taps are quantized from their bits into the Q1.15 engines,
        // which need exactly `engine_len_q15` i16 words.
        #[cfg(feature = "native-q15")]
        {
            let mut ws_f = vec![0i16; engine_len_q15(taps)];
            let conv_f = AutoConvolver::prepare_f32(&ir_f, &mut ws_f).unwrap();
            assert_eq!(conv_f.info().engine, select_q15(taps));
            assert_eq!(conv_f.info().domain, NumericDomain::FixedQ15);
        }

        let ir_q = vec![16i16; taps];
        let mut ws_q = vec![0i16; engine_len_q15(taps)];
        let conv_q = AutoConvolver::prepare_q15(&ir_q, 0, &mut ws_q).unwrap();
        assert_eq!(conv_q.info().engine, select_q15(taps));
    }
}

#[test]
fn workspace_len_covers_every_shorter_ir() {
    for max in [
        1usize, 50, 128, 129, 173, 174, 500, 900, 2000, 3000, 5120, 5121, 10_000, 40_000,
    ] {
        let wf = workspace_len_f32(max);
        let wq = workspace_len_q15(max);
        for taps in 1..=max {
            assert!(
                engine_len_f32(taps) + taps <= wf,
                "f32 max={max} taps={taps}"
            );
            assert!(
                engine_len_q15(taps) + 2 * taps <= wq,
                "q15 max={max} taps={taps}"
            );
        }
    }
}

#[test]
fn const_generic_workspace_len_evaluates_at_131072_taps() {
    use rtconv_simd::auto::{AutoWorkspace, workspace_len};

    // Compile-time evaluation of `workspace_len(131_072)` inside a const generic parameter.
    struct CheckConst<const N: usize>;
    let _check = CheckConst::<{ workspace_len(131_072) }>;
    let _check_f32 = CheckConst::<{ workspace_len_f32(131_072) }>;
    let _check_q15 = CheckConst::<{ workspace_len_q15(131_072) }>;
    let _ = core::mem::size_of::<AutoWorkspace<{ workspace_len(131_072) }>>();

    // Exhaustively verify O(1) `worst_engine_len_f32` matches brute-force scan up to 131_072.
    let brute_max = (1..=131_072usize).map(engine_len_f32).max().unwrap();
    assert_eq!(workspace_len_f32(131_072), brute_max + 131_072);
}

#[test]
fn selection_follows_cost_model() {
    assert_eq!(select_f32(1), AutoEngine::Direct);
    assert_eq!(select_f32(32), AutoEngine::Direct);
    assert!(matches!(select_f32(512), AutoEngine::Hybrid { .. }));
    assert!(matches!(select_f32(1024), AutoEngine::Hybrid { .. }));
    // Long IRs switch to cost-optimized NonUniform on all targets (SSE2, AVX2, P4, theoretical).
    assert_eq!(select_f32(16384), AutoEngine::NonUniform);
    assert_eq!(select_f32(65536), AutoEngine::NonUniform);
    assert_eq!(select_f32(262_144), AutoEngine::NonUniform);
    assert_eq!(select_q15(128), AutoEngine::FixedDirect);
    assert_eq!(select_q15(129), AutoEngine::FixedNonUniform);
}

fn oracle(ir: &[f64], x: &[f64]) -> Vec<f64> {
    (0..x.len())
        .map(|n| (0..ir.len().min(n + 1)).map(|k| ir[k] * x[n - k]).sum())
        .collect()
}

fn snr(reference: &[f64], actual: impl Iterator<Item = f64>) -> f64 {
    let (mut e2, mut s2) = (0.0, 0.0);
    for (r, a) in reference.iter().zip(actual) {
        e2 += (r - a) * (r - a);
        s2 += r * r;
    }
    10.0 * (s2 / e2.max(1e-300)).log10()
}

fn signal(len: usize, seed: u64, gain: f64) -> Vec<f64> {
    let mut x = seed;
    (0..len)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            ((x >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) * gain
        })
        .collect()
}

/// Stereo IEEE-float WAV writer (rePhase "32 bits IEEE stereo" layout).
fn wav_f32_stereo(l: &[f64], r: &[f64]) -> Vec<u8> {
    let mut data = Vec::new();
    for (a, b) in l.iter().zip(r) {
        data.extend_from_slice(&(*a as f32).to_le_bytes());
        data.extend_from_slice(&(*b as f32).to_le_bytes());
    }
    let mut v = Vec::new();
    v.extend_from_slice(b"RIFF");
    v.extend_from_slice(&(36 + data.len() as u32).to_le_bytes());
    v.extend_from_slice(b"WAVEfmt ");
    v.extend_from_slice(&16u32.to_le_bytes());
    v.extend_from_slice(&3u16.to_le_bytes());
    v.extend_from_slice(&2u16.to_le_bytes());
    v.extend_from_slice(&48_000u32.to_le_bytes());
    v.extend_from_slice(&(48_000u32 * 8).to_le_bytes());
    v.extend_from_slice(&8u16.to_le_bytes());
    v.extend_from_slice(&32u16.to_le_bytes());
    v.extend_from_slice(b"data");
    v.extend_from_slice(&(data.len() as u32).to_le_bytes());
    v.extend_from_slice(&data);
    v
}

fn fir(len: usize, seed: u64, peak: f64) -> Vec<f64> {
    // Linear-phase-like FIR with a centered peak, like a rePhase correction filter.
    let noise = signal(len, seed, 1.0);
    let mid = len / 2;
    (0..len)
        .map(|k| {
            let d = (k as f64 - mid as f64).abs();
            let env = (-d * 8.0 / len as f64).exp();
            if k == mid {
                peak
            } else {
                0.2 * peak * env * noise[k]
            }
        })
        .map(|v| (v as f32) as f64)
        .collect()
}

/// `f32` auto engines exist only with a hardware FPU (not under `native-q15`).
#[cfg(not(feature = "native-q15"))]
#[test]
fn f32_domain_stereo_wav_end_to_end_with_fragmented_blocks() {
    for taps in [48usize, 700, 3000] {
        let (l, r) = (fir(taps, 1, 0.8), fir(taps, 2, 0.5));
        let bytes = wav_f32_stereo(&l, &r);
        let mut ws = vec![0.0f32; 2 * workspace_len_f32(taps)];
        let mut st = AutoStereoConvolver::from_bytes(&bytes, &mut ws).unwrap();
        let info = st.left().info();
        assert_eq!(info.domain, NumericDomain::Float32);
        assert_eq!(info.engine, select_f32(taps));
        assert_eq!(
            (info.taps, info.latency_samples, info.sample_rate),
            (taps, 0, Some(48_000))
        );

        let x = signal(5000, 7, 0.5);
        let xf: Vec<f32> = x.iter().map(|&v| v as f32).collect();
        let (mut yl, mut yr) = (vec![0.0f32; x.len()], vec![0.0f32; x.len()]);
        let mut pos = 0;
        for (i, n) in [1usize, 31, 64, 100, 7, 256].iter().cycle().enumerate() {
            if pos >= x.len() || i > 10_000 {
                break;
            }
            let n = (*n).min(x.len() - pos);
            let (a, b) = (pos, pos + n);
            st.process_f32(&xf[a..b], &xf[a..b], &mut yl[a..b], &mut yr[a..b])
                .unwrap();
            pos = b;
        }
        let xq: Vec<f64> = xf.iter().map(|&v| v as f64).collect();
        assert!(
            snr(&oracle(&l, &xq), yl.iter().map(|&v| v as f64)) > 100.0,
            "L taps={taps}"
        );
        assert!(
            snr(&oracle(&r, &xq), yr.iter().map(|&v| v as f64)) > 100.0,
            "R taps={taps}"
        );
    }
}

#[test]
fn q15_domain_end_to_end_with_gain_above_unity() {
    // Boosting EQ: centre tap 1.6 (> 0 dBFS), loaded in the FPU-less domain.
    for taps in [100usize, 1500] {
        let h = fir(taps, 3, 1.6);
        let bytes = wav_f32_stereo(&h, &h);
        let art = FirArtifact::parse(&bytes).unwrap();
        let mut ws = vec![0i16; workspace_len_q15(taps)];
        let mut conv = AutoConvolver::from_artifact(&art, 0, &mut ws).unwrap();
        let info = conv.info();
        assert_eq!(info.domain, NumericDomain::FixedQ15);
        assert_eq!(info.engine, select_q15(taps));
        assert_eq!(info.gain_shift, 1);

        // Input level scaled by the filter's RMS gain so the boosted output stays well below
        // full scale (the oracle is unclipped).
        let l2 = h.iter().map(|v| v * v).sum::<f64>().sqrt();
        let x = signal(4000, 9, 0.15 / l2);
        let xq: Vec<i16> = x.iter().map(|&v| (v * 32768.0).round() as i16).collect();
        let mut y = vec![0i16; x.len()];
        for (xi, yo) in xq.chunks(37).zip(y.chunks_mut(37)) {
            conv.process_i16(xi, yo).unwrap();
        }
        let xr: Vec<f64> = xq.iter().map(|&v| v as f64 / 32768.0).collect();
        let reference = oracle(&h, &xr);
        let s = snr(&reference, y.iter().map(|&v| v as f64 / 32768.0));
        // Ceiling: one Q15 output rounding, minus `gain_shift` bits lost to headroom.
        let ideal: Vec<f64> = reference
            .iter()
            .map(|&r| ((r * 16384.0).round() * 2.0) / 32768.0)
            .collect();
        let ceiling = snr(&reference, ideal.into_iter());
        println!(
            "q15 auto taps={taps} engine={:?} snr={s:.2} dB ceiling={ceiling:.2} dB",
            info.engine
        );
        // Q1.15 coefficient rounding of the direct path costs a few dB below the output ceiling.
        assert!(
            s > ceiling - 6.0 && s > 65.0,
            "taps={taps} snr={s:.2} ceiling={ceiling:.2}"
        );
        assert_eq!(conv.saturation_count(), 0);

        // f32 I/O on the Q1.15 engine produces the same samples.
        conv.reset();
        let xf: Vec<f32> = xq.iter().map(|&v| v as f32 / 32768.0).collect();
        let mut yf = vec![0.0f32; x.len()];
        conv.process_f32(&xf, &mut yf).unwrap();
        for (a, b) in y.iter().zip(&yf) {
            assert_eq!(*a as f32 / 32768.0, *b);
        }
    }
}

#[cfg(not(feature = "native-q15"))]
#[test]
fn i16_io_on_f32_engine_and_interleaved_stereo() {
    let taps = 400;
    let (l, r) = (fir(taps, 4, 0.7), fir(taps, 5, 0.3));
    let bytes = wav_f32_stereo(&l, &r);
    let mut ws = vec![0.0f32; 2 * workspace_len_f32(taps)];
    let mut st = AutoStereoConvolver::from_bytes(&bytes, &mut ws).unwrap();
    let x = signal(3000, 11, 0.4);
    let xq: Vec<i16> = x.iter().map(|&v| (v * 32768.0).round() as i16).collect();
    let mut inter: Vec<i16> = xq.iter().flat_map(|&v| [v, v]).collect();
    st.process_interleaved_i16(&mut inter).unwrap();
    let xr: Vec<f64> = xq.iter().map(|&v| v as f64 / 32768.0).collect();
    let (rl, rr) = (oracle(&l, &xr), oracle(&r, &xr));
    for (i, (a, b)) in rl.iter().zip(&rr).enumerate() {
        assert!((inter[2 * i] as f64 / 32768.0 - a).abs() <= 1.0 / 32768.0);
        assert!((inter[2 * i + 1] as f64 / 32768.0 - b).abs() <= 1.0 / 32768.0);
    }
    assert!(st.process_interleaved_i16(&mut inter[..3]).is_err());
}

#[test]
fn mono_artifact_feeds_both_stereo_channels_and_errors_are_typed() {
    // Native domain of the build: f32 with an FPU, Q1.15 under `native-q15` (0.5 / 0.25 and
    // their products are exact in both).
    use rtconv_simd::auto::{NativeElem, workspace_len};
    let text = b"0.5\n0.25\n";
    let mut ws = vec![NativeElem::default(); 2 * workspace_len(2)];
    let mut st = AutoStereoConvolver::from_bytes(text, &mut ws).unwrap();
    let (mut a, mut b) = ([0.0f32; 3], [0.0f32; 3]);
    st.process_f32(&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0], &mut a, &mut b)
        .unwrap();
    assert_eq!(a, [0.5, 0.25, 0.0]);
    assert_eq!(b, [0.0, 0.5, 0.25]);

    let mut tiny = [NativeElem::default(); 3];
    assert!(matches!(
        AutoConvolver::from_bytes(text, 0, &mut tiny),
        Err(rtconv_simd::ConvError::WorkspaceTooSmall { .. })
    ));
    let mut ws = vec![NativeElem::default(); 64];
    assert!(matches!(
        AutoConvolver::from_bytes(text, 1, &mut ws),
        Err(rtconv_simd::ConvError::Artifact(_))
    ));
}

#[test]
fn native_elem_matches_native_domain() {
    use rtconv_simd::auto::{AutoElem, HAS_HARDWARE_FPU, NATIVE_DOMAIN, NativeElem};
    assert_eq!(<NativeElem as AutoElem>::DOMAIN, NATIVE_DOMAIN);
    assert_eq!(HAS_HARDWARE_FPU, NATIVE_DOMAIN == NumericDomain::Float32);
}

/// Host mirror of the `hardware/esp32c6` harness scenario: 300-tap raw `f32` `.bin` artifact
/// loaded in the Q1.15 domain (what an FPU-less ESP32-C6 build selects).
#[test]
fn esp32c6_harness_scenario_on_host() {
    let mut bin = Vec::new();
    for k in 0..300 {
        let v: f32 = match k {
            0 => 0.5,
            200 => -0.25,
            _ => 0.0,
        };
        bin.extend_from_slice(&v.to_le_bytes());
    }
    let mut ws = vec![0i16; workspace_len_q15(300)];
    let mut conv = AutoConvolver::from_bytes(&bin, 0, &mut ws).unwrap();
    assert_eq!(conv.info().engine, AutoEngine::FixedNonUniform);
    let mut x = [0i16; 32];
    let mut y = [0i16; 32];
    x[0] = 16384;
    for block in 0..8 {
        conv.process_i16(&x, &mut y).unwrap();
        x[0] = 0;
        if block == 0 {
            assert!((y[0] - 8192).abs() <= 1, "y0={}", y[0]);
        }
        if block == 6 {
            assert!((y[8] + 4096).abs() <= 1, "y200={}", y[8]);
        }
    }
}

#[cfg(not(feature = "native-q15"))]
#[test]
fn f32_auto_selects_optimized_nonuniform_on_long_ir_oracle_snr() {
    let taps = 16_384usize;
    let h = fir(taps, 42, 0.75);
    let hf: Vec<f32> = h.iter().map(|&v| v as f32).collect();
    let mut ws = vec![0.0f32; engine_len_f32(taps)];
    let mut conv = AutoConvolver::prepare_f32(&hf, &mut ws).unwrap();
    assert_eq!(conv.info().domain, NumericDomain::Float32);
    assert_eq!(conv.info().engine, AutoEngine::NonUniform);
    assert_eq!(conv.info().latency_samples, 0);

    let x = signal(20_000, 99, 0.4);
    let xf: Vec<f32> = x.iter().map(|&v| v as f32).collect();
    let mut y = vec![0.0f32; xf.len()];
    let mut pos = 0usize;
    for &n in [1usize, 31, 64, 100, 7, 256, 513].iter().cycle() {
        if pos >= xf.len() {
            break;
        }
        let n = n.min(xf.len() - pos);
        conv.process_f32(&xf[pos..pos + n], &mut y[pos..pos + n])
            .unwrap();
        pos += n;
    }

    let xq: Vec<f64> = xf.iter().map(|&v| v as f64).collect();
    let reference = oracle(&h, &xq);
    let s = snr(&reference, y.iter().map(|&v| v as f64));
    assert!(
        s > 100.0,
        "AutoConvolver f32 NonUniform 16k oracle SNR = {s:.2} dB (expected > 100 dB)"
    );

    // Verify reset() clears state cleanly and reproduces identical output.
    conv.reset();
    let mut y2 = vec![0.0f32; 512];
    conv.process_f32(&xf[..512], &mut y2).unwrap();
    for (i, (&a, &b)) in y[..512].iter().zip(&y2).enumerate() {
        assert!((a - b).abs() < 1e-6, "reset mismatch at {i}: {a} vs {b}");
    }
}

#[test]
fn q15_auto_optimized_nonuniform_long_ir() {
    let taps = 4_096usize;
    let h = fir(taps, 17, 0.8);
    let bytes = wav_f32_stereo(&h, &h);
    let art = FirArtifact::parse(&bytes).unwrap();
    let mut ws = vec![0i16; workspace_len_q15(taps)];
    let mut conv = AutoConvolver::from_artifact(&art, 0, &mut ws).unwrap();
    assert_eq!(conv.info().domain, NumericDomain::FixedQ15);
    assert_eq!(conv.info().engine, AutoEngine::FixedNonUniform);

    let l2 = h.iter().map(|v| v * v).sum::<f64>().sqrt();
    let x = signal(6_000, 23, 0.15 / l2);
    let xq: Vec<i16> = x.iter().map(|&v| (v * 32768.0).round() as i16).collect();
    let mut y = vec![0i16; x.len()];
    for (xi, yo) in xq.chunks(53).zip(y.chunks_mut(53)) {
        conv.process_i16(xi, yo).unwrap();
    }
    let xr: Vec<f64> = xq.iter().map(|&v| v as f64 / 32768.0).collect();
    let reference = oracle(&h, &xr);
    let s = snr(&reference, y.iter().map(|&v| v as f64 / 32768.0));
    assert_eq!(conv.saturation_count(), 0);
    assert!(
        s > 60.0,
        "AutoConvolver Q1.15 optimized NUPC 4k oracle SNR = {s:.2} dB"
    );
}

/// Raw little-endian `f32` mono artifact (rePhase `.bin`).
#[cfg(feature = "native-q15")]
fn raw_f32(h: &[f64]) -> Vec<u8> {
    h.iter().flat_map(|&v| (v as f32).to_le_bytes()).collect()
}

/// FPU-less build: `prepare_f32` (taps quantized from their IEEE bits, integer-only) prepares
/// exactly the same engine as loading the same taps from a raw `f32` artifact, and both
/// produce bit-identical output, including filters above 0 dBFS (headroom shift).
#[cfg(feature = "native-q15")]
#[test]
fn no_fpu_prepare_f32_matches_from_artifact_bit_exactly() {
    for (taps, peak) in [
        (1usize, 0.5),
        (64, 0.8),
        (128, 1.6),
        (129, 0.7),
        (300, 3.3),
        (2048, 0.9),
        (4096, 0.75),
    ] {
        let h = fir(taps, 31 + taps as u64, peak);
        let hf: Vec<f32> = h.iter().map(|&v| v as f32).collect();
        let bytes = raw_f32(&h);

        let mut ws_a = vec![0i16; engine_len_q15(taps)];
        let mut a = AutoConvolver::prepare_f32(&hf, &mut ws_a).unwrap();
        let mut ws_b = vec![0i16; workspace_len_q15(taps)];
        let mut b = AutoConvolver::from_bytes(&bytes, 0, &mut ws_b).unwrap();
        let (ia, ib) = (a.info(), b.info());
        assert_eq!(
            (
                ia.domain,
                ia.engine,
                ia.taps,
                ia.gain_shift,
                ia.latency_samples
            ),
            (
                ib.domain,
                ib.engine,
                ib.taps,
                ib.gain_shift,
                ib.latency_samples
            ),
            "taps={taps}"
        );
        assert_eq!(ia.domain, NumericDomain::FixedQ15);
        let l2 = h.iter().map(|v| v * v).sum::<f64>().sqrt();
        let x = signal(3000, 5, 0.2 / l2.max(1.0));
        let xq: Vec<i16> = x.iter().map(|&v| (v * 32768.0).round() as i16).collect();
        let (mut ya, mut yb) = (vec![0i16; x.len()], vec![0i16; x.len()]);
        let mut pos = 0usize;
        for &n in [1usize, 64, 37, 200].iter().cycle() {
            if pos >= x.len() {
                break;
            }
            let end = (pos + n).min(x.len());
            a.process_i16(&xq[pos..end], &mut ya[pos..end]).unwrap();
            b.process_i16(&xq[pos..end], &mut yb[pos..end]).unwrap();
            pos = end;
        }
        assert_eq!(ya, yb, "taps={taps}");
        assert_eq!(a.saturation_count(), b.saturation_count());
    }
}

#[cfg(feature = "native-q15")]
#[test]
fn no_fpu_prepare_f32_rejects_non_finite_and_empty() {
    let mut ws = vec![0i16; engine_len_q15(4)];
    assert!(matches!(
        AutoConvolver::prepare_f32(&[0.5, f32::NAN, 0.0, 0.0], &mut ws),
        Err(rtconv_simd::ConvError::NonFiniteCoefficient { index: 1 })
    ));
    assert!(matches!(
        AutoConvolver::prepare_f32(&[0.5, 0.1, f32::NEG_INFINITY, 0.0], &mut ws),
        Err(rtconv_simd::ConvError::NonFiniteCoefficient { index: 2 })
    ));
    assert!(matches!(
        AutoConvolver::prepare_f32(&[], &mut ws),
        Err(rtconv_simd::ConvError::EmptyIr)
    ));
}
