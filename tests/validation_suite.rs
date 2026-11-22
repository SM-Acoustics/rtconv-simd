//! Comprehensive correctness, phase-independence, FFT round-trip, cross-backend,
//! and randomized property test suite for `rtconv_simd`.

use rtconv_simd::fft::{FftBackend, FftSize, MicroFftBackend};
use rtconv_simd::simd_kernels::{cmac_scalar, cmac_simd, dot_product_scalar, dot_product_simd};
use rtconv_simd::{
    ConvError, DirectConfig, DirectConvolver, HybridConfig, HybridConvolver, PartitionedConfig,
    PartitionedConvolver, SimdBackend,
};
use std::f32::consts::PI;

/// Independent double-precision (`f64`) scalar reference convolution.
fn reference_convolve_f64(ir: &[f32], input: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; input.len()];
    for n in 0..input.len() {
        let mut acc = 0.0f64;
        let max_k = ir.len().min(n + 1);
        for k in 0..max_k {
            acc += (ir[k] as f64) * (input[n - k] as f64);
        }
        out[n] = acc as f32;
    }
    out
}

#[derive(Debug, Clone, Copy)]
struct ErrorMetrics {
    max_abs: f64,
    rms: f64,
    snr_db: f64,
}

fn compute_metrics(reference: &[f32], actual: &[f32]) -> ErrorMetrics {
    assert_eq!(reference.len(), actual.len());
    let mut max_abs = 0.0f64;
    let mut err_sq = 0.0f64;
    let mut sig_sq = 0.0f64;
    for (&r, &a) in reference.iter().zip(actual.iter()) {
        assert!(a.is_finite(), "non-finite output sample detected");
        let diff = (r as f64 - a as f64).abs();
        if diff > max_abs {
            max_abs = diff;
        }
        err_sq += diff * diff;
        sig_sq += (r as f64) * (r as f64);
    }
    let rms = (err_sq / reference.len().max(1) as f64).sqrt();
    let snr_db = if err_sq == 0.0 {
        f64::INFINITY
    } else if sig_sq == 0.0 {
        0.0
    } else {
        10.0 * (sig_sq / err_sq).log10()
    };
    ErrorMetrics {
        max_abs,
        rms,
        snr_db,
    }
}

/// Deterministic xorshift64 PRNG in `[-1.0, 1.0]`.
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn next_f32(&mut self) -> f32 {
        let v = (self.next_u64() >> 40) as f32 / ((1u32 << 24) as f32);
        v * 2.0 - 1.0
    }
    fn vec_f32(&mut self, len: usize, scale: f32) -> Vec<f32> {
        (0..len).map(|_| self.next_f32() * scale).collect()
    }
}

fn linear_phase_sinc_ir(len: usize, cutoff: f32) -> Vec<f32> {
    let mid = (len - 1) as f32 * 0.5;
    let mut h = Vec::with_capacity(len);
    for i in 0..len {
        let x = i as f32 - mid;
        let sinc = if x.abs() < 1e-6 {
            2.0 * cutoff
        } else {
            (2.0 * PI * cutoff * x).sin() / (PI * x)
        };
        let win = 0.54 - 0.46 * (2.0 * PI * i as f32 / (len.saturating_sub(1).max(1) as f32)).cos();
        h.push(sinc * win);
    }
    h
}

fn minimum_phase_ir(len: usize) -> Vec<f32> {
    // Causal damped resonant decay: strong energy at tap 0, asymmetric decay.
    (0..len)
        .map(|k| {
            let t = k as f32;
            (-t * 0.045).exp() * (0.37 * t).cos()
        })
        .collect()
}

fn mixed_phase_ir(len: usize) -> Vec<f32> {
    // Asymmetric pre-ringing + delayed peak + oscillatory tail.
    let peak = (len as f32) * 0.3;
    (0..len)
        .map(|k| {
            let d = k as f32 - peak;
            if d < 0.0 {
                0.35 * (d * 0.12).exp() * (0.5 * d).cos()
            } else {
                (-d * 0.03).exp() * (0.29 * d + 0.4).cos()
            }
        })
        .collect()
}

fn run_direct_block(
    ir: &[f32],
    input: &[f32],
    block_size: usize,
    backend: SimdBackend,
) -> Vec<f32> {
    let cfg = DirectConfig {
        max_ir_len: ir.len(),
    };
    let req = DirectConvolver::memory_requirements(cfg).unwrap();
    let mut ws = vec![0.0f32; req.total_f32];
    let mut conv = DirectConvolver::prepare(ir, cfg, &mut ws, backend).unwrap();
    assert_eq!(conv.latency_samples(), 0);
    let mut out = vec![0.0f32; input.len()];
    for (in_blk, out_blk) in input.chunks(block_size).zip(out.chunks_mut(block_size)) {
        conv.process_block(in_blk, out_blk).unwrap();
    }
    out
}

fn run_direct_sample(ir: &[f32], input: &[f32], backend: SimdBackend) -> Vec<f32> {
    let cfg = DirectConfig {
        max_ir_len: ir.len(),
    };
    let req = DirectConvolver::memory_requirements(cfg).unwrap();
    let mut ws = vec![0.0f32; req.total_f32];
    let mut conv = DirectConvolver::prepare(ir, cfg, &mut ws, backend).unwrap();
    input.iter().map(|&x| conv.process_sample(x)).collect()
}

fn run_partitioned(ir: &[f32], input: &[f32], block_size: usize, backend: SimdBackend) -> Vec<f32> {
    assert_eq!(input.len() % block_size, 0);
    let cfg = PartitionedConfig {
        block_size,
        max_ir_len: ir.len(),
    };
    let fft = MicroFftBackend::new();
    let req = PartitionedConvolver::memory_requirements(cfg, &fft).unwrap();
    let mut ws = vec![0.0f32; req.total_f32];
    let mut conv = PartitionedConvolver::prepare(ir, cfg, &mut ws, backend, fft).unwrap();
    assert_eq!(conv.latency_samples(), 0);
    let mut out = vec![0.0f32; input.len()];
    for (in_blk, out_blk) in input
        .chunks_exact(block_size)
        .zip(out.chunks_exact_mut(block_size))
    {
        conv.process_block(in_blk, out_blk).unwrap();
    }
    out
}

fn run_hybrid(
    ir: &[f32],
    input: &[f32],
    host_block: usize,
    tail_block: usize,
    backend: SimdBackend,
) -> Vec<f32> {
    let cfg = HybridConfig {
        head_len: tail_block,
        tail_block_size: tail_block,
        max_ir_len: ir.len(),
    };
    let fft = MicroFftBackend::new();
    let req = HybridConvolver::memory_requirements(cfg, &fft).unwrap();
    let mut ws = vec![0.0f32; req.total_f32];
    let mut conv = HybridConvolver::prepare(ir, cfg, &mut ws, backend, fft).unwrap();
    assert_eq!(conv.latency_samples(), 0);
    let mut out = vec![0.0f32; input.len()];
    for (in_blk, out_blk) in input.chunks(host_block).zip(out.chunks_mut(host_block)) {
        conv.process_block(in_blk, out_blk).unwrap();
    }
    out
}

fn assert_all_engines_match(
    name: &str,
    ir: &[f32],
    input: &[f32],
    max_abs_tol: f64,
    min_snr_db: f64,
) {
    let reference = reference_convolve_f64(ir, input);
    let backend = SimdBackend::detect_or_baseline();

    // Direct (sample-by-sample and block-by-block)
    let y_dir_s = run_direct_sample(ir, input, backend);
    let m_dir_s = compute_metrics(&reference, &y_dir_s);
    assert!(
        m_dir_s.max_abs <= max_abs_tol,
        "{name} direct_sample max_abs={} > {max_abs_tol}",
        m_dir_s.max_abs
    );

    let y_dir_b = run_direct_block(ir, input, 32, backend);
    let m_dir_b = compute_metrics(&reference, &y_dir_b);
    assert!(
        m_dir_b.max_abs <= max_abs_tol,
        "{name} direct_block max_abs={} > {max_abs_tol}",
        m_dir_b.max_abs
    );

    // Partitioned (block 32/FFT 64, block 64/FFT 128, block 128/FFT 256)
    for &b in &[32usize, 64, 128] {
        if input.len().is_multiple_of(b) {
            let y_part = run_partitioned(ir, input, b, backend);
            let m = compute_metrics(&reference, &y_part);
            assert!(
                m.max_abs <= max_abs_tol,
                "{name} partitioned(B={b}) max_abs={} > {max_abs_tol}, rms={}, snr={}",
                m.max_abs,
                m.rms,
                m.snr_db
            );
            if min_snr_db.is_finite() {
                assert!(
                    m.snr_db >= min_snr_db,
                    "{name} partitioned(B={b}) snr={} < {min_snr_db}",
                    m.snr_db
                );
            }
        }
    }

    // Hybrid (host block 32, tail block 64 and 128)
    for &tail_b in &[32usize, 64, 128] {
        let y_hyb = run_hybrid(ir, input, 32, tail_b, backend);
        let m = compute_metrics(&reference, &y_hyb);
        assert!(
            m.max_abs <= max_abs_tol,
            "{name} hybrid(B_host=32,B_tail={tail_b}) max_abs={} > {max_abs_tol}, rms={}, snr={}",
            m.max_abs,
            m.rms,
            m.snr_db
        );
        if min_snr_db.is_finite() {
            assert!(
                m.snr_db >= min_snr_db,
                "{name} hybrid(B_host=32,B_tail={tail_b}) snr={} < {min_snr_db}",
                m.snr_db
            );
        }
    }
}

#[test]
fn test_fft_roundtrip_and_spectral_multiply_64_128_256() {
    let fft = MicroFftBackend::new();
    let backend = SimdBackend::detect_or_baseline();
    let mut rng = Rng::new(0x1234_5678_9ABC_DEF0);

    for &size in &[FftSize::N64, FftSize::N128, FftSize::N256] {
        let n = size.len();
        let bins = size.spectrum_bins();
        let x = rng.vec_f32(n, 1.0);
        let mut h_time = vec![0.0f32; n];
        // Impulse at delay 5 with gain 0.75 + tap at 13 with gain -0.25 (circularly valid on padded x)
        h_time[5] = 0.75;
        h_time[13] = -0.25;

        let mut x_re = vec![0.0f32; bins];
        let mut x_im = vec![0.0f32; bins];
        let mut h_re = vec![0.0f32; bins];
        let mut h_im = vec![0.0f32; bins];
        let mut acc_re = vec![0.0f32; bins];
        let mut acc_im = vec![0.0f32; bins];
        let mut out = vec![0.0f32; n];

        // 1. Pure RFFT -> IFFT round-trip
        fft.rfft_forward(size, &x, &mut x_re, &mut x_im, &mut [])
            .unwrap();
        fft.rfft_inverse(size, &x_re, &x_im, &mut out, &mut [])
            .unwrap();
        let m_rt = compute_metrics(&x, &out);
        assert!(
            m_rt.max_abs < 5e-6 && m_rt.snr_db > 120.0,
            "FFT {n} roundtrip failed: {m_rt:?}"
        );

        // 2. Full RFFT -> CMAC -> IFFT circular convolution check
        fft.rfft_forward(size, &h_time, &mut h_re, &mut h_im, &mut [])
            .unwrap();
        cmac_simd(
            backend.level(),
            &mut acc_re,
            &mut acc_im,
            &x_re,
            &x_im,
            &h_re,
            &h_im,
        );
        fft.rfft_inverse(size, &acc_re, &acc_im, &mut out, &mut [])
            .unwrap();

        let mut expected = vec![0.0f32; n];
        for i in 0..n {
            expected[i] = 0.75 * x[(i + n - 5) % n] - 0.25 * x[(i + n - 13) % n];
        }
        let m_conv = compute_metrics(&expected, &out);
        assert!(
            m_conv.max_abs < 5e-6 && m_conv.snr_db > 115.0,
            "FFT {n} RFFT->CMAC->IFFT failed: {m_conv:?}"
        );
    }
}

#[test]
fn test_case_01_unit_and_delayed_impulses() {
    let mut rng = Rng::new(42);
    let ir = rng.vec_f32(317, 0.5);
    for &delay in &[0usize, 1, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129] {
        let mut input = vec![0.0f32; 1024];
        input[delay] = 1.0;
        assert_all_engines_match(&format!("impulse_delay_{delay}"), &ir, &input, 2e-5, 100.0);
    }
}

#[test]
fn test_case_02_identity_and_zero_fir() {
    let mut rng = Rng::new(77);
    let input = rng.vec_f32(512, 1.0);
    assert_all_engines_match("identity_fir", &[1.0], &input, 5e-6, 115.0);
    assert_all_engines_match("zero_fir", &[0.0; 137], &input, 1e-7, f64::NEG_INFINITY);
}

#[test]
fn test_case_03_04_05_06_phase_agnostic_firs() {
    let mut rng = Rng::new(999);
    let input = rng.vec_f32(1024, 0.8);

    // 3. Linear-phase symmetric (odd & even)
    assert_all_engines_match(
        "linear_phase_odd_199",
        &linear_phase_sinc_ir(199, 0.18),
        &input,
        2e-5,
        100.0,
    );
    assert_all_engines_match(
        "linear_phase_even_256",
        &linear_phase_sinc_ir(256, 0.22),
        &input,
        2e-5,
        100.0,
    );

    // 4. Minimum-phase asymmetric
    assert_all_engines_match("min_phase_211", &minimum_phase_ir(211), &input, 5e-5, 100.0);

    // 5. Mixed-phase asymmetric
    assert_all_engines_match("mixed_phase_307", &mixed_phase_ir(307), &input, 5e-5, 100.0);

    // 6. Random FIR
    let rand_ir = rng.vec_f32(523, 0.25);
    assert_all_engines_match("random_fir_523", &rand_ir, &input, 5e-5, 100.0);
}

#[test]
fn test_case_08_09_multitone_and_silence_transitions() {
    let ir = mixed_phase_ir(173);
    let n = 1024;
    let mut multitone = vec![0.0f32; n];
    for (i, sample) in multitone.iter_mut().enumerate() {
        let t = i as f32 / 48_000.0;
        *sample = 0.4 * (2.0 * PI * 440.0 * t).sin()
            + 0.3 * (2.0 * PI * 1760.0 * t).cos()
            + 0.2 * (2.0 * PI * 6311.0 * t).sin();
    }
    assert_all_engines_match("multitone", &ir, &multitone, 5e-5, 100.0);

    // Silence -> signal -> silence (including tail drain)
    let mut gated = vec![0.0f32; 1024];
    gated[128..512].copy_from_slice(&multitone[128..512]);
    assert_all_engines_match("silence_signal_silence", &ir, &gated, 5e-5, 100.0);
}

#[test]
fn test_case_10_reset_and_stream_reproducibility() {
    let mut rng = Rng::new(2026);
    let ir = rng.vec_f32(290, 0.3);
    let input = rng.vec_f32(512, 0.9);
    let backend = SimdBackend::detect_or_baseline();
    let fft = MicroFftBackend::new();

    // Direct reset
    let dcfg = DirectConfig {
        max_ir_len: ir.len(),
    };
    let mut dws = vec![
        0.0f32;
        DirectConvolver::memory_requirements(dcfg)
            .unwrap()
            .total_f32
    ];
    let mut dconv = DirectConvolver::prepare(&ir, dcfg, &mut dws, backend).unwrap();
    let mut out1 = vec![0.0f32; 512];
    let mut out2 = vec![0.0f32; 512];
    dconv.process_block(&input, &mut out1).unwrap();
    dconv.reset();
    dconv.process_block(&input, &mut out2).unwrap();
    assert_eq!(out1, out2);

    // Partitioned reset
    let pcfg = PartitionedConfig {
        block_size: 64,
        max_ir_len: ir.len(),
    };
    let mut pws = vec![
        0.0f32;
        PartitionedConvolver::memory_requirements(pcfg, &fft)
            .unwrap()
            .total_f32
    ];
    let mut pconv = PartitionedConvolver::prepare(&ir, pcfg, &mut pws, backend, fft).unwrap();
    for (i, o) in input
        .as_chunks::<64>()
        .0
        .iter()
        .zip(out1.as_chunks_mut::<64>().0.iter_mut())
    {
        pconv.process_block(i, o).unwrap();
    }
    pconv.reset();
    for (i, o) in input
        .as_chunks::<64>()
        .0
        .iter()
        .zip(out2.as_chunks_mut::<64>().0.iter_mut())
    {
        pconv.process_block(i, o).unwrap();
    }
    assert_eq!(out1, out2);

    // Hybrid reset
    let hcfg = HybridConfig {
        head_len: 128,
        tail_block_size: 128,
        max_ir_len: ir.len(),
    };
    let mut hws = vec![
        0.0f32;
        HybridConvolver::memory_requirements(hcfg, &fft)
            .unwrap()
            .total_f32
    ];
    let mut hconv = HybridConvolver::prepare(&ir, hcfg, &mut hws, backend, fft).unwrap();
    for (i, o) in input
        .as_chunks::<32>()
        .0
        .iter()
        .zip(out1.as_chunks_mut::<32>().0.iter_mut())
    {
        hconv.process_block(i, o).unwrap();
    }
    hconv.reset();
    for (i, o) in input
        .as_chunks::<32>()
        .0
        .iter()
        .zip(out2.as_chunks_mut::<32>().0.iter_mut())
    {
        hconv.process_block(i, o).unwrap();
    }
    assert_eq!(out1, out2);
}

#[test]
fn test_case_11_12_13_non_multiple_ir_boundaries_and_tail_drain() {
    let mut rng = Rng::new(314159);
    for &ir_len in &[1usize, 7, 31, 32, 33, 63, 64, 65, 127, 128, 129, 257, 513] {
        let ir = rng.vec_f32(ir_len, 0.25);
        // Active burst followed by full tail drain of zeros
        let mut input = vec![0.0f32; 1024];
        for sample in input.iter_mut().take(256) {
            *sample = rng.next_f32();
        }
        assert_all_engines_match(&format!("ir_len_{ir_len}"), &ir, &input, 5e-5, 95.0);
    }
}

#[test]
fn test_case_14_cross_backend_agreement() {
    let mut rng = Rng::new(7777);
    let ir = rng.vec_f32(333, 0.3);
    let input = rng.vec_f32(512, 0.9);
    let reference = reference_convolve_f64(&ir, &input);

    let mut backends = vec![SimdBackend::scalar(), SimdBackend::baseline()];
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if let Ok(b) = SimdBackend::try_sse2() {
            backends.push(b);
        }
        if let Ok(b) = SimdBackend::try_sse4_2() {
            backends.push(b);
        }
        if let Ok(b) = SimdBackend::try_avx2() {
            backends.push(b);
        }
        if let Ok(b) = SimdBackend::try_avx512() {
            backends.push(b);
        }
    }

    for backend in backends {
        // Kernels
        let d_sc = dot_product_scalar(&ir, &input[..ir.len()]);
        let d_sm = dot_product_simd(backend.level(), &ir, &input[..ir.len()]);
        assert!(
            (d_sc - d_sm).abs() < 1e-4,
            "dot mismatch for {}: scalar={d_sc}, simd={d_sm}",
            backend.name()
        );

        let mut acc_re_s = vec![0.0f32; 65];
        let mut acc_im_s = vec![0.0f32; 65];
        let mut acc_re_v = vec![0.0f32; 65];
        let mut acc_im_v = vec![0.0f32; 65];
        let xr = rng.vec_f32(65, 1.0);
        let xi = rng.vec_f32(65, 1.0);
        let hr = rng.vec_f32(65, 1.0);
        let hi = rng.vec_f32(65, 1.0);
        cmac_scalar(&mut acc_re_s, &mut acc_im_s, &xr, &xi, &hr, &hi);
        cmac_simd(
            backend.level(),
            &mut acc_re_v,
            &mut acc_im_v,
            &xr,
            &xi,
            &hr,
            &hi,
        );
        assert!(compute_metrics(&acc_re_s, &acc_re_v).max_abs < 1e-5);
        assert!(compute_metrics(&acc_im_s, &acc_im_v).max_abs < 1e-5);

        // Full engines
        let y_dir = run_direct_block(&ir, &input, 32, backend);
        let y_part = run_partitioned(&ir, &input, 64, backend);
        let y_hyb = run_hybrid(&ir, &input, 32, 128, backend);

        assert!(compute_metrics(&reference, &y_dir).max_abs < 2e-5);
        assert!(compute_metrics(&reference, &y_part).max_abs < 5e-5);
        assert!(compute_metrics(&reference, &y_hyb).max_abs < 5e-5);
    }
}

#[test]
fn test_randomized_property_sweep() {
    let mut rng = Rng::new(0xCAFE_BABE_1234_5678);
    for trial in 0..20 {
        let ir_len = 1 + (rng.next_u64() as usize % 400);
        let ir = rng.vec_f32(ir_len, 0.2);
        let input = rng.vec_f32(512, 0.8);
        assert_all_engines_match(
            &format!("prop_trial_{trial}_len_{ir_len}"),
            &ir,
            &input,
            8e-5,
            95.0,
        );
    }
}

#[test]
fn test_validation_errors() {
    let mut ws = vec![0.0f32; 4096];
    let backend = SimdBackend::detect_or_baseline();
    let fft = MicroFftBackend::new();

    assert_eq!(
        DirectConvolver::prepare(&[], DirectConfig { max_ir_len: 16 }, &mut ws, backend).err(),
        Some(ConvError::EmptyIr)
    );
    assert_eq!(
        DirectConvolver::prepare(
            &[1.0, f32::NAN],
            DirectConfig { max_ir_len: 16 },
            &mut ws,
            backend
        )
        .err(),
        Some(ConvError::NonFiniteCoefficient { index: 1 })
    );
    assert_eq!(
        DirectConvolver::prepare(
            &[1.0; 20],
            DirectConfig { max_ir_len: 16 },
            &mut ws,
            backend
        )
        .err(),
        Some(ConvError::IrTooLong {
            ir_len: 20,
            max_ir_len: 16
        })
    );
    assert!(matches!(
        DirectConvolver::prepare(
            &[1.0; 16],
            DirectConfig { max_ir_len: 16 },
            &mut ws[..10],
            backend
        )
        .err(),
        Some(ConvError::WorkspaceTooSmall { .. })
    ));
    assert_eq!(
        PartitionedConvolver::memory_requirements(
            PartitionedConfig {
                block_size: 48,
                max_ir_len: 128
            },
            &fft
        )
        .err(),
        Some(ConvError::InvalidBlockSize { block_size: 48 })
    );
    assert_eq!(
        HybridConvolver::memory_requirements(
            HybridConfig {
                head_len: 32,
                tail_block_size: 64,
                max_ir_len: 256
            },
            &fft
        )
        .err(),
        Some(ConvError::InvalidHybridConfig {
            head_len: 32,
            tail_block_size: 64
        })
    );
}
