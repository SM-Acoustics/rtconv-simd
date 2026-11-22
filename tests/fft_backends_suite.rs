//! Comprehensive WO-RTCONV-002 validation suite comparing:
//! - `MicroFftBackend` (baseline)
//! - `SimdFixedFftBackend` (no_std SIMD: Scalar, SSE2, AVX2)
//! - `PhastFftBackend` (desktop SIMD, under `--features phastft`)
//! - Independent `f64` DFT & convolution mathematical reference
//!
//! Covers round-trip, Parseval energy theorem, impulses, linear convolution,
//! asymmetric filters, random signals, block boundaries, and `HybridConvolver`
//! time-distributed tail spreading (`tail_spreading = true`).

use rtconv_simd::fft::{FftBackend, FftSize, MicroFftBackend};
use rtconv_simd::{
    HybridConfig, HybridConvolver, PartitionedConfig, PartitionedConvolver, SimdBackend,
    SimdFixedFftBackend,
};
use std::f64::consts::PI;

#[cfg(feature = "phastft")]
use rtconv_simd::PhastFftBackend;

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
        ((self.next_u64() >> 40) as f32 / ((1u32 << 24) as f32)) * 2.0 - 1.0
    }
    fn vec_f32(&mut self, len: usize, scale: f32) -> Vec<f32> {
        (0..len).map(|_| self.next_f32() * scale).collect()
    }
}

/// Independent `f64` reference DFT (bins 0..=N/2).
fn reference_rfft_f64(x: &[f32]) -> (Vec<f64>, Vec<f64>) {
    let n = x.len();
    let bins = n / 2 + 1;
    let mut re = vec![0.0f64; bins];
    let mut im = vec![0.0f64; bins];
    for k in 0..bins {
        let mut sr = 0.0f64;
        let mut si = 0.0f64;
        for (m, &xm) in x.iter().enumerate() {
            let angle = -2.0 * PI * (k as f64) * (m as f64) / (n as f64);
            sr += (xm as f64) * angle.cos();
            si += (xm as f64) * angle.sin();
        }
        re[k] = sr;
        im[k] = si;
    }
    (re, im)
}

/// Independent `f64` reference time-domain linear convolution.
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

fn max_abs_diff(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            assert!(x.is_finite() && y.is_finite());
            (x as f64 - y as f64).abs()
        })
        .fold(0.0f64, f64::max)
}

fn snr_db(reference: &[f32], actual: &[f32]) -> f64 {
    let mut sig = 0.0f64;
    let mut err = 0.0f64;
    for (&r, &a) in reference.iter().zip(actual.iter()) {
        sig += (r as f64) * (r as f64);
        let d = r as f64 - a as f64;
        err += d * d;
    }
    if err == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (sig / err).log10()
    }
}

fn validate_fft_backend<F: FftBackend>(label: &str, fft: &F) {
    let mut rng = Rng::new(0xBEEF_1234_5678);

    for &size in &[FftSize::N64, FftSize::N128, FftSize::N256] {
        assert!(fft.supports_size(size));
        let n = size.len();
        let bins = size.spectrum_bins();
        let x = rng.vec_f32(n, 1.0);
        let (ref_re, ref_im) = reference_rfft_f64(&x);

        let mut spec_re = vec![0.0f32; bins];
        let mut spec_im = vec![0.0f32; bins];
        let mut scratch = vec![0.0f32; fft.scratch_len_f32(size)];
        let mut time_out = vec![0.0f32; n];

        // 1. Forward RFFT vs f64 DFT
        fft.rfft_forward(size, &x, &mut spec_re, &mut spec_im, &mut scratch)
            .unwrap();
        let mut max_spec_err = 0.0f64;
        for k in 0..bins {
            let er = (spec_re[k] as f64 - ref_re[k]).abs();
            let ei = (spec_im[k] as f64 - ref_im[k]).abs();
            max_spec_err = max_spec_err.max(er).max(ei);
        }
        assert!(
            max_spec_err < 2e-4,
            "{label} N={n} forward spectrum error {max_spec_err} >= 2e-4"
        );
        // DC and Nyquist imaginary parts must be ~0
        assert!(spec_im[0].abs() < 1e-6, "{label} N={n} DC imag != 0");
        assert!(spec_im[bins - 1].abs() < 1e-6, "{label} N={n} Nyq imag != 0");

        // 2. Parseval energy theorem: sum |x[n]|^2 == (1/N) * (|X[0]|^2 + |X[N/2]|^2 + 2*sum_{1..N/2-1} |X[k]|^2)
        let time_energy: f64 = x.iter().map(|&v| (v as f64) * (v as f64)).sum();
        let mut freq_energy = (spec_re[0] as f64).powi(2) + (spec_re[bins - 1] as f64).powi(2);
        for k in 1..(bins - 1) {
            freq_energy += 2.0 * ((spec_re[k] as f64).powi(2) + (spec_im[k] as f64).powi(2));
        }
        freq_energy /= n as f64;
        let rel_parseval = (time_energy - freq_energy).abs() / time_energy.max(1e-12);
        assert!(
            rel_parseval < 1e-5,
            "{label} N={n} Parseval relative error {rel_parseval} >= 1e-5"
        );

        // 3. Round-trip RFFT -> IFFT
        fft.rfft_inverse(size, &spec_re, &spec_im, &mut time_out, &mut scratch)
            .unwrap();
        let rt_err = max_abs_diff(&x, &time_out);
        let rt_snr = snr_db(&x, &time_out);
        assert!(
            rt_err < 5e-6 && rt_snr > 120.0,
            "{label} N={n} round-trip err={rt_err:.3e}, snr={rt_snr:.2} dB"
        );
    }
}

fn validate_convolver_with_backend<F: FftBackend + Clone>(label: &str, fft: F) {
    let mut rng = Rng::new(0xFEED_FACE_9999);
    let backend = SimdBackend::detect();

    // Asymmetric mixed-phase IR of non-multiple length (517 taps)
    let ir: Vec<f32> = (0..517)
        .map(|k| {
            let d = k as f32 - 140.0;
            if d < 0.0 {
                0.3 * (d * 0.09).exp() * (0.41 * d).cos()
            } else {
                (-d * 0.025).exp() * (0.27 * d + 0.3).cos()
            }
        })
        .collect();
    let input = rng.vec_f32(1024, 0.8);
    let reference = reference_convolve_f64(&ir, &input);

    // PartitionedConvolver for B = 32, 64, 128
    for &b in &[32usize, 64, 128] {
        let cfg = PartitionedConfig {
            block_size: b,
            max_ir_len: ir.len(),
        };
        let req = PartitionedConvolver::memory_requirements(cfg, &fft).unwrap();
        let mut ws = vec![0.0f32; req.total_f32];
        let mut conv =
            PartitionedConvolver::prepare(&ir, cfg, &mut ws, backend, fft.clone()).unwrap();
        assert_eq!(conv.latency_samples(), 0);
        let mut out = vec![0.0f32; input.len()];
        for (ib, ob) in input.chunks_exact(b).zip(out.chunks_exact_mut(b)) {
            conv.process_block(ib, ob).unwrap();
        }
        let err = max_abs_diff(&reference, &out);
        let snr = snr_db(&reference, &out);
        assert!(
            err < 8e-5 && snr > 95.0,
            "{label} Partitioned(B={b}) err={err:.3e}, snr={snr:.2} dB"
        );
    }

    // HybridConvolver (H128/T128, host block = 32): burst AND tail_spreading = true
    for &spread in &[false, true] {
        let cfg = HybridConfig {
            head_len: 128,
            tail_block_size: 128,
            max_ir_len: ir.len(),
        };
        let req = HybridConvolver::memory_requirements(cfg, &fft).unwrap();
        let mut ws = vec![0.0f32; req.total_f32];
        let mut conv =
            HybridConvolver::prepare(&ir, cfg, &mut ws, backend, fft.clone()).unwrap();
        conv.set_tail_spreading(spread);
        assert_eq!(conv.latency_samples(), 0);
        let mut out = vec![0.0f32; input.len()];
        for (ib, ob) in input.chunks_exact(32).zip(out.chunks_exact_mut(32)) {
            conv.process_block(ib, ob).unwrap();
        }
        let err = max_abs_diff(&reference, &out);
        let snr = snr_db(&reference, &out);
        assert!(
            err < 8e-5 && snr > 95.0,
            "{label} Hybrid(H128/T128, spread={spread}) err={err:.3e}, snr={snr:.2} dB"
        );

        // Verify reset() bit-for-bit reproducibility
        conv.reset();
        let mut out2 = vec![0.0f32; input.len()];
        for (ib, ob) in input.chunks_exact(32).zip(out2.chunks_exact_mut(32)) {
            conv.process_block(ib, ob).unwrap();
        }
        assert_eq!(out, out2, "{label} Hybrid(spread={spread}) reset mismatch");
    }
}

#[test]
fn test_microfft_backend_full_validation() {
    let fft = MicroFftBackend::new();
    assert!(!fft.is_simd_accelerated());
    validate_fft_backend("MicroFftBackend", &fft);
    validate_convolver_with_backend("MicroFftBackend", fft);
}

#[test]
fn test_simd_fixed_fft_scalar_sse2_avx2_validation() {
    let scalar_fft = SimdFixedFftBackend::new(SimdBackend::scalar());
    assert!(!scalar_fft.is_simd_accelerated());
    validate_fft_backend("SimdFixedFft(Scalar)", &scalar_fft);
    validate_convolver_with_backend("SimdFixedFft(Scalar)", scalar_fft);

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if let Ok(sse2) = SimdBackend::try_sse2() {
            let fft = SimdFixedFftBackend::new(sse2);
            assert!(fft.is_simd_accelerated());
            validate_fft_backend("SimdFixedFft(SSE2)", &fft);
            validate_convolver_with_backend("SimdFixedFft(SSE2)", fft);
        }
        if let Ok(avx2) = SimdBackend::try_avx2() {
            let fft = SimdFixedFftBackend::new(avx2);
            assert!(fft.is_simd_accelerated());
            validate_fft_backend("SimdFixedFft(AVX2)", &fft);
            validate_convolver_with_backend("SimdFixedFft(AVX2)", fft);
        }
    }
}

#[cfg(feature = "phastft")]
#[test]
fn test_phastft_backend_full_validation() {
    let fft = PhastFftBackend::new();
    assert!(fft.is_simd_accelerated());
    validate_fft_backend("PhastFftBackend", &fft);
    validate_convolver_with_backend("PhastFftBackend", fft);
}

#[test]
fn test_hybrid_tail_spreading_impulse_and_long_ir_boundaries() {
    let mut rng = Rng::new(2026_0002);
    let ir = rng.vec_f32(2051, 0.15); // 128 head + 1923 tail -> 16 tail partitions
    let fft = SimdFixedFftBackend::new(SimdBackend::detect());
    let backend = SimdBackend::detect();

    // Test delayed impulses hitting every callback phase (0, 1, 2, 3) and boundary
    for &delay in &[0usize, 31, 32, 63, 64, 95, 96, 127, 128, 129, 255, 256] {
        let mut input = vec![0.0f32; 2560];
        input[delay] = 1.0;
        let reference = reference_convolve_f64(&ir, &input);

        let cfg = HybridConfig {
            head_len: 128,
            tail_block_size: 128,
            max_ir_len: ir.len(),
        };
        let req = HybridConvolver::memory_requirements(cfg, &fft).unwrap();
        let mut ws_burst = vec![0.0f32; req.total_f32];
        let mut ws_spread = vec![0.0f32; req.total_f32];

        let mut c_burst = HybridConvolver::prepare(&ir, cfg, &mut ws_burst, backend, fft).unwrap();
        let mut c_spread =
            HybridConvolver::prepare(&ir, cfg, &mut ws_spread, backend, fft).unwrap();
        c_spread.set_tail_spreading(true);

        let mut y_burst = vec![0.0f32; input.len()];
        let mut y_spread = vec![0.0f32; input.len()];
        for (i, (ob, os)) in input
            .chunks_exact(32)
            .zip(y_burst.chunks_exact_mut(32).zip(y_spread.chunks_exact_mut(32)))
        {
            c_burst.process_block(i, ob).unwrap();
            c_spread.process_block(i, os).unwrap();
        }

        let err_ref = max_abs_diff(&reference, &y_spread);
        let err_vs_burst = max_abs_diff(&y_burst, &y_spread);
        assert!(
            err_ref < 5e-5 && err_vs_burst < 1e-6,
            "delay={delay}: err_ref={err_ref:.3e}, err_vs_burst={err_vs_burst:.3e}"
        );
    }
}
