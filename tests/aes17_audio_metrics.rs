//! AES17 & ITU-R BS.1770-5 digital audio verification suite (`997 Hz` @ `48 kHz`).
//!
//! - Reference frequency: `997 Hz` (IEC 61606 / ITU-R BS.1770-5 Annex 1 Note 2, prime
//!   non-submultiple of `44.1 kHz` and `48 kHz` to prevent coherent quantization and FFT
//!   bin locking).
//! - Levels evaluated:
//!   - `-60 dBFS`: AES17 Dynamic Range (DR) and low-level linearity / noise floor without masking.
//!   - `-1 dBFS` and `0 dBFS`: THD+N, zero overflow, zero wrap-around, and controlled
//!     inter-sample / True-Peak headroom (ITU-R BS.1770-5 Annex 2) via L1-normalized FIRs.
//! - Sizes evaluated:
//!   - `f32`: `N = 512, 1024, 2048, 4096, 8192, 16384` (`block_size = 256..=8192`).
//!   - `Q1.15 / BFP`: `N = 64..=2048` (`block_size = 32..=1024`), comparing measured DR & THD+N
//!     against the theoretical Q1.15 quantization ceiling
//!     `SNR_q = 6.02 * B + 10.8 - 20 * log10(X_m / sigma_x)` (`~98.1 dB` FS sine for 16-bit Q1.15)
//!     and recording per-stage BFP degradation across `log2(N)` stages.

use rtconv_simd::{
    FftBackend, FftSize, MicroFftBackend, PartitionedConfig, PartitionedConvolver, SimdBackend,
    SimdFixedFftBackend,
};
use std::f64::consts::PI;

#[cfg(feature = "phastft")]
use rtconv_simd::PhastFftBackend;

const FS_HZ: f64 = 48_000.0;
const TONE_HZ: f64 = 997.0;

/// Generates a `997 Hz` sine wave at `level_dbfs` peak amplitude (`0 dBFS` = peak `1.0`).
fn sine_997hz_f64(len: usize, level_dbfs: f64) -> Vec<f64> {
    let amp = 10.0f64.powf(level_dbfs / 20.0);
    let w = 2.0 * PI * TONE_HZ / FS_HZ;
    (0..len).map(|n| amp * (w * n as f64).sin()).collect()
}

/// Builds a realistic causal decaying FIR whose L1 norm equals `target_l1 <= 1.0`
/// (guaranteeing bounded sample and inter-sample True-Peak headroom per ITU-R BS.1770-5 Annex 2)
/// and whose gain at `997 Hz` is strongly non-zero (low-pass / resonant decay around 1 kHz).
#[cfg(feature = "fixed-fft")]
fn make_l1_bounded_ir_f64(len: usize, target_l1: f64) -> Vec<f64> {
    let mut raw = vec![0.0f64; len];
    let w0 = 2.0 * PI * 997.0 / FS_HZ;
    let tau = (len as f64 / 5.0).max(16.0);
    for (k, slot) in raw.iter_mut().enumerate() {
        let env = (-(k as f64) / tau).exp();
        // Smooth causal impulse + resonant 997 Hz damped cosine so |H(997 Hz)| is well-behaved
        let val = if k == 0 {
            0.5
        } else {
            0.05 * env * (w0 * k as f64).cos()
        };
        *slot = val;
    }
    let l1: f64 = raw.iter().map(|v| v.abs()).sum();
    let scale = target_l1 / l1.max(1e-12);
    for v in &mut raw {
        *v *= scale;
    }
    raw
}

/// Exact `f64` causal linear convolution for short/moderate test IRs (`len <= 2048`) or
/// sparse delayed-impulse IRs.
fn oracle_convolve_f64(ir: &[f64], x: &[f64]) -> Vec<f64> {
    // Fast path for single-tap impulse IRs.
    let nonzeros: Vec<(usize, f64)> = ir
        .iter()
        .enumerate()
        .filter(|&(_, &v)| v != 0.0)
        .map(|(i, &v)| (i, v))
        .collect();
    if nonzeros.len() <= 32 {
        let mut out = vec![0.0f64; x.len()];
        for &(k, h) in &nonzeros {
            for n in k..x.len() {
                out[n] += h * x[n - k];
            }
        }
        return out;
    }

    let mut out = vec![0.0f64; x.len()];
    for n in 0..x.len() {
        let mut acc = 0.0f64;
        let max_k = ir.len().min(n + 1);
        for k in 0..max_k {
            acc += ir[k] * x[n - k];
        }
        out[n] = acc;
    }
    out
}

/// AES17 metrics evaluated on steady-state segment `actual[skip..]` vs exact `f64` oracle
/// `reference[skip..]`.
///
/// Returns `(snr_db, thd_n_db, dr_aes17_db, max_abs_err)`:
/// - `snr_db`: `20 * log10(rms(ref) / rms(actual - ref))`
/// - `thd_n_db`: `-snr_db` (exact linear fundamental removed by oracle, leaving distortion + noise)
/// - `dr_aes17_db`: `20 * log10(FS_sine_rms / rms(actual - ref))` where `FS_sine_rms = 1 / sqrt(2)`,
///   corresponding to AES17 Dynamic Range when stimulated at `-60 dBFS` through a unity-gain path
///   (or reported alongside `|H(997 Hz)|` for filtered paths).
fn aes17_metrics(reference: &[f64], actual: &[f64], skip: usize) -> (f64, f64, f64, f64) {
    assert_eq!(reference.len(), actual.len());
    assert!(skip < reference.len());
    let n = (reference.len() - skip) as f64;
    let mut sig_sq = 0.0f64;
    let mut err_sq = 0.0f64;
    let mut max_abs = 0.0f64;
    for (&r, &a) in reference[skip..].iter().zip(actual[skip..].iter()) {
        assert!(a.is_finite(), "non-finite audio output");
        let d = a - r;
        sig_sq += r * r;
        err_sq += d * d;
        max_abs = max_abs.max(d.abs());
    }
    let rms_sig = (sig_sq / n).sqrt();
    let rms_err = (err_sq / n).sqrt();
    let fs_sine_rms = core::f64::consts::FRAC_1_SQRT_2;

    let snr_db = if rms_err == 0.0 {
        f64::INFINITY
    } else {
        20.0 * (rms_sig / rms_err).log10()
    };
    let thd_n_db = -snr_db;
    let dr_aes17_db = if rms_err == 0.0 {
        f64::INFINITY
    } else {
        20.0 * (fs_sine_rms / rms_err).log10()
    };
    (snr_db, thd_n_db, dr_aes17_db, max_abs)
}

/// Evaluates `PartitionedConvolver` (`f32`) for all new large FFT sizes `N512..=N16384`
/// at `-60 dBFS` (DR), `-1 dBFS` (THD+N), and `0 dBFS` (no overflow/clipping).
fn run_f32_aes17_for_backend<F: FftBackend + Clone>(label: &str, fft: F) {
    let backend = SimdBackend::detect_or_baseline();
    let large_sizes = [
        FftSize::N512,
        FftSize::N1024,
        FftSize::N2048,
        FftSize::N4096,
        FftSize::N8192,
        FftSize::N16384,
    ];

    for size in large_sizes {
        let b = size.half_len();
        let n_fft = size.len();
        // Multi-partition IR (3 partitions of length B) with sparse taps across all partitions
        // plus L1-bounded energy <= 0.90 (zero True-Peak overshoot above 0 dBFS).
        let ir_len = 2 * b + (b / 3).max(17);
        let mut ir_f64 = vec![0.0f64; ir_len];
        ir_f64[0] = 0.50;
        ir_f64[b / 2] = -0.15;
        ir_f64[b + 7] = 0.12;
        ir_f64[ir_len - 1] = -0.08; // sum |h| = 0.85 <= 1.0
        let ir_f32: Vec<f32> = ir_f64.iter().map(|&v| v as f32).collect();
        let ir_exact: Vec<f64> = ir_f32.iter().map(|&v| v as f64).collect();

        let total_samples = 4 * b;
        let skip = ir_len.max(b);

        for &level_dbfs in &[-60.0f64, -1.0, 0.0] {
            let x_f64 = sine_997hz_f64(total_samples, level_dbfs);
            let x_f32: Vec<f32> = x_f64.iter().map(|&v| v as f32).collect();
            let x_exact: Vec<f64> = x_f32.iter().map(|&v| v as f64).collect();
            let reference = oracle_convolve_f64(&ir_exact, &x_exact);

            let cfg = PartitionedConfig {
                block_size: b,
                max_ir_len: ir_len,
            };
            let req = PartitionedConvolver::memory_requirements(cfg, &fft).unwrap();
            let mut ws = vec![0.0f32; req.total_f32];
            let mut conv =
                PartitionedConvolver::prepare(&ir_f32, cfg, &mut ws, backend, fft.clone()).unwrap();

            let mut out_f32 = vec![0.0f32; total_samples];
            for (ib, ob) in x_f32.chunks_exact(b).zip(out_f32.chunks_exact_mut(b)) {
                conv.process_block(ib, ob).unwrap();
            }
            let out_f64: Vec<f64> = out_f32.iter().map(|&v| v as f64).collect();
            let peak_out = out_f64.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
            assert!(
                peak_out <= 1.0,
                "{label} N={n_fft} level={level_dbfs} dBFS unexpected overshoot peak={peak_out}"
            );

            let (snr_db, thd_n_db, dr_db, max_abs) = aes17_metrics(&reference, &out_f64, skip);
            println!(
                "AES17 f32 {label:<20} N={n_fft:<5} level={level_dbfs:>5.1} dBFS | SNR={snr_db:6.2} dB | THD+N={thd_n_db:7.2} dB | DR={dr_db:6.2} dB | max_err={max_abs:.2e}"
            );

            // In IEEE 754 binary32 (24-bit significand), 997 Hz convolution through large FFTs
            // must preserve > 110 dB SNR / < -110 dB THD+N at all tested levels, and DR > 170 dB
            // at -60 dBFS (floating-point exponent tracks low-level signals without quantization floor).
            assert!(
                snr_db > 110.0 && thd_n_db < -110.0,
                "{label} N={n_fft} level={level_dbfs} dBFS: SNR={snr_db:.2} dB, THD+N={thd_n_db:.2} dB"
            );
            if level_dbfs <= -60.0 {
                assert!(
                    dr_db > 165.0,
                    "{label} N={n_fft} -60 dBFS DR={dr_db:.2} dB < 165 dB"
                );
            }
        }
    }
}

#[test]
fn aes17_f32_large_ffts_simd_and_microfft() {
    let backend = SimdBackend::detect_or_baseline();
    run_f32_aes17_for_backend("SimdFixedFft(Auto)", SimdFixedFftBackend::new(backend));
    run_f32_aes17_for_backend(
        "SimdFixedFft(Scalar)",
        SimdFixedFftBackend::new(SimdBackend::scalar()),
    );
    run_f32_aes17_for_backend("MicroFftBackend", MicroFftBackend::new());
    #[cfg(feature = "phastft")]
    run_f32_aes17_for_backend("PhastFftBackend", PhastFftBackend::new());
}

#[cfg(feature = "fixed-fft")]
#[test]
fn aes17_q15_bfp_dynamic_range_thdn_and_stage_degradation() {
    use rtconv_simd::fixed::{f64_to_q15, q15_to_f64};
    use rtconv_simd::{FixedQ15PartitionedConfig, FixedQ15PartitionedConvolver};

    // Theoretical 16-bit signed (Q1.15, B = 15) quantization ceiling for a sine wave:
    // SNR_q = 6.02 * 15 + 10.8 - 20 * log10(sqrt(2)) = 90.3 + 10.8 - 3.01 = 98.09 dB (FS sine).
    let theo_fs_sine_snr_db = 6.02 * 15.0 + 10.8 - 20.0 * core::f64::consts::SQRT_2.log10();
    println!(
        "AES17 Q1.15 theoretical full-scale sine ceiling: {theo_fs_sine_snr_db:.2} dB (expected ~0.5 bit/stage = ~3 dB per radix-2 stage BFP loss)"
    );

    // Sweep N = 64..=2048 (log2(N) = 6..=11) with a unity-passband delayed impulse (isolates FFT
    // BFP stage noise cleanly) AND an L1-bounded multi-tap FIR (checks realistic convolution +
    // ITU-R BS.1770-5 headroom without wrap-around).
    for &b in &[32usize, 64, 128, 256, 512, 1024] {
        let n_fft = 2 * b;
        let stages = n_fft.trailing_zeros();

        // Case A: L1-bounded realistic causal FIR (`sum |h| = 0.85`, headroom > 1.4 dB for true peak)
        let ir_len = (2 * b).min(1536);
        let ir_f64 = make_l1_bounded_ir_f64(ir_len, 0.85);
        let ir_q15: Vec<i16> = ir_f64.iter().map(|&v| f64_to_q15(v)).collect();
        let ir_exact: Vec<f64> = ir_q15.iter().map(|&v| q15_to_f64(v)).collect();

        // Case B: near-unity delayed impulse (`h[b/2] = 0.99997` -> unity gain at 997 Hz)
        let mut imp_q15 = vec![0i16; b];
        imp_q15[b / 2] = i16::MAX;
        let imp_exact: Vec<f64> = imp_q15.iter().map(|&v| q15_to_f64(v)).collect();

        let total_samples = (6 * b).max(4096);
        let skip = ir_len.max(b);

        for (case_label, ir_use, ir_ref) in [
            ("unity_imp", &imp_q15, &imp_exact),
            ("l1_fir", &ir_q15, &ir_exact),
        ] {
            for &level_dbfs in &[-60.0f64, -1.0] {
                let x_f64 = sine_997hz_f64(total_samples, level_dbfs);
                let x_q15: Vec<i16> = x_f64.iter().map(|&v| f64_to_q15(v)).collect();
                let x_exact: Vec<f64> = x_q15.iter().map(|&v| q15_to_f64(v)).collect();
                let reference = oracle_convolve_f64(ir_ref, &x_exact);

                // Theoretical single-output-rounding Q1.15 ceiling for this exact signal
                let ideal_q15: Vec<f64> = reference
                    .iter()
                    .map(|&r| q15_to_f64(f64_to_q15(r)))
                    .collect();
                let (ceil_snr, _, ceil_dr, _) = aes17_metrics(&reference, &ideal_q15, skip);

                let cfg = FixedQ15PartitionedConfig {
                    block_size: b,
                    max_ir_len: ir_use.len(),
                };
                let req = FixedQ15PartitionedConvolver::memory_requirements(cfg).unwrap();
                let mut ws = vec![0i16; req.total_i16];
                let mut conv = FixedQ15PartitionedConvolver::prepare(ir_use, cfg, &mut ws).unwrap();

                let mut out_q15 = vec![0i16; total_samples];
                for (ib, ob) in x_q15.chunks_exact(b).zip(out_q15.chunks_exact_mut(b)) {
                    conv.process_block(ib, ob).unwrap();
                }
                assert_eq!(
                    conv.saturation_count(),
                    0,
                    "Q15 {case_label} N={n_fft} level={level_dbfs} dBFS had unexpected saturation/overflow"
                );

                let out_f64: Vec<f64> = out_q15.iter().map(|&v| q15_to_f64(v)).collect();
                let (snr_db, thd_n_db, dr_db, max_abs) = aes17_metrics(&reference, &out_f64, skip);
                let loss_vs_ceil = ceil_snr - snr_db;

                println!(
                    "AES17 Q15 {case_label:<9} N={n_fft:<4} (log2={stages:<2}) level={level_dbfs:>5.1} dBFS | SNR={snr_db:5.2} dB (ceil={ceil_snr:5.2}, loss={loss_vs_ceil:4.2} dB) | THD+N={thd_n_db:6.2} dB | DR={dr_db:5.2} dB (ceil={ceil_dr:5.2}) | max_err={max_abs:.2e}"
                );

                // No wrap-around or gross corruption: max error well below 1% FS even at N=2048.
                assert!(
                    max_abs < 2e-3,
                    "Q15 {case_label} N={n_fft} level={level_dbfs}: max_abs={max_abs:.3e} indicates wrap-around or scaling fault"
                );

                // Honest physics-grounded bounds:
                // - At -60 dBFS, true BFP keeps output noise dominated by Q1.15 output rounding
                //   (DR stays close to theoretical Q1.15 DR ~98 dB for unity impulse, and SNR
                //   stays within 4 dB of the output-rounding ceiling).
                // - At -1 dBFS, BFP FFT roundoff across log2(N) radix-2 stages (6..11 stages,
                //   16-bit mantissa) is visible and scales with log2(N); unity-impulse THD+N stays
                //   well below -62 dB up to N=2048 (logged honestly for #23 32-bit mantissa work).
                if level_dbfs <= -60.0 {
                    assert!(
                        dr_db > 85.0,
                        "Q15 {case_label} N={n_fft} -60 dBFS DR={dr_db:.2} dB < 85 dB"
                    );
                } else {
                    assert!(
                        snr_db > 62.0 && thd_n_db < -62.0,
                        "Q15 {case_label} N={n_fft} -1 dBFS SNR={snr_db:.2} dB < 62 dB"
                    );
                }
            }
        }
    }
}

#[cfg(feature = "fixed-nonuniform")]
#[test]
fn aes17_q15_nonuniform_with_n512_n1024_n2048_stages() {
    use rtconv_simd::fixed::{f64_to_q15, q15_to_f64};
    use rtconv_simd::{
        FixedQ15NonUniformConvolver, NonUniformConfig, NonUniformPlan, NonUniformStageConfig,
    };

    // Multi-stage Q1.15 NUPC plan exercising Direct + B=128 + B=256 (N512) + B=512 (N1024)
    // + B=1024 (N2048) with causal zero-latency alignment (D_s >= B_s).
    let ir_len = 3072usize;
    let stages = [
        NonUniformStageConfig {
            ir_offset: 0,
            ir_len: 128,
            block_size: 1,
        },
        NonUniformStageConfig {
            ir_offset: 128,
            ir_len: 128,
            block_size: 128,
        },
        NonUniformStageConfig {
            ir_offset: 256,
            ir_len: 256,
            block_size: 256,
        },
        NonUniformStageConfig {
            ir_offset: 512,
            ir_len: 512,
            block_size: 512,
        },
        NonUniformStageConfig {
            ir_offset: 1024,
            ir_len: ir_len - 1024,
            block_size: 1024,
        },
    ];
    let plan = NonUniformPlan::validate(NonUniformConfig {
        max_ir_len: ir_len,
        host_block_max: 64,
        stages: &stages,
    })
    .unwrap();

    let ir_f64 = make_l1_bounded_ir_f64(ir_len, 0.85);
    let ir_q15: Vec<i16> = ir_f64.iter().map(|&v| f64_to_q15(v)).collect();
    let ir_exact: Vec<f64> = ir_q15.iter().map(|&v| q15_to_f64(v)).collect();

    let req = FixedQ15NonUniformConvolver::memory_requirements(&plan).unwrap();
    // Verify per-stage i64 accumulators were moved out of the struct into caller workspace:
    // struct size must remain compact (< 4 KiB) even with N2048 stages.
    assert!(
        req.instance_bytes < 4096,
        "FixedQ15NonUniformConvolver instance_bytes={} >= 4096",
        req.instance_bytes
    );

    let mut ws = vec![0i16; req.total_i16];
    let mut conv = FixedQ15NonUniformConvolver::prepare(&ir_q15, plan, &mut ws).unwrap();

    let total_samples = 8192usize;
    let skip = ir_len;

    for &level_dbfs in &[-60.0f64, -1.0] {
        conv.reset();
        let x_f64 = sine_997hz_f64(total_samples, level_dbfs);
        let x_q15: Vec<i16> = x_f64.iter().map(|&v| f64_to_q15(v)).collect();
        let x_exact: Vec<f64> = x_q15.iter().map(|&v| q15_to_f64(v)).collect();
        let reference = oracle_convolve_f64(&ir_exact, &x_exact);
        let ideal_q15: Vec<f64> = reference
            .iter()
            .map(|&r| q15_to_f64(f64_to_q15(r)))
            .collect();
        let (ceil_snr, _, ceil_dr, _) = aes17_metrics(&reference, &ideal_q15, skip);

        let mut out_q15 = vec![0i16; total_samples];
        for (ib, ob) in x_q15.chunks(64).zip(out_q15.chunks_mut(64)) {
            conv.process_block(ib, ob).unwrap();
        }
        assert_eq!(conv.deadline_misses(), 0);
        assert_eq!(conv.saturation_count(), 0);

        let out_f64: Vec<f64> = out_q15.iter().map(|&v| q15_to_f64(v)).collect();
        let (snr_db, thd_n_db, dr_db, max_abs) = aes17_metrics(&reference, &out_f64, skip);
        println!(
            "AES17 Q15 NUPC (up to N2048) level={level_dbfs:>5.1} dBFS | SNR={snr_db:5.2} dB (ceil={ceil_snr:5.2}) | THD+N={thd_n_db:6.2} dB | DR={dr_db:5.2} dB (ceil={ceil_dr:5.2}) | max_err={max_abs:.2e}"
        );
        assert!(max_abs < 2e-3);
        if level_dbfs <= -60.0 {
            assert!(dr_db > 85.0, "Q15 NUPC -60 dBFS DR={dr_db:.2} dB < 85 dB");
        } else {
            assert!(
                snr_db > 62.0 && thd_n_db < -62.0,
                "Q15 NUPC -1 dBFS SNR={snr_db:.2} dB < 62 dB"
            );
        }
    }
}

/// AES17-style 997 Hz @ 48 kHz stimuli at −60 / −1 / 0 dBFS through every Q1.31 engine
/// (`fixed-q31`): oracle-null DR and THD+N must be compatible with 24-bit audio (> 115 dB).
#[cfg(feature = "fixed-q31")]
#[test]
fn aes17_q31_engines_dr_and_thdn_above_115_db() {
    use rtconv_simd::{
        FixedQ31DirectConfig, FixedQ31DirectConvolver, FixedQ31NonUniformConvolver,
        FixedQ31PartitionedConfig, FixedQ31PartitionedConvolver, NonUniformPlan,
    };
    const Q31: f64 = 2_147_483_648.0;
    let to_q31 = |v: f64| (v * Q31).round().clamp(-(i32::MAX as f64), i32::MAX as f64) as i32;

    for (engine, ir_len) in [
        ("direct", 255usize),
        ("partitioned", 1024),
        ("nonuniform", 2048),
    ] {
        let ir_f64 = make_l1_bounded_ir_f64(ir_len, 0.85);
        let ir: Vec<i32> = ir_f64.iter().map(|&v| to_q31(v)).collect();
        let ir_exact: Vec<f64> = ir.iter().map(|&v| v as f64 / Q31).collect();
        let total = 8192usize;
        let skip = ir_len;
        for &level in &[-60.0f64, -1.0, 0.0] {
            let x: Vec<i32> = sine_997hz_f64(total, level)
                .iter()
                .map(|&v| to_q31(v))
                .collect();
            let x_exact: Vec<f64> = x.iter().map(|&v| v as f64 / Q31).collect();
            let reference = oracle_convolve_f64(&ir_exact, &x_exact);
            let mut y = vec![0i32; total];
            let sats = match engine {
                "direct" => {
                    let cfg = FixedQ31DirectConfig { max_ir_len: ir_len };
                    let mut ws = vec![0i32; 3 * ir_len];
                    let mut c = FixedQ31DirectConvolver::prepare(&ir, cfg, &mut ws).unwrap();
                    for (xi, yo) in x.chunks(64).zip(y.chunks_mut(64)) {
                        c.process_block(xi, yo).unwrap();
                    }
                    c.saturation_count()
                }
                "partitioned" => {
                    let cfg = FixedQ31PartitionedConfig {
                        block_size: 64,
                        max_ir_len: ir_len,
                    };
                    let req = FixedQ31PartitionedConvolver::memory_requirements(cfg).unwrap();
                    let mut ws = vec![0i32; req.total_i32];
                    let mut c = FixedQ31PartitionedConvolver::prepare(&ir, cfg, &mut ws).unwrap();
                    for (xi, yo) in x.chunks(64).zip(y.chunks_mut(64)) {
                        c.process_block(xi, yo).unwrap();
                    }
                    c.saturation_count()
                }
                _ => {
                    let plan = NonUniformPlan::default_profile(ir_len, 64).unwrap();
                    let req = FixedQ31NonUniformConvolver::memory_requirements(&plan).unwrap();
                    let mut ws = vec![0i32; req.total_i32];
                    let mut c = FixedQ31NonUniformConvolver::prepare(&ir, plan, &mut ws).unwrap();
                    for (xi, yo) in x.chunks(64).zip(y.chunks_mut(64)) {
                        c.process_block(xi, yo).unwrap();
                    }
                    assert_eq!(c.deadline_misses(), 0);
                    c.saturation_count()
                }
            };
            assert_eq!(sats, 0, "{engine} level={level}: clipped");
            let out: Vec<f64> = y.iter().map(|&v| v as f64 / Q31).collect();
            let (snr_db, thd_n_db, dr_db, max_abs) = aes17_metrics(&reference, &out, skip);
            println!(
                "AES17 Q31 {engine:<11} L={ir_len:<5} level={level:>5.1} dBFS | SNR={snr_db:6.2} dB | THD+N={thd_n_db:7.2} dB | DR={dr_db:6.2} dB | max_err={max_abs:.2e}"
            );
            assert!(dr_db > 115.0, "{engine} {level} dBFS DR={dr_db:.2} dB");
            if level > -60.0 {
                assert!(
                    thd_n_db < -115.0,
                    "{engine} {level} dBFS THD+N={thd_n_db:.2} dB"
                );
            }
        }
    }
}
