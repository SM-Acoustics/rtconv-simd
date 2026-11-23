#![cfg(feature = "fixed-q15")]

use rtconv_simd::fixed::{f64_to_q15, q15_to_f64, round_sat_q15_from_i64};
use rtconv_simd::{FixedQ15DirectConfig, FixedQ15DirectConvolver};

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        let bits = (self.0 >> 11) as f64 / ((1u64 << 53) as f64);
        bits * 2.0 - 1.0
    }
}

fn ref_convolve_q15_f64(ir_q15: &[i16], input_q15: &[i16]) -> Vec<f64> {
    let mut out = vec![0.0f64; input_q15.len()];
    for n in 0..input_q15.len() {
        let mut acc = 0.0f64;
        let max_k = ir_q15.len().min(n + 1);
        for k in 0..max_k {
            acc += q15_to_f64(ir_q15[k]) * q15_to_f64(input_q15[n - k]);
        }
        out[n] = acc.clamp(-1.0, 32767.0 / 32768.0);
    }
    out
}

fn q15_metrics(reference: &[f64], actual_q15: &[i16]) -> (f64, f64, f64) {
    let mut max_abs = 0.0f64;
    let mut err_sq = 0.0f64;
    let mut sig_sq = 0.0f64;
    for (&r, &a) in reference.iter().zip(actual_q15.iter()) {
        let af = q15_to_f64(a);
        let d = r - af;
        max_abs = max_abs.max(d.abs());
        err_sq += d * d;
        sig_sq += r * r;
    }
    let rms = (err_sq / reference.len() as f64).sqrt();
    let snr = if err_sq == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (sig_sq / err_sq).log10()
    };
    (max_abs, rms, snr)
}

#[test]
fn test_q15_impulse_unity_attenuation_and_extreme_coeffs() {
    // 1. Unity-like impulse (i16::MAX = +0.9999695) and negative full-scale (i16::MIN = -1.0)
    let ir = [i16::MIN, 16384, -8192, i16::MAX];
    let mut input = [0i16; 16];
    input[0] = i16::MIN; // -1.0 -> (-1.0) * (-1.0) = +1.0 -> saturates to i16::MAX (32767)

    let cfg = FixedQ15DirectConfig { max_ir_len: 8 };
    let req = FixedQ15DirectConvolver::memory_requirements(cfg).unwrap();
    let mut ws = vec![0i16; req.total_i16];
    let mut conv = FixedQ15DirectConvolver::prepare(&ir, cfg, &mut ws).unwrap();
    assert_eq!(conv.latency_samples(), 0);

    let mut out = [0i16; 16];
    conv.process_block(&input, &mut out).unwrap();
    assert_eq!(out[0], i16::MAX, "(-1.0)*(-1.0) must saturate to i16::MAX");
    assert_eq!(conv.saturation_count(), 1);
    assert_eq!(out[1], -16384);
    assert_eq!(out[2], 8192);
    assert_eq!(out[3], -32767);
}

#[test]
fn test_q15_rounding_boundaries_and_overflow_saturation() {
    // Exact half-LSB boundary: +16384 rounds up to +1; +16383 rounds down to 0
    assert_eq!(round_sat_q15_from_i64(16384), (1, false));
    assert_eq!(round_sat_q15_from_i64(16383), (0, false));
    assert_eq!(round_sat_q15_from_i64(-16384), (0, false));
    assert_eq!(round_sat_q15_from_i64(-16385), (-1, false));

    // Deliberate positive & negative overflow saturation with 4 taps of 0.75 * 0.75
    let ir = [f64_to_q15(0.75); 4];
    let input = [
        f64_to_q15(0.75),
        f64_to_q15(0.75),
        f64_to_q15(0.75),
        f64_to_q15(0.75),
        f64_to_q15(-0.75),
        f64_to_q15(-0.75),
        f64_to_q15(-0.75),
        f64_to_q15(-0.75),
    ];
    let cfg = FixedQ15DirectConfig { max_ir_len: 4 };
    let mut ws = vec![
        0i16;
        FixedQ15DirectConvolver::memory_requirements(cfg)
            .unwrap()
            .total_i16
    ];
    let mut conv = FixedQ15DirectConvolver::prepare(&ir, cfg, &mut ws).unwrap();
    let mut out = [0i16; 8];
    conv.process_block(&input, &mut out).unwrap();
    assert_eq!(out[3], i16::MAX);
    assert_eq!(out[7], i16::MIN);
    assert!(conv.saturation_count() >= 2);
}

#[test]
fn test_q15_long_ir_noise_and_near_zero_against_f64_reference() {
    let mut rng = Rng::new(0x004C_2026);
    // 512-tap decaying IR normalized so sum(|h|) < 0.8 (no clipping)
    let raw_ir: Vec<f64> = (0..512)
        .map(|k| rng.next_f64() * (-0.01 * k as f64).exp())
        .collect();
    let l1: f64 = raw_ir.iter().map(|v| v.abs()).sum();
    let ir_q15: Vec<i16> = raw_ir.iter().map(|&v| f64_to_q15(v * 0.7 / l1)).collect();
    let input_q15: Vec<i16> = (0..2048)
        .map(|_| f64_to_q15(rng.next_f64() * 0.8))
        .collect();

    let expected = ref_convolve_q15_f64(&ir_q15, &input_q15);
    let cfg = FixedQ15DirectConfig { max_ir_len: 512 };
    let mut ws = vec![
        0i16;
        FixedQ15DirectConvolver::memory_requirements(cfg)
            .unwrap()
            .total_i16
    ];
    let mut conv = FixedQ15DirectConvolver::prepare(&ir_q15, cfg, &mut ws).unwrap();

    let mut out1 = vec![0i16; input_q15.len()];
    conv.process_block(&input_q15, &mut out1).unwrap();
    assert_eq!(conv.saturation_count(), 0);

    // Quantization error from exact i64 accumulation + single final round-half-up is <= 0.5 LSB!
    let (max_abs, rms, snr) = q15_metrics(&expected, &out1);
    assert!(
        max_abs <= 1.0 / 32768.0,
        "Q15 direct max_abs={max_abs:.3e} > 1 LSB"
    );
    // Theoretical Q1.15 quantization RMS floor is 2^-15 / sqrt(12) ≈ 8.81e-6;
    // for this low-RMS decaying random IR (L1=0.7), theoretical SNR is ~69.7 dB.
    assert!(
        rms <= 0.5 / 32768.0 && snr > 68.0,
        "Q15 direct rms={rms:.3e}, snr={snr:.2} dB"
    );

    // Reset bit-for-bit reproducibility
    conv.reset();
    let mut out2 = vec![0i16; input_q15.len()];
    conv.process_block(&input_q15, &mut out2).unwrap();
    assert_eq!(out1, out2);
}
