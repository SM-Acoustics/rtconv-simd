//! Q1.31 engine family (`fixed-q31`) against an exact `f64` convolution oracle.
//!
//! Every engine is fed Q1.31 noise at 0 dBFS and −40 dBFS through IRs with `‖h‖₁ <= 0.85` (no
//! output clipping), with random host-block fragmentation where the API allows it, and must
//! reach SNR >= 120 dB (0 dBFS) and >= 95 dB (−40 dBFS). The direct engine's FIR Type I folding
//! must be bit-exact with the unfolded sum.
#![cfg(feature = "fixed-q31")]

use rtconv_simd::{
    ClearancePolicy, CostModel, FixedQ31DirectConfig, FixedQ31DirectConvolver,
    FixedQ31NonUniformConvolver, FixedQ31PartitionedConfig, FixedQ31PartitionedConvolver,
    FixedQ31Symmetry, NonUniformPlan, OptimizerScratchCell, optimizer_scratch_len,
};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

const Q31: f64 = 2_147_483_648.0;

fn to_q31(v: f64) -> i32 {
    (v * Q31).round().clamp(-(i32::MAX as f64), i32::MAX as f64) as i32
}

/// Decaying, sign-alternating IR scaled to `‖h‖₁ = l1`.
fn ir_q31(rng: &mut Rng, len: usize, l1: f64) -> Vec<i32> {
    let raw: Vec<f64> = (0..len)
        .map(|k| (-(k as f64) * 6.0 / len as f64).exp() * rng.unit())
        .collect();
    let s: f64 = raw.iter().map(|v| v.abs()).sum();
    raw.iter().map(|&v| to_q31(v * l1 / s)).collect()
}

fn noise_q31(rng: &mut Rng, len: usize, level_db: f64) -> Vec<i32> {
    let g = 10f64.powf(level_db / 20.0);
    (0..len).map(|_| to_q31(rng.unit() * g)).collect()
}

fn oracle(h: &[i32], x: &[i32]) -> Vec<f64> {
    (0..x.len())
        .map(|n| {
            (0..h.len().min(n + 1))
                .map(|k| h[k] as f64 / Q31 * (x[n - k] as f64 / Q31))
                .sum()
        })
        .collect()
}

fn snr(reference: &[f64], y: &[i32]) -> f64 {
    let (mut s, mut e) = (0.0, 0.0);
    for (&r, &v) in reference.iter().zip(y) {
        s += r * r;
        e += (r - v as f64 / Q31).powi(2);
    }
    10.0 * (s / e.max(1e-300)).log10()
}

/// Random host fragmentation `1..=max_block`.
fn fragments(rng: &mut Rng, total: usize, max_block: usize) -> Vec<usize> {
    let mut out = Vec::new();
    let mut left = total;
    while left > 0 {
        let b = (1 + rng.below(max_block)).min(left);
        out.push(b);
        left -= b;
    }
    out
}

fn check(label: &str, level: f64, s: f64) {
    std::println!("Q31 {label:<44} level={level:>5.1} dBFS  SNR={s:7.2} dB");
    let floor = if level >= 0.0 { 120.0 } else { 95.0 };
    assert!(
        s >= floor,
        "{label} level={level} SNR {s:.2} dB < {floor} dB"
    );
}

#[test]
fn direct_oracle_with_fragmentation_and_type1_folding() {
    let mut rng = Rng(0xD1EC7);
    for &len in &[1usize, 2, 3, 17, 64, 255, 256, 600] {
        let h = ir_q31(&mut rng, len, 0.85);
        let cfg = FixedQ31DirectConfig { max_ir_len: len };
        let req = FixedQ31DirectConvolver::memory_requirements(cfg).unwrap();
        assert_eq!(req.total_i32, 3 * len);
        let mut ws = vec![0i32; req.total_i32];
        let mut conv = FixedQ31DirectConvolver::prepare(&h, cfg, &mut ws).unwrap();
        for level in [0.0, -40.0] {
            conv.reset();
            let x = noise_q31(&mut rng, 3000, level);
            let mut y = vec![0i32; x.len()];
            let mut pos = 0;
            for b in fragments(&mut rng, x.len(), 97) {
                conv.process_block(&x[pos..pos + b], &mut y[pos..pos + b])
                    .unwrap();
                pos += b;
            }
            check(&format!("direct L={len}"), level, snr(&oracle(&h, &x), &y));
            assert_eq!(conv.saturation_count(), 0);
        }
    }
}

/// Linear-phase Type I crossover (255 taps, windowed-sinc low-pass at 1200 Hz / 48 kHz):
/// folded (128 MACs) and unfolded (255 MACs) outputs are bit-identical.
#[test]
fn type1_crossover_folding_is_bit_exact_with_128_macs() {
    let n = 255usize;
    let m = (n - 1) / 2;
    let fc = 1200.0 / 48_000.0;
    let raw: Vec<f64> = (0..n)
        .map(|k| {
            let t = k as f64 - m as f64;
            let sinc = if t == 0.0 {
                2.0 * fc
            } else {
                (2.0 * core::f64::consts::PI * fc * t).sin() / (core::f64::consts::PI * t)
            };
            let w = 0.54 - 0.46 * (2.0 * core::f64::consts::PI * k as f64 / (n - 1) as f64).cos();
            sinc * w
        })
        .collect();
    let dc: f64 = raw.iter().sum();
    let mut h: Vec<i32> = raw.iter().map(|&v| to_q31(v / dc * 0.98)).collect();
    // Exact symmetry after rounding.
    for k in 0..m {
        h[n - 1 - k] = h[k];
    }
    let cfg = FixedQ31DirectConfig { max_ir_len: n };
    let mut ws_a = vec![0i32; 3 * n];
    let mut ws_b = vec![0i32; 3 * n];
    let mut folded = FixedQ31DirectConvolver::prepare(&h, cfg, &mut ws_a).unwrap();
    let mut plain = FixedQ31DirectConvolver::prepare_unfolded(&h, cfg, &mut ws_b).unwrap();
    assert_eq!(folded.capabilities().symmetry, FixedQ31Symmetry::TypeIEven);
    assert_eq!(folded.capabilities().macs_per_sample, 128);
    assert_eq!(plain.capabilities().symmetry, FixedQ31Symmetry::None);
    assert_eq!(plain.capabilities().macs_per_sample, 255);
    let mut rng = Rng(0x5EA5);
    let x = noise_q31(&mut rng, 6000, -0.5);
    let (mut ya, mut yb) = (vec![0i32; x.len()], vec![0i32; x.len()]);
    for (xi, (a, b)) in x.chunks(64).zip(ya.chunks_mut(64).zip(yb.chunks_mut(64))) {
        folded.process_block(xi, a).unwrap();
        plain.process_block(xi, b).unwrap();
    }
    assert_eq!(ya, yb, "folding must not change a single bit");
    let s = snr(&oracle(&h, &x), &ya);
    check("direct Type I 255-tap crossover (folded)", 0.0, s);

    // A non-symmetric or even-length IR is never folded.
    let mut asym = h.clone();
    asym[0] += 1;
    let mut ws = vec![0i32; 3 * n];
    let c = FixedQ31DirectConvolver::prepare(&asym, cfg, &mut ws).unwrap();
    assert_eq!(c.capabilities().symmetry, FixedQ31Symmetry::None);
}

#[test]
fn direct_saturates_symmetrically_and_handles_high_gain() {
    // +12 dB gain (‖h‖₁ = 4): products are pre-shifted, results still match the oracle.
    let h = vec![to_q31(0.999), to_q31(0.999), to_q31(0.999), to_q31(0.999)];
    let cfg = FixedQ31DirectConfig { max_ir_len: 4 };
    let mut ws = vec![0i32; 12];
    let mut conv = FixedQ31DirectConvolver::prepare(&h, cfg, &mut ws).unwrap();
    assert!(conv.capabilities().product_shift > 0);
    let x = vec![to_q31(0.1); 8];
    let mut y = vec![0i32; 8];
    conv.process_block(&x, &mut y).unwrap();
    let r = oracle(&h, &x);
    for (a, b) in r.iter().zip(&y) {
        assert!((a - *b as f64 / Q31).abs() < 4.0 / Q31);
    }
    // Full-scale input through the 4x gain clips to ±(2^31 − 1) and is counted.
    conv.reset();
    let x = vec![i32::MIN, i32::MIN, i32::MAX, i32::MAX];
    let mut y = vec![0i32; 4];
    conv.process_block(&x, &mut y).unwrap();
    // y[1] = 0.999·(−2) clips to −(2^31 − 1); y[3] = 0.999·(−2^-30) ≈ −2 LSB.
    assert_eq!(y[1], -i32::MAX);
    assert!(y[3].abs() <= 2, "y[3]={}", y[3]);
    assert_eq!(conv.saturation_count(), 1);
    let mut ws = vec![0i32; 2];
    assert!(FixedQ31DirectConvolver::prepare(&h, cfg, &mut ws).is_err());
    assert!(FixedQ31DirectConvolver::prepare(&[], cfg, &mut [0i32; 12]).is_err());
}

#[test]
fn partitioned_oracle_all_block_sizes() {
    let mut rng = Rng(0xFA57);
    for &b in &[32usize, 64, 128, 256, 512, 1024] {
        let len = 3 * b + b / 3;
        let h = ir_q31(&mut rng, len, 0.85);
        let cfg = FixedQ31PartitionedConfig {
            block_size: b,
            max_ir_len: len,
        };
        let req = FixedQ31PartitionedConvolver::memory_requirements(cfg).unwrap();
        let mut ws = vec![0i32; req.total_i32];
        let mut conv = FixedQ31PartitionedConvolver::prepare(&h, cfg, &mut ws).unwrap();
        for level in [0.0, -40.0] {
            conv.reset();
            let x = noise_q31(&mut rng, 8 * b, level);
            let mut y = vec![0i32; x.len()];
            for (xi, yo) in x.chunks(b).zip(y.chunks_mut(b)) {
                conv.process_block(xi, yo).unwrap();
            }
            check(
                &format!("partitioned B={b}"),
                level,
                snr(&oracle(&h, &x), &y),
            );
            assert_eq!(conv.saturation_count(), 0);
        }
    }
}

#[test]
fn nonuniform_oracle_default_and_optimized_plans_fragmented() {
    let mut rng = Rng(0x0A11);
    for &len in &[300usize, 1024, 2048, 4096] {
        let h = ir_q31(&mut rng, len, 0.85);
        let mut plans = vec![(
            "default_profile",
            NonUniformPlan::default_profile(len, 64).unwrap(),
        )];
        let max_b = 1024.min((len / 4).max(32));
        let mut sc = vec![OptimizerScratchCell::ZERO; optimizer_scratch_len(len, max_b)];
        let (opt, _) = NonUniformPlan::optimize_with_max_block_size(
            len,
            64,
            max_b,
            &CostModel::esp32_c6_q15_estimate(),
            ClearancePolicy::Strategy2,
            &mut sc,
        )
        .unwrap();
        plans.push(("optimized", opt));
        for (name, plan) in plans {
            let req = FixedQ31NonUniformConvolver::memory_requirements(&plan).unwrap();
            assert!(req.instance_bytes < 4096, "instance {}", req.instance_bytes);
            let mut ws = vec![0i32; req.total_i32];
            let mut conv = FixedQ31NonUniformConvolver::prepare(&h, plan, &mut ws).unwrap();
            assert_eq!(conv.latency_samples(), 0);
            for level in [0.0, -40.0] {
                conv.reset();
                let x = noise_q31(&mut rng, 3 * len + 500, level);
                let mut y = vec![0i32; x.len()];
                let mut pos = 0;
                for b in fragments(&mut rng, x.len(), 64) {
                    conv.process_block(&x[pos..pos + b], &mut y[pos..pos + b])
                        .unwrap();
                    pos += b;
                }
                assert_eq!(conv.deadline_misses(), 0);
                assert_eq!(conv.saturation_count(), 0);
                check(
                    &format!("nonuniform {name} L={len} stages={}", plan.stage_count()),
                    level,
                    snr(&oracle(&h, &x), &y),
                );
            }
        }
    }
}
