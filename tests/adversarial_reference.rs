//! Adversarial cross-engine verification against an independent `f64` direct-form oracle.
//!
//! Sweeps IR lengths straddling every partition boundary, irregular host block sizes
//! (including 1-sample and larger-than-partition callbacks), every FFT backend, scalar and
//! detected SIMD levels, BURST and SPREAD hybrid modes, and NUPC plans with positive slack.

use rtconv_simd::{
    DirectConfig, DirectConvolver, FftBackend, HybridConfig, HybridConvolver, MicroFftBackend,
    PartitionedConfig, PartitionedConvolver, SimdBackend, SimdFixedFftBackend,
};

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
        ((self.next_u64() >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
    fn vec(&mut self, len: usize, scale: f32) -> Vec<f32> {
        (0..len).map(|_| self.next_f32() * scale).collect()
    }
}

fn oracle(ir: &[f32], x: &[f32]) -> Vec<f64> {
    (0..x.len())
        .map(|n| {
            let mut acc = 0.0f64;
            for k in 0..ir.len().min(n + 1) {
                acc += ir[k] as f64 * x[n - k] as f64;
            }
            acc
        })
        .collect()
}

/// Returns `(max_abs_error, snr_db)` against the oracle.
fn compare(reference: &[f64], actual: &[f32]) -> (f64, f64) {
    let mut max_abs = 0.0f64;
    let mut e2 = 0.0f64;
    let mut s2 = 0.0f64;
    for (&r, &a) in reference.iter().zip(actual) {
        assert!(a.is_finite());
        let d = (r - a as f64).abs();
        max_abs = max_abs.max(d);
        e2 += d * d;
        s2 += r * r;
    }
    let snr = if e2 == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (s2 / e2).log10()
    };
    (max_abs, snr)
}

fn host_blocks(rng: &mut Rng, total: usize, max_block: usize) -> Vec<usize> {
    let mut v = Vec::new();
    let mut left = total;
    while left > 0 {
        let b = (1 + rng.below(max_block)).min(left);
        v.push(b);
        left -= b;
    }
    v
}

const IR_LENGTHS: &[usize] = &[
    1, 2, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 1000, 1025, 4097,
];

fn backends() -> Vec<SimdBackend> {
    vec![SimdBackend::scalar(), SimdBackend::detect_or_baseline()]
}

#[test]
fn direct_matches_oracle_for_all_lengths_and_fragmentations() {
    let mut rng = Rng::new(0xD1EC7);
    for &len in IR_LENGTHS {
        let ir = rng.vec(len, 0.5);
        let x = rng.vec(3000, 1.0);
        let reference = oracle(&ir, &x);
        for backend in backends() {
            let req = DirectConvolver::memory_requirements(DirectConfig {
                max_ir_len: len + 7,
            })
            .unwrap();
            let mut ws = vec![0.0f32; req.total_f32];
            let mut conv = DirectConvolver::prepare(
                &ir,
                DirectConfig {
                    max_ir_len: len + 7,
                },
                &mut ws,
                backend,
            )
            .unwrap();
            let mut out = vec![0.0f32; x.len()];
            let mut pos = 0;
            for b in host_blocks(&mut rng, x.len(), 300) {
                conv.process_block(&x[pos..pos + b], &mut out[pos..pos + b])
                    .unwrap();
                pos += b;
            }
            let (max_abs, snr) = compare(&reference, &out);
            assert!(
                max_abs < 1e-4 && snr > 100.0,
                "direct len={len} backend={} max_abs={max_abs:e} snr={snr:.1}",
                backend.name()
            );
        }
    }
}

fn run_partitioned<F: FftBackend + Copy>(fft: F, backend: SimdBackend, b: usize) {
    let mut rng = Rng::new(0xBADC0DE ^ b as u64);
    for &len in IR_LENGTHS {
        let ir = rng.vec(len, 0.5);
        let n_blocks = 40;
        let x = rng.vec(b * n_blocks, 1.0);
        let reference = oracle(&ir, &x);
        let cfg = PartitionedConfig {
            block_size: b,
            max_ir_len: len + 3,
        };
        let req = PartitionedConvolver::memory_requirements(cfg, &fft).unwrap();
        let mut ws = vec![0.0f32; req.total_f32];
        let mut conv = PartitionedConvolver::prepare(&ir, cfg, &mut ws, backend, fft).unwrap();
        let mut out = vec![0.0f32; x.len()];
        for blk in 0..n_blocks {
            conv.process_block(&x[blk * b..(blk + 1) * b], &mut out[blk * b..(blk + 1) * b])
                .unwrap();
        }
        let (max_abs, snr) = compare(&reference, &out);
        assert!(
            snr > 100.0,
            "partitioned B={b} len={len} fft={} simd={} max_abs={max_abs:e} snr={snr:.1}",
            fft.name(),
            backend.name()
        );
    }
}

#[test]
fn partitioned_matches_oracle_for_every_backend_and_block() {
    for &b in &[32usize, 64, 128] {
        for backend in backends() {
            run_partitioned(MicroFftBackend::new(), backend, b);
            run_partitioned(SimdFixedFftBackend::new(backend), backend, b);
        }
    }
}

fn run_hybrid<F: FftBackend + Copy>(fft: F, backend: SimdBackend, b: usize, spread: bool) {
    let mut rng = Rng::new(0x48A9 ^ (b as u64) ^ ((spread as u64) << 20));
    for &len in IR_LENGTHS {
        let ir = rng.vec(len, 0.5);
        let x = rng.vec(4000, 1.0);
        let reference = oracle(&ir, &x);
        let cfg = HybridConfig {
            head_len: b,
            tail_block_size: b,
            max_ir_len: len + 11,
        };
        let req = HybridConvolver::memory_requirements(cfg, &fft).unwrap();
        let mut ws = vec![0.0f32; req.total_f32];
        let mut conv =
            HybridConvolver::prepare_with_tail_spreading(&ir, cfg, &mut ws, backend, fft, spread)
                .unwrap();
        let mut out = vec![0.0f32; x.len()];
        let mut pos = 0;
        for hb in host_blocks(&mut rng, x.len(), 3 * b) {
            conv.process_block(&x[pos..pos + hb], &mut out[pos..pos + hb])
                .unwrap();
            pos += hb;
        }
        let (max_abs, snr) = compare(&reference, &out);
        assert!(
            snr > 100.0,
            "hybrid B={b} spread={spread} len={len} fft={} max_abs={max_abs:e} snr={snr:.1}",
            fft.name()
        );
    }
}

#[test]
fn hybrid_burst_and_spread_match_oracle_with_fragmented_host_blocks() {
    for &b in &[32usize, 64, 128] {
        for backend in backends() {
            for spread in [false, true] {
                run_hybrid(MicroFftBackend::new(), backend, b, spread);
                run_hybrid(SimdFixedFftBackend::new(backend), backend, b, spread);
            }
        }
    }
}

/// Independent `f64` radix-2 FFT for exact `f64` linear convolution up to 131k+ taps.
fn cfft_f64(re: &mut [f64], im: &mut [f64], inverse: bool) {
    let n = re.len();
    let bits = n.trailing_zeros();
    for i in 0..n {
        let rev = i.reverse_bits() >> (usize::BITS - bits);
        if i < rev {
            re.swap(i, rev);
            im.swap(i, rev);
        }
    }
    let sign = if inverse { 1.0 } else { -1.0 };
    let mut half = 1usize;
    while half < n {
        let step = half << 1;
        let ang_step = sign * 2.0 * core::f64::consts::PI / (step as f64);
        for base in (0..n).step_by(step) {
            for j in 0..half {
                let ang = ang_step * (j as f64);
                let (si, co) = ang.sin_cos();
                let i0 = base + j;
                let i1 = i0 + half;
                let tr = re[i1] * co - im[i1] * si;
                let ti = re[i1] * si + im[i1] * co;
                re[i1] = re[i0] - tr;
                im[i1] = im[i0] - ti;
                re[i0] += tr;
                im[i0] += ti;
            }
        }
        half = step;
    }
    if inverse {
        let inv = 1.0 / (n as f64);
        for (r, i) in re.iter_mut().zip(im.iter_mut()) {
            *r *= inv;
            *i *= inv;
        }
    }
}

/// Fast exact `f64` causal linear convolution returning `y[0..x.len()]`.
fn oracle_fast_f64(ir: &[f32], x: &[f32]) -> Vec<f64> {
    let total = ir.len() + x.len() - 1;
    let n = total.next_power_of_two();
    let mut hr = vec![0.0f64; n];
    let mut hi = vec![0.0f64; n];
    let mut xr = vec![0.0f64; n];
    let mut xi = vec![0.0f64; n];
    for (i, &v) in ir.iter().enumerate() {
        hr[i] = v as f64;
    }
    for (i, &v) in x.iter().enumerate() {
        xr[i] = v as f64;
    }
    cfft_f64(&mut hr, &mut hi, false);
    cfft_f64(&mut xr, &mut xi, false);
    for k in 0..n {
        let re = hr[k] * xr[k] - hi[k] * xi[k];
        let im = hr[k] * xi[k] + hi[k] * xr[k];
        hr[k] = re;
        hi[k] = im;
    }
    cfft_f64(&mut hr, &mut hi, true);
    hr.truncate(x.len());
    hr
}

#[test]
fn large_ir_sweep_up_to_131k_taps_matches_f64_oracle_over_100db() {
    let mut rng = Rng::new(0x131_072);
    // Cross-verify the fast f64 FFT oracle against the direct O(L*M) f64 oracle.
    {
        let h_chk = rng.vec(1025, 0.25);
        let x_chk = rng.vec(1500, 0.8);
        let d_ref = oracle(&h_chk, &x_chk);
        let f_ref = oracle_fast_f64(&h_chk, &x_chk);
        let diff = d_ref
            .iter()
            .zip(&f_ref)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        assert!(diff < 1e-11, "fast f64 oracle mismatch: {diff:.3e}");
    }

    let backend = SimdBackend::detect_or_baseline();
    let simd_fft = SimdFixedFftBackend::new(backend);
    let micro_fft = MicroFftBackend::new();

    // 1. PartitionedConvolver across all new large block sizes B = 256..=8192 up to 131_072 taps.
    for &(ir_len, b) in &[
        (16_384usize, 256usize),
        (32_769, 512),
        (65_536, 1024),
        (65_537, 2048),
        (131_072, 4096),
        (131_072, 8192),
    ] {
        // Decaying broadband room-like IR
        let ir: Vec<f32> = (0..ir_len)
            .map(|k| rng.next_f32() * 0.2 * (-(k as f32) * 5.0 / ir_len as f32).exp())
            .collect();
        let x_len = (4 * b).max(16_384);
        let x = rng.vec(x_len, 0.8);
        let reference = oracle_fast_f64(&ir, &x);

        let cfg = PartitionedConfig {
            block_size: b,
            max_ir_len: ir_len,
        };
        for is_simd in [true, false] {
            let mut out = vec![0.0f32; x_len];
            if is_simd {
                let req = PartitionedConvolver::memory_requirements(cfg, &simd_fft).unwrap();
                let mut ws = vec![0.0f32; req.total_f32];
                let mut conv =
                    PartitionedConvolver::prepare(&ir, cfg, &mut ws, backend, simd_fft).unwrap();
                for (ib, ob) in x.chunks_exact(b).zip(out.chunks_exact_mut(b)) {
                    conv.process_block(ib, ob).unwrap();
                }
            } else {
                let req = PartitionedConvolver::memory_requirements(cfg, &micro_fft).unwrap();
                let mut ws = vec![0.0f32; req.total_f32];
                let mut conv =
                    PartitionedConvolver::prepare(&ir, cfg, &mut ws, backend, micro_fft).unwrap();
                for (ib, ob) in x.chunks_exact(b).zip(out.chunks_exact_mut(b)) {
                    conv.process_block(ib, ob).unwrap();
                }
            }
            let (max_abs, snr) = compare(&reference, &out);
            assert!(
                snr > 100.0,
                "large partitioned ir_len={ir_len} B={b} simd={is_simd} max_abs={max_abs:e} snr={snr:.2} dB"
            );
        }
    }

    // 2. HybridConvolver (SPREAD) with large tail blocks B = 256, 512, 1024 on fragmented host blocks.
    for &(ir_len, b) in &[(16_385usize, 256usize), (65_536, 512), (131_072, 1024)] {
        let ir: Vec<f32> = (0..ir_len)
            .map(|k| rng.next_f32() * 0.2 * (-(k as f32) * 5.0 / ir_len as f32).exp())
            .collect();
        let x = rng.vec(8192, 0.8);
        let reference = oracle_fast_f64(&ir, &x);
        let cfg = HybridConfig {
            head_len: b,
            tail_block_size: b,
            max_ir_len: ir_len,
        };
        let req = HybridConvolver::memory_requirements(cfg, &simd_fft).unwrap();
        let mut ws = vec![0.0f32; req.total_f32];
        let mut conv = HybridConvolver::prepare_with_tail_spreading(
            &ir, cfg, &mut ws, backend, simd_fft, true,
        )
        .unwrap();
        let mut out = vec![0.0f32; x.len()];
        let mut pos = 0;
        for hb in host_blocks(&mut rng, x.len(), 256) {
            conv.process_block(&x[pos..pos + hb], &mut out[pos..pos + hb])
                .unwrap();
            pos += hb;
        }
        let (max_abs, snr) = compare(&reference, &out);
        assert!(
            snr > 100.0,
            "large hybrid SPREAD ir_len={ir_len} B={b} max_abs={max_abs:e} snr={snr:.2} dB"
        );
    }

    // 3. Multi-stage NonUniformConvolver up to 131_072 taps with stages up to B=8192 (N16384).
    #[cfg(feature = "nonuniform")]
    {
        use rtconv_simd::{
            NonUniformConfig, NonUniformConvolver, NonUniformPlan, NonUniformStageConfig,
        };
        let ir_len = 131_072usize;
        let ir: Vec<f32> = (0..ir_len)
            .map(|k| rng.next_f32() * 0.15 * (-(k as f32) * 5.0 / ir_len as f32).exp())
            .collect();
        // Process 140_000 samples (> ir_len) so every stage including the B=8192 tail fires
        // multiple full blocks and aligns causally into the output stream.
        let x = rng.vec(140_000, 0.8);
        let reference = oracle_fast_f64(&ir, &x);

        let stages = [
            NonUniformStageConfig {
                ir_offset: 0,
                ir_len: 64,
                block_size: 1,
            },
            NonUniformStageConfig {
                ir_offset: 64,
                ir_len: 192,
                block_size: 64,
            },
            NonUniformStageConfig {
                ir_offset: 256,
                ir_len: 768,
                block_size: 256,
            },
            NonUniformStageConfig {
                ir_offset: 1024,
                ir_len: 3072,
                block_size: 1024,
            },
            NonUniformStageConfig {
                ir_offset: 4096,
                ir_len: 12_288,
                block_size: 4096,
            },
            NonUniformStageConfig {
                ir_offset: 16_384,
                ir_len: ir_len - 16_384,
                block_size: 8192,
            },
        ];
        let plan = NonUniformPlan::validate(NonUniformConfig {
            max_ir_len: ir_len,
            host_block_max: 64,
            stages: &stages,
        })
        .unwrap();
        let req = NonUniformConvolver::memory_requirements(&plan, &simd_fft).unwrap();
        let mut ws = vec![0.0f32; req.total_f32];
        let mut conv = NonUniformConvolver::prepare(&ir, plan, &mut ws, backend, simd_fft).unwrap();
        let mut out = vec![0.0f32; x.len()];
        let mut pos = 0;
        for hb in host_blocks(&mut rng, x.len(), 64) {
            conv.process_block(&x[pos..pos + hb], &mut out[pos..pos + hb])
                .unwrap();
            pos += hb;
        }
        assert_eq!(conv.deadline_misses(), 0);
        let (max_abs, snr) = compare(&reference, &out);
        println!(
            "adversarial NUPC 131_072 taps (up to N16384): max_abs={max_abs:.3e} snr={snr:.2} dB"
        );
        assert!(
            snr > 100.0,
            "large NUPC 131_072 taps max_abs={max_abs:e} snr={snr:.2} dB"
        );
    }
}

#[cfg(feature = "nonuniform")]
mod nupc {
    use super::*;
    use rtconv_simd::{
        NonUniformConfig, NonUniformConvolver, NonUniformPlan, NonUniformStageConfig,
    };

    fn run(
        plan: NonUniformPlan,
        ir: &[f32],
        x: &[f32],
        rng: &mut Rng,
        max_host: usize,
    ) -> Vec<f32> {
        let backend = SimdBackend::detect_or_baseline();
        let fft = SimdFixedFftBackend::new(backend);
        let req = NonUniformConvolver::memory_requirements(&plan, &fft).unwrap();
        let mut ws = vec![0.0f32; req.total_f32];
        let mut conv = NonUniformConvolver::prepare(ir, plan, &mut ws, backend, fft).unwrap();
        let mut out = vec![0.0f32; x.len()];
        let mut pos = 0;
        for hb in host_blocks(rng, x.len(), max_host) {
            conv.process_block(&x[pos..pos + hb], &mut out[pos..pos + hb])
                .unwrap();
            pos += hb;
        }
        assert_eq!(conv.deadline_misses(), 0);
        out
    }

    #[test]
    fn default_profile_matches_oracle() {
        let mut rng = Rng::new(0x9A9A);
        for &len in IR_LENGTHS {
            let ir = rng.vec(len, 0.5);
            let x = rng.vec(5000, 1.0);
            let reference = oracle(&ir, &x);
            let plan = NonUniformPlan::default_profile(len + 5, 64).unwrap();
            let out = run(plan, &ir, &x, &mut rng, 64);
            let (max_abs, snr) = compare(&reference, &out);
            assert!(
                snr > 100.0,
                "nupc len={len} max_abs={max_abs:e} snr={snr:.1}"
            );
        }
    }

    #[test]
    fn custom_plan_with_large_slack_and_misaligned_offsets() {
        let mut rng = Rng::new(0x51AC);
        let stages = [
            NonUniformStageConfig {
                ir_offset: 0,
                ir_len: 100,
                block_size: 1,
            },
            NonUniformStageConfig {
                ir_offset: 100,
                ir_len: 333,
                block_size: 32,
            },
            NonUniformStageConfig {
                ir_offset: 433,
                ir_len: 900,
                block_size: 128,
            },
            NonUniformStageConfig {
                ir_offset: 1333,
                ir_len: 1500,
                block_size: 64,
            },
        ];
        for host in [1usize, 7, 32, 100] {
            let plan = NonUniformPlan::validate(NonUniformConfig {
                max_ir_len: 2833,
                host_block_max: host,
                stages: &stages,
            })
            .unwrap();
            let ir = rng.vec(2833, 0.3);
            let x = rng.vec(6000, 1.0);
            let reference = oracle(&ir, &x);
            let out = run(plan, &ir, &x, &mut rng, host);
            let (max_abs, snr) = compare(&reference, &out);
            assert!(
                snr > 100.0,
                "custom nupc host={host} max_abs={max_abs:e} snr={snr:.1}"
            );
        }
    }

    #[test]
    fn budget_estimate_scales_with_host_block() {
        // A host callback of 512 samples triggers 16 B=32 FDL blocks; a conservative
        // per-callback bound must grow accordingly.
        let small = NonUniformPlan::default_profile(256, 32).unwrap();
        let large = NonUniformPlan::default_profile(256, 512).unwrap();
        let direct_delta = 32 * (512 - 32);
        assert!(
            large.estimated_worst_ops_per_callback()
                > small.estimated_worst_ops_per_callback() * 8 + direct_delta,
            "small={} large={}",
            small.estimated_worst_ops_per_callback(),
            large.estimated_worst_ops_per_callback()
        );
    }
}

#[cfg(feature = "fixed-nonuniform")]
mod fixed {
    use super::*;
    use rtconv_simd::fixed::{f64_to_q15, q15_to_f64};
    use rtconv_simd::{
        FixedQ15NonUniformConvolver, FixedQ15PartitionedConfig, FixedQ15PartitionedConvolver,
        NonUniformPlan,
    };

    fn snr_q15(reference: &[f64], out: &[i16]) -> f64 {
        let mut e2 = 0.0;
        let mut s2 = 0.0;
        for (&r, &o) in reference.iter().zip(out) {
            let d = r - q15_to_f64(o);
            e2 += d * d;
            s2 += r * r;
        }
        10.0 * (s2 / e2.max(1e-300)).log10()
    }

    fn decaying_ir(rng: &mut Rng, len: usize) -> Vec<i16> {
        // Realistic room/speaker IR: strong head, exponentially decaying tail (-60 dB at end).
        (0..len)
            .map(|k| {
                let env = (-(k as f64) * 6.9 / len as f64).exp();
                f64_to_q15(0.7 * env * rng.next_f32() as f64)
            })
            .collect()
    }

    fn oracle_q15(ir: &[i16], x: &[i16]) -> Vec<f64> {
        let irf: Vec<f32> = ir.iter().map(|&v| q15_to_f64(v) as f32).collect();
        let xf: Vec<f32> = x.iter().map(|&v| q15_to_f64(v) as f32).collect();
        // Use f64 directly to avoid f32 rounding of the exact Q15 grid.
        (0..xf.len())
            .map(|n| {
                let mut acc = 0.0f64;
                for k in 0..ir.len().min(n + 1) {
                    acc += q15_to_f64(ir[k]) * q15_to_f64(x[n - k]);
                }
                let _ = (&irf, &xf);
                acc
            })
            .collect()
    }

    /// rePhase-style linear-phase FIR (centre peak, side lobes decaying to about -80 dB): the
    /// absolute Q1.15 coefficient grid (`LSB = -90 dBFS`) keeps only a few bits of the tails,
    /// whereas `prepare_q31` stores every FFT partition with its own block exponent.
    #[test]
    fn q31_partition_coefficients_beat_absolute_q15_taps() {
        let mut rng = Rng::new(0xC0EF);
        let len = 4096;
        let mid = len / 2;
        let h: Vec<f64> = (0..len)
            .map(|k| {
                let d = (k as f64 - mid as f64).abs();
                if k == mid {
                    0.9
                } else {
                    0.05 * (-d * 9.2 / mid as f64).exp() * rng.next_f32() as f64
                }
            })
            .collect();
        let h_q15: Vec<i16> = h.iter().map(|&v| f64_to_q15(v)).collect();
        let h_q31: Vec<i32> = h
            .iter()
            .map(|&v| (v * 2_147_483_648.0).round() as i32)
            .collect();
        let x: Vec<i16> = (0..12_000)
            .map(|_| f64_to_q15(rng.next_f32() as f64 * 0.3))
            .collect();
        let xr: Vec<f64> = x.iter().map(|&v| q15_to_f64(v)).collect();
        let reference: Vec<f64> = (0..xr.len())
            .map(|n| (0..len.min(n + 1)).map(|k| h[k] * xr[n - k]).sum())
            .collect();
        let ideal: Vec<i16> = reference.iter().map(|&r| f64_to_q15(r)).collect();
        let ceiling = snr_q15(&reference, &ideal);

        let plan = NonUniformPlan::default_profile(len, 32).unwrap();
        let req = FixedQ15NonUniformConvolver::memory_requirements(&plan).unwrap();
        let (mut w1, mut w2) = (vec![0i16; req.total_i16], vec![0i16; req.total_i16]);
        let mut a = FixedQ15NonUniformConvolver::prepare(&h_q15, plan, &mut w1).unwrap();
        let mut b = FixedQ15NonUniformConvolver::prepare_q31(&h_q31, plan, None, &mut w2).unwrap();
        let (mut ya, mut yb) = (vec![0i16; x.len()], vec![0i16; x.len()]);
        a.process_block(&x, &mut ya).unwrap();
        b.process_block(&x, &mut yb).unwrap();
        let (sa, sb) = (snr_q15(&reference, &ya), snr_q15(&reference, &yb));
        std::println!(
            "rePhase-style 4096 taps: ceiling={ceiling:.2} dB  q15-taps={sa:.2} dB  q31-partition-bfp={sb:.2} dB"
        );
        // Per-partition coefficient exponents recover the low-level tails; with the
        // 32-bit-mantissa real-packed FFT core (issue #23) the FFT path no longer caps the SNR
        // (~70 dB with 16-bit butterflies): it must reach the Q1.15 output-rounding ceiling
        // minus at most 3 dB.
        assert!(sb > sa + 3.0, "q31={sb:.2} q15={sa:.2}");
        assert!(sb > 66.0, "q31={sb:.2}");
        assert!(
            sb >= ceiling - 3.0,
            "FFT-path SNR {sb:.2} dB below output ceiling {ceiling:.2} dB - 3 dB"
        );
    }

    /// Reports SNR at several input levels. The level sweep exposes whether the BFP path keeps
    /// precision for quiet program material (the dominant case in audio).
    #[test]
    fn fixed_engines_snr_vs_input_level() {
        let mut rng = Rng::new(0xF1ED);
        let len = 1500;
        let ir = decaying_ir(&mut rng, len);
        // Normalize IR so output never clips at 0 dBFS input: scale by L1 norm.
        let l1: f64 = ir.iter().map(|&v| q15_to_f64(v).abs()).sum();
        let ir: Vec<i16> = ir
            .iter()
            .map(|&v| f64_to_q15(q15_to_f64(v) / l1.max(1.0)))
            .collect();
        for level_db in [0.0f64, -20.0, -40.0] {
            let g = 10f64.powf(level_db / 20.0) * 0.9;
            let x: Vec<i16> = (0..6000)
                .map(|_| f64_to_q15(rng.next_f32() as f64 * g))
                .collect();
            let reference = oracle_q15(&ir, &x);

            let cfg = FixedQ15PartitionedConfig {
                block_size: 64,
                max_ir_len: len,
            };
            let req = FixedQ15PartitionedConvolver::memory_requirements(cfg).unwrap();
            let mut ws = vec![0i16; req.total_i16];
            let mut conv = FixedQ15PartitionedConvolver::prepare(&ir, cfg, &mut ws).unwrap();
            let mut out = vec![0i16; x.len()];
            for (xi, yo) in x.chunks(64).zip(out.chunks_mut(64)) {
                if xi.len() == 64 {
                    conv.process_block(xi, yo).unwrap();
                }
            }
            let n_full = (x.len() / 64) * 64;
            let snr_part = snr_q15(&reference[..n_full], &out[..n_full]);

            let plan = NonUniformPlan::default_profile(len, 32).unwrap();
            let req = FixedQ15NonUniformConvolver::memory_requirements(&plan).unwrap();
            let mut ws = vec![0i16; req.total_i16];
            let mut conv = FixedQ15NonUniformConvolver::prepare(&ir, plan, &mut ws).unwrap();
            let mut out = vec![0i16; x.len()];
            conv.process_block(&x, &mut out).unwrap();
            let snr_nupc = snr_q15(&reference, &out);
            // Ceiling: the exact result rounded once to the Q1.15 output grid.
            let ideal: Vec<i16> = reference.iter().map(|&r| f64_to_q15(r)).collect();
            let snr_ceiling = snr_q15(&reference, &ideal);
            std::println!(
                "fixed level={level_db:>5.1} dBFS  Q15-output ceiling={snr_ceiling:6.2} dB  \
                 partitioned SNR={snr_part:6.2} dB  nupc SNR={snr_nupc:6.2} dB"
            );
            // True BFP keeps the algorithmic noise close to the unavoidable output rounding.
            assert!(
                snr_part > snr_ceiling - 3.0,
                "partitioned {snr_part} vs ceiling {snr_ceiling}"
            );
            assert!(
                snr_nupc > snr_ceiling - 3.0,
                "nupc {snr_nupc} vs ceiling {snr_ceiling}"
            );
            if level_db == 0.0 {
                assert!(snr_part > 65.0 && snr_nupc > 65.0);
            }
        }
    }
}
