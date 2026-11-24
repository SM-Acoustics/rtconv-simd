//! Verification suite for Issue #16 (PR-C):
//! - Bit-exact staged vs. monolithic FFT/IFFT across `N64..=N16384` (`SimdFixedFftBackend`,
//!   Scalar & SIMD) and `N64..=N2048` (`fft_q15` BFP).
//! - Bit-exact output equivalence between time-distributed staged `NonUniformConvolver` and
//!   monolithic-transform `NonUniformConvolver` (plus `f64` oracle `SNR > 100 dB`).
//! - `FixedQ15NonUniformConvolver` deferral + staged BFP FFT equivalence vs `f64` oracle and
//!   AES17 997 Hz @ −60 dBFS / −1 dBFS.
//! - Deterministic per-sample and per-callback op-count traces proving
//!   `actual_callback_ops <= plan.estimated_worst_ops_per_callback()` across all alignments and
//!   `peak_callback_ops <= declared_factor * mean_callback_ops`.
//! - Power-of-two `2 * B_s` delay ring & shared real input ring RAM reduction verification.
//! - Zero deadline misses under fragmented/irregular host block sequences (`f32` and `Q1.15`).

use rtconv_simd::{ConvError, FftBackend, FftSize, SimdBackend, SimdFixedFftBackend};

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        let bits = (self.0 >> 40) as u32;
        (bits as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
    }
    fn vec_f32(&mut self, len: usize, scale: f32) -> Vec<f32> {
        (0..len).map(|_| self.next_f32() * scale).collect()
    }
}

// ------------------------------------------------------------------------------------------
// 1. Bit-exact staged vs. monolithic `SimdFixedFftBackend` (`N64..=N16384`, Scalar & SIMD)
// ------------------------------------------------------------------------------------------

#[test]
fn simd_fixed_fft_staged_is_bit_exact_with_monolithic_all_sizes_scalar_and_simd() {
    let mut rng = Rng::new(0x0016_F320);
    for backend in [SimdBackend::scalar(), SimdBackend::detect_or_baseline()] {
        let fft = SimdFixedFftBackend::new(backend);
        assert!(fft.supports_staged_fft());

        for size in FftSize::ALL {
            let n = size.len();
            let bins = size.spectrum_bins();
            let fwd_steps = fft.staged_forward_steps(size);
            let inv_steps = fft.staged_inverse_steps(size);
            assert_eq!(fwd_steps, (size.log2() as usize) - 1);
            assert_eq!(inv_steps, (size.log2() as usize) - 1);

            let time_in = rng.vec_f32(n, 0.85);
            let mut mono_re = vec![0.0f32; bins];
            let mut mono_im = vec![0.0f32; bins];
            let mut scratch = vec![0.0f32; fft.scratch_len_f32(size)];
            fft.rfft_forward(size, &time_in, &mut mono_re, &mut mono_im, &mut scratch)
                .unwrap();

            // Staged forward operates in-place on spec_re/spec_im with empty scratch.
            let mut staged_re = vec![0.0f32; bins];
            let mut staged_im = vec![0.0f32; bins];
            for step in 0..fwd_steps {
                let t_slice: &[f32] = if step == 0 { &time_in } else { &[] };
                fft.rfft_forward_step(size, step, t_slice, &mut staged_re, &mut staged_im, &mut [])
                    .unwrap();
            }
            for k in 0..bins {
                assert_eq!(
                    staged_re[k].to_bits(),
                    mono_re[k].to_bits(),
                    "forward Re bit mismatch backend={} size={size:?} bin={k}",
                    backend.name()
                );
                assert_eq!(
                    staged_im[k].to_bits(),
                    mono_im[k].to_bits(),
                    "forward Im bit mismatch backend={} size={size:?} bin={k}",
                    backend.name()
                );
            }

            // Out-of-range forward step must fail cleanly.
            assert_eq!(
                fft.rfft_forward_step(
                    size,
                    fwd_steps,
                    &time_in,
                    &mut staged_re,
                    &mut staged_im,
                    &mut []
                )
                .unwrap_err(),
                ConvError::InvalidStateTransition
            );

            // Monolithic vs staged inverse (in-place on staged_re/staged_im, empty scratch).
            let mut mono_out = vec![0.0f32; n];
            fft.rfft_inverse(size, &mono_re, &mono_im, &mut mono_out, &mut scratch)
                .unwrap();

            let mut staged_out = vec![0.0f32; n];
            for step in 0..inv_steps {
                let o_slice: &mut [f32] = if step + 1 == inv_steps {
                    &mut staged_out
                } else {
                    &mut []
                };
                fft.rfft_inverse_step(size, step, &mut staged_re, &mut staged_im, o_slice, &mut [])
                    .unwrap();
            }
            for i in 0..n {
                assert_eq!(
                    staged_out[i].to_bits(),
                    mono_out[i].to_bits(),
                    "inverse bit mismatch backend={} size={size:?} sample={i}",
                    backend.name()
                );
            }

            // Out-of-range inverse step must fail cleanly.
            assert_eq!(
                fft.rfft_inverse_step(
                    size,
                    inv_steps,
                    &mut staged_re,
                    &mut staged_im,
                    &mut staged_out,
                    &mut []
                )
                .unwrap_err(),
                ConvError::InvalidStateTransition
            );
        }
    }
}

// ------------------------------------------------------------------------------------------
// 2. Bit-exact staged vs. monolithic Q1.15 BFP FFT/IFFT (`N64..=N2048`)
// ------------------------------------------------------------------------------------------

#[cfg(feature = "fixed-fft")]
#[test]
fn q15_bfp_fft_staged_is_bit_exact_with_monolithic_n64_to_n2048() {
    use rtconv_simd::fixed::{
        FixedStagedFftState, f32_to_q15, fixed_fft_scratch_len_i32, fixed_fft_staged_steps,
        irfft_q15_bfp_step_with_scratch, irfft_q15_bfp_with_scratch,
        rfft_q15_bfp_step_with_scratch, rfft_q15_bfp_with_scratch,
    };

    let mut rng = Rng::new(0x0016_0015);
    let q15_sizes = [
        FftSize::N64,
        FftSize::N128,
        FftSize::N256,
        FftSize::N512,
        FftSize::N1024,
        FftSize::N2048,
    ];

    for size in q15_sizes {
        let n = size.len();
        let bins = size.spectrum_bins();
        let steps = fixed_fft_staged_steps(size);
        assert_eq!(steps, size.log2() as usize);

        // Test zero, quiet (-60 dBFS), and full-scale signals.
        let cases: [Vec<i16>; 3] = [
            vec![0i16; n],
            rng.vec_f32(n, 0.001).into_iter().map(f32_to_q15).collect(),
            rng.vec_f32(n, 0.95).into_iter().map(f32_to_q15).collect(),
        ];

        for (case_idx, time_in) in cases.iter().enumerate() {
            let mut mono_re = vec![0i16; bins];
            let mut mono_im = vec![0i16; bins];
            let mut sc_mono = vec![0i32; fixed_fft_scratch_len_i32(size)];
            let mono_exp =
                rfft_q15_bfp_with_scratch(size, time_in, &mut mono_re, &mut mono_im, &mut sc_mono)
                    .unwrap();

            let mut staged_re = vec![0i16; bins];
            let mut staged_im = vec![0i16; bins];
            let mut sc_staged = vec![0i32; fixed_fft_scratch_len_i32(size)];
            let mut fstate = FixedStagedFftState::new();
            let mut staged_exp = None;
            for step in 0..steps {
                let t_slice: &[i16] = if step == 0 { time_in } else { &[] };
                let res = rfft_q15_bfp_step_with_scratch(
                    size,
                    step,
                    t_slice,
                    &mut staged_re,
                    &mut staged_im,
                    &mut sc_staged,
                    &mut fstate,
                )
                .unwrap();
                if step + 1 < steps {
                    assert!(res.is_none());
                } else {
                    staged_exp = res;
                }
            }
            assert_eq!(
                staged_exp,
                Some(mono_exp),
                "Q15 forward exp mismatch size={size:?} case={case_idx}"
            );
            assert_eq!(
                staged_re, mono_re,
                "Q15 forward Re mismatch size={size:?} case={case_idx}"
            );
            assert_eq!(
                staged_im, mono_im,
                "Q15 forward Im mismatch size={size:?} case={case_idx}"
            );

            // Inverse comparison
            let mut mono_out = vec![0i16; n];
            let mono_sats = irfft_q15_bfp_with_scratch(
                size,
                &mono_re,
                &mono_im,
                mono_exp,
                &mut mono_out,
                &mut sc_mono,
            )
            .unwrap();

            let mut staged_out = vec![0i16; n];
            let mut istate = FixedStagedFftState::new();
            let mut staged_sats = None;
            for step in 0..steps {
                let o_slice: &mut [i16] = if step + 1 == steps {
                    &mut staged_out
                } else {
                    &mut []
                };
                let res = irfft_q15_bfp_step_with_scratch(
                    size,
                    step,
                    &mono_re,
                    &mono_im,
                    mono_exp,
                    o_slice,
                    &mut sc_staged,
                    &mut istate,
                )
                .unwrap();
                if step + 1 < steps {
                    assert!(res.is_none());
                } else {
                    staged_sats = res;
                }
            }
            assert_eq!(
                staged_sats,
                Some(mono_sats),
                "Q15 inverse sat mismatch size={size:?} case={case_idx}"
            );
            assert_eq!(
                staged_out, mono_out,
                "Q15 inverse output mismatch size={size:?} case={case_idx}"
            );
        }
    }

    // Negative cases: unsupported size and out-of-range step.
    let mut state = FixedStagedFftState::new();
    let mut sc = vec![0i32; fixed_fft_scratch_len_i32(FftSize::N64)];
    let mut re = vec![0i16; FftSize::N64.spectrum_bins()];
    let mut im = vec![0i16; FftSize::N64.spectrum_bins()];
    assert_eq!(
        rfft_q15_bfp_step_with_scratch(
            FftSize::N4096,
            0,
            &[],
            &mut re,
            &mut im,
            &mut sc,
            &mut state
        )
        .unwrap_err(),
        ConvError::UnsupportedFftSize { fft_size: 4096 }
    );
    assert_eq!(
        rfft_q15_bfp_step_with_scratch(
            FftSize::N64,
            fixed_fft_staged_steps(FftSize::N64),
            &[],
            &mut re,
            &mut im,
            &mut sc,
            &mut state
        )
        .unwrap_err(),
        ConvError::InvalidStateTransition
    );
}

// ------------------------------------------------------------------------------------------
// 3..7. NUPC time-distributed end-to-end equivalence, op-trace bounds, RAM & fragmentation
// ------------------------------------------------------------------------------------------

#[cfg(feature = "nonuniform")]
mod nonuniform_pr_c {
    use super::*;
    use rtconv_simd::nonuniform::{NonUniformScheduler, stage_work_item_ops};
    use rtconv_simd::{
        ClearancePolicy, CostModel, MicroFftBackend, NonUniformBudget, NonUniformConfig,
        NonUniformConvolver, NonUniformPlan, NonUniformStageConfig, NonUniformStageKind,
        OptimizerScratchCell, optimizer_scratch_len,
    };

    /// Wraps `SimdFixedFftBackend` with `supports_staged_fft() == false` (`1` monolithic step)
    /// so we can prove bit-exact equality between time-distributed staged execution and
    /// monolithic-transform execution using the exact same FFT arithmetic.
    #[derive(Clone, Copy)]
    struct MonolithicSimdFft(SimdFixedFftBackend);

    impl FftBackend for MonolithicSimdFft {
        fn name(&self) -> &'static str {
            self.0.name()
        }
        fn is_simd_accelerated(&self) -> bool {
            self.0.is_simd_accelerated()
        }
        fn supports_size(&self, size: FftSize) -> bool {
            self.0.supports_size(size)
        }
        fn scratch_len_f32(&self, size: FftSize) -> usize {
            self.0.scratch_len_f32(size)
        }
        fn rfft_forward(
            &self,
            size: FftSize,
            time_in: &[f32],
            spec_re: &mut [f32],
            spec_im: &mut [f32],
            scratch: &mut [f32],
        ) -> Result<(), ConvError> {
            self.0
                .rfft_forward(size, time_in, spec_re, spec_im, scratch)
        }
        fn rfft_inverse(
            &self,
            size: FftSize,
            spec_re: &[f32],
            spec_im: &[f32],
            time_out: &mut [f32],
            scratch: &mut [f32],
        ) -> Result<(), ConvError> {
            self.0
                .rfft_inverse(size, spec_re, spec_im, time_out, scratch)
        }
    }

    fn oracle_f64(ir: &[f32], input: &[f32]) -> Vec<f64> {
        (0..input.len())
            .map(|n| {
                let mut acc = 0.0f64;
                let max_k = ir.len().min(n + 1);
                for k in 0..max_k {
                    acc += (ir[k] as f64) * (input[n - k] as f64);
                }
                acc
            })
            .collect()
    }

    fn snr_db(reference: &[f64], actual: &[f32]) -> f64 {
        let mut s2 = 0.0f64;
        let mut e2 = 0.0f64;
        for (&r, &a) in reference.iter().zip(actual.iter()) {
            let d = r - (a as f64);
            s2 += r * r;
            e2 += d * d;
        }
        10.0 * (s2 / e2.max(1e-300)).log10()
    }

    #[test]
    fn nonuniform_f32_time_distributed_is_bit_identical_to_monolithic_and_f64_oracle() {
        let mut rng = Rng::new(0x0016_B17E);
        let backend = SimdBackend::detect_or_baseline();
        let staged_fft = SimdFixedFftBackend::new(backend);
        let mono_fft = MonolithicSimdFft(staged_fft);
        let model = CostModel::for_backend_f32(backend);

        // Plan A: default_profile(2048, 32) — large deferral on B=128@1024 (defer=769).
        let plan_default = NonUniformPlan::default_profile(2048, 32).unwrap();

        // Plan B: Strategy-2 optimized 16k plan (H=64).
        let mut sc = vec![OptimizerScratchCell::ZERO; optimizer_scratch_len(16_384, 8192)];
        let (plan_s2, _) =
            NonUniformPlan::optimize(16_384, 64, &model, ClearancePolicy::Strategy2, &mut sc)
                .unwrap();

        // Plan C: Strategy-1 optimized 8k plan (H=64, full B_s clearance on every stage).
        let (plan_s1, _) =
            NonUniformPlan::optimize(8_192, 64, &model, ClearancePolicy::Strategy1, &mut sc)
                .unwrap();

        // Plan D: Extreme deferral custom plan (B=1024 at offset 6144 -> defer=4097).
        let plan_defer = NonUniformPlan::validate(NonUniformConfig {
            max_ir_len: 9216,
            host_block_max: 64,
            stages: &[
                NonUniformStageConfig {
                    ir_offset: 0,
                    ir_len: 64,
                    block_size: 1,
                },
                NonUniformStageConfig {
                    ir_offset: 64,
                    ir_len: 960,
                    block_size: 64,
                },
                NonUniformStageConfig {
                    ir_offset: 1024,
                    ir_len: 5120,
                    block_size: 256,
                },
                NonUniformStageConfig {
                    ir_offset: 6144,
                    ir_len: 3072,
                    block_size: 1024,
                },
            ],
        })
        .unwrap();

        for (label, plan, x_len) in [
            ("default_2048", plan_default, 4096usize),
            ("strategy1_8k", plan_s1, 10_240usize),
            ("strategy2_16k", plan_s2, 18_432usize),
            ("extreme_defer_9k", plan_defer, 12_288usize),
        ] {
            let ir: Vec<f32> = (0..plan.max_ir_len())
                .map(|k| {
                    let env = (-(k as f32) * 4.5 / plan.max_ir_len() as f32).exp();
                    rng.next_f32() * 0.25 * env
                })
                .collect();
            let input = rng.vec_f32(x_len, 0.8);

            let req_s = NonUniformConvolver::memory_requirements(&plan, &staged_fft).unwrap();
            let req_m = NonUniformConvolver::memory_requirements(&plan, &mono_fft).unwrap();
            assert_eq!(
                req_s, req_m,
                "{label}: staged transforms require 0 extra RAM"
            );

            let mut ws_s = vec![0.0f32; req_s.total_f32];
            let mut ws_m = vec![0.0f32; req_m.total_f32];
            let mut conv_s =
                NonUniformConvolver::prepare(&ir, plan, &mut ws_s, backend, staged_fft).unwrap();
            let mut conv_m =
                NonUniformConvolver::prepare(&ir, plan, &mut ws_m, backend, mono_fft).unwrap();

            let mut out_s = vec![0.0f32; x_len];
            let mut out_m = vec![0.0f32; x_len];
            for (ib, (os, om)) in input
                .chunks(64)
                .zip(out_s.chunks_mut(64).zip(out_m.chunks_mut(64)))
            {
                conv_s.process_block(ib, os).unwrap();
                conv_m.process_block(ib, om).unwrap();
            }

            assert_eq!(conv_s.deadline_misses(), 0, "{label} staged misses");
            assert_eq!(conv_m.deadline_misses(), 0, "{label} mono misses");

            // Exact bit-for-bit equality between time-distributed and monolithic transforms!
            for i in 0..x_len {
                assert_eq!(
                    out_s[i].to_bits(),
                    out_m[i].to_bits(),
                    "{label}: bit mismatch at sample {i}: staged={} mono={}",
                    out_s[i],
                    out_m[i]
                );
            }

            // Oracle verification against f64 reference (SNR > 100 dB).
            let check_len = x_len.min(4096);
            let ref_f64 = oracle_f64(&ir, &input[..check_len]);
            let snr = snr_db(&ref_f64, &out_s[..check_len]);
            assert!(
                snr > 100.0,
                "{label}: f64 oracle SNR={snr:.2} dB below 100 dB"
            );
        }
    }

    #[test]
    fn deterministic_op_trace_proves_worst_ops_bound_and_peak_to_mean_flattening() {
        let model = CostModel::x86_avx2_calibrated();
        let micro = MicroFftBackend::new();
        let simd_fft = SimdFixedFftBackend::new(SimdBackend::scalar());
        let mut sc = vec![OptimizerScratchCell::ZERO; optimizer_scratch_len(65_536, 8192)];

        // Trace Strategy-1 (full B_s execution span -> duty cycle 1.0x -> peak/mean <= 2.5x)
        // and Strategy-2 (compressed clearance span -> e.g. B=4096 in 6 callbacks at 64k gives
        // mathematical floor ~B/span = 10.7x for that stage, flattened to <= 8.0x overall vs >14x
        // monolithic).
        let cases = [
            (
                "S1_4k_H64",
                4_096usize,
                64usize,
                ClearancePolicy::Strategy1,
                2.5f64,
            ),
            ("S1_16k_H64", 16_384, 64, ClearancePolicy::Strategy1, 2.5),
            ("S1_64k_H64", 65_536, 64, ClearancePolicy::Strategy1, 2.5),
            ("S2_16k_H64", 16_384, 64, ClearancePolicy::Strategy2, 5.0),
            ("S2_64k_H64", 65_536, 64, ClearancePolicy::Strategy2, 8.0),
            ("S1_16k_H32", 16_384, 32, ClearancePolicy::Strategy1, 3.0),
            ("S2_16k_H128", 16_384, 128, ClearancePolicy::Strategy2, 5.0),
            ("S2_64k_H128", 65_536, 128, ClearancePolicy::Strategy2, 8.0),
        ];

        for (label, ir_len, h, policy, max_peak_to_mean) in cases {
            let (plan, _) = NonUniformPlan::optimize(ir_len, h, &model, policy, &mut sc).unwrap();
            let max_b = plan
                .stages()
                .iter()
                .map(|s| s.block_size)
                .max()
                .unwrap_or(64);
            let trace_samples = (4 * max_b).max(8192);
            let direct_ops = plan.direct_sample_ops();

            // 1. Native f32 staged trace (r = i = log2(B))
            let mut sched_f32 = NonUniformScheduler::new(&plan);
            let mut per_sample_f32 = vec![0usize; trace_samples];
            for slot in &mut per_sample_f32 {
                let mut ops = direct_ops;
                for (idx, st) in plan.stages().iter().enumerate() {
                    if st.kind == NonUniformStageKind::Fdl {
                        let steps = st.block_size.trailing_zeros() as usize;
                        let item = sched_f32.on_fdl_sample_with_steps(idx, &plan, steps, steps);
                        ops += stage_work_item_ops(&plan, idx, item, steps, steps);
                    }
                }
                sched_f32.advance_sample();
                *slot = ops;
            }
            assert_eq!(sched_f32.total_deadline_misses(), 0, "{label} f32 misses");

            // Check EVERY possible contiguous H-sample sliding window (all H alignments!).
            let bound_plan = plan.estimated_worst_ops_per_callback();
            let bound_simd = plan.estimated_worst_ops_for_backend(&simd_fft).unwrap();
            assert!(bound_simd <= bound_plan);

            let mut win_sum: usize = per_sample_f32[..h].iter().sum();
            let mut max_win = win_sum;
            for i in h..trace_samples {
                win_sum = win_sum + per_sample_f32[i] - per_sample_f32[i - h];
                if win_sum > max_win {
                    max_win = win_sum;
                }
            }
            assert!(
                max_win <= bound_simd,
                "{label} f32: actual max callback ops ({max_win}) exceeded backend bound ({bound_simd})"
            );
            assert!(
                max_win <= bound_plan,
                "{label} f32: actual max callback ops ({max_win}) exceeded plan bound ({bound_plan})"
            );

            // Steady-state mean over complete cycles after initial ramp (last 2 * max_b samples).
            let steady_start = trace_samples - 2 * max_b;
            let steady_total: usize = per_sample_f32[steady_start..].iter().sum();
            let mean_cb = (steady_total as f64) * (h as f64) / ((2 * max_b) as f64);
            let ratio = (max_win as f64) / mean_cb;
            assert!(
                ratio <= max_peak_to_mean,
                "{label} f32: peak/mean = {ratio:.2} exceeds declared factor {max_peak_to_mean:.2} (max={max_win}, mean={mean_cb:.1})"
            );

            // 2. Native Q1.15 staged trace (r = i = log2(2*B))
            let mut sched_q15 = NonUniformScheduler::new(&plan);
            let mut per_sample_q15 = vec![0usize; trace_samples];
            for slot in &mut per_sample_q15 {
                let mut ops = direct_ops;
                for (idx, st) in plan.stages().iter().enumerate() {
                    if st.kind == NonUniformStageKind::Fdl {
                        let steps = st.fft_size.trailing_zeros() as usize;
                        let item = sched_q15.on_fdl_sample_with_steps(idx, &plan, steps, steps);
                        ops += stage_work_item_ops(&plan, idx, item, steps, steps);
                    }
                }
                sched_q15.advance_sample();
                *slot = ops;
            }
            assert_eq!(sched_q15.total_deadline_misses(), 0, "{label} q15 misses");
            let mut q_win: usize = per_sample_q15[..h].iter().sum();
            let mut q_max = q_win;
            for i in h..trace_samples {
                q_win = q_win + per_sample_q15[i] - per_sample_q15[i - h];
                if q_win > q_max {
                    q_max = q_win;
                }
            }
            assert!(
                q_max <= bound_plan,
                "{label} q15: actual max callback ops ({q_max}) exceeded plan bound ({bound_plan})"
            );

            // 3. Monolithic backend (`MicroFftBackend`, r = i = 1) bounded by `estimated_worst_ops_for_backend`
            let bound_micro = plan.estimated_worst_ops_for_backend(&micro).unwrap();
            let has_multi_cb_stage = plan.stages().iter().any(|s| {
                s.kind == NonUniformStageKind::Fdl
                    && s.slack_samples.min(s.block_size.saturating_sub(1)) + 1 > h
            });
            if has_multi_cb_stage {
                assert!(
                    bound_simd < bound_micro,
                    "{label}: staged bound ({bound_simd}) must be strictly smaller than monolithic bound ({bound_micro})"
                );
            }
            let mut sched_mono = NonUniformScheduler::new(&plan);
            let mut per_sample_mono = vec![0usize; trace_samples];
            for slot in &mut per_sample_mono {
                let mut ops = direct_ops;
                for (idx, st) in plan.stages().iter().enumerate() {
                    if st.kind == NonUniformStageKind::Fdl {
                        let item = sched_mono.on_fdl_sample_with_steps(idx, &plan, 1, 1);
                        ops += stage_work_item_ops(&plan, idx, item, 1, 1);
                    }
                }
                sched_mono.advance_sample();
                *slot = ops;
            }
            assert_eq!(sched_mono.total_deadline_misses(), 0, "{label} mono misses");
            let mut m_win: usize = per_sample_mono[..h].iter().sum();
            let mut m_max = m_win;
            for i in h..trace_samples {
                m_win = m_win + per_sample_mono[i] - per_sample_mono[i - h];
                if m_win > m_max {
                    m_max = m_win;
                }
            }
            assert!(
                m_max <= bound_micro,
                "{label} mono: actual max ({m_max}) exceeded microfft bound ({bound_micro})"
            );
            // When the largest stage's transform itself spans multiple callbacks
            // (`h * total_units < span * r_steps`), subdividing the RFFT/IFFT strictly lowers
            // observed peak callback ops below monolithic RFFT/IFFT (even with manual RFFT/CMAC/IFFT
            // separation!). Otherwise both execute one full RFFT + half the CMACs inside one
            // callback (agreeing up to `div_ceil` rounding of `<= r_steps` ops).
            let largest_splits_transform = plan
                .stages()
                .iter()
                .rfind(|s| s.kind == NonUniformStageKind::Fdl)
                .is_some_and(|s| {
                    let span = s.slack_samples.min(s.block_size.saturating_sub(1)) + 1;
                    let r_steps = s.block_size.trailing_zeros() as usize;
                    let total_units = 2 * r_steps + s.num_partitions;
                    h * total_units < span * r_steps
                });
            if largest_splits_transform {
                assert!(
                    max_win < m_max,
                    "{label}: staged peak ops ({max_win}) must be strictly smaller than monolithic peak ops ({m_max})"
                );
            } else {
                assert!(
                    max_win <= m_max + 32,
                    "{label}: staged peak ops ({max_win}) unexpectedly above monolithic ({m_max})"
                );
            }
            println!(
                "op_trace {label:<12}: staged max={max_win:>7} (peak/mean={ratio:4.2}x, bound={bound_simd:>7}) | mono max={m_max:>7} (peak/mean={:4.2}x, bound={bound_micro:>7}) | q15 max={q_max:>7}",
                (m_max as f64) / mean_cb
            );

            // Budget admission accepts staged SimdFixedFftBackend at `bound_simd` while rejecting
            // monolithic MicroFftBackend under the exact same tight ops ceiling!
            let tight_budget = NonUniformBudget {
                sample_rate_hz: 48_000,
                max_frames_per_callback: h,
                deadline_us: 1333,
                max_memory_bytes: 0,
                max_ops_per_callback: bound_simd,
                safety_margin_percent: 0,
            };
            assert_eq!(
                plan.check_budget_for_backend(tight_budget, 0, &simd_fft)
                    .unwrap(),
                rtconv_simd::BudgetVerdict::WithinKnownBounds
            );
            assert_eq!(
                plan.check_budget_for_backend(tight_budget, 0, &micro)
                    .unwrap(),
                rtconv_simd::BudgetVerdict::Exceeded
            );
        }
    }

    #[test]
    fn delay_ring_and_shared_input_ram_reduction_verified() {
        let fft = SimdFixedFftBackend::new(SimdBackend::scalar());
        // High-offset stage matching prompt_eng.md §3 PR-C (B=8192 at offset ~30k).
        let stages = [
            NonUniformStageConfig {
                ir_offset: 0,
                ir_len: 64,
                block_size: 1,
            },
            NonUniformStageConfig {
                ir_offset: 64,
                ir_len: 448,
                block_size: 64,
            },
            NonUniformStageConfig {
                ir_offset: 512,
                ir_len: 3_584,
                block_size: 256,
            },
            NonUniformStageConfig {
                ir_offset: 4_096,
                ir_len: 25_904,
                block_size: 1024,
            },
            NonUniformStageConfig {
                ir_offset: 30_000,
                ir_len: 35_536,
                block_size: 8192,
            },
        ];
        let plan = NonUniformPlan::validate(NonUniformConfig {
            max_ir_len: 65_536,
            host_block_max: 64,
            stages: &stages,
        })
        .unwrap();
        let req = NonUniformConvolver::memory_requirements(&plan, &fft).unwrap();

        // Delay ring is strictly 2 * B_s per FDL stage (power of two).
        let expected_delay_f32: usize = plan
            .stages()
            .iter()
            .filter(|s| s.kind == NonUniformStageKind::Fdl)
            .map(|s| 2 * s.block_size)
            .sum();
        assert_eq!(req.delay_f32, expected_delay_f32);
        assert_eq!(req.delay_f32, 2 * (64 + 256 + 1024 + 8192));

        // Legacy delay sum was sum(D_s + B_s) = 44,208 floats (> 2.3x larger!).
        let legacy_delay_f32: usize = plan
            .stages()
            .iter()
            .filter(|s| s.kind == NonUniformStageKind::Fdl)
            .map(|s| s.ir_offset + s.block_size)
            .sum();
        assert_eq!(legacy_delay_f32, 44_208);
        assert!(req.delay_f32 * 2 < legacy_delay_f32);

        // Single shared input ring replaces all per-stage overlap_in[2*B_s]:
        // state_f32 = 2*64 (direct) + max(2*8192, 30001) = 128 + 30001 = 30129.
        assert_eq!(req.state_f32, 2 * 64 + 30_001);
        let legacy_state_plus_delay = 2 * 64 + expected_delay_f32 + legacy_delay_f32;
        assert!(req.state_f32 + req.delay_f32 < legacy_state_plus_delay);
    }

    #[test]
    fn fragmented_host_blocks_zero_deadline_misses_f32() {
        let mut rng = Rng::new(0x0016_F4A6);
        let backend = SimdBackend::detect_or_baseline();
        let fft = SimdFixedFftBackend::new(backend);
        let model = CostModel::for_backend_f32(backend);
        let mut sc = vec![OptimizerScratchCell::ZERO; optimizer_scratch_len(16_384, 8192)];

        for policy in [ClearancePolicy::Strategy1, ClearancePolicy::Strategy2] {
            let (plan, _) = NonUniformPlan::optimize(8_192, 64, &model, policy, &mut sc).unwrap();
            let ir = rng.vec_f32(8_192, 0.15);
            let input = rng.vec_f32(10_000, 0.75);
            let req = NonUniformConvolver::memory_requirements(&plan, &fft).unwrap();

            // Stream 1: irregular/prime host chunks
            let mut ws_frag = vec![0.0f32; req.total_f32];
            let mut conv_frag =
                NonUniformConvolver::prepare(&ir, plan, &mut ws_frag, backend, fft).unwrap();
            let mut out_frag = vec![0.0f32; input.len()];
            let chunks = [1usize, 7, 13, 31, 64, 100, 17, 257, 3, 511, 29];
            let mut pos = 0usize;
            let mut ci = 0usize;
            while pos < input.len() {
                let len = chunks[ci % chunks.len()].min(input.len() - pos);
                conv_frag
                    .process_block(&input[pos..pos + len], &mut out_frag[pos..pos + len])
                    .unwrap();
                pos += len;
                ci += 1;
            }
            assert_eq!(conv_frag.deadline_misses(), 0);

            // Stream 2: sample-by-sample reference -> must be bit-for-bit identical!
            let mut ws_1 = vec![0.0f32; req.total_f32];
            let mut conv_1 =
                NonUniformConvolver::prepare(&ir, plan, &mut ws_1, backend, fft).unwrap();
            for (i, (&x, &yf)) in input.iter().zip(out_frag.iter()).enumerate() {
                let y1 = conv_1.process_sample(x).unwrap();
                assert_eq!(
                    yf.to_bits(),
                    y1.to_bits(),
                    "policy={policy:?} fragmented vs sample-by-sample mismatch at {i}"
                );
            }
            assert_eq!(conv_1.deadline_misses(), 0);

            // Reset reproducibility check
            conv_frag.reset();
            let mut out_reset = vec![0.0f32; 2048];
            conv_frag
                .process_block(&input[..2048], &mut out_reset)
                .unwrap();
            assert_eq!(out_reset, out_frag[..2048]);
        }
    }
}

#[cfg(feature = "fixed-nonuniform")]
mod fixed_nonuniform_pr_c {
    use super::*;
    use rtconv_simd::fixed::{f64_to_q15, q15_to_f64};
    use rtconv_simd::{
        FixedQ15NonUniformConvolver, NonUniformConfig, NonUniformPlan, NonUniformStageConfig,
        NonUniformStageKind, preset_esp32_c6_q15,
    };

    fn oracle_q15_f64(ir: &[i16], input: &[i16]) -> Vec<f64> {
        (0..input.len())
            .map(|n| {
                let mut acc = 0.0f64;
                let max_k = ir.len().min(n + 1);
                for k in 0..max_k {
                    acc += q15_to_f64(ir[k]) * q15_to_f64(input[n - k]);
                }
                acc
            })
            .collect()
    }

    fn snr_q15_db(reference: &[f64], actual: &[i16]) -> f64 {
        let mut s2 = 0.0f64;
        let mut e2 = 0.0f64;
        for (&r, &a) in reference.iter().zip(actual.iter()) {
            let d = r - q15_to_f64(a);
            s2 += r * r;
            e2 += d * d;
        }
        10.0 * (s2 / e2.max(1e-300)).log10()
    }

    #[test]
    fn nonuniform_q15_time_distributed_deferral_fragmentation_and_aes17() {
        let mut rng = Rng::new(0x0016_0C60);

        // 1. RAM reduction check on high-offset Q1.15 plan (B=1024 at offset 8192).
        let plan_defer = NonUniformPlan::validate(NonUniformConfig {
            max_ir_len: 12_288,
            host_block_max: 64,
            stages: &[
                NonUniformStageConfig {
                    ir_offset: 0,
                    ir_len: 64,
                    block_size: 1,
                },
                NonUniformStageConfig {
                    ir_offset: 64,
                    ir_len: 960,
                    block_size: 32,
                },
                NonUniformStageConfig {
                    ir_offset: 1024,
                    ir_len: 7168,
                    block_size: 256,
                },
                NonUniformStageConfig {
                    ir_offset: 8192,
                    ir_len: 4096,
                    block_size: 1024,
                },
            ],
        })
        .unwrap();
        let req_defer = FixedQ15NonUniformConvolver::memory_requirements(&plan_defer).unwrap();
        let expected_delay_i16: usize = plan_defer
            .stages()
            .iter()
            .filter(|s| s.kind == NonUniformStageKind::Fdl)
            .map(|s| 4 * s.block_size)
            .sum();
        assert_eq!(req_defer.delay_i16, expected_delay_i16);
        // Legacy delay_i16 was 2 * ((64+32) + (1024+256) + (8192+1024)) = 21,184 words!
        assert_eq!(req_defer.delay_i16, 4 * (32 + 256 + 1024));
        // Shared scratch no longer duplicates 4*max_fft (saves 16 KiB at N2048!).
        assert_eq!(req_defer.scratch_i16, 2 * 1025 + 2048);

        // 2. End-to-end oracle + fragmented blocks + reset on default_profile & ESP32-C6 preset.
        let plan_c6 = preset_esp32_c6_q15(4096, 64).unwrap();
        let plan_def = NonUniformPlan::default_profile(2048, 32).unwrap();

        for (label, plan) in [("default_2048", plan_def), ("esp32_c6_4096", plan_c6)] {
            let ir_len = plan.max_ir_len();
            // Direct impulse + stage-spanning decaying echo tail (l1 <= 0.85 for ITU-R BS.1770-5
            // Annex 2 headroom, keeping |H(997 Hz)| ~ 0.5 so -60 dBFS sits ~30 dB above the Q1.15 floor).
            let mut raw = vec![0.0f64; ir_len];
            raw[0] = 0.50;
            // Place explicit reflections in every stage (including the deferred tail!) plus diffuse tail.
            let mut tail_l1 = 0.0f64;
            for (k, slot) in raw.iter_mut().enumerate().skip(1) {
                let env = (-(k as f64) * 5.0 / ir_len as f64).exp();
                let v = (rng.next_f32() as f64) * env;
                *slot = v;
                tail_l1 += v.abs();
            }
            for slot in raw.iter_mut().skip(1) {
                *slot *= 0.35 / tail_l1.max(1e-12);
            }
            let ir: Vec<i16> = raw.iter().map(|&v| f64_to_q15(v)).collect();

            // AES17 997 Hz @ -1 dBFS & -60 dBFS (48 kHz)
            let fs = 48_000.0f64;
            let amp_m1 = 10.0f64.powf(-1.0 / 20.0);
            let amp_m60 = 10.0f64.powf(-60.0 / 20.0);
            let n_sig = ir_len + 2048;

            for (lvl_name, amp, min_snr) in [("-1 dBFS", amp_m1, 70.0), ("-60 dBFS", amp_m60, 28.0)]
            {
                let input: Vec<i16> = (0..n_sig)
                    .map(|n| {
                        let s = amp * (2.0 * core::f64::consts::PI * 997.0 * (n as f64) / fs).sin();
                        f64_to_q15(s)
                    })
                    .collect();

                let req = FixedQ15NonUniformConvolver::memory_requirements(&plan).unwrap();
                let mut ws = vec![0i16; req.total_i16];
                let mut conv = FixedQ15NonUniformConvolver::prepare(&ir, plan, &mut ws).unwrap();
                let mut out = vec![0i16; n_sig];

                // Process with fragmented prime/irregular host chunks
                let chunks = [1usize, 7, 13, 31, 64, 19, 100, 3, 127, 29];
                let mut pos = 0usize;
                let mut ci = 0usize;
                while pos < n_sig {
                    let len = chunks[ci % chunks.len()].min(n_sig - pos);
                    conv.process_block(&input[pos..pos + len], &mut out[pos..pos + len])
                        .unwrap();
                    pos += len;
                    ci += 1;
                }
                assert_eq!(conv.deadline_misses(), 0, "{label} {lvl_name} misses");
                assert_eq!(conv.saturation_count(), 0, "{label} {lvl_name} saturation");

                // Verify reset() produces bit-identical output
                conv.reset();
                let mut out_reset = vec![0i16; 1024];
                conv.process_block(&input[..1024], &mut out_reset).unwrap();
                assert_eq!(out_reset, out[..1024], "{label} {lvl_name} reset mismatch");

                let ref_f64 = oracle_q15_f64(&ir, &input);
                let snr = snr_q15_db(&ref_f64[ir_len..], &out[ir_len..]);
                println!("Q1.15 staged AES17 {label:<14} {lvl_name:<8}: SNR={snr:.2} dB");
                assert!(
                    snr > min_snr,
                    "{label} {lvl_name}: AES17 997 Hz steady-state SNR={snr:.2} dB <= {min_snr:.1} dB"
                );
            }
        }
    }
}
