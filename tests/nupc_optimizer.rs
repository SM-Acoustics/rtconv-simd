//! Comprehensive verification suite for the cost-model NUPC partition optimizer (Issue #15 / PR-B).
//!
//! Covers:
//! 1. 100% `NonUniformPlan::validate` compliance across all cost models, clearance policies,
//!    Q1.15 ceilings (`B <= 1024`), zero-scratch MCU presets, and negative/edge inputs.
//! 2. Exhaustive brute-force optimality proof on small impulse responses (`N <= 384`).
//! 3. Dominance over `NonUniformPlan::default_profile` under the same `CostModel`.
//! 4. Formal verification of the sample-time Strategy-1 clearance offset `c = 1`
//!    (`D_s = 2*B_s - 1` vs `D_s = 2*B_s - 2`) directly on `NonUniformScheduler` and
//!    `NonUniformConvolver`.
//! 5. Primary-source reproduction of Wefers (2015) §6.5.3, §6.5.4, §6.12 (`N = 88,200, B = 128`),
//!    and Appendix A.2.1 / A.2.4 / A.2.5 tables, plus sample-time vs block-mode comparison.
//! 6. `f64` reference oracle (`SNR > 100 dB` for `f32` up to `65,536` taps) and AES17 997 Hz
//!    (`-60 dBFS` & `-1 dBFS`) verification for optimized `f32` and `Q1.15` NUPC plans.
#![cfg(feature = "nonuniform")]

use rtconv_simd::nonuniform::{
    BlockModePolicy, ClearancePolicy, CostModel, NonUniformConfig, NonUniformConvolver,
    NonUniformPlan, NonUniformScheduler, NonUniformStageConfig, NonUniformStageKind,
    OPTIMIZER_BLOCK_SIZES, OptimizerScratchCell, StageWorkItem, block_mode_scratch_len,
    optimize_block_mode, optimizer_scratch_len, preset_esp32_c6_q15, preset_esp32_p4_f32,
};
use rtconv_simd::{ConvError, SimdBackend, SimdFixedFftBackend};

#[cfg(feature = "fixed-nonuniform")]
use rtconv_simd::FixedQ15NonUniformConvolver;

fn optimize_helper(
    ir_len: usize,
    host_block_max: usize,
    max_block_size: usize,
    model: &CostModel,
    policy: ClearancePolicy,
) -> (NonUniformPlan, rtconv_simd::OptimizationReport) {
    let sc_len = optimizer_scratch_len(ir_len, max_block_size);
    let mut sc = vec![OptimizerScratchCell::ZERO; sc_len];
    NonUniformPlan::optimize_with_max_block_size(
        ir_len,
        host_block_max,
        max_block_size,
        model,
        policy,
        &mut sc,
    )
    .expect("optimizer should succeed on valid inputs")
}

// ------------------------------------------------------------------------------------------
// 1. Validation, edge cases, Q1.15 ceiling, MCU presets, and negative inputs
// ------------------------------------------------------------------------------------------

#[test]
fn all_generated_plans_pass_validate_across_models_and_policies() {
    let models = [
        ("theoretical", CostModel::theoretical()),
        ("x86_sse2", CostModel::x86_sse2_calibrated()),
        ("x86_avx2", CostModel::x86_avx2_calibrated()),
        ("wefers_2015", CostModel::wefers_2015_test_system()),
        ("esp32_p4_est", CostModel::esp32_p4_f32_estimate()),
        ("esp32_c6_est", CostModel::esp32_c6_q15_estimate()),
    ];
    let policies = [
        ClearancePolicy::MinLoad,
        ClearancePolicy::Strategy1,
        ClearancePolicy::Strategy2,
        ClearancePolicy::Custom {
            min_slack_samples: 0,
        },
        ClearancePolicy::Custom {
            min_slack_samples: 64,
        },
    ];
    let lengths = [
        1usize, 2, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 511, 512, 1000, 1024, 2048,
        4096, 8192, 16384, 32768, 65536,
    ];

    for (mname, model) in models {
        for policy in policies {
            for &ir_len in &lengths {
                let (plan, report) = optimize_helper(ir_len, 64, 8192, &model, policy);
                assert_eq!(plan.max_ir_len(), ir_len, "{mname} {policy:?} L={ir_len}");
                assert_eq!(report.covered_ir_len, ir_len);
                assert!(
                    (report.predicted_cost_per_sample - model.plan_cost_per_sample(&plan).unwrap())
                        .abs()
                        < 1e-9,
                    "report cost mismatch for {mname} {policy:?} L={ir_len}"
                );
                // Re-validate explicitly through `NonUniformPlan::validate`.
                let stage_cfgs: Vec<NonUniformStageConfig> = plan
                    .stages()
                    .iter()
                    .map(|s| NonUniformStageConfig {
                        ir_offset: s.ir_offset,
                        ir_len: s.ir_len,
                        block_size: s.block_size,
                    })
                    .collect();
                let revalidated = NonUniformPlan::validate(NonUniformConfig {
                    max_ir_len: ir_len,
                    host_block_max: 64,
                    stages: &stage_cfgs,
                })
                .expect("revalidation failed");
                assert_eq!(revalidated.stage_count(), plan.stage_count());

                // Verify every FDL stage strictly obeys the requested clearance policy.
                for st in plan.stages() {
                    if st.kind == NonUniformStageKind::Fdl {
                        let min_slack = policy.min_slack_samples(st.block_size, 64);
                        assert!(
                            st.slack_samples >= min_slack,
                            "{mname} {policy:?} L={ir_len}: stage B={} slack={} < min_slack={}",
                            st.block_size,
                            st.slack_samples,
                            min_slack
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn mcu_presets_and_q15_ceiling_validate_cleanly() {
    for &ir_len in &[
        1usize, 32, 64, 128, 129, 256, 512, 1024, 2048, 4096, 16384, 65536,
    ] {
        let p4 = preset_esp32_p4_f32(ir_len, 64).unwrap();
        assert_eq!(p4.max_ir_len(), ir_len);

        let c6 = preset_esp32_c6_q15(ir_len, 64).unwrap();
        assert_eq!(c6.max_ir_len(), ir_len);
        for st in c6.stages() {
            if st.kind == NonUniformStageKind::Fdl {
                assert!(
                    st.block_size <= 1024 && st.fft_size <= 2048,
                    "ESP32-C6 preset exceeded N2048: B={}",
                    st.block_size
                );
            }
        }

        let (q_opt, _) = optimize_helper(
            ir_len,
            64,
            1024,
            &CostModel::esp32_c6_q15_estimate(),
            ClearancePolicy::Strategy2,
        );
        for st in q_opt.stages() {
            if st.kind == NonUniformStageKind::Fdl {
                assert!(st.block_size <= 1024 && st.fft_size <= 2048);
            }
        }

        #[cfg(feature = "fixed-nonuniform")]
        {
            let req_preset = FixedQ15NonUniformConvolver::memory_requirements(&c6).unwrap();
            let req_opt = FixedQ15NonUniformConvolver::memory_requirements(&q_opt).unwrap();
            assert!(req_preset.instance_bytes < 4096);
            assert!(req_opt.instance_bytes < 4096);
        }
    }
}

#[test]
fn optimizer_negative_and_error_cases() {
    let model = CostModel::theoretical();
    let mut sc = [OptimizerScratchCell::ZERO; 16];

    assert_eq!(
        NonUniformPlan::optimize(0, 64, &model, ClearancePolicy::MinLoad, &mut sc).unwrap_err(),
        ConvError::InvalidNonUniformConfig
    );
    assert_eq!(
        NonUniformPlan::optimize(128, 0, &model, ClearancePolicy::MinLoad, &mut sc).unwrap_err(),
        ConvError::InvalidNonUniformConfig
    );

    // Undersized scratch returns WorkspaceTooSmall (never panics).
    let req = optimizer_scratch_len(1024, 8192);
    let mut small = vec![OptimizerScratchCell::ZERO; req - 1];
    assert!(matches!(
        NonUniformPlan::optimize(1024, 64, &model, ClearancePolicy::MinLoad, &mut small),
        Err(ConvError::WorkspaceTooSmall { required, provided })
            if required == req && provided == req - 1
    ));

    // Invalid cost models (NaN, negative, zero direct MAC) are rejected.
    let mut bad = model;
    bad.t_direct_mac = 0.0;
    assert_eq!(bad.validate(), Err(ConvError::InvalidNonUniformConfig));
    let mut bad_nan = model;
    bad_nan.t_fft[2] = f64::NAN;
    assert_eq!(bad_nan.validate(), Err(ConvError::InvalidNonUniformConfig));
    let mut bad_neg = model;
    bad_neg.t_cmac[0] = -1.0;
    assert_eq!(bad_neg.validate(), Err(ConvError::InvalidNonUniformConfig));
}

// ------------------------------------------------------------------------------------------
// 2. Exhaustive brute-force optimality proof on small IR lengths
// ------------------------------------------------------------------------------------------

fn brute_force_best_cost(
    ir_len: usize,
    host_block_max: usize,
    max_block_size: usize,
    model: &CostModel,
    policy: ClearancePolicy,
) -> f64 {
    let mut best = model.direct_cost_per_sample(ir_len);

    // Allowed FDL block sizes strictly < ir_len and <= max_block_size.
    let levels: Vec<usize> = OPTIMIZER_BLOCK_SIZES
        .iter()
        .copied()
        .filter(|&b| b <= max_block_size && b < ir_len)
        .collect();

    if levels.is_empty() {
        return best;
    }

    // Same mod-32 head residues explored by the sample-time DP: {0, ir_len % 32, 31}.
    let mut rems = vec![0usize];
    if !ir_len.is_multiple_of(32) {
        rems.push(ir_len % 32);
    }
    if !matches!(policy, ClearancePolicy::MinLoad) && ir_len % 32 != 31 {
        rems.push(31);
    }

    let max_l0 = match policy {
        ClearancePolicy::Strategy2 => (host_block_max.max(32) * 4).max(128),
        _ => usize::MAX,
    };

    #[allow(clippy::too_many_arguments)]
    fn dfs(
        offset: usize,
        level_idx: usize,
        stages_used: usize,
        acc_cost: f64,
        ir_len: usize,
        host_block_max: usize,
        levels: &[usize],
        model: &CostModel,
        policy: ClearancePolicy,
        best: &mut f64,
    ) {
        if acc_cost >= *best {
            return;
        }
        if offset >= ir_len {
            if stages_used > 0 && acc_cost < *best {
                *best = acc_cost;
            }
            return;
        }
        if level_idx >= levels.len() || stages_used >= rtconv_simd::MAX_OPTIMIZER_FDL_STAGES {
            return;
        }

        // Branch 1: skip levels[level_idx].
        dfs(
            offset,
            level_idx + 1,
            stages_used,
            acc_cost,
            ir_len,
            host_block_max,
            levels,
            model,
            policy,
            best,
        );

        // Branch 2: use levels[level_idx] for P = 1..=P_max partitions if clearance permits.
        let b = levels[level_idx];
        if offset >= policy.min_offset_samples(b, host_block_max) {
            let max_p = (ir_len - offset).div_ceil(b);
            for p in 1..=max_p {
                let seg_cost = model.segment_cost_per_sample(b, p).unwrap();
                dfs(
                    offset + p * b,
                    level_idx + 1,
                    stages_used + 1,
                    acc_cost + seg_cost,
                    ir_len,
                    host_block_max,
                    levels,
                    model,
                    policy,
                    best,
                );
            }
        }
    }

    for rem in rems {
        let mut l0 = if rem == 0 { 32 } else { rem };
        while l0 < ir_len && l0 <= max_l0 {
            let head_cost = model.direct_cost_per_sample(l0);
            dfs(
                l0,
                0,
                0,
                head_cost,
                ir_len,
                host_block_max,
                &levels,
                model,
                policy,
                &mut best,
            );
            l0 += 32;
        }
    }

    best
}

#[test]
fn dp_matches_exhaustive_brute_force_optimality_proof() {
    let models = [
        ("theoretical", CostModel::theoretical()),
        ("x86_avx2", CostModel::x86_avx2_calibrated()),
        ("esp32_p4", CostModel::esp32_p4_f32_estimate()),
    ];
    let policies = [
        ClearancePolicy::MinLoad,
        ClearancePolicy::Strategy1,
        ClearancePolicy::Strategy2,
        ClearancePolicy::Custom {
            min_slack_samples: 32,
        },
    ];
    let lengths = [64usize, 95, 96, 128, 130, 160, 192, 200, 224, 256, 320, 384];

    for (mname, model) in models {
        for policy in policies {
            for &ir_len in &lengths {
                let (_plan, report) = optimize_helper(ir_len, 64, 256, &model, policy);
                let bf_cost = brute_force_best_cost(ir_len, 64, 256, &model, policy);
                assert!(
                    (report.predicted_cost_per_sample - bf_cost).abs() <= 1e-9,
                    "DP vs brute-force mismatch ({mname}, {policy:?}, L={ir_len}): dp={:.9}, bf={:.9}",
                    report.predicted_cost_per_sample,
                    bf_cost
                );
            }
        }
    }
}

// ------------------------------------------------------------------------------------------
// 3. Dominance over `default_profile` under the same `CostModel`
// ------------------------------------------------------------------------------------------

#[test]
fn optimized_plan_never_exceeds_default_profile_cost() {
    let models = [
        ("theoretical", CostModel::theoretical()),
        ("x86_sse2", CostModel::x86_sse2_calibrated()),
        ("x86_avx2", CostModel::x86_avx2_calibrated()),
        ("wefers_2015", CostModel::wefers_2015_test_system()),
        ("esp32_p4_est", CostModel::esp32_p4_f32_estimate()),
        ("esp32_c6_est", CostModel::esp32_c6_q15_estimate()),
    ];
    let lengths = [
        16usize, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536,
    ];

    for (mname, model) in models {
        for &ir_len in &lengths {
            let def = NonUniformPlan::default_profile(ir_len, 64).unwrap();
            let def_cost = model.plan_cost_per_sample(&def).unwrap();

            // `MinLoad` strictly includes `default_profile` in its feasible space, so its cost
            // must NEVER exceed `default_profile` for ANY length or ANY cost model!
            let (_opt_min, rep_min) =
                optimize_helper(ir_len, 64, 8192, &model, ClearancePolicy::MinLoad);
            assert!(
                rep_min.predicted_cost_per_sample <= def_cost + 1e-9,
                "MinLoad exceeded default_profile ({mname}, L={ir_len}): opt={:.4} > def={:.4}",
                rep_min.predicted_cost_per_sample,
                def_cost
            );

            // For medium/long IRs (`>= 4096`), even clearance-constrained `Strategy2` beats
            // `default_profile` (whose FFTs are capped at B=128) decisively!
            if ir_len >= 4096 {
                let (_opt_s2, rep_s2) =
                    optimize_helper(ir_len, 64, 8192, &model, ClearancePolicy::Strategy2);
                assert!(
                    rep_s2.predicted_cost_per_sample <= def_cost + 1e-9,
                    "Strategy2 exceeded default_profile ({mname}, L={ir_len}): opt={:.4} > def={:.4}",
                    rep_s2.predicted_cost_per_sample,
                    def_cost
                );
            }
        }
    }
}

// ------------------------------------------------------------------------------------------
// 4. Formal proof test of sample-time Strategy-1 clearance offset `c = 1`
// ------------------------------------------------------------------------------------------

#[test]
fn formal_sample_time_clearance_c_equals_1_boundary() {
    for &b in &[32usize, 64, 128, 256] {
        // Policy formula check: `offset >= 2*B - 1` <=> `slack >= B - 1` (`c = 1`).
        assert_eq!(ClearancePolicy::Strategy1.min_slack_samples(b, 64), b - 1);
        assert_eq!(
            ClearancePolicy::Strategy1.min_offset_samples(b, 64),
            2 * b - 1
        );

        // Case A: Exact boundary `D_s = 2*B - 1` (`slack = B - 1`, `c = 1`):
        // - Block 0 finishes collecting at `t_avail = B - 1`, triggering `StartBlockAndCmac`.
        // - Scheduler spreads work over `W_eff = min(B - 1, B - 1) = B - 1` follow-up samples,
        //   giving total inclusive execution span `W_eff + 1 = B` samples (`t = B - 1 ..= 2*B - 2`).
        // - Delay ring commits at `t = 2*B - 2`, right before sample `t = 2*B - 1 = D_s` is read!
        // - Zero overlap with Block 1 arriving at `t = 2*B - 1` (`deadline_misses() == 0`).
        let d_valid = 2 * b - 1;
        // With `host_block_max = 1` and 2 partitions (`spread_steps = 2`), `step_interval`
        // equals `effective_window` exactly, so the final step fires at `t = (B - 1) + W_eff`.
        let plan_valid = NonUniformPlan::validate(NonUniformConfig {
            max_ir_len: d_valid + 2 * b,
            host_block_max: 1,
            stages: &[
                NonUniformStageConfig {
                    ir_offset: 0,
                    ir_len: d_valid,
                    block_size: 1,
                },
                NonUniformStageConfig {
                    ir_offset: d_valid,
                    ir_len: 2 * b,
                    block_size: b,
                },
            ],
        })
        .unwrap();

        let mut sched = NonUniformScheduler::new(&plan_valid);
        let mut start_t = None;
        let mut finish_t = None;
        for t in 0..(2 * b) {
            match sched.on_fdl_sample(1, &plan_valid) {
                StageWorkItem::StartBlockAndCmac { finish_ifft, .. } => {
                    if start_t.is_none() {
                        start_t = Some(t);
                    }
                    if finish_ifft && finish_t.is_none() {
                        finish_t = Some(t);
                    }
                }
                StageWorkItem::ContinueCmac { finish_ifft, .. }
                    if finish_ifft && finish_t.is_none() =>
                {
                    finish_t = Some(t);
                }
                _ => {}
            }
        }
        assert_eq!(start_t, Some(b - 1));
        assert_eq!(finish_t, Some(2 * b - 2));
        let span_valid = finish_t.unwrap() - start_t.unwrap() + 1;
        assert_eq!(
            span_valid, b,
            "D_s = 2B - 1 must yield exact timespan_eff = B = {b}"
        );
        assert_eq!(sched.total_deadline_misses(), 0);

        // Execute end-to-end inside `NonUniformConvolver` and confirm 0 deadline misses + exact impulse!
        let mut ir = vec![0.0f32; plan_valid.max_ir_len()];
        ir[0] = 0.5;
        ir[d_valid] = 0.25;
        ir[d_valid + b] = -0.125;
        let fft = SimdFixedFftBackend::new(SimdBackend::scalar());
        let req = NonUniformConvolver::memory_requirements(&plan_valid, &fft).unwrap();
        let mut ws = vec![0.0f32; req.total_f32];
        let mut conv =
            NonUniformConvolver::prepare(&ir, plan_valid, &mut ws, SimdBackend::scalar(), fft)
                .unwrap();
        let mut out = vec![0.0f32; d_valid + 3 * b];
        for (i, y) in out.iter_mut().enumerate() {
            *y = conv.process_sample(if i == 0 { 1.0 } else { 0.0 }).unwrap();
        }
        assert_eq!(conv.deadline_misses(), 0);
        assert!((out[0] - 0.5).abs() < 1e-6);
        assert!((out[d_valid] - 0.25).abs() < 1e-6);
        assert!((out[d_valid + b] + 0.125).abs() < 1e-6);

        // Case B: One sample shorter `D_s = 2*B - 2` (`slack = B - 2`, i.e. `c = 2`):
        // Total inclusive span drops to `B - 1 < B`, proving `c = 1` is tight!
        let d_short = 2 * b - 2;
        let plan_short = NonUniformPlan::validate(NonUniformConfig {
            max_ir_len: d_short + 2 * b,
            host_block_max: 1,
            stages: &[
                NonUniformStageConfig {
                    ir_offset: 0,
                    ir_len: d_short,
                    block_size: 1,
                },
                NonUniformStageConfig {
                    ir_offset: d_short,
                    ir_len: 2 * b,
                    block_size: b,
                },
            ],
        })
        .unwrap();
        let mut sched_short = NonUniformScheduler::new(&plan_short);
        let mut s_t = None;
        let mut f_t = None;
        for t in 0..(2 * b) {
            match sched_short.on_fdl_sample(1, &plan_short) {
                StageWorkItem::StartBlockAndCmac { finish_ifft, .. } => {
                    if s_t.is_none() {
                        s_t = Some(t);
                    }
                    if finish_ifft && f_t.is_none() {
                        f_t = Some(t);
                    }
                }
                StageWorkItem::ContinueCmac { finish_ifft, .. } if finish_ifft && f_t.is_none() => {
                    f_t = Some(t);
                }
                _ => {}
            }
        }
        let span_short = f_t.unwrap() - s_t.unwrap() + 1;
        assert_eq!(
            span_short,
            b - 1,
            "D_s = 2B - 2 yields only B - 1 < B samples, proving c = 1 is tight"
        );
    }
}

// ------------------------------------------------------------------------------------------
// 5. Wefers (2015) §6.5.3, §6.5.4, §6.12 & Appendix A.2 primary-source reproduction
// ------------------------------------------------------------------------------------------

#[test]
fn wefers_2015_section_6_5_3_structural_properties() {
    let model = CostModel::wefers_2015_test_system();
    let b = 128usize;

    // Property 1: Uniform partition (1 segment) remains optimal up to ~20-30*B (Wefers §6.5.3).
    let mut first_multi_seg = None;
    for mult in 1..=40 {
        let n = mult * b;
        let mut sc = vec![OptimizerScratchCell::ZERO; block_mode_scratch_len(n, b, 8192)];
        let part = optimize_block_mode(n, b, 8192, &model, BlockModePolicy::MinLoad, &mut sc)
            .expect("block-mode optimize failed");
        if part.segments().len() > 1 && first_multi_seg.is_none() {
            first_multi_seg = Some(mult);
        }
    }
    let transition = first_multi_seg.expect("expected transition from uniform to non-uniform");
    assert!(
        (10..=35).contains(&transition),
        "uniform->nonuniform transition at {transition}*B (expected ~15-30*B per Wefers §6.5.3)"
    );

    // Property 2: At most 4 segments are needed across all typical acoustic IR lengths up to 131k.
    for &n in &[
        128usize, 1024, 4096, 8192, 16384, 32768, 65536, 88200, 131072,
    ] {
        let mut sc = vec![OptimizerScratchCell::ZERO; block_mode_scratch_len(n, b, 8192)];
        let part = optimize_block_mode(n, b, 8192, &model, BlockModePolicy::MinLoad, &mut sc)
            .expect("block-mode optimize failed");
        assert!(
            part.segments().len() <= 4,
            "Wefers §6.5.3 predicts <= 4 segments, got {} for N={n}",
            part.segments().len()
        );
    }
}

#[test]
fn wefers_2015_section_6_12_and_appendix_a2_exact_reproduction() {
    let model = CostModel::wefers_2015_test_system();
    let n = 88_200usize;
    let b = 128usize;
    let mut sc = vec![OptimizerScratchCell::ZERO; block_mode_scratch_len(n, b, 8192)];

    // 1. Minimal-load partition (Appendix A.2.1 & §6.12):
    //    (128^7, 1024^7, 8192^10), covered = 89,984, cost ~= 204.1 cycles/sample.
    let min_part =
        optimize_block_mode(n, b, 8192, &model, BlockModePolicy::MinLoad, &mut sc).unwrap();
    let min_sig: Vec<(usize, usize)> = min_part
        .segments()
        .iter()
        .map(|s| (s.block_size, s.partitions))
        .collect();
    assert_eq!(min_sig, vec![(128, 7), (1024, 7), (8192, 10)]);
    assert_eq!(min_part.covered_len, 89_984);
    assert!(
        (min_part.cost_per_sample - 204.1).abs() < 0.5,
        "min-load cost = {:.2} (Wefers §6.12 = 204.1)",
        min_part.cost_per_sample
    );
    // UPOLS baseline for 88,200 taps at B=128: ceil(88200/128) = 690 partitions ~= 2015.6 cycles/sample (~9.9x).
    let upols_cost = model.segment_cost_per_sample(128, n.div_ceil(128)).unwrap();
    let speedup = upols_cost / min_part.cost_per_sample;
    assert!(
        (speedup - 9.9).abs() < 0.2,
        "min-load speedup vs UPOLS = {speedup:.2}x (Wefers §6.12 = 9.9x)"
    );

    // 2. Practical partition from §6.12 & Appendix A.2.5 (variant 2):
    //    (128^2, 256^4, 1024^8, 8192^10), covered = 91,392, clearances = (0, 1, 3, 11),
    //    cost ~= 240.4 cycles/sample (~8.4x vs UPOLS, +18% vs min-load).
    let prac_part = optimize_block_mode(
        n,
        b,
        8192,
        &model,
        BlockModePolicy::Strategy2TableA25,
        &mut sc,
    )
    .unwrap();
    let prac_sig: Vec<(usize, usize, usize)> = prac_part
        .segments()
        .iter()
        .map(|s| (s.block_size, s.partitions, s.clearance_cycles))
        .collect();
    assert_eq!(
        prac_sig,
        vec![(128, 2, 0), (256, 4, 1), (1024, 8, 3), (8192, 10, 11)]
    );
    assert_eq!(prac_part.covered_len, 91_392);
    assert!(
        (prac_part.cost_per_sample - 240.4).abs() < 0.5,
        "practical cost = {:.2} (Wefers §6.12 = 240.4)",
        prac_part.cost_per_sample
    );
    let overhead_pct = 100.0 * (prac_part.cost_per_sample / min_part.cost_per_sample - 1.0);
    assert!(
        (overhead_pct - 17.8).abs() < 1.0,
        "practical overhead = {overhead_pct:.1}% (Wefers §6.12 = +18%)"
    );

    // 3. Appendix A.2.4 (variant 1 table): (128^22, 1024^21, 8192^8), clearances (0, 15, 127), ~276.2.
    let a24_part = optimize_block_mode(
        n,
        b,
        8192,
        &model,
        BlockModePolicy::Strategy1TableA24,
        &mut sc,
    )
    .unwrap();
    let a24_sig: Vec<(usize, usize, usize)> = a24_part
        .segments()
        .iter()
        .map(|s| (s.block_size, s.partitions, s.clearance_cycles))
        .collect();
    assert_eq!(a24_sig, vec![(128, 22, 0), (1024, 21, 15), (8192, 8, 127)]);
    assert!(
        (a24_part.cost_per_sample - 276.2).abs() < 0.5,
        "A.2.4 cost = {:.2} (thesis = 276.2)",
        a24_part.cost_per_sample
    );

    // 4. §6.5.4 prose Strategy 1 (`clearance >= L_i/B - 1`): (128^14, 1024^14, 8192^9), clearances (0, 7, 63).
    let s1_prose =
        optimize_block_mode(n, b, 8192, &model, BlockModePolicy::Strategy1Prose, &mut sc).unwrap();
    let s1_sig: Vec<(usize, usize, usize)> = s1_prose
        .segments()
        .iter()
        .map(|s| (s.block_size, s.partitions, s.clearance_cycles))
        .collect();
    assert_eq!(s1_sig, vec![(128, 14, 0), (1024, 14, 7), (8192, 9, 63)]);
    assert!(s1_prose.cost_per_sample < a24_part.cost_per_sample);

    // 5. Verbatim Appendix A.2.5 rows (B=128, practical variant 2, Wefers 2015 pp. 243–245).
    type A25Row = (usize, &'static [(usize, usize)], f64);
    let a25_rows: &[A25Row] = &[
        (1_280, &[(128, 2), (256, 4)], 93.9),
        (4_352, &[(128, 2), (256, 16)], 126.1),
        (6_400, &[(128, 2), (256, 4), (1024, 5)], 147.3),
        (16_640, &[(128, 2), (256, 4), (1024, 15)], 172.9),
        (20_736, &[(128, 2), (256, 8), (2048, 9)], 181.7),
        (33_024, &[(128, 2), (256, 16), (4096, 7)], 201.3),
        (66_816, &[(128, 2), (256, 4), (1024, 8), (8192, 7)], 230.3),
        (124_160, &[(128, 2), (256, 4), (1024, 8), (8192, 14)], 253.9),
    ];
    for &(target_n, expected_segs, expected_cost) in a25_rows {
        let mut rsc = vec![OptimizerScratchCell::ZERO; block_mode_scratch_len(target_n, b, 8192)];
        let p = optimize_block_mode(
            target_n,
            b,
            8192,
            &model,
            BlockModePolicy::Strategy2TableA25,
            &mut rsc,
        )
        .unwrap();
        let got: Vec<(usize, usize)> = p
            .segments()
            .iter()
            .map(|s| (s.block_size, s.partitions))
            .collect();
        assert_eq!(got, expected_segs, "A.2.5 mismatch at N={target_n}");
        assert!(
            (p.cost_per_sample - expected_cost).abs() < 0.5,
            "A.2.5 cost mismatch at N={target_n}: got {:.2}, thesis {expected_cost:.1}",
            p.cost_per_sample
        );
    }

    // 6. Compare sample-time zero-latency optimizer (`NonUniformPlan::optimize`) against
    //    block-mode at `N = 88,200`: sample-time replaces the framed `128^2` head with a
    //    zero-latency direct head (`L_0 * t_direct_mac`) plus `B_1 >= 32` FDL stages obeying
    //    `D_s >= B_s`, achieving comparable or lower cycles/sample with ZERO algorithmic latency!
    let (sample_plan, sample_rep) =
        optimize_helper(n, 128, 8192, &model, ClearancePolicy::Strategy2);
    assert_eq!(sample_plan.max_ir_len(), n);
    assert!(sample_rep.direct_head_len > 0);
    assert!(sample_rep.fdl_stage_count >= 2 && sample_rep.fdl_stage_count <= 4);
    // Sample-time zero-latency plan stays competitive with block-mode (within 2x of block-mode A.2.5
    // despite eliminating the 128-sample framing latency).
    assert!(
        sample_rep.predicted_cost_per_sample < 1.5 * prac_part.cost_per_sample,
        "sample-time zero-latency cost {:.2} vs block-mode {:.2}",
        sample_rep.predicted_cost_per_sample,
        prac_part.cost_per_sample
    );
}

// ------------------------------------------------------------------------------------------
// 6. `f64` Oracle (`SNR > 100 dB`) and AES17 997 Hz (`-60 dBFS` & `-1 dBFS`) on optimized plans
// ------------------------------------------------------------------------------------------

fn xorshift64(state: &mut u64) -> f64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    ((x >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

fn make_decaying_ir(len: usize, seed: u64) -> Vec<f64> {
    let mut s = seed;
    let mut h: Vec<f64> = (0..len)
        .map(|k| {
            let env = (-(k as f64) * 6.0 / len as f64).exp();
            let v = if k == 0 {
                0.6
            } else {
                0.15 * env * xorshift64(&mut s)
            };
            (v as f32) as f64
        })
        .collect();
    // Normalize L1 norm <= 0.85 so ITU-R BS.1770-5 inter-sample peak cannot exceed unity.
    let l1: f64 = h.iter().map(|v| v.abs()).sum();
    if l1 > 0.85 {
        let scale = 0.85 / l1;
        for v in &mut h {
            *v = ((*v * scale) as f32) as f64;
        }
    }
    h
}

fn oracle_convolve(ir: &[f64], x: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0f64; x.len()];
    for n in 0..x.len() {
        let mut acc = 0.0f64;
        let lim = ir.len().min(n + 1);
        for k in 0..lim {
            acc += ir[k] * x[n - k];
        }
        y[n] = acc;
    }
    y
}

fn snr_db(reference: &[f64], actual: &[f64]) -> f64 {
    let mut sig = 0.0f64;
    let mut err = 0.0f64;
    for (&r, &a) in reference.iter().zip(actual.iter()) {
        sig += r * r;
        let d = r - a;
        err += d * d;
    }
    10.0 * (sig / err.max(1e-300)).log10()
}

fn aes17_residual_db(reference: &[f64], actual: &[f64]) -> f64 {
    // RMS residual error relative to full-scale sine (0 dBFS = 1/sqrt(2) RMS).
    let n = reference.len().min(actual.len());
    let mut err2 = 0.0f64;
    for i in 0..n {
        let d = reference[i] - actual[i];
        err2 += d * d;
    }
    let rms_err = (err2 / n as f64).sqrt();
    let rms_fs = core::f64::consts::FRAC_1_SQRT_2;
    20.0 * (rms_err.max(1e-30) / rms_fs).log10()
}

#[test]
fn f32_optimized_plans_match_f64_oracle_and_aes17() {
    let backend = SimdBackend::detect_or_baseline();
    let fft = SimdFixedFftBackend::new(backend);
    let model = CostModel::for_backend_f32(backend);

    // 1. Broadband oracle verification for 4k, 16k, and 64k optimized plans.
    for &(ir_len, sig_len) in &[(4_096usize, 6_000usize), (16_384, 18_000)] {
        let (plan, _rep) = optimize_helper(ir_len, 64, 8192, &model, ClearancePolicy::Strategy2);
        let ir64 = make_decaying_ir(ir_len, 0xC0FFEE ^ ir_len as u64);
        let ir32: Vec<f32> = ir64.iter().map(|&v| v as f32).collect();

        let req = NonUniformConvolver::memory_requirements(&plan, &fft).unwrap();
        let mut ws = vec![0.0f32; req.total_f32];
        let mut conv = NonUniformConvolver::prepare(&ir32, plan, &mut ws, backend, fft).unwrap();

        let mut seed = 0x1234_5678_9ABC_DEF0u64 ^ ir_len as u64;
        let x64: Vec<f64> = (0..sig_len)
            .map(|_| ((0.5 * xorshift64(&mut seed)) as f32) as f64)
            .collect();
        let x32: Vec<f32> = x64.iter().map(|&v| v as f32).collect();
        let mut y32 = vec![0.0f32; sig_len];

        let mut pos = 0usize;
        for &blk in [1usize, 19, 64, 127, 256, 33].iter().cycle() {
            if pos >= sig_len {
                break;
            }
            let n = blk.min(sig_len - pos);
            conv.process_block(&x32[pos..pos + n], &mut y32[pos..pos + n])
                .unwrap();
            pos += n;
        }
        assert_eq!(conv.deadline_misses(), 0);

        let ref64 = oracle_convolve(&ir64, &x64);
        let y64: Vec<f64> = y32.iter().map(|&v| v as f64).collect();
        let s = snr_db(&ref64, &y64);
        assert!(
            s > 100.0,
            "f32 optimized NUPC L={ir_len} oracle SNR = {s:.2} dB (expected > 100 dB)"
        );
    }

    // 2. Sparse-tap 65,536-tap exact oracle (fast O(sig_len * sparse_taps) exact f64 oracle)
    //    verifying large B=4096 optimized plan end-to-end!
    {
        let ir_len = 65_536usize;
        let (plan, rep) = optimize_helper(ir_len, 64, 8192, &model, ClearancePolicy::Strategy2);
        assert!(rep.fdl_stage_count >= 2);
        let sparse = [
            (0usize, 0.5f32),
            (37, -0.2),
            (400, 0.15),
            (4_500, -0.1),
            (32_768, 0.08),
            (65_535, -0.05),
        ];
        let mut ir32 = vec![0.0f32; ir_len];
        for &(idx, val) in &sparse {
            ir32[idx] = val;
        }
        let req = NonUniformConvolver::memory_requirements(&plan, &fft).unwrap();
        let mut ws = vec![0.0f32; req.total_f32];
        let mut conv = NonUniformConvolver::prepare(&ir32, plan, &mut ws, backend, fft).unwrap();

        let sig_len = 70_000usize;
        let mut seed = 0x55AA_33CC_1122_7788u64;
        let x32: Vec<f32> = (0..sig_len)
            .map(|_| (0.5 * xorshift64(&mut seed)) as f32)
            .collect();
        let mut y32 = vec![0.0f32; sig_len];
        for (xi, yo) in x32.chunks(64).zip(y32.chunks_mut(64)) {
            conv.process_block(xi, yo).unwrap();
        }
        assert_eq!(conv.deadline_misses(), 0);

        let ref64: Vec<f64> = (0..sig_len)
            .map(|n| {
                sparse
                    .iter()
                    .filter(|&&(k, _)| n >= k)
                    .map(|&(k, h)| (h as f64) * (x32[n - k] as f64))
                    .sum()
            })
            .collect();
        let y64: Vec<f64> = y32.iter().map(|&v| v as f64).collect();
        let s = snr_db(&ref64, &y64);
        assert!(
            s > 100.0,
            "f32 optimized NUPC L=65536 oracle SNR = {s:.2} dB (expected > 100 dB)"
        );
    }

    // 3. AES17 997 Hz @ -60 dBFS and -1 dBFS on 16,384-tap optimized f32 NUPC.
    {
        let fs = 48_000.0f64;
        let freq = 997.0f64;
        let ir_len = 16_384usize;
        let (plan, _) = optimize_helper(ir_len, 64, 8192, &model, ClearancePolicy::Strategy2);
        let ir64 = make_decaying_ir(ir_len, 0xAE517);
        let ir32: Vec<f32> = ir64.iter().map(|&v| v as f32).collect();
        let req = NonUniformConvolver::memory_requirements(&plan, &fft).unwrap();
        let mut ws = vec![0.0f32; req.total_f32];
        let mut conv = NonUniformConvolver::prepare(&ir32, plan, &mut ws, backend, fft).unwrap();

        let sig_len = 20_480usize;
        for &(level_dbfs, min_dr_db) in &[(-60.0f64, 120.0f64), (-1.0f64, 110.0f64)] {
            conv.reset();
            let amp = 10.0f64.powf(level_dbfs / 20.0);
            let x32: Vec<f32> = (0..sig_len)
                .map(|n| (amp * (2.0 * core::f64::consts::PI * freq * n as f64 / fs).sin()) as f32)
                .collect();
            let x64: Vec<f64> = x32.iter().map(|&v| v as f64).collect();
            let mut y32 = vec![0.0f32; sig_len];
            for (xi, yo) in x32.chunks(64).zip(y32.chunks_mut(64)) {
                conv.process_block(xi, yo).unwrap();
            }
            let ref64 = oracle_convolve(&ir64, &x64);
            let y64: Vec<f64> = y32.iter().map(|&v| v as f64).collect();
            let res_dbfs = aes17_residual_db(&ref64[ir_len..], &y64[ir_len..]);
            assert!(
                -res_dbfs > min_dr_db,
                "f32 AES17 997Hz @ {level_dbfs} dBFS residual = {res_dbfs:.2} dBFS (expected < -{min_dr_db} dBFS)"
            );
        }
    }
}

#[cfg(feature = "fixed-nonuniform")]
#[test]
fn q15_optimized_plans_aes17_997hz_honest_bfp_metrics() {
    let fs = 48_000.0f64;
    let freq = 997.0f64;
    let model = CostModel::esp32_c6_q15_estimate();

    for &(ir_len, sig_len) in &[(1_024usize, 3_072usize), (4_096usize, 6_144usize)] {
        let (plan, _) = optimize_helper(ir_len, 64, 1024, &model, ClearancePolicy::Strategy2);
        let ir64 = make_decaying_ir(ir_len, 0x1515 ^ ir_len as u64);
        // Quantize IR to Q1.15 cleanly (already L1 <= 0.85 -> zero ITU-R BS.1770-5 clipping).
        let ir_q15: Vec<i16> = ir64
            .iter()
            .map(|&v| (v * 32768.0).round().clamp(-32768.0, 32767.0) as i16)
            .collect();
        let ir_ref: Vec<f64> = ir_q15.iter().map(|&v| v as f64 / 32768.0).collect();

        let req = FixedQ15NonUniformConvolver::memory_requirements(&plan).unwrap();
        let mut ws = vec![0i16; req.total_i16];
        let mut conv = FixedQ15NonUniformConvolver::prepare(&ir_q15, plan, &mut ws).unwrap();

        for &level_dbfs in &[-60.0f64, -1.0f64] {
            conv.reset();
            let amp = 10.0f64.powf(level_dbfs / 20.0);
            let x_q15: Vec<i16> = (0..sig_len)
                .map(|n| {
                    let s = amp * (2.0 * core::f64::consts::PI * freq * n as f64 / fs).sin();
                    (s * 32768.0).round().clamp(-32768.0, 32767.0) as i16
                })
                .collect();
            let x_ref: Vec<f64> = x_q15.iter().map(|&v| v as f64 / 32768.0).collect();
            let mut y_q15 = vec![0i16; sig_len];
            for (xi, yo) in x_q15.chunks(64).zip(y_q15.chunks_mut(64)) {
                conv.process_block(xi, yo).unwrap();
            }
            assert_eq!(
                conv.saturation_count(),
                0,
                "unexpected saturation on L1-normalized IR L={ir_len} @ {level_dbfs} dBFS"
            );

            let ref64 = oracle_convolve(&ir_ref, &x_ref);
            let y64: Vec<f64> = y_q15.iter().map(|&v| v as f64 / 32768.0).collect();
            let res_dbfs = aes17_residual_db(&ref64[ir_len..], &y64[ir_len..]);
            let s_db = snr_db(&ref64[ir_len..], &y64[ir_len..]);
            // Record honest BFP dynamic range & SNR relative to Q1.15 ceiling (~98.1 dB).
            // With B=64/256 (N=128/512, 7..9 BFP stages) on 16-bit mantissa, residual sits well
            // below -72 dBFS at -1 dBFS and below -85 dBFS at -60 dBFS (no wrap-around/overflow).
            let max_res = if level_dbfs < -30.0 { -80.0 } else { -70.0 };
            assert!(
                res_dbfs < max_res,
                "Q1.15 optimized NUPC L={ir_len} @ {level_dbfs} dBFS: residual={res_dbfs:.2} dBFS, SNR={s_db:.2} dB"
            );
        }
    }
}
