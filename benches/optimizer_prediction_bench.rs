//! Predicted vs Measured Cost-Model Validation Benchmark (Issue #15 / PR-B, Wefers Ch. 7
//! Figs. 7.5 & 7.6 analogue, `HOST_ONLY`).
//!
//! Compares `CostModel` predictions (`ns/sample`) against measured wall-clock execution time
//! (`ns/sample` and per-callback `us`) for:
//! 1. Uniform Partitioned Convolution (`PartitionedConvolver`, `B = 128..=4096`, Fig. 7.5 analogue).
//! 2. Cost-optimized NUPC (`NonUniformPlan::optimize`, `MinLoad` / `Strategy1` / `Strategy2`) vs
//!    `default_profile` and `HybridConvolver` at `4k`, `16k`, and `64k` taps (Fig. 7.6 analogue).
//!
//! Reports GUM (ISO/IEC Guide 98-3 / JCGM 100:2008) Type A statistics (`n`, `mean ± U` at `k = 2`,
//! `s`, `u_A`), tail percentiles (`p50`/`p95`/`p99`/`max`), relative error
//! `(T_meas - T_pred) / T_meas`, and ISO 80000-1 Annex B / ABNT NBR 5891 half-to-even rounding.
//!
//! Empirical host measurements only — NOT formal WCET proofs.

use rtconv_simd::nonuniform::{
    ClearancePolicy, CostModel, NonUniformConvolver, NonUniformPlan, OptimizerScratchCell,
    optimizer_scratch_len,
};
use rtconv_simd::{
    HybridConfig, HybridConvolver, PartitionedConfig, PartitionedConvolver, SimdBackend,
    SimdFixedFftBackend,
};
use std::hint::black_box;
use std::time::Instant;

const HOST_BLOCK: usize = 64;
const WARMUP_BLOCKS: usize = 512;
const MEAS_BLOCKS: usize = 4096;

fn make_ir(len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| ((i as f32) * 0.17).sin() * (-5.0 * i as f32 / len as f32).exp() * 0.25)
        .collect()
}

/// Round-half-to-even to `decimals` fractional digits (ISO 80000-1 Annex B / ABNT NBR 5891).
fn round_half_even(x: f64, decimals: usize) -> f64 {
    let scale = 10.0f64.powi(decimals as i32);
    (x * scale).round_ties_even() / scale
}

/// Formats `mean ± U` with 2 significant digits on `U` and `mean` aligned to the same decimal
/// place using half-to-even rounding (ISO 80000-1 Annex B / ABNT NBR 5891).
fn format_gum(mean: f64, u_exp: f64) -> String {
    if !u_exp.is_finite() || u_exp <= 0.0 {
        return format!("{mean:.2} ± 0");
    }
    let exp = u_exp.log10().floor() as i32;
    let decimals = (1 - exp).max(0) as usize;
    let r_mean = round_half_even(mean, decimals);
    let r_u = round_half_even(u_exp, decimals);
    format!("{r_mean:.decimals$} ± {r_u:.decimals$}")
}

fn summarize_ns_per_sample(label: &str, pred_ns: f64, block_ns: &[f64], block_len: usize) {
    let n = block_ns.len();
    let per_sample: Vec<f64> = block_ns.iter().map(|&t| t / block_len as f64).collect();
    let mean = per_sample.iter().sum::<f64>() / n as f64;
    let var = per_sample
        .iter()
        .map(|&x| {
            let d = x - mean;
            d * d
        })
        .sum::<f64>()
        / (n - 1) as f64;
    let s = var.sqrt();
    let u_a = s / (n as f64).sqrt();
    let u_exp = 2.0 * u_a;

    let mut sorted = block_ns.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let pct_us = |q: f64| -> f64 { sorted[((n - 1) as f64 * q).round() as usize] / 1000.0 };
    let p50 = pct_us(0.50);
    let p95 = pct_us(0.95);
    let p99 = pct_us(0.99);
    let max = sorted[n - 1] / 1000.0;

    let rel_err_pct = 100.0 * (mean - pred_ns) / mean;
    let gum = format_gum(mean, u_exp);

    println!(
        "{label:<36} | pred={pred_ns:6.1} | meas={gum:<16} ns/smp (s={s:5.1}, uA={u_a:4.2}) | err={rel_err_pct:+6.1}% | cb us: p50={p50:5.2} p95={p95:5.2} p99={p99:6.2} max={max:6.2}"
    );
}

fn bench_upols(ir: &[f32], block_size: usize, model: &CostModel, backend: SimdBackend) {
    let fft = SimdFixedFftBackend::new(backend);
    let cfg = PartitionedConfig {
        block_size,
        max_ir_len: ir.len(),
    };
    let req = PartitionedConvolver::memory_requirements(cfg, &fft).unwrap();
    let mut ws = vec![0.0f32; req.total_f32];
    let mut conv = PartitionedConvolver::prepare(ir, cfg, &mut ws, backend, fft).unwrap();

    let inp = vec![0.25f32; block_size];
    let mut out = vec![0.0f32; block_size];
    let warmup = (WARMUP_BLOCKS * HOST_BLOCK).div_ceil(block_size).max(32);
    let iters = (MEAS_BLOCKS * HOST_BLOCK)
        .div_ceil(block_size)
        .clamp(256, 4096);
    for _ in 0..warmup {
        conv.process_block(&inp, &mut out).unwrap();
    }
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        conv.process_block(black_box(&inp), black_box(&mut out))
            .unwrap();
        times.push(t0.elapsed().as_nanos() as f64);
    }

    let parts = ir.len().div_ceil(block_size);
    let pred_ns = model.segment_cost_per_sample(block_size, parts).unwrap();
    let label = format!("UPOLS B={block_size:<4} (P={parts:<3})");
    summarize_ns_per_sample(&label, pred_ns, &times, block_size);
}

fn bench_hybrid(ir: &[f32], block_size: usize, model: &CostModel, backend: SimdBackend) {
    let fft = SimdFixedFftBackend::new(backend);
    let cfg = HybridConfig {
        head_len: block_size,
        tail_block_size: block_size,
        max_ir_len: ir.len(),
    };
    let req = HybridConvolver::memory_requirements(cfg, &fft).unwrap();
    let mut ws = vec![0.0f32; req.total_f32];
    let mut conv =
        HybridConvolver::prepare_with_tail_spreading(ir, cfg, &mut ws, backend, fft, true).unwrap();

    let inp = [0.25f32; HOST_BLOCK];
    let mut out = [0.0f32; HOST_BLOCK];
    for _ in 0..WARMUP_BLOCKS {
        conv.process_block(&inp, &mut out).unwrap();
    }
    let mut times = Vec::with_capacity(MEAS_BLOCKS);
    for _ in 0..MEAS_BLOCKS {
        let t0 = Instant::now();
        conv.process_block(black_box(&inp), black_box(&mut out))
            .unwrap();
        times.push(t0.elapsed().as_nanos() as f64);
    }

    let pred_ns = model.hybrid_cost_per_sample(ir.len(), block_size).unwrap();
    let label = format!("Hybrid(H{block_size}/T{block_size}) SPREAD");
    summarize_ns_per_sample(&label, pred_ns, &times, HOST_BLOCK);
}

fn bench_nupc_plan(
    label: &str,
    ir: &[f32],
    plan: NonUniformPlan,
    model: &CostModel,
    backend: SimdBackend,
) {
    let fft = SimdFixedFftBackend::new(backend);
    let pred_ns = model.plan_cost_per_sample(&plan).unwrap();
    let req = NonUniformConvolver::memory_requirements(&plan, &fft).unwrap();
    let mut ws = vec![0.0f32; req.total_f32];
    let mut conv = NonUniformConvolver::prepare(ir, plan, &mut ws, backend, fft).unwrap();

    let inp = [0.25f32; HOST_BLOCK];
    let mut out = [0.0f32; HOST_BLOCK];
    for _ in 0..WARMUP_BLOCKS {
        conv.process_block(&inp, &mut out).unwrap();
    }
    let mut times = Vec::with_capacity(MEAS_BLOCKS);
    for _ in 0..MEAS_BLOCKS {
        let t0 = Instant::now();
        conv.process_block(black_box(&inp), black_box(&mut out))
            .unwrap();
        times.push(t0.elapsed().as_nanos() as f64);
    }
    assert_eq!(conv.deadline_misses(), 0);
    summarize_ns_per_sample(label, pred_ns, &times, HOST_BLOCK);
}

fn format_plan_compact(plan: &NonUniformPlan) -> String {
    let mut parts = Vec::new();
    for st in plan.stages() {
        if st.block_size == 1 {
            parts.push(format!("D{}", st.ir_len));
        } else {
            parts.push(format!("{}^{}", st.block_size, st.num_partitions));
        }
    }
    parts.join(" ")
}

fn run_for_backend(backend: SimdBackend, model: &CostModel, model_label: &str) {
    println!(
        "\n====================================================================================================="
    );
    println!(
        "Backend: {} | Model: {model_label} | block={HOST_BLOCK} | warmup={WARMUP_BLOCKS} | n={MEAS_BLOCKS} (k=2, HOST_ONLY)",
        backend.name()
    );
    println!(
        "====================================================================================================="
    );

    // Part 1: Wefers Fig. 7.5 analogue — UPOLS predicted vs measured across B = 128..=4096 at L = 16,384
    let ir_16k = make_ir(16_384);
    println!("\n--- Fig. 7.5 Analogue: UPOLS L = 16,384 taps ---");
    for &b in &[128usize, 256, 512, 1024, 2048, 4096] {
        bench_upols(&ir_16k, b, model, backend);
    }

    // Part 2: Wefers Fig. 7.6 analogue — Optimized NUPC vs Hybrid vs default_profile at 4k, 16k, 64k
    for &ir_len in &[4_096usize, 16_384, 65_536] {
        println!("\n--- Fig. 7.6 Analogue: L = {ir_len} taps ---");
        let ir = make_ir(ir_len);

        let best_h_b = [32usize, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
            .into_iter()
            .min_by(|&a, &b| {
                let ca = model.hybrid_cost_per_sample(ir_len, a).unwrap();
                let cb = model.hybrid_cost_per_sample(ir_len, b).unwrap();
                ca.total_cmp(&cb)
            })
            .unwrap();
        bench_hybrid(&ir, best_h_b, model, backend);

        let def_plan = NonUniformPlan::default_profile(ir_len, HOST_BLOCK).unwrap();
        bench_nupc_plan("NUPC default_profile", &ir, def_plan, model, backend);

        let sc_len = optimizer_scratch_len(ir_len, 8192);
        let mut sc = vec![OptimizerScratchCell::ZERO; sc_len];
        for (pname, policy) in [
            ("MinLoad", ClearancePolicy::MinLoad),
            ("Strategy1", ClearancePolicy::Strategy1),
            ("Strategy2", ClearancePolicy::Strategy2),
        ] {
            let t0 = Instant::now();
            let (plan, _rep) =
                NonUniformPlan::optimize(ir_len, HOST_BLOCK, model, policy, &mut sc).unwrap();
            let opt_us = t0.elapsed().as_nanos() as f64 / 1000.0;
            let desc = format_plan_compact(&plan);
            let label = format!("NUPC {pname:<9} [{desc}]");
            println!(
                "  (prepare DP time: {opt_us:.1} us, scratch: {:.1} KiB)",
                (sc_len * 8) as f64 / 1024.0
            );
            bench_nupc_plan(&label, &ir, plan, model, backend);
        }
    }
}

fn main() {
    println!("=== PR-B (#15) Optimizer Cost-Model Prediction vs Measurement Benchmark ===");
    println!("Provenance: HOST_ONLY (empirical timing; NOT a formal WCET proof).");

    let detected = SimdBackend::detect_or_baseline();
    let model_det = CostModel::for_backend_f32(detected);
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let is_avx = !detected.is_scalar()
        && (detected.level().as_avx2().is_some() || detected.level().as_avx512().is_some());
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    let is_avx = false;

    let det_label = if is_avx {
        "x86_avx2_calibrated"
    } else {
        "for_backend_f32"
    };
    run_for_backend(detected, &model_det, det_label);

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_avx && let Ok(sse2) = SimdBackend::try_sse2() {
        let model_sse2 = CostModel::x86_sse2_calibrated();
        run_for_backend(sse2, &model_sse2, "x86_sse2_calibrated");
    }
}
