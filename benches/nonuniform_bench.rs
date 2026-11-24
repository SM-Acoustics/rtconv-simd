//! Comparative NUPC vs Hybrid benchmark at 16k and 64k taps (`HOST_ONLY`).
//!
//! Computes GUM (ISO/IEC Guide 98-3 / JCGM 100:2008) Type A uncertainty (`n`, `mean`, `s`,
//! `u_A = s/sqrt(n)`, `U = 2*u_A`), tail percentiles (`p50`/`p95`/`p99`/`max`), and ISO 80000-1
//! Annex B / ABNT NBR 5891 half-to-even rounded `mean ± U` summaries after warm-up.
//!
//! Empirical host measurements only — NOT formal WCET proofs.

use rtconv_simd::nonuniform::{
    ClearancePolicy, CostModel, OptimizerScratchCell, optimizer_scratch_len,
};
use rtconv_simd::{
    HybridConfig, HybridConvolver, NonUniformConfig, NonUniformConvolver, NonUniformPlan,
    NonUniformStageConfig, SimdBackend, SimdFixedFftBackend,
};
use std::hint::black_box;
use std::time::Instant;

const SAMPLE_RATE_HZ: f64 = 48_000.0;
const HOST_BLOCK: usize = 32;
const WARMUP_ITERS: usize = 512;
const MEAS_ITERS: usize = 4096;

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
        return format!("{mean:.3} ± 0");
    }
    let exp = u_exp.log10().floor() as i32;
    let decimals = (1 - exp).max(0) as usize;
    let r_mean = round_half_even(mean, decimals);
    let r_u = round_half_even(u_exp, decimals);
    format!("{r_mean:.decimals$} ± {r_u:.decimals$}")
}

fn summarize(label: &str, samples_us: &[f64], ws_bytes: usize) {
    let n = samples_us.len();
    let mean = samples_us.iter().sum::<f64>() / n as f64;
    let var = samples_us
        .iter()
        .map(|&x| {
            let d = x - mean;
            d * d
        })
        .sum::<f64>()
        / (n - 1) as f64;
    let std_dev = var.sqrt();
    let u_a = std_dev / (n as f64).sqrt();
    let u_exp = 2.0 * u_a;

    let mut sorted = samples_us.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let pct = |q: f64| -> f64 { sorted[((n - 1) as f64 * q).round() as usize] };
    let p50 = pct(0.50);
    let p95 = pct(0.95);
    let p99 = pct(0.99);
    let max = sorted[n - 1];
    let deadline_us = (HOST_BLOCK as f64 / SAMPLE_RATE_HZ) * 1e6;
    let gum_str = format_gum(mean, u_exp);

    println!(
        "{label:<34} | n={n:<4} | {gum_str:<16} us (s={std_dev:5.2}, uA={u_a:4.2}) | p50={p50:6.2} | p95={p95:6.2} | p99={p99:6.2} ({:5.2}%) | max={max:6.2} us | ws={:>8.2} KiB",
        (p99 / deadline_us) * 100.0,
        ws_bytes as f64 / 1024.0
    );
}

fn bench_hybrid(
    label: &str,
    ir: &[f32],
    block_size: usize,
    spread: bool,
    backend: SimdBackend,
    fft: SimdFixedFftBackend,
) {
    let input = [0.25f32; HOST_BLOCK];
    let mut out = [0.0f32; HOST_BLOCK];
    let hcfg = HybridConfig {
        head_len: block_size,
        tail_block_size: block_size,
        max_ir_len: ir.len(),
    };
    let hreq = HybridConvolver::memory_requirements(hcfg, &fft).unwrap();
    let mut hws = vec![0.0f32; hreq.total_f32];
    let mut conv =
        HybridConvolver::prepare_with_tail_spreading(ir, hcfg, &mut hws, backend, fft, spread)
            .unwrap();
    for _ in 0..WARMUP_ITERS {
        conv.process_block(&input, &mut out).unwrap();
    }
    let mut times = Vec::with_capacity(MEAS_ITERS);
    for _ in 0..MEAS_ITERS {
        let t0 = Instant::now();
        conv.process_block(black_box(&input), black_box(&mut out))
            .unwrap();
        times.push(t0.elapsed().as_nanos() as f64 / 1000.0);
    }
    summarize(label, &times, hreq.total_bytes);
}

fn bench_nupc(
    label: &str,
    ir: &[f32],
    plan: NonUniformPlan,
    backend: SimdBackend,
    fft: SimdFixedFftBackend,
) {
    let input = [0.25f32; HOST_BLOCK];
    let mut out = [0.0f32; HOST_BLOCK];
    let nreq = NonUniformConvolver::memory_requirements(&plan, &fft).unwrap();
    let mut nws = vec![0.0f32; nreq.total_f32];
    let mut conv = NonUniformConvolver::prepare(ir, plan, &mut nws, backend, fft).unwrap();
    for _ in 0..WARMUP_ITERS {
        conv.process_block(&input, &mut out).unwrap();
    }
    let mut times = Vec::with_capacity(MEAS_ITERS);
    for _ in 0..MEAS_ITERS {
        let t0 = Instant::now();
        conv.process_block(black_box(&input), black_box(&mut out))
            .unwrap();
        times.push(t0.elapsed().as_nanos() as f64 / 1000.0);
    }
    assert_eq!(conv.deadline_misses(), 0);
    summarize(label, &times, nreq.total_bytes);
}

fn main() {
    let backend = SimdBackend::detect_or_baseline();
    let fft = SimdFixedFftBackend::new(backend);

    println!(
        "=== NUPC (Large FFTs + Optimizer + Time-Distributed FFT) vs Hybrid Benchmark (48 kHz, block={HOST_BLOCK}, budget=666.67 us, backend={}, warmup={WARMUP_ITERS}, n={MEAS_ITERS}, k=2) ===",
        backend.name()
    );
    println!("Provenance: HOST_ONLY (empirical timing; NOT a formal WCET proof).");

    for &ir_len in &[16_384usize, 65_536usize] {
        println!("\n--- IR length = {ir_len} taps ---");
        let ir = make_ir(ir_len);

        bench_hybrid("Hybrid(H128/T128) BURST", &ir, 128, false, backend, fft);
        bench_hybrid("Hybrid(H128/T128) SPREAD", &ir, 128, true, backend, fft);
        bench_hybrid("Hybrid(H256/T256) SPREAD", &ir, 256, true, backend, fft);
        bench_hybrid("Hybrid(H512/T512) SPREAD", &ir, 512, true, backend, fft);

        let default_plan = NonUniformPlan::default_profile(ir_len, HOST_BLOCK).unwrap();
        bench_nupc(
            "NUPC default (D32/32/64/128)",
            &ir,
            default_plan,
            backend,
            fft,
        );

        // Large-FFT Gardner-style NUPC schedule exploiting N512..=N16384
        let stages_16k = [
            NonUniformStageConfig {
                ir_offset: 0,
                ir_len: 32,
                block_size: 1,
            },
            NonUniformStageConfig {
                ir_offset: 32,
                ir_len: 96,
                block_size: 32,
            },
            NonUniformStageConfig {
                ir_offset: 128,
                ir_len: 384,
                block_size: 128,
            },
            NonUniformStageConfig {
                ir_offset: 512,
                ir_len: 1536,
                block_size: 512,
            },
            NonUniformStageConfig {
                ir_offset: 2048,
                ir_len: 6144,
                block_size: 2048,
            },
            NonUniformStageConfig {
                ir_offset: 8192,
                ir_len: ir_len - 8192,
                block_size: 4096,
            },
        ];
        let plan_4k = NonUniformPlan::validate(NonUniformConfig {
            max_ir_len: ir_len,
            host_block_max: HOST_BLOCK,
            stages: &stages_16k,
        })
        .unwrap();
        bench_nupc(
            "NUPC large (D32/32/128/512/2k/4k)",
            &ir,
            plan_4k,
            backend,
            fft,
        );

        if ir_len >= 65_536 {
            let stages_64k = [
                NonUniformStageConfig {
                    ir_offset: 0,
                    ir_len: 32,
                    block_size: 1,
                },
                NonUniformStageConfig {
                    ir_offset: 32,
                    ir_len: 96,
                    block_size: 32,
                },
                NonUniformStageConfig {
                    ir_offset: 128,
                    ir_len: 384,
                    block_size: 128,
                },
                NonUniformStageConfig {
                    ir_offset: 512,
                    ir_len: 1536,
                    block_size: 512,
                },
                NonUniformStageConfig {
                    ir_offset: 2048,
                    ir_len: 6144,
                    block_size: 2048,
                },
                NonUniformStageConfig {
                    ir_offset: 8192,
                    ir_len: ir_len - 8192,
                    block_size: 8192,
                },
            ];
            let plan_8k = NonUniformPlan::validate(NonUniformConfig {
                max_ir_len: ir_len,
                host_block_max: HOST_BLOCK,
                stages: &stages_64k,
            })
            .unwrap();
            bench_nupc(
                "NUPC large (D32/32/128/512/2k/8k)",
                &ir,
                plan_8k,
                backend,
                fft,
            );
        }

        let model = CostModel::for_backend_f32(backend);
        let mut sc = vec![OptimizerScratchCell::ZERO; optimizer_scratch_len(ir_len, 8192)];
        let (opt_s1, _) = NonUniformPlan::optimize(
            ir_len,
            HOST_BLOCK,
            &model,
            ClearancePolicy::Strategy1,
            &mut sc,
        )
        .unwrap();
        bench_nupc("NUPC optimized (Strategy1)", &ir, opt_s1, backend, fft);

        let (opt_s2, _) = NonUniformPlan::optimize(
            ir_len,
            HOST_BLOCK,
            &model,
            ClearancePolicy::Strategy2,
            &mut sc,
        )
        .unwrap();
        bench_nupc("NUPC optimized (Strategy2)", &ir, opt_s2, backend, fft);
    }
}
