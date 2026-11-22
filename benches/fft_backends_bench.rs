//! WO-RTCONV-002 Benchmark Suite:
//! - Separated stage breakdown: RFFT, IFFT, CMAC, and full FDL
//! - Backend comparison: MicroFftBackend vs SimdFixedFft (Scalar, SSE2, AVX2) vs PhastFftBackend
//! - Phase-by-phase (0, 1, 2, 3) measurement of `Hybrid(H128/T128)` @ 32-sample host blocks
//!   BEFORE (burst) vs AFTER (`tail_spreading = true`).
//!
//! Empirical host measurements only — NOT formal WCET proofs.

use rtconv_simd::fft::{FftBackend, FftSize, MicroFftBackend};
use rtconv_simd::{
    HybridConfig, HybridConvolver, PartitionedConfig, PartitionedConvolver, SimdBackend,
    SimdFixedFftBackend,
};
use std::hint::black_box;
use std::time::Instant;

#[cfg(feature = "phastft")]
use rtconv_simd::PhastFftBackend;

const SAMPLE_RATE_HZ: f64 = 48_000.0;
const WARMUP: usize = 200;
const ITERS: usize = 2000;

struct Stats {
    med_us: f64,
    p99_us: f64,
    max_us: f64,
}

fn measure<F: FnMut()>(mut f: F, warmup: usize, iters: usize) -> Stats {
    for _ in 0..warmup {
        f();
    }
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        f();
        times.push(t0.elapsed().as_nanos() as f64 / 1000.0);
    }
    times.sort_by(|a, b| a.total_cmp(b));
    let idx = |q: f64| -> usize { ((iters as f64 - 1.0) * q).round() as usize };
    Stats {
        med_us: times[idx(0.50)],
        p99_us: times[idx(0.99)],
        max_us: times[iters - 1],
    }
}

fn make_signal(len: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..len)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / ((1u32 << 24) as f32)) * 2.0 - 1.0
        })
        .collect()
}

fn bench_fft_stages<F: FftBackend>(label: &str, fft: &F, size: FftSize, micro_rt_ns: Option<f64>) -> f64 {
    let n = size.len();
    let bins = size.spectrum_bins();
    let x = make_signal(n, 42 + n as u64);
    let mut re = vec![0.0f32; bins];
    let mut im = vec![0.0f32; bins];
    let mut out = vec![0.0f32; n];
    let mut scratch = vec![0.0f32; fft.scratch_len_f32(size)];

    let reps = 32usize;
    let st_fwd = measure(
        || {
            for _ in 0..reps {
                fft.rfft_forward(size, black_box(&x), &mut re, &mut im, &mut scratch)
                    .unwrap();
            }
            black_box(&re);
        },
        WARMUP,
        ITERS,
    );
    let fwd_ns = st_fwd.med_us * 1000.0 / reps as f64;

    let st_inv = measure(
        || {
            for _ in 0..reps {
                fft.rfft_inverse(size, black_box(&re), black_box(&im), &mut out, &mut scratch)
                    .unwrap();
            }
            black_box(&out);
        },
        WARMUP,
        ITERS,
    );
    let inv_ns = st_inv.med_us * 1000.0 / reps as f64;
    let total_ns = fwd_ns + inv_ns;
    let speedup = match micro_rt_ns {
        Some(base) => format!("{:.2}x", base / total_ns),
        None => "1.00x".to_string(),
    };

    println!(
        "{:<24} {:<6} {:<10.1} {:<10.1} {:<10.1} {:<10} {:<10}",
        label,
        n,
        fwd_ns,
        inv_ns,
        total_ns,
        speedup,
        fft.scratch_len_f32(size) * 4
    );
    total_ns
}

fn bench_partitioned<F: FftBackend + Clone>(
    label: &str,
    fft: F,
    cmac_backend: SimdBackend,
    ir_len: usize,
    block_size: usize,
) {
    let ir = make_signal(ir_len, 700 + ir_len as u64);
    let input = make_signal(block_size, 800);
    let mut out = vec![0.0f32; block_size];
    let cfg = PartitionedConfig {
        block_size,
        max_ir_len: ir_len,
    };
    let req = PartitionedConvolver::memory_requirements(cfg, &fft).unwrap();
    let mut ws = vec![0.0f32; req.total_f32];
    let mut conv = PartitionedConvolver::prepare(&ir, cfg, &mut ws, cmac_backend, fft).unwrap();

    let st = measure(
        || {
            conv.process_block(black_box(&input), black_box(&mut out)).unwrap();
        },
        WARMUP,
        ITERS,
    );
    let deadline_us = (block_size as f64 / SAMPLE_RATE_HZ) * 1e6;
    println!(
        "{:<24} {:<6} {:<5} {:<8.2} {:<8.2} {:<8.2} {:<9.2}% {:<8.2}",
        label,
        ir_len,
        block_size,
        st.med_us,
        st.p99_us,
        st.max_us,
        (st.p99_us / deadline_us) * 100.0,
        req.total_bytes as f64 / 1024.0
    );
}

fn bench_hybrid_4phases<F: FftBackend + Clone>(
    label: &str,
    fft: F,
    cmac_backend: SimdBackend,
    ir_len: usize,
    spread: bool,
) {
    let ir = make_signal(ir_len, 900 + ir_len as u64);
    let input = make_signal(32, 950);
    let mut out = vec![0.0f32; 32];
    let cfg = HybridConfig {
        head_len: 128,
        tail_block_size: 128,
        max_ir_len: ir_len,
    };
    let req = HybridConvolver::memory_requirements(cfg, &fft).unwrap();
    let mut ws = vec![0.0f32; req.total_f32];
    let mut conv = HybridConvolver::prepare(&ir, cfg, &mut ws, cmac_backend, fft).unwrap();
    conv.set_tail_spreading(spread);

    // Warmup 100 full 4-phase cycles
    for _ in 0..400 {
        conv.process_block(&input, &mut out).unwrap();
    }

    let cycles = 1000usize;
    let mut phases: [Vec<f64>; 4] = [
        Vec::with_capacity(cycles),
        Vec::with_capacity(cycles),
        Vec::with_capacity(cycles),
        Vec::with_capacity(cycles),
    ];

    for _ in 0..cycles {
        for p in 0..4 {
            let t0 = Instant::now();
            conv.process_block(black_box(&input), black_box(&mut out)).unwrap();
            phases[p].push(t0.elapsed().as_nanos() as f64 / 1000.0);
        }
    }

    let deadline_us = (32.0 / SAMPLE_RATE_HZ) * 1e6;
    let mut meds = [0.0f64; 4];
    let mut p99s = [0.0f64; 4];
    let mut maxs = [0.0f64; 4];
    for p in 0..4 {
        phases[p].sort_by(|a, b| a.total_cmp(b));
        meds[p] = phases[p][cycles / 2];
        p99s[p] = phases[p][((cycles as f64 - 1.0) * 0.99).round() as usize];
        maxs[p] = phases[p][cycles - 1];
    }
    let worst_p99 = p99s.iter().copied().fold(0.0f64, f64::max);
    println!(
        "{:<24} {:<6} {:<6} {:<7.2} {:<7.2} {:<7.2} {:<7.2} | {:<7.2} {:<7.2} {:<7.2} {:<7.2} | {:<6.2}%",
        label,
        ir_len,
        if spread { "SPREAD" } else { "BURST" },
        meds[0],
        meds[1],
        meds[2],
        meds[3],
        p99s[0],
        p99s[1],
        p99s[2],
        p99s[3],
        (worst_p99 / deadline_us) * 100.0
    );
}

fn main() {
    let best = SimdBackend::detect();
    let sse2 = SimdBackend::try_sse2().unwrap_or(SimdBackend::baseline());
    let avx2 = SimdBackend::try_avx2().unwrap_or(best);

    println!("=== WO-RTCONV-002: SIMD FFT Backends & Tail-Spreading Benchmarks ===");
    println!("Host best SIMD backend: {}", best.name());
    println!();

    // 1. Separated RFFT / IFFT Stage Breakdown (N = 64, 128, 256)
    println!("--- 1. Separated FFT Stage Costs: RFFT vs IFFT (ns/call, 32 reps x 2000 iters) ---");
    println!(
        "{:<24} {:<6} {:<10} {:<10} {:<10} {:<10} {:<10}",
        "Backend", "N", "RFFT(ns)", "IFFT(ns)", "Total(ns)", "vs Micro", "Scratch(B)"
    );
    let micro = MicroFftBackend::new();
    let fixed_sc = SimdFixedFftBackend::new(SimdBackend::scalar());
    let fixed_sse2 = SimdFixedFftBackend::new(sse2);
    let fixed_avx2 = SimdFixedFftBackend::new(avx2);
    #[cfg(feature = "phastft")]
    let phast = PhastFftBackend::new();

    for &size in &[FftSize::N64, FftSize::N128, FftSize::N256] {
        let base = bench_fft_stages("MicroFftBackend", &micro, size, None);
        bench_fft_stages("SimdFixedFft(Scalar)", &fixed_sc, size, Some(base));
        bench_fft_stages("SimdFixedFft(SSE2)", &fixed_sse2, size, Some(base));
        bench_fft_stages("SimdFixedFft(AVX2)", &fixed_avx2, size, Some(base));
        #[cfg(feature = "phastft")]
        bench_fft_stages("PhastFftBackend", &phast, size, Some(base));
        println!();
    }

    // 2. Full PartitionedConvolver Comparison (IR = 256, 2048, 16383; B = 32, 64, 128)
    println!("--- 2. Full PartitionedConvolver FDL Comparison (48 kHz) ---");
    println!(
        "{:<24} {:<6} {:<5} {:<8} {:<8} {:<8} {:<10} {:<8}",
        "FFT Backend", "IR", "Blk", "Med(us)", "p99(us)", "Max(us)", "p99/Dead%", "Mem(KiB)"
    );
    for &(ir_len, b) in &[
        (256usize, 32usize),
        (2048, 32),
        (2048, 64),
        (2048, 128),
        (16383, 64),
        (16383, 128),
    ] {
        bench_partitioned("MicroFftBackend", micro, best, ir_len, b);
        bench_partitioned("SimdFixedFft(SSE2)", fixed_sse2, best, ir_len, b);
        bench_partitioned("SimdFixedFft(AVX2)", fixed_avx2, best, ir_len, b);
        #[cfg(feature = "phastft")]
        bench_partitioned("PhastFftBackend", phast.clone(), best, ir_len, b);
        println!();
    }

    // 3. Fase E: Hybrid(H128/T128) @ B_host = 32 — 4-Phase Callback Breakdown (BURST vs SPREAD)
    println!("--- 3. Fase E: Hybrid(H128/T128) @ Blk=32 (4-Callback Cycle: Med P0..P3 | p99 P0..P3 | Worst p99/666.67us) ---");
    println!(
        "{:<24} {:<6} {:<6} {:<7} {:<7} {:<7} {:<7} | {:<7} {:<7} {:<7} {:<7} | {:<7}",
        "FFT Backend", "IR", "Mode", "MedP0", "MedP1", "MedP2", "MedP3", "p99P0", "p99P1", "p99P2", "p99P3", "Peak%"
    );
    for &ir_len in &[2048usize, 16383] {
        bench_hybrid_4phases("MicroFftBackend", micro, best, ir_len, false);
        bench_hybrid_4phases("MicroFftBackend", micro, best, ir_len, true);
        bench_hybrid_4phases("SimdFixedFft(SSE2)", fixed_sse2, best, ir_len, false);
        bench_hybrid_4phases("SimdFixedFft(SSE2)", fixed_sse2, best, ir_len, true);
        bench_hybrid_4phases("SimdFixedFft(AVX2)", fixed_avx2, best, ir_len, false);
        bench_hybrid_4phases("SimdFixedFft(AVX2)", fixed_avx2, best, ir_len, true);
        #[cfg(feature = "phastft")]
        {
            bench_hybrid_4phases("PhastFftBackend", phast.clone(), best, ir_len, false);
            bench_hybrid_4phases("PhastFftBackend", phast.clone(), best, ir_len, true);
        }
        println!();
    }
}
