//! Real executable benchmark suite for `rtconv_simd`.
//!
//! Measures median, p95, p99, max observed, time/sample, throughput (MSamples/s),
//! SIMD speedup over scalar, workspace memory, algorithmic latency, and % of
//! the 666.67 us deadline (32 samples @ 48 kHz).
//!
//! Note: Measured percentiles and observed maxima are empirical host observations,
//! NOT formal WCET proofs.

use rtconv_simd::fft::{FftBackend, FftSize, MicroFftBackend};
use rtconv_simd::simd_kernels::{cmac_fdl_scalar, cmac_fdl_simd, dot_product_scalar, dot_product_simd};
use rtconv_simd::{
    DirectConfig, DirectConvolver, HybridConfig, HybridConvolver, PartitionedConfig,
    PartitionedConvolver, SimdBackend,
};
use std::hint::black_box;
use std::time::Instant;

const SAMPLE_RATE_HZ: f64 = 48_000.0;
const WARMUP: usize = 200;
const SAMPLES: usize = 2000;

struct Stats {
    median_us: f64,
    p95_us: f64,
    p99_us: f64,
    max_us: f64,
}

fn measure_us<F: FnMut()>(mut f: F, warmup: usize, iters: usize) -> Stats {
    for _ in 0..warmup {
        f();
    }
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        f();
        let dt = t0.elapsed().as_nanos() as f64 / 1000.0;
        times.push(dt);
    }
    times.sort_by(|a, b| a.total_cmp(b));
    let idx = |q: f64| -> usize { ((iters as f64 - 1.0) * q).round() as usize };
    Stats {
        median_us: times[idx(0.50)],
        p95_us: times[idx(0.95)],
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

fn main() {
    let best = SimdBackend::detect();
    let scalar = SimdBackend::scalar();
    println!("=== RTConv SIMD Benchmark Suite ===");
    println!("Detected best SIMD backend: {}", best.name());
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        println!(
            "Available x86 backends: SSE2={}, SSE4.2={}, AVX2={}, AVX512={}",
            SimdBackend::try_sse2().is_ok(),
            SimdBackend::try_sse4_2().is_ok(),
            SimdBackend::try_avx2().is_ok(),
            SimdBackend::try_avx512().is_ok(),
        );
    }
    println!();

    // 1. Microbenchmark: dot_product (Scalar vs SSE2 vs AVX2)
    println!("--- 1. Kernel Microbenchmark: dot_product (ns/call, 2000 iters, 64 reps batched) ---");
    println!(
        "{:<8} {:<12} {:<12} {:<12} {:<12}",
        "Taps", "Scalar(ns)", "SSE2(ns)", "AVX2(ns)", "AVX2/Scalar"
    );
    for &taps in &[32usize, 64, 128, 256, 512, 1024] {
        let a = make_signal(taps, 1);
        let b = make_signal(taps, 2);
        let reps = 64usize;

        let s_sc = measure_us(
            || {
                let mut acc = 0.0f32;
                for _ in 0..reps {
                    acc += dot_product_scalar(black_box(&a), black_box(&b));
                }
                black_box(acc);
            },
            WARMUP,
            SAMPLES,
        );
        let sc_ns = s_sc.median_us * 1000.0 / reps as f64;

        let sse2_ns = if let Ok(sse2) = SimdBackend::try_sse2() {
            let st = measure_us(
                || {
                    let mut acc = 0.0f32;
                    for _ in 0..reps {
                        acc += dot_product_simd(sse2.level(), black_box(&a), black_box(&b));
                    }
                    black_box(acc);
                },
                WARMUP,
                SAMPLES,
            );
            st.median_us * 1000.0 / reps as f64
        } else {
            f64::NAN
        };

        let avx2_ns = if let Ok(avx2) = SimdBackend::try_avx2() {
            let st = measure_us(
                || {
                    let mut acc = 0.0f32;
                    for _ in 0..reps {
                        acc += dot_product_simd(avx2.level(), black_box(&a), black_box(&b));
                    }
                    black_box(acc);
                },
                WARMUP,
                SAMPLES,
            );
            st.median_us * 1000.0 / reps as f64
        } else {
            f64::NAN
        };

        println!(
            "{:<8} {:<12.1} {:<12.1} {:<12.1} {:<12.2}x",
            taps,
            sc_ns,
            sse2_ns,
            avx2_ns,
            sc_ns / avx2_ns
        );
    }
    println!();

    // 2. Microbenchmark: CMAC FDL (Scalar vs SSE2 vs AVX2)
    println!("--- 2. Kernel Microbenchmark: CMAC FDL (us/call, 64 partitions) ---");
    println!(
        "{:<10} {:<12} {:<12} {:<12} {:<12}",
        "FFT(bins)", "Scalar(us)", "SSE2(us)", "AVX2(us)", "AVX2/Scalar"
    );
    for &size in &[FftSize::N64, FftSize::N128, FftSize::N256] {
        let bins = size.spectrum_bins();
        let parts = 64usize;
        let fdl_re = make_signal(parts * bins, 10);
        let fdl_im = make_signal(parts * bins, 11);
        let ir_re = make_signal(parts * bins, 12);
        let ir_im = make_signal(parts * bins, 13);
        let mut acc_re = vec![0.0f32; bins];
        let mut acc_im = vec![0.0f32; bins];

        let st_sc = measure_us(
            || {
                acc_re.fill(0.0);
                acc_im.fill(0.0);
                cmac_fdl_scalar(
                    black_box(&mut acc_re),
                    black_box(&mut acc_im),
                    &fdl_re,
                    &fdl_im,
                    &ir_re,
                    &ir_im,
                    parts,
                    bins,
                    0,
                );
            },
            WARMUP,
            SAMPLES,
        );

        let sse2_us = if let Ok(sse2) = SimdBackend::try_sse2() {
            measure_us(
                || {
                    acc_re.fill(0.0);
                    acc_im.fill(0.0);
                    cmac_fdl_simd(
                        sse2.level(),
                        black_box(&mut acc_re),
                        black_box(&mut acc_im),
                        &fdl_re,
                        &fdl_im,
                        &ir_re,
                        &ir_im,
                        parts,
                        bins,
                        0,
                    );
                },
                WARMUP,
                SAMPLES,
            )
            .median_us
        } else {
            f64::NAN
        };

        let avx2_us = if let Ok(avx2) = SimdBackend::try_avx2() {
            measure_us(
                || {
                    acc_re.fill(0.0);
                    acc_im.fill(0.0);
                    cmac_fdl_simd(
                        avx2.level(),
                        black_box(&mut acc_re),
                        black_box(&mut acc_im),
                        &fdl_re,
                        &fdl_im,
                        &ir_re,
                        &ir_im,
                        parts,
                        bins,
                        0,
                    );
                },
                WARMUP,
                SAMPLES,
            )
            .median_us
        } else {
            f64::NAN
        };

        println!(
            "{:<10} {:<12.2} {:<12.2} {:<12.2} {:<12.2}x",
            format!("{}({})", size.len(), bins),
            st_sc.median_us,
            sse2_us,
            avx2_us,
            st_sc.median_us / avx2_us
        );
    }
    println!();

    // 3. FFT 64 vs 128 vs 256 full path (RFFT -> CMAC -> IFFT)
    println!("--- 3. FFT Comparison: RFFT -> CMAC(1 part) -> IFFT (microfft 0.6.0, scalar FFT + AVX2 CMAC) ---");
    println!(
        "{:<8} {:<10} {:<10} {:<10} {:<10} {:<14} {:<12}",
        "FFT N", "Med(us)", "p95(us)", "p99(us)", "Max(us)", "MaxAbsErr", "SNR(dB)"
    );
    let fft = MicroFftBackend::new();
    for &size in &[FftSize::N64, FftSize::N128, FftSize::N256] {
        let n = size.len();
        let bins = size.spectrum_bins();
        let x = make_signal(n, 100 + n as u64);
        let mut h = vec![0.0f32; n];
        h[0] = 1.0;
        let mut x_re = vec![0.0f32; bins];
        let mut x_im = vec![0.0f32; bins];
        let mut h_re = vec![0.0f32; bins];
        let mut h_im = vec![0.0f32; bins];
        let mut acc_re = vec![0.0f32; bins];
        let mut acc_im = vec![0.0f32; bins];
        let mut out = vec![0.0f32; n];
        fft.rfft_forward(size, &h, &mut h_re, &mut h_im, &mut []).unwrap();

        let st = measure_us(
            || {
                fft.rfft_forward(size, black_box(&x), &mut x_re, &mut x_im, &mut [])
                    .unwrap();
                acc_re.fill(0.0);
                acc_im.fill(0.0);
                cmac_fdl_simd(
                    best.level(),
                    &mut acc_re,
                    &mut acc_im,
                    &x_re,
                    &x_im,
                    &h_re,
                    &h_im,
                    1,
                    bins,
                    0,
                );
                fft.rfft_inverse(size, &acc_re, &acc_im, black_box(&mut out), &mut [])
                    .unwrap();
            },
            WARMUP,
            SAMPLES,
        );

        let mut max_err = 0.0f64;
        let mut err_sq = 0.0f64;
        let mut sig_sq = 0.0f64;
        for (&r, &a) in x.iter().zip(out.iter()) {
            let d = (r as f64 - a as f64).abs();
            if d > max_err {
                max_err = d;
            }
            err_sq += d * d;
            sig_sq += (r as f64) * (r as f64);
        }
        let snr = 10.0 * (sig_sq / err_sq).log10();
        println!(
            "{:<8} {:<10.2} {:<10.2} {:<10.2} {:<10.2} {:<14.3e} {:<12.2}",
            n, st.median_us, st.p95_us, st.p99_us, st.max_us, max_err, snr
        );
    }
    println!();

    // 4. Full Engine Benchmarks @ 48 kHz
    println!("--- 4. Full Engine Benchmarks (48 kHz) ---");
    println!(
        "{:<24} {:<6} {:<5} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8} {:<9} {:<9} {:<8}",
        "Engine/Config",
        "IR",
        "Blk",
        "Med(us)",
        "p95(us)",
        "p99(us)",
        "Max(us)",
        "ns/smp",
        "MSmp/s",
        "Speedup",
        "Deadline%",
        "Mem(KiB)"
    );

    // DirectConvolver (32-sample blocks)
    for &ir_len in &[64usize, 256, 1024] {
        let ir = make_signal(ir_len, 500 + ir_len as u64);
        let input = make_signal(32, 600);
        let mut out = vec![0.0f32; 32];
        let cfg = DirectConfig { max_ir_len: ir_len };
        let req = DirectConvolver::memory_requirements(cfg).unwrap();

        let mut ws_sc = vec![0.0f32; req.total_f32];
        let mut conv_sc = DirectConvolver::prepare(&ir, cfg, &mut ws_sc, scalar).unwrap();
        let st_sc = measure_us(
            || {
                conv_sc.process_block(black_box(&input), black_box(&mut out)).unwrap();
            },
            WARMUP,
            SAMPLES,
        );

        let mut ws_sm = vec![0.0f32; req.total_f32];
        let mut conv_sm = DirectConvolver::prepare(&ir, cfg, &mut ws_sm, best).unwrap();
        let st_sm = measure_us(
            || {
                conv_sm.process_block(black_box(&input), black_box(&mut out)).unwrap();
            },
            WARMUP,
            SAMPLES,
        );

        let deadline_us = (32.0 / SAMPLE_RATE_HZ) * 1e6;
        let ns_per_smp = st_sm.median_us * 1000.0 / 32.0;
        let msmp_s = 32.0 / st_sm.median_us;
        println!(
            "{:<24} {:<6} {:<5} {:<8.2} {:<8.2} {:<8.2} {:<8.2} {:<8.1} {:<8.2} {:<8.2}x {:<8.2}% {:<8.2}",
            "Direct(SIMD)",
            ir_len,
            32,
            st_sm.median_us,
            st_sm.p95_us,
            st_sm.p99_us,
            st_sm.max_us,
            ns_per_smp,
            msmp_s,
            st_sc.median_us / st_sm.median_us,
            (st_sm.median_us / deadline_us) * 100.0,
            req.total_bytes as f64 / 1024.0
        );
    }

    // PartitionedConvolver (block 32, 64, 128 across short/medium/long FIRs)
    for &(ir_len, block_size) in &[
        (256usize, 32usize),
        (2048, 32),
        (2048, 64),
        (2048, 128),
        (4096, 64),
        (16383, 64),
        (16383, 128),
    ] {
        let ir = make_signal(ir_len, 700 + ir_len as u64);
        let input = make_signal(block_size, 800);
        let mut out = vec![0.0f32; block_size];
        let cfg = PartitionedConfig {
            block_size,
            max_ir_len: ir_len,
        };
        let req = PartitionedConvolver::memory_requirements(cfg, &fft).unwrap();

        let mut ws_sc = vec![0.0f32; req.total_f32];
        let mut conv_sc =
            PartitionedConvolver::prepare(&ir, cfg, &mut ws_sc, scalar, fft).unwrap();
        let st_sc = measure_us(
            || {
                conv_sc.process_block(black_box(&input), black_box(&mut out)).unwrap();
            },
            WARMUP,
            SAMPLES,
        );

        let mut ws_sm = vec![0.0f32; req.total_f32];
        let mut conv_sm = PartitionedConvolver::prepare(&ir, cfg, &mut ws_sm, best, fft).unwrap();
        let st_sm = measure_us(
            || {
                conv_sm.process_block(black_box(&input), black_box(&mut out)).unwrap();
            },
            WARMUP,
            SAMPLES,
        );

        let deadline_us = (block_size as f64 / SAMPLE_RATE_HZ) * 1e6;
        let ns_per_smp = st_sm.median_us * 1000.0 / block_size as f64;
        let msmp_s = block_size as f64 / st_sm.median_us;
        println!(
            "{:<24} {:<6} {:<5} {:<8.2} {:<8.2} {:<8.2} {:<8.2} {:<8.1} {:<8.2} {:<8.2}x {:<8.2}% {:<8.2}",
            format!("Part(FFT{})", block_size * 2),
            ir_len,
            block_size,
            st_sm.median_us,
            st_sm.p95_us,
            st_sm.p99_us,
            st_sm.max_us,
            ns_per_smp,
            msmp_s,
            st_sc.median_us / st_sm.median_us,
            (st_sm.median_us / deadline_us) * 100.0,
            req.total_bytes as f64 / 1024.0
        );
    }

    // HybridConvolver (host block 32, tail block 64 and 128)
    for &(ir_len, tail_b) in &[(2048usize, 64usize), (2048, 128), (16383, 128)] {
        let ir = make_signal(ir_len, 900 + ir_len as u64);
        let input = make_signal(32, 950);
        let mut out = vec![0.0f32; 32];
        let cfg = HybridConfig {
            head_len: tail_b,
            tail_block_size: tail_b,
            max_ir_len: ir_len,
        };
        let req = HybridConvolver::memory_requirements(cfg, &fft).unwrap();

        let mut ws_sc = vec![0.0f32; req.total_f32];
        let mut conv_sc = HybridConvolver::prepare(&ir, cfg, &mut ws_sc, scalar, fft).unwrap();
        let st_sc = measure_us(
            || {
                conv_sc.process_block(black_box(&input), black_box(&mut out)).unwrap();
            },
            WARMUP,
            SAMPLES,
        );

        let mut ws_sm = vec![0.0f32; req.total_f32];
        let mut conv_sm = HybridConvolver::prepare(&ir, cfg, &mut ws_sm, best, fft).unwrap();
        let st_sm = measure_us(
            || {
                conv_sm.process_block(black_box(&input), black_box(&mut out)).unwrap();
            },
            WARMUP,
            SAMPLES,
        );

        let deadline_us = (32.0 / SAMPLE_RATE_HZ) * 1e6;
        let ns_per_smp = st_sm.median_us * 1000.0 / 32.0;
        let msmp_s = 32.0 / st_sm.median_us;
        println!(
            "{:<24} {:<6} {:<5} {:<8.2} {:<8.2} {:<8.2} {:<8.2} {:<8.1} {:<8.2} {:<8.2}x {:<8.2}% {:<8.2}",
            format!("Hybrid(H{}/T{})", tail_b, tail_b),
            ir_len,
            32,
            st_sm.median_us,
            st_sm.p95_us,
            st_sm.p99_us,
            st_sm.max_us,
            ns_per_smp,
            msmp_s,
            st_sc.median_us / st_sm.median_us,
            (st_sm.p99_us / deadline_us) * 100.0,
            req.total_bytes as f64 / 1024.0
        );
    }
}
