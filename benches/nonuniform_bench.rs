//! Comparative benchmark: Hybrid BURST vs Hybrid SPREAD vs NonUniformConvolver (NUPC).

use rtconv_simd::{
    HybridConfig, HybridConvolver, NonUniformConvolver, NonUniformPlan, SimdBackend,
    SimdFixedFftBackend,
};
use std::hint::black_box;
use std::time::Instant;

const SAMPLE_RATE_HZ: f64 = 48_000.0;
const HOST_BLOCK: usize = 32;

fn make_ir(len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| ((i as f32) * 0.17).sin() * (-0.0005 * i as f32).exp() * 0.25)
        .collect()
}

fn summarize(label: &str, samples_us: &mut [f64], ws_bytes: usize) {
    samples_us.sort_by(|a, b| a.total_cmp(b));
    let n = samples_us.len();
    let mean = samples_us.iter().sum::<f64>() / n as f64;
    let p95 = samples_us[((n - 1) as f64 * 0.95).round() as usize];
    let p99 = samples_us[((n - 1) as f64 * 0.99).round() as usize];
    let max = samples_us[n - 1];
    let deadline_us = (HOST_BLOCK as f64 / SAMPLE_RATE_HZ) * 1e6;
    println!(
        "{label:<30} | mean={mean:6.2} us | p95={p95:6.2} us | p99={p99:6.2} us ({:4.2}%) | max={max:6.2} us | ws={ws_bytes} B",
        (p99 / deadline_us) * 100.0
    );
}

fn main() {
    let backend = SimdBackend::detect_or_baseline();
    let fft = SimdFixedFftBackend::new(backend);
    let input = [0.25f32; HOST_BLOCK];
    let mut out = [0.0f32; HOST_BLOCK];
    let iters = 2000usize;

    println!(
        "=== WO-RTCONV-004B NUPC Benchmark (48 kHz, block={}, budget=666.67 us, backend={}) ===",
        HOST_BLOCK,
        backend.name()
    );

    for &ir_len in &[2048usize, 16_383] {
        println!("\n--- IR length = {ir_len} taps ---");
        let ir = make_ir(ir_len);

        // Hybrid BURST
        let hcfg = HybridConfig {
            head_len: 128,
            tail_block_size: 128,
            max_ir_len: ir_len,
        };
        let hreq = HybridConvolver::memory_requirements(hcfg, &fft).unwrap();
        let mut hws = vec![0.0f32; hreq.total_f32];
        let mut hb = HybridConvolver::prepare(&ir, hcfg, &mut hws, backend, fft).unwrap();
        for _ in 0..64 {
            hb.process_block(&input, &mut out).unwrap();
        }
        let mut times = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            hb.process_block(black_box(&input), black_box(&mut out))
                .unwrap();
            times.push(t0.elapsed().as_nanos() as f64 / 1000.0);
        }
        summarize("Hybrid(H128/T128) BURST", &mut times, hreq.total_bytes);

        // Hybrid SPREAD
        let mut hs =
            HybridConvolver::prepare_with_tail_spreading(&ir, hcfg, &mut hws, backend, fft, true)
                .unwrap();
        for _ in 0..64 {
            hs.process_block(&input, &mut out).unwrap();
        }
        times.clear();
        for _ in 0..iters {
            let t0 = Instant::now();
            hs.process_block(black_box(&input), black_box(&mut out))
                .unwrap();
            times.push(t0.elapsed().as_nanos() as f64 / 1000.0);
        }
        summarize("Hybrid(H128/T128) SPREAD", &mut times, hreq.total_bytes);

        // NonUniformConvolver (default 4-stage profile: D32 / B32 / B64 / B128)
        let plan = NonUniformPlan::default_profile(ir_len, HOST_BLOCK).unwrap();
        let nreq = NonUniformConvolver::memory_requirements(&plan, &fft).unwrap();
        let mut nws = vec![0.0f32; nreq.total_f32];
        let mut nu = NonUniformConvolver::prepare(&ir, plan, &mut nws, backend, fft).unwrap();
        for _ in 0..64 {
            nu.process_block(&input, &mut out).unwrap();
        }
        times.clear();
        for _ in 0..iters {
            let t0 = Instant::now();
            nu.process_block(black_box(&input), black_box(&mut out))
                .unwrap();
            times.push(t0.elapsed().as_nanos() as f64 / 1000.0);
        }
        summarize("NonUniform(D32/32/64/128)", &mut times, nreq.total_bytes);
    }
}
