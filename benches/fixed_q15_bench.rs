//! Benchmark comparing FixedQ15DirectConvolver vs DirectConvolver (f32 scalar & SIMD).

use rtconv_simd::fixed::f32_to_q15;
use rtconv_simd::{
    DirectConfig, DirectConvolver, FixedQ15DirectConfig, FixedQ15DirectConvolver, SimdBackend,
};
use std::hint::black_box;
use std::time::Instant;

const BLOCK: usize = 32;
const SAMPLE_RATE_HZ: f64 = 48_000.0;

fn main() {
    let deadline_us = (BLOCK as f64 / SAMPLE_RATE_HZ) * 1e6;
    let simd = SimdBackend::detect_or_baseline();
    let scalar = SimdBackend::scalar();
    let iters = 4000usize;

    println!(
        "=== WO-RTCONV-004C Q15 Direct vs f32 Direct (48 kHz, block={BLOCK}, budget={deadline_us:.2} us) ==="
    );
    for &taps in &[32usize, 64, 128, 256, 512] {
        let ir_f32: Vec<f32> = (0..taps)
            .map(|k| ((k as f32) * 0.13).sin() * 0.01)
            .collect();
        let ir_q15: Vec<i16> = ir_f32.iter().map(|&v| f32_to_q15(v)).collect();
        let in_f32 = [0.25f32; BLOCK];
        let mut out_f32 = [0.0f32; BLOCK];
        let in_q15 = [f32_to_q15(0.25); BLOCK];
        let mut out_q15 = [0i16; BLOCK];

        // f32 scalar
        let dcfg = DirectConfig { max_ir_len: taps };
        let dreq = DirectConvolver::memory_requirements(dcfg).unwrap();
        let mut dws = vec![0.0f32; dreq.total_f32];
        let mut ds = DirectConvolver::prepare(&ir_f32, dcfg, &mut dws, scalar).unwrap();
        let t0 = Instant::now();
        for _ in 0..iters {
            ds.process_block(black_box(&in_f32), black_box(&mut out_f32))
                .unwrap();
        }
        let f32_scalar_us = t0.elapsed().as_nanos() as f64 / (iters as f64 * 1000.0);

        // f32 SIMD
        let mut dv = DirectConvolver::prepare(&ir_f32, dcfg, &mut dws, simd).unwrap();
        let t0 = Instant::now();
        for _ in 0..iters {
            dv.process_block(black_box(&in_f32), black_box(&mut out_f32))
                .unwrap();
        }
        let f32_simd_us = t0.elapsed().as_nanos() as f64 / (iters as f64 * 1000.0);

        // Q15 direct
        let qcfg = FixedQ15DirectConfig { max_ir_len: taps };
        let qreq = FixedQ15DirectConvolver::memory_requirements(qcfg).unwrap();
        let mut qws = vec![0i16; qreq.total_i16];
        let mut qc = FixedQ15DirectConvolver::prepare(&ir_q15, qcfg, &mut qws).unwrap();
        let t0 = Instant::now();
        for _ in 0..iters {
            qc.process_block(black_box(&in_q15), black_box(&mut out_q15))
                .unwrap();
        }
        let q15_us = t0.elapsed().as_nanos() as f64 / (iters as f64 * 1000.0);

        println!(
            "taps={taps:>3} | Q15={q15_us:6.2} us ({:4.2}%, ws={:>4} B) | f32-scalar={f32_scalar_us:6.2} us | f32-simd={f32_simd_us:6.2} us (ws={:>4} B)",
            (q15_us / deadline_us) * 100.0,
            qreq.total_bytes,
            dreq.total_bytes
        );
    }
}
