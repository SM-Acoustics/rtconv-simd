//! Subnormal (IEEE 754 denormal) robustness benchmark.
//!
//! Times `DirectConvolver` and `HybridConvolver` per 32-sample block for normal inputs, tiny
//! normal inputs whose products with the taps would be subnormal, subnormal inputs, and an IR
//! whose deep tail decodes to subnormal `f32` taps. With `DENORMAL_GUARD` every case should cost
//! the same as the normal case (no microcode-assist cliff on x86 without FTZ/DAZ).
//!
//! `cargo bench --bench denormal_bench`

use rtconv_simd::{
    DirectConfig, DirectConvolver, HybridConfig, HybridConvolver, SimdBackend, SimdFixedFftBackend,
};
use std::hint::black_box;
use std::time::Instant;

fn time_us<F: FnMut()>(mut f: F) -> f64 {
    for _ in 0..500 {
        f();
    }
    let iters = 8000;
    let t = Instant::now();
    for _ in 0..iters {
        f();
    }
    t.elapsed().as_secs_f64() * 1e6 / iters as f64
}

fn run(label: &str, ir: &[f32], level: f32) {
    let backend = SimdBackend::detect_or_baseline();
    let taps = ir.len();
    let fft = SimdFixedFftBackend::new(backend);
    let cfg = HybridConfig {
        head_len: 128,
        tail_block_size: 128,
        max_ir_len: taps,
    };
    let req = HybridConvolver::memory_requirements(cfg, &fft).unwrap();
    let mut ws = vec![0.0f32; req.total_f32];
    let mut hybrid =
        HybridConvolver::prepare_with_tail_spreading(ir, cfg, &mut ws, backend, fft, true).unwrap();
    let mut dws = vec![0.0f32; 2 * taps];
    let mut direct =
        DirectConvolver::prepare(ir, DirectConfig { max_ir_len: taps }, &mut dws, backend).unwrap();
    let input: Vec<f32> = (0..32)
        .map(|i| level * if i % 2 == 0 { 1.0 } else { -0.7 })
        .collect();
    let mut out = [0.0f32; 32];
    let th = time_us(|| {
        hybrid.process_block(black_box(&input), &mut out).unwrap();
        black_box(&out);
    });
    let td = time_us(|| {
        direct.process_block(black_box(&input), &mut out).unwrap();
        black_box(&out);
    });
    println!("{label:<44} | hybrid(1024) = {th:7.2} us/block | direct(1024) = {td:7.2} us/block");
}

fn main() {
    let taps = 1024;
    let normal: Vec<f32> = (0..taps)
        .map(|k| 0.5 * (-(k as f32) / 150.0).exp() * ((k as f32) * 0.3).sin())
        .collect();
    // f64 design whose tail decays to about -900 dB, decoded to f32: ~150 subnormal taps.
    let deep: Vec<f32> = (0..taps)
        .map(|k| (0.5 * (-(k as f64) / 9.0).exp() * ((k as f64) * 0.3).sin()) as f32)
        .collect();
    println!("backend: {}", SimdBackend::detect_or_baseline().name());
    run("normal IR, input 0.1", &normal, 0.1);
    run(
        "normal IR, input 1e-35 (subnormal products)",
        &normal,
        1e-35,
    );
    run(
        "normal IR, input 1e-37 (subnormal products)",
        &normal,
        1e-37,
    );
    run("normal IR, input 1e-39 (subnormal input)", &normal, 1e-39);
    run("deep-tail IR (subnormal taps), input 0.1", &deep, 0.1);
}
