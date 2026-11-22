//! Zero-latency hybrid FIR convolution with time-distributed FDL tail spreading.

use rtconv_simd::{HybridConfig, HybridConvolver, SimdBackend, SimdFixedFftBackend};

fn main() {
    let ir: Vec<f32> = (0..1024)
        .map(|k| (-0.008 * k as f32).exp() * (0.21 * k as f32).cos() * 0.25)
        .collect();

    let backend = SimdBackend::detect_or_baseline();
    let fft = SimdFixedFftBackend::new(backend);
    let cfg = HybridConfig {
        head_len: 128,
        tail_block_size: 128,
        max_ir_len: ir.len(),
    };
    let req = HybridConvolver::memory_requirements(cfg, &fft).expect("memory requirements");
    let mut workspace = vec![0.0f32; req.total_f32];

    let mut conv =
        HybridConvolver::prepare_with_tail_spreading(&ir, cfg, &mut workspace, backend, fft, true)
            .expect("prepare");

    let mut input = [0.0f32; 32];
    let mut output = [0.0f32; 32];
    input[0] = 1.0;
    conv.process_block(&input, &mut output).expect("block 0");
    println!(
        "HybridConvolver (latency={} smp, spread={}, mem={} B) first 4: {:?}",
        conv.latency_samples(),
        conv.tail_spreading(),
        req.total_bytes,
        &output[..4]
    );
}
