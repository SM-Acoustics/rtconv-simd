//! Direct time-domain FIR convolution example (`DirectConvolver`).

use rtconv_simd::{DirectConfig, DirectConvolver, SimdBackend};

fn main() {
    let ir = [0.5f32, 0.25, -0.125, 0.0625];
    let cfg = DirectConfig {
        max_ir_len: ir.len(),
    };
    let req = DirectConvolver::memory_requirements(cfg).expect("valid config");
    let mut workspace = vec![0.0f32; req.total_f32];

    let backend = SimdBackend::detect_or_baseline();
    let mut conv = DirectConvolver::prepare(&ir, cfg, &mut workspace, backend).expect("prepare");

    let mut input = [0.0f32; 32];
    input[0] = 1.0;
    let mut output = [0.0f32; 32];
    conv.process_block(&input, &mut output).expect("process");

    println!(
        "DirectConvolver ({}) first 6: {:?}",
        backend.name(),
        &output[..6]
    );
}
