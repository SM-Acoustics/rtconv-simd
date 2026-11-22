//! Pure `#![no_std]`-style static buffer usage example using `SimdFixedFftBackend`
//! and `HybridConvolver` with zero heap allocation.

use rtconv_simd::{HybridConfig, HybridConvolver, SimdBackend, SimdFixedFftBackend};

const MAX_IR: usize = 256;
const BLOCK: usize = 32;

fn main() {
    let ir = [0.5f32, 0.25, -0.125, 0.0625];
    let mut input = [0.0f32; BLOCK];
    input[0] = 1.0;
    let mut output = [0.0f32; BLOCK];

    let backend = SimdBackend::baseline();
    let fft = SimdFixedFftBackend::new(backend);
    let cfg = HybridConfig {
        head_len: 64,
        tail_block_size: 64,
        max_ir_len: MAX_IR,
    };
    let req = HybridConvolver::memory_requirements(cfg, &fft).expect("valid config");
    assert!(req.total_f32 <= 2048);

    let mut workspace = [0.0f32; 2048];
    let mut conv =
        HybridConvolver::prepare(&ir, cfg, &mut workspace[..req.total_f32], backend, fft)
            .expect("prepare");
    conv.set_tail_spreading(true).expect("tail spreading");
    conv.process_block(&input, &mut output).expect("process");

    assert!((output[0] - 0.5).abs() < 1e-6);
    assert!((output[1] - 0.25).abs() < 1e-6);
    assert!((output[2] + 0.125).abs() < 1e-6);
    assert!((output[3] - 0.0625).abs() < 1e-6);
}
