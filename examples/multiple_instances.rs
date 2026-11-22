//! Multi-channel processing via independent mono `HybridConvolver` instances (stereo).

use rtconv_simd::{HybridConfig, HybridConvolver, SimdBackend, SimdFixedFftBackend};

fn main() {
    let ir_left = [0.8f32, -0.2, 0.1, 0.05];
    let ir_right = [-0.6f32, 0.3, -0.15, 0.075];

    let backend = SimdBackend::detect_or_baseline();
    let fft = SimdFixedFftBackend::new(backend);
    let cfg = HybridConfig {
        head_len: 64,
        tail_block_size: 64,
        max_ir_len: 128,
    };
    let req = HybridConvolver::memory_requirements(cfg, &fft).expect("requirements");

    let mut ws_left = vec![0.0f32; req.total_f32];
    let mut ws_right = vec![0.0f32; req.total_f32];
    let mut conv_l = HybridConvolver::prepare(&ir_left, cfg, &mut ws_left, backend, fft).unwrap();
    let mut conv_r = HybridConvolver::prepare(&ir_right, cfg, &mut ws_right, backend, fft).unwrap();

    let mut in_l = [0.0f32; 32];
    let mut in_r = [0.0f32; 32];
    in_l[0] = 1.0;
    in_r[0] = 1.0;

    let mut out_l = [0.0f32; 32];
    let mut out_r = [0.0f32; 32];
    conv_l.process_block(&in_l, &mut out_l).unwrap();
    conv_r.process_block(&in_r, &mut out_r).unwrap();

    println!("Left  first 4: {:?}", &out_l[..4]);
    println!("Right first 4: {:?}", &out_r[..4]);
}
