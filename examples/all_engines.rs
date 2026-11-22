//! Example demonstrating `DirectConvolver`, `PartitionedConvolver`, and `HybridConvolver`
//! using purely static/stack buffers (zero heap allocation, `no_std`-compatible pattern).

use rtconv_simd::{
    DirectConfig, DirectConvolver, HybridConfig, HybridConvolver, MicroFftBackend,
    PartitionedConfig, PartitionedConvolver, SimdBackend,
};

fn main() {
    let ir = [0.5f32, 0.25, -0.125, 0.0625];
    let mut input = [0.0f32; 64];
    input[0] = 1.0;

    // 1. DirectConvolver with static workspace
    let mut direct_ws = [0.0f32; 16];
    let mut direct = DirectConvolver::prepare(
        &ir,
        DirectConfig { max_ir_len: 8 },
        &mut direct_ws,
        SimdBackend::baseline(),
    )
    .expect("direct prepare");
    let mut out_direct = [0.0f32; 64];
    direct.process_block(&input, &mut out_direct).unwrap();

    // 2. PartitionedConvolver with static workspace (B=32, FFT=64)
    let fft = MicroFftBackend::new();
    let p_cfg = PartitionedConfig {
        block_size: 32,
        max_ir_len: 64,
    };
    let mut part_ws = [0.0f32; 512];
    let mut part =
        PartitionedConvolver::prepare(&ir, p_cfg, &mut part_ws, SimdBackend::baseline(), fft)
            .expect("partitioned prepare");
    let mut out_part = [0.0f32; 64];
    part.process_block(&input[..32], &mut out_part[..32])
        .unwrap();
    part.process_block(&input[32..], &mut out_part[32..])
        .unwrap();

    // 3. HybridConvolver with static workspace (head=64, tail=64, host block=32)
    let h_cfg = HybridConfig {
        head_len: 64,
        tail_block_size: 64,
        max_ir_len: 128,
    };
    let mut hyb_ws = [0.0f32; 1024];
    let mut hyb = HybridConvolver::prepare(&ir, h_cfg, &mut hyb_ws, SimdBackend::baseline(), fft)
        .expect("hybrid prepare");
    let mut out_hyb = [0.0f32; 64];
    hyb.process_block(&input[..32], &mut out_hyb[..32]).unwrap();
    hyb.process_block(&input[32..], &mut out_hyb[32..]).unwrap();

    println!("Direct      first 4: {:?}", &out_direct[..4]);
    println!("Partitioned first 4: {:?}", &out_part[..4]);
    println!("Hybrid      first 4: {:?}", &out_hyb[..4]);
    println!("Capabilities: {:?}", hyb.capabilities());
}
