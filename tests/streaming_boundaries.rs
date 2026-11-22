//! Streaming boundary, partial block, denormal, silence, and multi-instance tests.

use rtconv_simd::{
    DirectConfig, DirectConvolver, HybridConfig, HybridConvolver, MicroFftBackend,
    PartitionedConfig, PartitionedConvolver, SimdBackend, SimdFixedFftBackend,
};

fn ref_convolve_f64(ir: &[f32], input: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; input.len()];
    for n in 0..input.len() {
        let mut acc = 0.0f64;
        for k in 0..ir.len().min(n + 1) {
            acc += (ir[k] as f64) * (input[n - k] as f64);
        }
        out[n] = acc as f32;
    }
    out
}

#[test]
fn test_denormal_inputs_and_coefficients_stay_finite() {
    // Subnormal f32 values (~1e-40)
    let ir = [1.0e-40f32, -2.5e-40, 0.5, 1.0e-41, -0.25];
    let mut input = vec![1.0e-40f32; 256];
    input[64] = 1.0;
    input[128] = -1.0e-42;
    let expected = ref_convolve_f64(&ir, &input);
    let backend = SimdBackend::detect_or_baseline();

    // Direct
    let dcfg = DirectConfig {
        max_ir_len: ir.len(),
    };
    let mut dws = vec![
        0.0f32;
        DirectConvolver::memory_requirements(dcfg)
            .unwrap()
            .total_f32
    ];
    let mut dconv = DirectConvolver::prepare(&ir, dcfg, &mut dws, backend).unwrap();
    let mut out = vec![0.0f32; 256];
    dconv.process_block(&input, &mut out).unwrap();
    for (&r, &a) in expected.iter().zip(out.iter()) {
        assert!(a.is_finite());
        assert!((r - a).abs() < 1e-5);
    }

    // Partitioned (SimdFixedFft)
    let fft = SimdFixedFftBackend::new(backend);
    let pcfg = PartitionedConfig {
        block_size: 64,
        max_ir_len: ir.len(),
    };
    let mut pws = vec![
        0.0f32;
        PartitionedConvolver::memory_requirements(pcfg, &fft)
            .unwrap()
            .total_f32
    ];
    let mut pconv = PartitionedConvolver::prepare(&ir, pcfg, &mut pws, backend, fft).unwrap();
    for (ib, ob) in input
        .as_chunks::<64>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<64>().0.iter_mut())
    {
        pconv.process_block(ib, ob).unwrap();
    }
    for (&r, &a) in expected.iter().zip(out.iter()) {
        assert!(a.is_finite());
        assert!((r - a).abs() < 1e-5);
    }
}

#[test]
fn test_multiple_independent_instances_stereo_isolation() {
    let ir_l = [0.8f32, -0.2, 0.1, 0.05];
    let ir_r = [-0.5f32, 0.4, -0.1, 0.2];
    let mut in_l = vec![0.0f32; 256];
    let mut in_r = vec![0.0f32; 256];
    in_l[0] = 1.0;
    in_r[17] = 1.0;

    let ref_l = ref_convolve_f64(&ir_l, &in_l);
    let ref_r = ref_convolve_f64(&ir_r, &in_r);

    let backend = SimdBackend::detect_or_baseline();
    let fft = MicroFftBackend::new();
    let cfg = HybridConfig {
        head_len: 64,
        tail_block_size: 64,
        max_ir_len: 128,
    };
    let req = HybridConvolver::memory_requirements(cfg, &fft).unwrap();
    let mut ws_l = vec![0.0f32; req.total_f32];
    let mut ws_r = vec![0.0f32; req.total_f32];

    let mut conv_l =
        HybridConvolver::prepare_with_tail_spreading(&ir_l, cfg, &mut ws_l, backend, fft, true)
            .unwrap();
    let mut conv_r =
        HybridConvolver::prepare_with_tail_spreading(&ir_r, cfg, &mut ws_r, backend, fft, true)
            .unwrap();

    let mut out_l = vec![0.0f32; 256];
    let mut out_r = vec![0.0f32; 256];
    for i in (0..256).step_by(32) {
        conv_l
            .process_block(&in_l[i..i + 32], &mut out_l[i..i + 32])
            .unwrap();
        conv_r
            .process_block(&in_r[i..i + 32], &mut out_r[i..i + 32])
            .unwrap();
    }

    for i in 0..256 {
        assert!((ref_l[i] - out_l[i]).abs() < 1e-5);
        assert!((ref_r[i] - out_r[i]).abs() < 1e-5);
    }
}
