#![cfg(feature = "nonuniform")]

use rtconv_simd::{NonUniformConvolver, NonUniformPlan, SimdBackend, SimdFixedFftBackend};

fn ref_convolve_f64(ir: &[f32], input: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; input.len()];
    for n in 0..input.len() {
        let mut acc = 0.0f64;
        let max_k = ir.len().min(n + 1);
        for k in 0..max_k {
            acc += (ir[k] as f64) * (input[n - k] as f64);
        }
        out[n] = acc as f32;
    }
    out
}

#[test]
fn test_nupc_arbitrary_and_dynamic_host_block_fragmentation() {
    let ir: Vec<f32> = (0..1357)
        .map(|i| ((i as f32) * 0.13).sin() * (-0.002 * i as f32).exp() * 0.3)
        .collect();
    let input: Vec<f32> = (0..2345).map(|i| ((i as f32) * 0.29).cos() * 0.7).collect();
    let expected = ref_convolve_f64(&ir, &input);

    let plan = NonUniformPlan::default_profile(ir.len(), 32).unwrap();
    let backend = SimdBackend::detect_or_baseline();
    let fft = SimdFixedFftBackend::new(backend);
    let req = NonUniformConvolver::memory_requirements(&plan, &fft).unwrap();
    let mut ws = vec![0.0f32; req.total_f32];
    let mut conv = NonUniformConvolver::prepare(&ir, plan, &mut ws, backend, fft).unwrap();

    // Dynamic varying host block sizes (prime, sub-block, super-block, 1-sample)
    let pattern = [1usize, 7, 13, 32, 3, 64, 19, 100, 17, 5, 31];
    let mut out = vec![0.0f32; input.len()];
    let mut pos = 0usize;
    let mut p_idx = 0usize;
    while pos < input.len() {
        let chunk = pattern[p_idx % pattern.len()].min(input.len() - pos);
        conv.process_block(&input[pos..pos + chunk], &mut out[pos..pos + chunk])
            .unwrap();
        pos += chunk;
        p_idx += 1;
    }

    let mut max_err = 0.0f32;
    for (&r, &a) in expected.iter().zip(out.iter()) {
        assert!(a.is_finite());
        max_err = max_err.max((r - a).abs());
    }
    assert!(
        max_err < 1e-4,
        "fragmented NUPC max_err={max_err:.3e} exceeds tolerance"
    );
    assert_eq!(conv.deadline_misses(), 0);
}

#[test]
fn test_nupc_multiple_independent_stereo_instances() {
    let ir_l: Vec<f32> = (0..600)
        .map(|i| if i == 0 || i == 300 { 0.5 } else { 0.0 })
        .collect();
    let ir_r: Vec<f32> = (0..600)
        .map(|i| if i == 40 || i == 500 { -0.4 } else { 0.0 })
        .collect();
    let mut in_l = vec![0.0f32; 1024];
    let mut in_r = vec![0.0f32; 1024];
    in_l[5] = 1.0;
    in_r[19] = 1.0;

    let ref_l = ref_convolve_f64(&ir_l, &in_l);
    let ref_r = ref_convolve_f64(&ir_r, &in_r);

    let plan = NonUniformPlan::default_profile(600, 32).unwrap();
    let backend = SimdBackend::detect_or_baseline();
    let fft = SimdFixedFftBackend::new(backend);
    let req = NonUniformConvolver::memory_requirements(&plan, &fft).unwrap();

    let mut ws_l = vec![0.0f32; req.total_f32];
    let mut ws_r = vec![0.0f32; req.total_f32];
    let mut conv_l = NonUniformConvolver::prepare(&ir_l, plan, &mut ws_l, backend, fft).unwrap();
    let mut conv_r = NonUniformConvolver::prepare(&ir_r, plan, &mut ws_r, backend, fft).unwrap();

    let mut out_l = vec![0.0f32; 1024];
    let mut out_r = vec![0.0f32; 1024];
    conv_l.process_block(&in_l, &mut out_l).unwrap();
    conv_r.process_block(&in_r, &mut out_r).unwrap();

    for i in 0..1024 {
        assert!((ref_l[i] - out_l[i]).abs() < 1e-5);
        assert!((ref_r[i] - out_r[i]).abs() < 1e-5);
    }
}
