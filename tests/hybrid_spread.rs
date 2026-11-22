//! State integrity and partial/variable block size tests for `HybridConvolver`
//! time-distributed tail spreading (`BURST` vs `SPREAD`).

use rtconv_simd::{ConvError, HybridConfig, HybridConvolver, SimdBackend, SimdFixedFftBackend};

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
fn test_set_tail_spreading_rejects_mid_cycle_transition() {
    let ir: Vec<f32> = (0..512).map(|k| (-0.01 * k as f32).exp() * 0.2).collect();
    let backend = SimdBackend::detect_or_baseline();
    let fft = SimdFixedFftBackend::new(backend);
    let cfg = HybridConfig {
        head_len: 128,
        tail_block_size: 128,
        max_ir_len: 512,
    };
    let req = HybridConvolver::memory_requirements(cfg, &fft).unwrap();
    let mut ws = vec![0.0f32; req.total_f32];
    let mut conv = HybridConvolver::prepare(&ir, cfg, &mut ws, backend, fft).unwrap();

    // Allowed before streaming
    assert!(conv.set_tail_spreading(true).is_ok());
    assert!(conv.tail_spreading());

    // Process 1 partial sub-block (32 of 128) -> mid-cycle
    let input = [0.5f32; 32];
    let mut output = [0.0f32; 32];
    conv.process_block(&input, &mut output).unwrap();

    // Idempotent call with same value succeeds
    assert!(conv.set_tail_spreading(true).is_ok());
    // Changing mode mid-cycle MUST fail with InvalidStateTransition
    assert_eq!(
        conv.set_tail_spreading(false),
        Err(ConvError::InvalidStateTransition)
    );

    // After reset(), mode change succeeds cleanly
    conv.reset();
    assert!(conv.set_tail_spreading(false).is_ok());
    assert!(!conv.tail_spreading());
}

#[test]
fn test_hybrid_spread_arbitrary_and_partial_block_sizes() {
    let mut s = 0x1234_5678_9ABC_DEF0u64;
    let mut next_f32 = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 40) as f32 / ((1u32 << 24) as f32)) * 2.0 - 1.0
    };

    let ir: Vec<f32> = (0..777).map(|_| next_f32() * 0.2).collect();
    let input: Vec<f32> = (0..1200).map(|_| next_f32() * 0.8).collect();
    let reference = ref_convolve_f64(&ir, &input);

    let backend = SimdBackend::detect_or_baseline();
    let fft = SimdFixedFftBackend::new(backend);
    let cfg = HybridConfig {
        head_len: 128,
        tail_block_size: 128,
        max_ir_len: ir.len(),
    };
    let req = HybridConvolver::memory_requirements(cfg, &fft).unwrap();

    // Test fixed non-power-of-two block sizes and a varying chunk pattern
    let patterns: &[&[usize]] = &[
        &[1],
        &[7],
        &[19],
        &[32],
        &[48],
        &[128],
        &[200],
        &[13, 37, 5, 73, 64],
    ];

    for pattern in patterns {
        let mut ws = vec![0.0f32; req.total_f32];
        let mut conv =
            HybridConvolver::prepare_with_tail_spreading(&ir, cfg, &mut ws, backend, fft, true)
                .unwrap();

        let mut out = vec![0.0f32; input.len()];
        let mut pos = 0;
        let mut p_idx = 0;
        while pos < input.len() {
            let chunk = pattern[p_idx % pattern.len()].min(input.len() - pos);
            conv.process_block(&input[pos..pos + chunk], &mut out[pos..pos + chunk])
                .unwrap();
            pos += chunk;
            p_idx += 1;
        }

        let max_err = reference
            .iter()
            .zip(out.iter())
            .map(|(&r, &a)| (r as f64 - a as f64).abs())
            .fold(0.0f64, f64::max);
        assert!(
            max_err < 8e-5,
            "pattern {:?} failed: max_err={max_err:.3e}",
            pattern
        );
    }
}
