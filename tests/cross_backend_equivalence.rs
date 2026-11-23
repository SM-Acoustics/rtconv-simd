#![cfg(feature = "fixed-nonuniform")]

use rtconv_simd::fixed::{f64_to_q15, q15_to_f64};
use rtconv_simd::{
    BudgetVerdict, FixedQ15NonUniformConvolver, NonUniformBudget, NonUniformConvolver,
    NonUniformPlan, SimdBackend, SimdFixedFftBackend,
};

fn ref_convolve_q15_f64(ir: &[i16], input: &[i16]) -> Vec<f64> {
    let mut out = vec![0.0f64; input.len()];
    for n in 0..input.len() {
        let mut acc = 0.0f64;
        let max_k = ir.len().min(n + 1);
        for k in 0..max_k {
            acc += q15_to_f64(ir[k]) * q15_to_f64(input[n - k]);
        }
        out[n] = acc.clamp(-1.0, 32767.0 / 32768.0);
    }
    out
}

#[test]
fn test_fixed_nupc_matches_f32_nupc_and_f64_reference_with_fragmentation() {
    let mut seed = 0x004E_2026u64;
    let mut next_f64 = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((seed >> 11) as f64 / ((1u64 << 53) as f64)) * 2.0 - 1.0
    };

    // 1,350-tap decaying IR spanning all 4 NUPC stages (Direct 32, B32, B64, B128)
    let raw_ir: Vec<f64> = (0..1350)
        .map(|k| next_f64() * (-0.004 * k as f64).exp())
        .collect();
    let l1: f64 = raw_ir.iter().map(|v| v.abs()).sum();
    let ir_q15: Vec<i16> = raw_ir.iter().map(|&v| f64_to_q15(v * 0.7 / l1)).collect();
    let ir_f32: Vec<f32> = ir_q15.iter().map(|&v| q15_to_f64(v) as f32).collect();

    let in_q15: Vec<i16> = (0..2200).map(|_| f64_to_q15(next_f64() * 0.7)).collect();
    let in_f32: Vec<f32> = in_q15.iter().map(|&v| q15_to_f64(v) as f32).collect();
    let expected = ref_convolve_q15_f64(&ir_q15, &in_q15);

    let plan = NonUniformPlan::default_profile(ir_q15.len(), 32).unwrap();
    let qreq = FixedQ15NonUniformConvolver::memory_requirements(&plan).unwrap();
    let mut qws = vec![0i16; qreq.total_i16];

    let budget = NonUniformBudget {
        sample_rate_hz: 48_000,
        max_frames_per_callback: 32,
        deadline_us: 666,
        max_memory_bytes: qreq.total_bytes * 2,
        max_ops_per_callback: 0,
        safety_margin_percent: 20,
    };
    let mut qconv =
        FixedQ15NonUniformConvolver::prepare_with_budget(&ir_q15, plan, Some(budget), &mut qws)
            .unwrap();
    assert_eq!(
        qconv.capabilities().budget_verdict,
        BudgetVerdict::WithinKnownBounds
    );

    let backend = SimdBackend::detect_or_baseline();
    let fft = SimdFixedFftBackend::new(backend);
    let freq = NonUniformConvolver::memory_requirements(&plan, &fft).unwrap();
    let mut fws = vec![0.0f32; freq.total_f32];
    let mut fconv = NonUniformConvolver::prepare(&ir_f32, plan, &mut fws, backend, fft).unwrap();

    // Fragmented host blocks (1, 7, 13, 32, 19, 64, 5)
    let pattern = [1usize, 7, 13, 32, 19, 64, 5];
    let mut qout = vec![0i16; in_q15.len()];
    let mut fout = vec![0.0f32; in_f32.len()];
    let mut pos = 0usize;
    let mut pi = 0usize;
    while pos < in_q15.len() {
        let n = pattern[pi % pattern.len()].min(in_q15.len() - pos);
        qconv
            .process_block(&in_q15[pos..pos + n], &mut qout[pos..pos + n])
            .unwrap();
        fconv
            .process_block(&in_f32[pos..pos + n], &mut fout[pos..pos + n])
            .unwrap();
        pos += n;
        pi += 1;
    }

    assert_eq!(qconv.deadline_misses(), 0);
    assert_eq!(fconv.deadline_misses(), 0);
    assert_eq!(qconv.saturation_count(), 0);

    let mut sig = 0.0f64;
    let mut err_q = 0.0f64;
    let mut max_q = 0.0f64;
    let mut max_f = 0.0f64;
    for i in 0..expected.len() {
        let r = expected[i];
        let qv = q15_to_f64(qout[i]);
        let fv = fout[i] as f64;
        sig += r * r;
        err_q += (r - qv) * (r - qv);
        max_q = max_q.max((r - qv).abs());
        max_f = max_f.max((r - fv).abs());
    }
    let snr_q = 10.0 * (sig / err_q).log10();
    assert!(
        snr_q > 42.0 && max_q < 5e-4,
        "Fixed NUPC vs f64: snr={snr_q:.2} dB, max_err={max_q:.3e}"
    );
    assert!(max_f < 1e-4, "f32 NUPC vs f64: max_err={max_f:.3e}");

    // Deterministic reset
    qconv.reset();
    let mut qout2 = vec![0i16; in_q15.len()];
    qconv.process_block(&in_q15, &mut qout2).unwrap();
    assert_eq!(qout, qout2);
}
