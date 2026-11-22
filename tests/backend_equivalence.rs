//! Cross-backend equivalence tests across SIMD levels and FFT backends.

#[cfg(feature = "phastft")]
use rtconv_simd::PhastFftBackend;
use rtconv_simd::{
    FftBackend, HybridConfig, HybridConvolver, MicroFftBackend, SimdBackend, SimdFixedFftBackend,
};

fn run_hybrid_stream<F: FftBackend>(
    ir: &[f32],
    input: &[f32],
    backend: SimdBackend,
    fft: F,
    spread: bool,
) -> Vec<f32> {
    let cfg = HybridConfig {
        head_len: 128,
        tail_block_size: 128,
        max_ir_len: ir.len(),
    };
    let req = HybridConvolver::memory_requirements(cfg, &fft).unwrap();
    let mut ws = vec![0.0f32; req.total_f32];
    let mut conv =
        HybridConvolver::prepare_with_tail_spreading(ir, cfg, &mut ws, backend, fft, spread)
            .unwrap();
    let mut out = vec![0.0f32; input.len()];
    for (ib, ob) in input.chunks(32).zip(out.chunks_mut(32)) {
        conv.process_block(ib, ob).unwrap();
    }
    out
}

#[test]
fn test_all_available_backends_agree() {
    let mut s = 0xDEAD_BEEF_1029_3847u64;
    let mut next_f32 = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 40) as f32 / ((1u32 << 24) as f32)) * 2.0 - 1.0
    };

    let ir: Vec<f32> = (0..641).map(|_| next_f32() * 0.2).collect();
    let input: Vec<f32> = (0..1024).map(|_| next_f32() * 0.8).collect();

    let scalar = SimdBackend::scalar();
    let baseline = SimdBackend::baseline();
    let best = SimdBackend::detect_or_baseline();

    let y_micro_sc = run_hybrid_stream(&ir, &input, scalar, MicroFftBackend::new(), false);
    let y_fixed_sc = run_hybrid_stream(&ir, &input, scalar, SimdFixedFftBackend::new(scalar), true);
    let y_fixed_base = run_hybrid_stream(
        &ir,
        &input,
        baseline,
        SimdFixedFftBackend::new(baseline),
        true,
    );
    let y_fixed_best = run_hybrid_stream(&ir, &input, best, SimdFixedFftBackend::new(best), true);

    for (label, candidate) in [
        ("SimdFixed(Scalar)", &y_fixed_sc),
        ("SimdFixed(Baseline)", &y_fixed_base),
        ("SimdFixed(Best)", &y_fixed_best),
    ] {
        let max_diff = y_micro_sc
            .iter()
            .zip(candidate.iter())
            .map(|(&a, &b)| (a as f64 - b as f64).abs())
            .fold(0.0f64, f64::max);
        assert!(
            max_diff < 5e-5,
            "{label} diverged from MicroFft(Scalar): max_diff={max_diff:.3e}"
        );
    }

    #[cfg(feature = "phastft")]
    {
        let y_phast = run_hybrid_stream(&ir, &input, best, PhastFftBackend::new(), true);
        let max_diff = y_micro_sc
            .iter()
            .zip(y_phast.iter())
            .map(|(&a, &b)| (a as f64 - b as f64).abs())
            .fold(0.0f64, f64::max);
        assert!(
            max_diff < 5e-5,
            "PhastFftBackend diverged from MicroFft(Scalar): max_diff={max_diff:.3e}"
        );
    }
}
