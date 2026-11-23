#![cfg(feature = "nonuniform")]

use rtconv_simd::{
    BudgetVerdict, ConvError, MicroFftBackend, NonUniformBudget, NonUniformConfig,
    NonUniformConvolver, NonUniformPlan, NonUniformStageConfig, SimdBackend, SimdFixedFftBackend,
};
use std::f32::consts::PI;

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        let bits = (self.0 >> 40) as u32;
        (bits as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
    }
    fn vec_f32(&mut self, len: usize, scale: f32) -> Vec<f32> {
        (0..len).map(|_| self.next_f32() * scale).collect()
    }
}

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

fn metrics(reference: &[f32], actual: &[f32]) -> (f32, f64) {
    let mut max_abs = 0.0f32;
    let mut sig = 0.0f64;
    let mut err = 0.0f64;
    for (&r, &a) in reference.iter().zip(actual.iter()) {
        let d = (r as f64) - (a as f64);
        max_abs = max_abs.max(d.abs() as f32);
        sig += (r as f64) * (r as f64);
        err += d * d;
    }
    let snr = if err == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (sig / err).log10()
    };
    (max_abs, snr)
}

#[test]
fn test_nupc_impulses_on_every_stage_boundary() {
    let max_ir = 2048;
    let plan = NonUniformPlan::default_profile(max_ir, 32).unwrap();
    let backend = SimdBackend::detect_or_baseline();
    let fft = SimdFixedFftBackend::new(backend);
    let req = NonUniformConvolver::memory_requirements(&plan, &fft).unwrap();
    let mut ws = vec![0.0f32; req.total_f32];

    let mut input = vec![0.0f32; 2560];
    input[0] = 1.0;

    for &tap in &[0usize, 1, 31, 32, 33, 255, 256, 257, 1023, 1024, 1025, 2047] {
        let mut ir = vec![0.0f32; tap + 1];
        ir[tap] = 1.0;
        let expected = ref_convolve_f64(&ir, &input);

        let mut conv = NonUniformConvolver::prepare(&ir, plan, &mut ws, backend, fft).unwrap();
        assert_eq!(conv.latency_samples(), 0);

        let mut out = vec![0.0f32; input.len()];
        for (ib, ob) in input
            .as_chunks::<32>()
            .0
            .iter()
            .zip(out.as_chunks_mut::<32>().0.iter_mut())
        {
            conv.process_block(ib, ob).unwrap();
        }
        assert_eq!(conv.deadline_misses(), 0);
        let (max_abs, _) = metrics(&expected, &out);
        assert!(
            max_abs < 1e-5,
            "impulse at tap={tap} misaligned: max_abs={max_abs:.3e}"
        );
    }
}

#[test]
fn test_nupc_random_phase_and_non_multiple_irs_against_f64() {
    let mut rng = Rng::new(0x004B_2026);
    let backend = SimdBackend::detect_or_baseline();
    let input = rng.vec_f32(3072, 0.8);

    // Non-multiple lengths spanning 1, 2, 3, and 4 stages
    for &ir_len in &[
        1usize, 31, 32, 33, 173, 255, 256, 257, 1023, 1024, 1025, 2051,
    ] {
        let ir = rng.vec_f32(ir_len, 0.2);
        let expected = ref_convolve_f64(&ir, &input);
        let plan = NonUniformPlan::default_profile(ir_len, 32).unwrap();

        // MicroFftBackend
        let mfft = MicroFftBackend::new();
        let req_m = NonUniformConvolver::memory_requirements(&plan, &mfft).unwrap();
        let mut ws_m = vec![0.0f32; req_m.total_f32];
        let mut cm = NonUniformConvolver::prepare(&ir, plan, &mut ws_m, backend, mfft).unwrap();
        let mut out_m = vec![0.0f32; input.len()];
        for (ib, ob) in input
            .as_chunks::<32>()
            .0
            .iter()
            .zip(out_m.as_chunks_mut::<32>().0.iter_mut())
        {
            cm.process_block(ib, ob).unwrap();
        }
        let (err_m, snr_m) = metrics(&expected, &out_m);
        assert!(
            err_m < 1e-4 && snr_m > 95.0,
            "MicroFft NUPC ir_len={ir_len}: err={err_m:.3e}, snr={snr_m:.2} dB"
        );
        assert_eq!(cm.deadline_misses(), 0);

        // SimdFixedFftBackend
        let sfft = SimdFixedFftBackend::new(backend);
        let req_s = NonUniformConvolver::memory_requirements(&plan, &sfft).unwrap();
        let mut ws_s = vec![0.0f32; req_s.total_f32];
        let mut cs = NonUniformConvolver::prepare(&ir, plan, &mut ws_s, backend, sfft).unwrap();
        let mut out_s = vec![0.0f32; input.len()];
        for (ib, ob) in input
            .as_chunks::<32>()
            .0
            .iter()
            .zip(out_s.as_chunks_mut::<32>().0.iter_mut())
        {
            cs.process_block(ib, ob).unwrap();
        }
        let (err_s, snr_s) = metrics(&expected, &out_s);
        assert!(
            err_s < 1e-4 && snr_s > 95.0,
            "SimdFixedFft NUPC ir_len={ir_len}: err={err_s:.3e}, snr={snr_s:.2} dB"
        );
        assert_eq!(cs.deadline_misses(), 0);
    }
}

#[test]
fn test_nupc_multitone_silence_drain_and_reset_reproducibility() {
    let mut rng = Rng::new(777);
    let ir = rng.vec_f32(1500, 0.25);
    let mut gated = vec![0.0f32; 4096];
    for (i, sample) in gated.iter_mut().enumerate().skip(128).take(1024) {
        let t = i as f32 / 48_000.0;
        *sample = 0.5 * (2.0 * PI * 440.0 * t).sin() + 0.3 * (2.0 * PI * 2713.0 * t).cos();
    }
    let expected = ref_convolve_f64(&ir, &gated);

    let plan = NonUniformPlan::default_profile(ir.len(), 32).unwrap();
    let backend = SimdBackend::detect_or_baseline();
    let fft = SimdFixedFftBackend::new(backend);
    let req = NonUniformConvolver::memory_requirements(&plan, &fft).unwrap();
    let mut ws = vec![0.0f32; req.total_f32];

    let mut conv = NonUniformConvolver::prepare(&ir, plan, &mut ws, backend, fft).unwrap();
    let mut out1 = vec![0.0f32; gated.len()];
    let mut out2 = vec![0.0f32; gated.len()];
    conv.process_block(&gated, &mut out1).unwrap();
    conv.reset();
    conv.process_block(&gated, &mut out2).unwrap();

    assert_eq!(out1, out2, "reset() must produce bit-identical output");
    let (err, snr) = metrics(&expected, &out1);
    assert!(err < 1e-4 && snr > 95.0, "err={err:.3e}, snr={snr:.2} dB");
}

#[test]
fn test_nupc_rejects_non_causal_and_budget_exceeded_configs() {
    // Non-causal: FDL block_size=64 at ir_offset=32 (< 64)
    let bad_causal = NonUniformConfig {
        max_ir_len: 256,
        host_block_max: 32,
        stages: &[
            NonUniformStageConfig {
                ir_offset: 0,
                ir_len: 32,
                block_size: 1,
            },
            NonUniformStageConfig {
                ir_offset: 32,
                ir_len: 224,
                block_size: 64,
            },
        ],
    };
    assert_eq!(
        NonUniformPlan::validate(bad_causal).unwrap_err(),
        ConvError::InvalidNonUniformConfig
    );

    // Budget rejection (memory limit too tight)
    let plan = NonUniformPlan::default_profile(2048, 32).unwrap();
    let backend = SimdBackend::detect_or_baseline();
    let fft = MicroFftBackend::new();
    let req = NonUniformConvolver::memory_requirements(&plan, &fft).unwrap();
    let mut ws = vec![0.0f32; req.total_f32];
    let ir = vec![0.1f32; 2048];
    let tight_budget = NonUniformBudget {
        sample_rate_hz: 48_000,
        max_frames_per_callback: 32,
        deadline_us: 666,
        max_memory_bytes: 1024,
        max_ops_per_callback: 0,
        safety_margin_percent: 20,
    };
    assert_eq!(
        plan.check_budget(tight_budget, req.total_bytes),
        BudgetVerdict::Exceeded
    );
    assert!(matches!(
        NonUniformConvolver::prepare_with_budget(
            &ir,
            plan,
            Some(tight_budget),
            &mut ws,
            backend,
            fft
        ),
        Err(ConvError::InvalidNonUniformConfig)
    ));
}
