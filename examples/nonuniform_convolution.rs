#[cfg(feature = "nonuniform")]
fn main() {
    use rtconv_simd::{
        NonUniformBudget, NonUniformConvolver, NonUniformPlan, SimdBackend, SimdFixedFftBackend,
    };

    let ir: Vec<f32> = (0..2048)
        .map(|i| {
            if i == 0 {
                1.0
            } else {
                0.25 * (-0.003 * i as f32).exp() * ((i as f32) * 0.1).sin()
            }
        })
        .collect();
    let plan = NonUniformPlan::default_profile(ir.len(), 32).unwrap();
    let backend = SimdBackend::detect_or_baseline();
    let fft = SimdFixedFftBackend::new(backend);
    let req = NonUniformConvolver::memory_requirements(&plan, &fft).unwrap();

    let budget = NonUniformBudget {
        sample_rate_hz: 48_000,
        max_frames_per_callback: 32,
        deadline_us: 666,
        max_memory_bytes: req.total_bytes * 2,
        max_ops_per_callback: 0,
        safety_margin_percent: 20,
    };

    let mut ws = vec![0.0f32; req.total_f32];
    let mut conv =
        NonUniformConvolver::prepare_with_budget(&ir, plan, Some(budget), &mut ws, backend, fft)
            .unwrap();

    let mut input = [0.0f32; 32];
    input[0] = 1.0;
    let mut output = [0.0f32; 32];
    conv.process_block(&input, &mut output).unwrap();
    let caps = conv.capabilities();
    println!(
        "NUPC stages={}, latency={}, workspace={} B, y[0]={:.4}, verdict={:?}",
        caps.stage_count, caps.latency_samples, req.total_bytes, output[0], caps.budget_verdict
    );
}

#[cfg(not(feature = "nonuniform"))]
fn main() {
    println!("Run with `--features nonuniform` to execute the NUPC example.");
}
