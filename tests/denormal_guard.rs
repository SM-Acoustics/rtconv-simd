//! IEEE 754 subnormal protection (`simd_kernels::DENORMAL_GUARD`): tiny inputs and coefficients
//! are flushed to zero consistently in every `f32` engine, so decaying signals end in exact
//! zeros and IRs with subnormal taps behave exactly like their flushed counterparts.

use rtconv_simd::simd_kernels::{DENORMAL_GUARD, flush_tiny};
use rtconv_simd::{
    DirectConfig, DirectConvolver, HybridConfig, HybridConvolver, PartitionedConfig,
    PartitionedConvolver, SimdBackend, SimdFixedFftBackend,
};

#[test]
fn guard_constant_is_exactly_two_to_minus_63_and_products_stay_normal() {
    assert_eq!(DENORMAL_GUARD, 2.0f32.powi(-63));
    // The smallest non-zero product of two guarded values is the smallest normal f32.
    assert_eq!(DENORMAL_GUARD * DENORMAL_GUARD, f32::MIN_POSITIVE);
    assert!((DENORMAL_GUARD * DENORMAL_GUARD).is_normal());
    assert_eq!(flush_tiny(1e-39), 0.0);
    assert_eq!(flush_tiny(-1e-20), 0.0);
    assert_eq!(flush_tiny(DENORMAL_GUARD), DENORMAL_GUARD);
    assert_eq!(flush_tiny(-0.25), -0.25);
    assert!(flush_tiny(f32::NAN).is_nan());
    assert_eq!(flush_tiny(f32::INFINITY), f32::INFINITY);
}

/// IR with a deep tail: values below the guard (including true subnormals) mixed with normal taps.
fn deep_tail_ir(len: usize) -> Vec<f32> {
    (0..len)
        .map(|k| (0.5 * (-(k as f64) / 9.0).exp() * ((k as f64) * 0.3).sin()) as f32)
        .collect()
}

/// Input that decays from 0.5 through the subnormal range and then stays silent.
fn decaying_input(len: usize) -> Vec<f32> {
    (0..len).map(|n| 0.5 * 0.9f32.powi(n as i32)).collect()
}

fn run_all<F: FnMut(&[f32], &[f32]) -> Vec<f32>>(mut engine: F) {
    let ir = deep_tail_ir(1024);
    assert!(ir.iter().any(|v| *v != 0.0 && v.abs() < f32::MIN_POSITIVE));
    let ir_flushed: Vec<f32> = ir.iter().map(|&v| flush_tiny(v)).collect();
    let x = decaying_input(4096);
    let x_flushed: Vec<f32> = x.iter().map(|&v| flush_tiny(v)).collect();

    // 1. Subnormal / tiny coefficients and inputs behave exactly as if pre-flushed.
    let y = engine(&ir, &x);
    let y_ref = engine(&ir_flushed, &x_flushed);
    assert_eq!(y, y_ref);

    // 2. Once the input is below the guard and the IR has drained, the output is exact zero
    //    (no lingering subnormal residue) and never subnormal before that.
    let last_nonzero_input = x_flushed.iter().rposition(|v| *v != 0.0).unwrap();
    let drained = last_nonzero_input + ir.len() + 2 * 128 + 1;
    assert!(y[drained..].iter().all(|v| *v == 0.0));
    let subnormal_out = y.iter().filter(|v| v.is_subnormal()).count();
    assert_eq!(subnormal_out, 0, "outputs must not be subnormal");
}

#[test]
fn direct_flushes_inputs_and_coefficients() {
    for backend in [SimdBackend::scalar(), SimdBackend::detect_or_baseline()] {
        run_all(|ir, x| {
            let cfg = DirectConfig {
                max_ir_len: ir.len(),
            };
            let mut ws = vec![0.0f32; 2 * ir.len()];
            let mut c = DirectConvolver::prepare(ir, cfg, &mut ws, backend).unwrap();
            let mut y = vec![0.0f32; x.len()];
            for (xi, yo) in x.chunks(37).zip(y.chunks_mut(37)) {
                c.process_block(xi, yo).unwrap();
            }
            y
        });
    }
}

#[test]
fn partitioned_flushes_inputs_and_coefficients() {
    let backend = SimdBackend::detect_or_baseline();
    run_all(|ir, x| {
        let fft = SimdFixedFftBackend::new(backend);
        let cfg = PartitionedConfig {
            block_size: 128,
            max_ir_len: ir.len(),
        };
        let req = PartitionedConvolver::memory_requirements(cfg, &fft).unwrap();
        let mut ws = vec![0.0f32; req.total_f32];
        let mut c = PartitionedConvolver::prepare(ir, cfg, &mut ws, backend, fft).unwrap();
        let mut y = vec![0.0f32; x.len()];
        for (xi, yo) in x.chunks(128).zip(y.chunks_mut(128)) {
            c.process_block(xi, yo).unwrap();
        }
        y
    });
}

#[test]
fn hybrid_spread_flushes_inputs_and_coefficients() {
    let backend = SimdBackend::detect_or_baseline();
    run_all(|ir, x| {
        let fft = SimdFixedFftBackend::new(backend);
        let cfg = HybridConfig {
            head_len: 64,
            tail_block_size: 64,
            max_ir_len: ir.len(),
        };
        let req = HybridConvolver::memory_requirements(cfg, &fft).unwrap();
        let mut ws = vec![0.0f32; req.total_f32];
        let mut c =
            HybridConvolver::prepare_with_tail_spreading(ir, cfg, &mut ws, backend, fft, true)
                .unwrap();
        let mut y = vec![0.0f32; x.len()];
        for (xi, yo) in x.chunks(48).zip(y.chunks_mut(48)) {
            c.process_block(xi, yo).unwrap();
        }
        y
    });
}

#[cfg(feature = "nonuniform")]
#[test]
fn nonuniform_flushes_inputs_and_coefficients() {
    use rtconv_simd::{NonUniformConvolver, NonUniformPlan};
    let backend = SimdBackend::detect_or_baseline();
    run_all(|ir, x| {
        let fft = SimdFixedFftBackend::new(backend);
        let plan = NonUniformPlan::default_profile(ir.len(), 32).unwrap();
        let req = NonUniformConvolver::memory_requirements(&plan, &fft).unwrap();
        let mut ws = vec![0.0f32; req.total_f32];
        let mut c = NonUniformConvolver::prepare(ir, plan, &mut ws, backend, fft).unwrap();
        let mut y = vec![0.0f32; x.len()];
        c.process_block(x, &mut y).unwrap();
        y
    });
}

#[test]
fn large_ffts_n512_to_n16384_flush_subnormals_on_simd_and_microfft() {
    use rtconv_simd::MicroFftBackend;

    let backend = SimdBackend::detect_or_baseline();
    let simd_fft = SimdFixedFftBackend::new(backend);
    let micro_fft = MicroFftBackend::new();

    for &b in &[256usize, 512, 1024, 2048, 4096, 8192] {
        // IR spans at least 2 partitions of size B and >= 1024 taps so deep_tail_ir reaches true subnormals
        let ir_len = (b + 256).max(1024);
        let ir = deep_tail_ir(ir_len);
        assert!(ir.iter().any(|v| *v != 0.0 && v.abs() < f32::MIN_POSITIVE));
        let ir_flushed: Vec<f32> = ir.iter().map(|&v| flush_tiny(v)).collect();

        // Input has >= 3 blocks (and long enough to drain ir_len)
        let x_len = (3 * b).max(ir_len + 2 * b + 1024).next_multiple_of(b);
        let x = decaying_input(x_len);
        let x_flushed: Vec<f32> = x.iter().map(|&v| flush_tiny(v)).collect();
        let last_nz = x_flushed.iter().rposition(|v| *v != 0.0).unwrap();
        let drained = (last_nz + ir_len + 2 * b + 1).min(x_len);

        // PartitionedConvolver (SimdFixedFftBackend & MicroFftBackend)
        let pcfg = PartitionedConfig {
            block_size: b,
            max_ir_len: ir_len,
        };
        for is_simd in [true, false] {
            let run_p = |h: &[f32], inp: &[f32]| -> Vec<f32> {
                let mut y = vec![0.0f32; inp.len()];
                if is_simd {
                    let req = PartitionedConvolver::memory_requirements(pcfg, &simd_fft).unwrap();
                    let mut ws = vec![0.0f32; req.total_f32];
                    let mut c =
                        PartitionedConvolver::prepare(h, pcfg, &mut ws, backend, simd_fft).unwrap();
                    for (xi, yo) in inp.chunks_exact(b).zip(y.chunks_exact_mut(b)) {
                        c.process_block(xi, yo).unwrap();
                    }
                } else {
                    let req = PartitionedConvolver::memory_requirements(pcfg, &micro_fft).unwrap();
                    let mut ws = vec![0.0f32; req.total_f32];
                    let mut c = PartitionedConvolver::prepare(h, pcfg, &mut ws, backend, micro_fft)
                        .unwrap();
                    for (xi, yo) in inp.chunks_exact(b).zip(y.chunks_exact_mut(b)) {
                        c.process_block(xi, yo).unwrap();
                    }
                }
                y
            };
            let y = run_p(&ir, &x);
            let y_ref = run_p(&ir_flushed, &x_flushed);
            assert_eq!(y, y_ref, "Partitioned B={b} simd={is_simd} mismatch");
            assert_eq!(
                y.iter().filter(|v| v.is_subnormal()).count(),
                0,
                "Partitioned B={b} simd={is_simd} emitted subnormal output"
            );
            assert!(y[drained..].iter().all(|v| *v == 0.0));
        }

        // HybridConvolver SPREAD for B = 256..=2048
        if b <= 2048 {
            let hcfg = HybridConfig {
                head_len: b,
                tail_block_size: b,
                max_ir_len: ir_len,
            };
            let req = HybridConvolver::memory_requirements(hcfg, &simd_fft).unwrap();
            let run_h = |h: &[f32], inp: &[f32]| -> Vec<f32> {
                let mut ws = vec![0.0f32; req.total_f32];
                let mut c = HybridConvolver::prepare_with_tail_spreading(
                    h, hcfg, &mut ws, backend, simd_fft, true,
                )
                .unwrap();
                let mut y = vec![0.0f32; inp.len()];
                for (xi, yo) in inp.chunks(64).zip(y.chunks_mut(64)) {
                    c.process_block(xi, yo).unwrap();
                }
                y
            };
            let y = run_h(&ir, &x);
            let y_ref = run_h(&ir_flushed, &x_flushed);
            assert_eq!(y, y_ref, "Hybrid SPREAD B={b} mismatch");
            assert_eq!(y.iter().filter(|v| v.is_subnormal()).count(), 0);
        }
    }
}

#[cfg(feature = "nonuniform")]
#[test]
fn nonuniform_large_stages_up_to_n16384_flush_subnormals() {
    use rtconv_simd::{
        NonUniformConfig, NonUniformConvolver, NonUniformPlan, NonUniformStageConfig,
    };

    let backend = SimdBackend::detect_or_baseline();
    let fft = SimdFixedFftBackend::new(backend);
    let ir_len = 16_384;
    let ir = deep_tail_ir(ir_len);
    let ir_flushed: Vec<f32> = ir.iter().map(|&v| flush_tiny(v)).collect();
    let x = decaying_input(20_480);
    let x_flushed: Vec<f32> = x.iter().map(|&v| flush_tiny(v)).collect();

    let stages = [
        NonUniformStageConfig {
            ir_offset: 0,
            ir_len: 64,
            block_size: 1,
        },
        NonUniformStageConfig {
            ir_offset: 64,
            ir_len: 192,
            block_size: 64,
        },
        NonUniformStageConfig {
            ir_offset: 256,
            ir_len: 768,
            block_size: 256,
        },
        NonUniformStageConfig {
            ir_offset: 1024,
            ir_len: 3072,
            block_size: 1024,
        },
        NonUniformStageConfig {
            ir_offset: 4096,
            ir_len: 4096,
            block_size: 4096,
        },
        NonUniformStageConfig {
            ir_offset: 8192,
            ir_len: 8192,
            block_size: 8192,
        },
    ];
    let plan = NonUniformPlan::validate(NonUniformConfig {
        max_ir_len: ir_len,
        host_block_max: 64,
        stages: &stages,
    })
    .unwrap();
    let req = NonUniformConvolver::memory_requirements(&plan, &fft).unwrap();

    let run_nu = |h: &[f32], inp: &[f32]| -> Vec<f32> {
        let mut ws = vec![0.0f32; req.total_f32];
        let mut c = NonUniformConvolver::prepare(h, plan, &mut ws, backend, fft).unwrap();
        let mut y = vec![0.0f32; inp.len()];
        c.process_block(inp, &mut y).unwrap();
        assert_eq!(c.deadline_misses(), 0);
        y
    };

    let y = run_nu(&ir, &x);
    let y_ref = run_nu(&ir_flushed, &x_flushed);
    assert_eq!(y, y_ref, "NUPC large stages mismatch");
    assert_eq!(y.iter().filter(|v| v.is_subnormal()).count(), 0);
}
