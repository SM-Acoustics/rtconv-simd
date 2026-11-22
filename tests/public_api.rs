//! Public API surface, `Capabilities`, and feature-flag contract tests.

use rtconv_simd::{
    ConvError, DirectConfig, DirectConvolver, EngineKind, FftBackend, FftSize, HybridConfig,
    HybridConvolver, MicroFftBackend, PartitionedConfig, PartitionedConvolver, SimdBackend,
    SimdFixedFftBackend,
};

#[test]
fn test_capabilities_distinguish_scalar_vs_simd_and_fft_simd() {
    let ir = [1.0f32, 0.5, 0.25];
    let scalar = SimdBackend::scalar();
    assert!(scalar.is_scalar());
    assert!(!scalar.is_simd_active());

    let dcfg = DirectConfig { max_ir_len: 8 };
    let mut dws = [0.0f32; 16];
    let dconv = DirectConvolver::prepare(&ir, dcfg, &mut dws, scalar).unwrap();
    let dcaps = dconv.capabilities();
    assert_eq!(dcaps.engine, EngineKind::Direct);
    assert!(!dcaps.simd_active);
    assert!(!dcaps.fft_is_simd);
    assert_eq!(dcaps.latency_samples, 0);

    let micro = MicroFftBackend::new();
    assert!(!micro.is_simd_accelerated());
    assert_eq!(micro.scratch_len_f32(FftSize::N64), 0);

    let pcfg = PartitionedConfig {
        block_size: 32,
        max_ir_len: 64,
    };
    let preq = PartitionedConvolver::memory_requirements(pcfg, &micro).unwrap();
    let mut pws = vec![0.0f32; preq.total_f32];
    let pconv = PartitionedConvolver::prepare(&ir, pcfg, &mut pws, scalar, micro).unwrap();
    let pcaps = pconv.capabilities();
    assert_eq!(pcaps.engine, EngineKind::Partitioned);
    assert!(!pcaps.simd_active);
    assert!(!pcaps.fft_is_simd);
    assert_eq!(pcaps.block_size, Some(32));
    assert_eq!(pcaps.fft_size, Some(64));

    let fixed_sc = SimdFixedFftBackend::new(scalar);
    assert!(!fixed_sc.is_simd_accelerated());
    assert_eq!(fixed_sc.scratch_len_f32(FftSize::N64), 64);

    let hcfg = HybridConfig {
        head_len: 64,
        tail_block_size: 64,
        max_ir_len: 128,
    };
    let hreq = HybridConvolver::memory_requirements(hcfg, &fixed_sc).unwrap();
    let mut hws = vec![0.0f32; hreq.total_f32];
    let hconv =
        HybridConvolver::prepare_with_tail_spreading(&ir, hcfg, &mut hws, scalar, fixed_sc, true)
            .unwrap();
    let hcaps = hconv.capabilities();
    assert_eq!(hcaps.engine, EngineKind::Hybrid);
    assert!(hconv.tail_spreading());
    assert_eq!(hcaps.latency_samples, 0);
}

#[test]
fn test_checked_arithmetic_overflow_fails_cleanly() {
    let err = DirectConvolver::memory_requirements(DirectConfig {
        max_ir_len: usize::MAX,
    })
    .unwrap_err();
    assert_eq!(err, ConvError::ArithmeticOverflow);

    let fft = MicroFftBackend::new();
    let err = PartitionedConvolver::memory_requirements(
        PartitionedConfig {
            block_size: 64,
            max_ir_len: usize::MAX,
        },
        &fft,
    )
    .unwrap_err();
    assert_eq!(err, ConvError::ArithmeticOverflow);
}
