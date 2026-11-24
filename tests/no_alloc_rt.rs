//! Standalone (`harness = false`) zero-allocation real-time contract verifier.
//!
//! # Why `harness = false`?
//! Running under Rust's default `libtest` harness spawns a worker thread while the main
//! harness thread remains active (capturing output and managing channels). On macOS,
//! background harness/runtime allocations on another thread intermittently raced with a
//! process-wide `GUARD_ACTIVE` flag. Using `harness = false` gives a single-threaded
//! process with zero `libtest` runner threads, and `GlobalAlloc` uses only lock-free
//! atomics (avoiding TLS recursion observed on Rust 1.89.0).
//!
//! # Limitations
//! This binary verifies that `rtconv_simd` performs zero Rust `GlobalAlloc` allocations
//! or deallocations on the host process during `prepare`, `process_sample`,
//! `process_block`, `reset`, and metadata queries. It does not intercept direct OS
//! syscalls bypassing `GlobalAlloc` (none exist in `#![no_std]` `#![forbid(unsafe_code)]`
//! core code).

#[cfg(feature = "phastft")]
use rtconv_simd::PhastFftBackend;
use rtconv_simd::{
    DirectConfig, DirectConvolver, HybridConfig, HybridConvolver, MicroFftBackend,
    PartitionedConfig, PartitionedConvolver, SimdBackend, SimdFixedFftBackend,
};
use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

struct CountingAllocator;

static GUARD_ACTIVE: AtomicBool = AtomicBool::new(false);
static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
static DEALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);

// SAFETY: Delegates directly to `System` after updating lock-free atomic counters.
// No TLS, no formatting, no locks, and no recursive allocations inside `alloc`/`dealloc`.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if GUARD_ACTIVE.load(Ordering::SeqCst) {
            ALLOC_COUNT.fetch_add(1, Ordering::SeqCst);
        }
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if GUARD_ACTIVE.load(Ordering::SeqCst) {
            DEALLOC_COUNT.fetch_add(1, Ordering::SeqCst);
        }
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn measure_allocs<F: FnOnce()>(f: F) -> (usize, usize) {
    ALLOC_COUNT.store(0, Ordering::SeqCst);
    DEALLOC_COUNT.store(0, Ordering::SeqCst);
    GUARD_ACTIVE.store(true, Ordering::SeqCst);
    f();
    GUARD_ACTIVE.store(false, Ordering::SeqCst);
    (
        ALLOC_COUNT.load(Ordering::SeqCst),
        DEALLOC_COUNT.load(Ordering::SeqCst),
    )
}

fn assert_zero_alloc<F: FnOnce()>(phase: &str, f: F) {
    let (allocs, deallocs) = measure_allocs(f);
    assert_eq!(
        allocs, 0,
        "phase `{phase}`: detected {allocs} heap allocations"
    );
    assert_eq!(
        deallocs, 0,
        "phase `{phase}`: detected {deallocs} heap deallocations"
    );
}

fn verify_positive_control() {
    let (allocs, deallocs) = measure_allocs(|| {
        let boxed = black_box(Box::new([0xA5u64; 16]));
        drop(boxed);
    });
    assert!(
        allocs >= 1 && deallocs >= 1,
        "positive control failed: instrumentation did not detect deliberate Box allocation (allocs={allocs}, deallocs={deallocs})"
    );
}

fn verify_microfft_engines(backend: SimdBackend) {
    let ir = [0.25f32; 512];
    let input = [0.5f32; 128];
    let mut output = [0.0f32; 128];
    let fft = MicroFftBackend::new();

    let d_cfg = DirectConfig { max_ir_len: 512 };
    let mut d_ws = vec![
        0.0f32;
        DirectConvolver::memory_requirements(d_cfg)
            .unwrap()
            .total_f32
    ];
    let p_cfg = PartitionedConfig {
        block_size: 64,
        max_ir_len: 512,
    };
    let mut p_ws = vec![
        0.0f32;
        PartitionedConvolver::memory_requirements(p_cfg, &fft)
            .unwrap()
            .total_f32
    ];
    let h_cfg = HybridConfig {
        head_len: 128,
        tail_block_size: 128,
        max_ir_len: 512,
    };
    let mut h_ws = vec![
        0.0f32;
        HybridConvolver::memory_requirements(h_cfg, &fft)
            .unwrap()
            .total_f32
    ];

    // DirectConvolver: prepare, process_sample, process_block, reset, metadata
    let mut d_conv = None;
    assert_zero_alloc("direct::prepare", || {
        d_conv = Some(DirectConvolver::prepare(&ir, d_cfg, &mut d_ws, backend).unwrap());
    });
    let d_conv = d_conv.as_mut().unwrap();
    assert_zero_alloc("direct::process_sample", || {
        for &s in &input[..64] {
            black_box(d_conv.process_sample(s));
        }
    });
    assert_zero_alloc("direct::process_block", || {
        for _ in 0..16 {
            d_conv
                .process_block(&input[..32], &mut output[..32])
                .unwrap();
        }
    });
    assert_zero_alloc("direct::reset", || d_conv.reset());
    assert_zero_alloc("direct::metadata", || {
        black_box(d_conv.latency_samples());
        black_box(d_conv.capabilities());
    });

    // PartitionedConvolver (MicroFft): prepare, process_block, reset, metadata
    let mut p_conv = None;
    assert_zero_alloc("partitioned_microfft::prepare", || {
        p_conv = Some(PartitionedConvolver::prepare(&ir, p_cfg, &mut p_ws, backend, fft).unwrap());
    });
    let p_conv = p_conv.as_mut().unwrap();
    assert_zero_alloc("partitioned_microfft::process_block", || {
        for _ in 0..16 {
            p_conv
                .process_block(&input[..64], &mut output[..64])
                .unwrap();
        }
    });
    assert_zero_alloc("partitioned_microfft::reset", || p_conv.reset());
    assert_zero_alloc("partitioned_microfft::metadata", || {
        black_box(p_conv.latency_samples());
        black_box(p_conv.capabilities());
    });

    // HybridConvolver (MicroFft): prepare, process_block, reset, metadata
    let mut h_conv = None;
    assert_zero_alloc("hybrid_microfft::prepare", || {
        h_conv = Some(HybridConvolver::prepare(&ir, h_cfg, &mut h_ws, backend, fft).unwrap());
    });
    let h_conv = h_conv.as_mut().unwrap();
    assert_zero_alloc("hybrid_microfft::process_block", || {
        for _ in 0..32 {
            h_conv
                .process_block(&input[..32], &mut output[..32])
                .unwrap();
        }
    });
    assert_zero_alloc("hybrid_microfft::reset", || h_conv.reset());
    assert_zero_alloc("hybrid_microfft::metadata", || {
        black_box(h_conv.latency_samples());
        black_box(h_conv.capabilities());
    });
}

fn verify_simd_fixed_and_tail_spreading(backend: SimdBackend) {
    let ir = [0.25f32; 1024];
    let input = [0.5f32; 128];
    let mut output = [0.0f32; 128];
    let fft = SimdFixedFftBackend::new(backend);

    let p_cfg = PartitionedConfig {
        block_size: 128,
        max_ir_len: 1024,
    };
    let mut p_ws = vec![
        0.0f32;
        PartitionedConvolver::memory_requirements(p_cfg, &fft)
            .unwrap()
            .total_f32
    ];
    let h_cfg = HybridConfig {
        head_len: 128,
        tail_block_size: 128,
        max_ir_len: 1024,
    };
    let mut h_ws = vec![
        0.0f32;
        HybridConvolver::memory_requirements(h_cfg, &fft)
            .unwrap()
            .total_f32
    ];

    let mut p_conv = None;
    assert_zero_alloc("partitioned_simdfixed::prepare", || {
        p_conv = Some(PartitionedConvolver::prepare(&ir, p_cfg, &mut p_ws, backend, fft).unwrap());
    });
    let p_conv = p_conv.as_mut().unwrap();
    assert_zero_alloc("partitioned_simdfixed::process_block", || {
        for _ in 0..8 {
            p_conv.process_block(&input, &mut output).unwrap();
        }
    });
    assert_zero_alloc("partitioned_simdfixed::reset", || p_conv.reset());
    assert_zero_alloc("partitioned_simdfixed::metadata", || {
        black_box(p_conv.latency_samples());
        black_box(p_conv.capabilities());
    });

    let mut h_conv = None;
    assert_zero_alloc("hybrid_simdfixed_spread::prepare", || {
        h_conv = Some(
            HybridConvolver::prepare_with_tail_spreading(&ir, h_cfg, &mut h_ws, backend, fft, true)
                .unwrap(),
        );
    });
    let h_conv = h_conv.as_mut().unwrap();
    assert_zero_alloc("hybrid_simdfixed_spread::process_block", || {
        for _ in 0..32 {
            h_conv
                .process_block(&input[..32], &mut output[..32])
                .unwrap();
        }
    });
    assert_zero_alloc("hybrid_simdfixed_spread::reset", || h_conv.reset());
    assert_zero_alloc("hybrid_simdfixed_spread::metadata", || {
        black_box(h_conv.latency_samples());
        black_box(h_conv.capabilities());
    });
}

#[cfg(feature = "phastft")]
fn verify_phastft_audio_callback(backend: SimdBackend) {
    let ir = [0.25f32; 1024];
    let input = [0.5f32; 128];
    let mut output = [0.0f32; 128];

    // Planner is constructed ONCE outside the RT audio callback.
    let fft = PhastFftBackend::new();
    let h_cfg = HybridConfig {
        head_len: 128,
        tail_block_size: 128,
        max_ir_len: 1024,
    };
    let mut h_ws = vec![
        0.0f32;
        HybridConvolver::memory_requirements(h_cfg, &fft)
            .unwrap()
            .total_f32
    ];
    let mut h_conv =
        HybridConvolver::prepare_with_tail_spreading(&ir, h_cfg, &mut h_ws, backend, fft, true)
            .unwrap();

    assert_zero_alloc("hybrid_phastft_spread::process_block", || {
        for _ in 0..32 {
            h_conv
                .process_block(&input[..32], &mut output[..32])
                .unwrap();
        }
    });
    assert_zero_alloc("hybrid_phastft_spread::reset", || h_conv.reset());
    assert_zero_alloc("hybrid_phastft_spread::metadata", || {
        black_box(h_conv.latency_samples());
        black_box(h_conv.capabilities());
    });
}

#[cfg(feature = "nonuniform")]
fn verify_nonuniform(backend: SimdBackend) {
    use rtconv_simd::{NonUniformConvolver, NonUniformPlan};

    let ir = [0.25f32; 2048];
    let input = [0.5f32; 32];
    let mut output = [0.0f32; 32];
    let fft = SimdFixedFftBackend::new(backend);
    let plan = NonUniformPlan::default_profile(2048, 32).unwrap();
    let req = NonUniformConvolver::memory_requirements(&plan, &fft).unwrap();
    let mut ws = vec![0.0f32; req.total_f32];

    let mut conv = None;
    assert_zero_alloc("nonuniform::prepare", || {
        conv = Some(NonUniformConvolver::prepare(&ir, plan, &mut ws, backend, fft).unwrap());
    });
    let conv = conv.as_mut().unwrap();
    assert_zero_alloc("nonuniform::process_block", || {
        for _ in 0..32 {
            conv.process_block(&input, &mut output).unwrap();
        }
    });
    assert_zero_alloc("nonuniform::reset", || conv.reset());
    assert_zero_alloc("nonuniform::metadata", || {
        black_box(conv.latency_samples());
        black_box(conv.capabilities());
        black_box(conv.deadline_misses());
    });
}

#[cfg(feature = "fixed-q15")]
fn verify_fixed_q15_direct() {
    use rtconv_simd::{FixedQ15DirectConfig, FixedQ15DirectConvolver};

    let ir = [8192i16; 128];
    let input = [4096i16; 32];
    let mut output = [0i16; 32];
    let cfg = FixedQ15DirectConfig { max_ir_len: 128 };
    let req = FixedQ15DirectConvolver::memory_requirements(cfg).unwrap();
    let mut ws = vec![0i16; req.total_i16];

    let mut conv = None;
    assert_zero_alloc("fixed_q15_direct::prepare", || {
        conv = Some(FixedQ15DirectConvolver::prepare(&ir, cfg, &mut ws).unwrap());
    });
    let conv = conv.as_mut().unwrap();
    assert_zero_alloc("fixed_q15_direct::process_sample", || {
        for &x in &input {
            black_box(conv.process_sample(x));
        }
    });
    assert_zero_alloc("fixed_q15_direct::process_block", || {
        for _ in 0..16 {
            conv.process_block(&input, &mut output).unwrap();
        }
    });
    assert_zero_alloc("fixed_q15_direct::reset", || conv.reset());
    assert_zero_alloc("fixed_q15_direct::metadata", || {
        black_box(conv.latency_samples());
        black_box(conv.capabilities());
        black_box(conv.saturation_count());
    });
}

#[cfg(feature = "fixed-fft")]
fn verify_fixed_q15_partitioned() {
    use rtconv_simd::{FixedQ15PartitionedConfig, FixedQ15PartitionedConvolver};

    let ir = [1024i16; 512];
    let input = [4096i16; 64];
    let mut output = [0i16; 64];
    let cfg = FixedQ15PartitionedConfig {
        block_size: 64,
        max_ir_len: 512,
    };
    let req = FixedQ15PartitionedConvolver::memory_requirements(cfg).unwrap();
    let mut ws = vec![0i16; req.total_i16];

    let mut conv = None;
    assert_zero_alloc("fixed_q15_partitioned::prepare", || {
        conv = Some(FixedQ15PartitionedConvolver::prepare(&ir, cfg, &mut ws).unwrap());
    });
    let conv = conv.as_mut().unwrap();
    assert_zero_alloc("fixed_q15_partitioned::process_block", || {
        for _ in 0..16 {
            conv.process_block(&input, &mut output).unwrap();
        }
    });
    assert_zero_alloc("fixed_q15_partitioned::reset", || conv.reset());
    assert_zero_alloc("fixed_q15_partitioned::metadata", || {
        black_box(conv.latency_samples());
        black_box(conv.capabilities());
        black_box(conv.saturation_count());
    });
}

#[cfg(feature = "fixed-nonuniform")]
fn verify_fixed_q15_nonuniform() {
    use rtconv_simd::{FixedQ15NonUniformConvolver, NonUniformPlan};

    let ir = [512i16; 1024];
    let input = [4096i16; 32];
    let mut output = [0i16; 32];
    let plan = NonUniformPlan::default_profile(1024, 32).unwrap();
    let req = FixedQ15NonUniformConvolver::memory_requirements(&plan).unwrap();
    let mut ws = vec![0i16; req.total_i16];

    let mut conv = None;
    assert_zero_alloc("fixed_q15_nonuniform::prepare", || {
        conv = Some(FixedQ15NonUniformConvolver::prepare(&ir, plan, &mut ws).unwrap());
    });
    let conv = conv.as_mut().unwrap();
    assert_zero_alloc("fixed_q15_nonuniform::process_block", || {
        for _ in 0..32 {
            conv.process_block(&input, &mut output).unwrap();
        }
    });
    assert_zero_alloc("fixed_q15_nonuniform::reset", || conv.reset());
    assert_zero_alloc("fixed_q15_nonuniform::metadata", || {
        black_box(conv.latency_samples());
        black_box(conv.capabilities());
        black_box(conv.deadline_misses());
        black_box(conv.saturation_count());
    });
}

#[cfg(feature = "auto")]
fn verify_auto() {
    use rtconv_simd::auto::{
        AutoConvolver, AutoStereoConvolver, workspace_len_f32, workspace_len_q15,
    };

    // 2,000-tap stereo IEEE-float WAV (rePhase "32 bits IEEE stereo" layout), built before arming.
    let taps = 2000usize;
    let mut data = Vec::new();
    for k in 0..taps {
        let v = if k == 1000 {
            0.9f32
        } else {
            1e-3 * ((k % 7) as f32 - 3.0)
        };
        data.extend_from_slice(&v.to_le_bytes());
        data.extend_from_slice(&(-v).to_le_bytes());
    }
    let mut wav = Vec::new();
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(36 + data.len() as u32).to_le_bytes());
    wav.extend_from_slice(b"WAVEfmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    for v in [3u16, 2] {
        wav.extend_from_slice(&v.to_le_bytes());
    }
    for v in [48_000u32, 48_000 * 8] {
        wav.extend_from_slice(&v.to_le_bytes());
    }
    for v in [8u16, 32] {
        wav.extend_from_slice(&v.to_le_bytes());
    }
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&(data.len() as u32).to_le_bytes());
    wav.extend_from_slice(&data);

    let mut wsf = vec![0.0f32; 2 * workspace_len_f32(taps)];
    let mut wsq = vec![0i16; workspace_len_q15(taps)];
    let in_f = [0.25f32; 37];
    let mut out_f = [0.0f32; 37];
    let mut out_r = [0.0f32; 37];
    let in_q = [8192i16; 37];
    let mut out_q = [0i16; 37];
    let mut inter = [4096i16; 64];

    let mut stereo = None;
    assert_zero_alloc("auto::stereo_f32::from_bytes", || {
        stereo = Some(AutoStereoConvolver::from_bytes(&wav, &mut wsf).unwrap());
    });
    let stereo = stereo.as_mut().unwrap();
    assert_zero_alloc("auto::stereo_f32::process", || {
        for _ in 0..64 {
            stereo
                .process_f32(&in_f, &in_f, &mut out_f, &mut out_r)
                .unwrap();
            stereo.process_interleaved_i16(&mut inter).unwrap();
        }
    });
    assert_zero_alloc("auto::stereo_f32::reset", || stereo.reset());

    let mut q = None;
    assert_zero_alloc("auto::q15::from_bytes", || {
        q = Some(AutoConvolver::from_bytes(&wav, 1, &mut wsq).unwrap());
    });
    let q = q.as_mut().unwrap();
    assert_zero_alloc("auto::q15::process", || {
        for _ in 0..64 {
            q.process_i16(&in_q, &mut out_q).unwrap();
            q.process_f32(&in_f, &mut out_f).unwrap();
        }
        black_box(q.info());
        black_box(q.saturation_count());
    });
    assert_zero_alloc("auto::q15::reset", || q.reset());
}

fn verify_large_ffts_f32(backend: SimdBackend) {
    use rtconv_simd::{FftBackend, FftSize};

    let micro = MicroFftBackend::new();
    let simd = SimdFixedFftBackend::new(backend);
    #[cfg(feature = "phastft")]
    let phast = PhastFftBackend::new();

    for &size in &[
        FftSize::N512,
        FftSize::N1024,
        FftSize::N2048,
        FftSize::N4096,
        FftSize::N8192,
        FftSize::N16384,
    ] {
        let n = size.len();
        let bins = size.spectrum_bins();
        let x = vec![0.25f32; n];
        let mut re = vec![0.0f32; bins];
        let mut im = vec![0.0f32; bins];
        let mut out = vec![0.0f32; n];
        let mut sc_micro = vec![0.0f32; micro.scratch_len_f32(size)];
        let mut sc_simd = vec![0.0f32; simd.scratch_len_f32(size)];

        assert_zero_alloc("microfft_large::rfft_irfft", || {
            micro
                .rfft_forward(size, &x, &mut re, &mut im, &mut sc_micro)
                .unwrap();
            micro
                .rfft_inverse(size, &re, &im, &mut out, &mut sc_micro)
                .unwrap();
            black_box(&out);
        });
        assert_zero_alloc("simdfixed_large::rfft_irfft", || {
            simd.rfft_forward(size, &x, &mut re, &mut im, &mut sc_simd)
                .unwrap();
            simd.rfft_inverse(size, &re, &im, &mut out, &mut sc_simd)
                .unwrap();
            black_box(&out);
        });
        #[cfg(feature = "phastft")]
        {
            let mut sc_phast = vec![0.0f32; phast.scratch_len_f32(size)];
            assert_zero_alloc("phastft_large::rfft_irfft", || {
                phast
                    .rfft_forward(size, &x, &mut re, &mut im, &mut sc_phast)
                    .unwrap();
                phast
                    .rfft_inverse(size, &re, &im, &mut out, &mut sc_phast)
                    .unwrap();
                black_box(&out);
            });
        }
    }

    // PartitionedConvolver with B = 8192 (FFT N16384)
    let ir_16k = vec![0.01f32; 16_384];
    let in_8k = vec![0.25f32; 8192];
    let mut out_8k = vec![0.0f32; 8192];
    let p_cfg = PartitionedConfig {
        block_size: 8192,
        max_ir_len: 16_384,
    };
    let mut p_ws = vec![
        0.0f32;
        PartitionedConvolver::memory_requirements(p_cfg, &simd)
            .unwrap()
            .total_f32
    ];
    let mut p_conv = None;
    assert_zero_alloc("partitioned_n16384::prepare", || {
        p_conv =
            Some(PartitionedConvolver::prepare(&ir_16k, p_cfg, &mut p_ws, backend, simd).unwrap());
    });
    let p_conv = p_conv.as_mut().unwrap();
    assert_zero_alloc("partitioned_n16384::process_block", || {
        p_conv.process_block(&in_8k, &mut out_8k).unwrap();
        p_conv.process_block(&in_8k, &mut out_8k).unwrap();
    });
    assert_zero_alloc("partitioned_n16384::reset", || p_conv.reset());

    // NonUniformConvolver with large stages up to B = 8192 (FFT N16384)
    #[cfg(feature = "nonuniform")]
    {
        use rtconv_simd::{
            NonUniformConfig, NonUniformConvolver, NonUniformPlan, NonUniformStageConfig,
        };
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
            max_ir_len: 16_384,
            host_block_max: 64,
            stages: &stages,
        })
        .unwrap();
        let req = NonUniformConvolver::memory_requirements(&plan, &simd).unwrap();
        let mut ws = vec![0.0f32; req.total_f32];
        let in_64 = [0.25f32; 64];
        let mut out_64 = [0.0f32; 64];
        let mut nu = None;
        assert_zero_alloc("nonuniform_n16384::prepare", || {
            nu = Some(NonUniformConvolver::prepare(&ir_16k, plan, &mut ws, backend, simd).unwrap());
        });
        let nu = nu.as_mut().unwrap();
        assert_zero_alloc("nonuniform_n16384::process_block", || {
            for _ in 0..130 {
                nu.process_block(&in_64, &mut out_64).unwrap();
            }
        });
        assert_zero_alloc("nonuniform_n16384::reset", || nu.reset());
    }
}

#[cfg(feature = "fixed-fft")]
fn verify_large_ffts_q15() {
    use rtconv_simd::FftSize;
    use rtconv_simd::fixed::{
        fixed_fft_scratch_len_i32, irfft_q15_bfp_with_scratch, rfft_q15_bfp_with_scratch,
    };
    use rtconv_simd::{FixedQ15PartitionedConfig, FixedQ15PartitionedConvolver};

    for &size in &[FftSize::N512, FftSize::N1024, FftSize::N2048] {
        let n = size.len();
        let bins = size.spectrum_bins();
        let x = vec![4096i16; n];
        let mut re = vec![0i16; bins];
        let mut im = vec![0i16; bins];
        let mut out = vec![0i16; n];
        let mut sc = vec![0i32; fixed_fft_scratch_len_i32(size)];
        assert_zero_alloc("fixed_fft_large::rfft_irfft", || {
            let exp = rfft_q15_bfp_with_scratch(size, &x, &mut re, &mut im, &mut sc).unwrap();
            let sats = irfft_q15_bfp_with_scratch(size, &re, &im, exp, &mut out, &mut sc).unwrap();
            black_box((sats, &out));
        });
    }

    let ir = vec![64i16; 2048];
    let input = vec![4096i16; 1024];
    let mut output = vec![0i16; 1024];
    let cfg = FixedQ15PartitionedConfig {
        block_size: 1024,
        max_ir_len: 2048,
    };
    let req = FixedQ15PartitionedConvolver::memory_requirements(cfg).unwrap();
    let mut ws = vec![0i16; req.total_i16];
    let mut conv = None;
    assert_zero_alloc("fixed_q15_partitioned_n2048::prepare", || {
        conv = Some(FixedQ15PartitionedConvolver::prepare(&ir, cfg, &mut ws).unwrap());
    });
    let conv = conv.as_mut().unwrap();
    assert_zero_alloc("fixed_q15_partitioned_n2048::process_block", || {
        conv.process_block(&input, &mut output).unwrap();
        conv.process_block(&input, &mut output).unwrap();
    });
    assert_zero_alloc("fixed_q15_partitioned_n2048::reset", || conv.reset());

    #[cfg(feature = "fixed-nonuniform")]
    {
        use rtconv_simd::{
            FixedQ15NonUniformConvolver, NonUniformConfig, NonUniformPlan, NonUniformStageConfig,
        };
        let stages = [
            NonUniformStageConfig {
                ir_offset: 0,
                ir_len: 128,
                block_size: 1,
            },
            NonUniformStageConfig {
                ir_offset: 128,
                ir_len: 128,
                block_size: 128,
            },
            NonUniformStageConfig {
                ir_offset: 256,
                ir_len: 256,
                block_size: 256,
            },
            NonUniformStageConfig {
                ir_offset: 512,
                ir_len: 512,
                block_size: 512,
            },
            NonUniformStageConfig {
                ir_offset: 1024,
                ir_len: 1024,
                block_size: 1024,
            },
        ];
        let plan = NonUniformPlan::validate(NonUniformConfig {
            max_ir_len: 2048,
            host_block_max: 64,
            stages: &stages,
        })
        .unwrap();
        let req = FixedQ15NonUniformConvolver::memory_requirements(&plan).unwrap();
        let mut ws = vec![0i16; req.total_i16];
        let in_64 = [4096i16; 64];
        let mut out_64 = [0i16; 64];
        let mut nu = None;
        assert_zero_alloc("fixed_q15_nonuniform_n2048::prepare", || {
            nu = Some(FixedQ15NonUniformConvolver::prepare(&ir, plan, &mut ws).unwrap());
        });
        let nu = nu.as_mut().unwrap();
        assert_zero_alloc("fixed_q15_nonuniform_n2048::process_block", || {
            for _ in 0..34 {
                nu.process_block(&in_64, &mut out_64).unwrap();
            }
        });
        assert_zero_alloc("fixed_q15_nonuniform_n2048::reset", || nu.reset());
    }
}

#[cfg(feature = "nonuniform")]
fn verify_nupc_optimizer(backend: SimdBackend) {
    use rtconv_simd::nonuniform::{
        BlockModePolicy, ClearancePolicy, CostModel, NonUniformConvolver, NonUniformPlan,
        OptimizerScratchCell, block_mode_scratch_len, optimize_block_mode, optimizer_scratch_len,
        preset_esp32_c6_q15, preset_esp32_p4_f32,
    };

    let model = CostModel::for_backend_f32(backend);
    let ir_len = 16_384usize;
    let mut sc = vec![OptimizerScratchCell::ZERO; optimizer_scratch_len(ir_len, 8192)];
    let mut opt_plan = None;
    assert_zero_alloc("nupc_optimizer::optimize_sample_time", || {
        let (plan, rep) =
            NonUniformPlan::optimize(ir_len, 64, &model, ClearancePolicy::Strategy2, &mut sc)
                .unwrap();
        black_box(rep);
        opt_plan = Some(plan);
    });
    let opt_plan = opt_plan.unwrap();

    // Block-mode Viterbi DP (Wefers §6.12: 88,200 taps, B=128)
    let w_model = CostModel::wefers_2015_test_system();
    let mut bsc = vec![OptimizerScratchCell::ZERO; block_mode_scratch_len(88_200, 128, 8192)];
    assert_zero_alloc("nupc_optimizer::optimize_block_mode", || {
        let part = optimize_block_mode(
            88_200,
            128,
            8192,
            &w_model,
            BlockModePolicy::Strategy2TableA25,
            &mut bsc,
        )
        .unwrap();
        black_box(part);
    });

    // Zero-scratch MCU presets
    assert_zero_alloc("nupc_optimizer::mcu_presets", || {
        black_box(preset_esp32_p4_f32(16_384, 64).unwrap());
        black_box(preset_esp32_c6_q15(16_384, 64).unwrap());
    });

    // Prepare + process optimized NUPC plan with zero heap allocation
    let fft = SimdFixedFftBackend::new(backend);
    let req = NonUniformConvolver::memory_requirements(&opt_plan, &fft).unwrap();
    let mut ws = vec![0.0f32; req.total_f32];
    let ir = vec![0.01f32; ir_len];
    let inp = [0.25f32; 64];
    let mut out = [0.0f32; 64];
    let mut nu = None;
    assert_zero_alloc("nupc_optimizer::convolver_prepare", || {
        nu = Some(NonUniformConvolver::prepare(&ir, opt_plan, &mut ws, backend, fft).unwrap());
    });
    let nu = nu.as_mut().unwrap();
    assert_zero_alloc("nupc_optimizer::convolver_process", || {
        for _ in 0..32 {
            nu.process_block(&inp, &mut out).unwrap();
        }
    });
    assert_zero_alloc("nupc_optimizer::convolver_reset", || nu.reset());

    // AutoConvolver selecting cost-optimized AutoEngine::NonUniform (16,384 taps, in-place DP)
    #[cfg(feature = "auto")]
    {
        use rtconv_simd::auto::{AutoConvolver, AutoEngine, engine_len_f32};
        let mut auto_ws = vec![0.0f32; engine_len_f32(ir_len)];
        let mut aconv = None;
        assert_zero_alloc("auto::nonuniform_16k::prepare_f32", || {
            aconv = Some(AutoConvolver::prepare_f32(&ir, &mut auto_ws).unwrap());
        });
        let aconv = aconv.as_mut().unwrap();
        assert_eq!(aconv.info().engine, AutoEngine::NonUniform);
        assert_zero_alloc("auto::nonuniform_16k::process_f32", || {
            for _ in 0..32 {
                aconv.process_f32(&inp, &mut out).unwrap();
            }
        });
        assert_zero_alloc("auto::nonuniform_16k::reset", || aconv.reset());
    }
}

fn main() {
    // Warm up runtime environment and CPU feature detection before arming instrumentation.
    let backend = SimdBackend::detect_or_baseline();
    println!(
        "no_alloc_rt: standalone harness starting (simd_active={})",
        backend.is_simd_active()
    );

    verify_positive_control();
    verify_microfft_engines(backend);
    verify_simd_fixed_and_tail_spreading(backend);
    verify_large_ffts_f32(backend);
    #[cfg(feature = "phastft")]
    verify_phastft_audio_callback(backend);
    #[cfg(feature = "nonuniform")]
    verify_nonuniform(backend);
    #[cfg(feature = "nonuniform")]
    verify_nupc_optimizer(backend);
    #[cfg(feature = "fixed-q15")]
    verify_fixed_q15_direct();
    #[cfg(feature = "fixed-fft")]
    verify_fixed_q15_partitioned();
    #[cfg(feature = "fixed-fft")]
    verify_large_ffts_q15();
    #[cfg(feature = "fixed-nonuniform")]
    verify_fixed_q15_nonuniform();
    #[cfg(feature = "auto")]
    verify_auto();

    println!("no_alloc_rt: ALL ZERO-ALLOCATION AND POSITIVE-CONTROL CHECKS PASSED");
}
