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
    #[cfg(feature = "phastft")]
    verify_phastft_audio_callback(backend);
    #[cfg(feature = "nonuniform")]
    verify_nonuniform(backend);
    #[cfg(feature = "fixed-q15")]
    verify_fixed_q15_direct();
    #[cfg(feature = "fixed-fft")]
    verify_fixed_q15_partitioned();

    println!("no_alloc_rt: ALL ZERO-ALLOCATION AND POSITIVE-CONTROL CHECKS PASSED");
}
