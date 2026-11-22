//! Allocation-free real-time contract verification using a serialized `GlobalAlloc`.
//!
//! Uses a static `Mutex<()>` to serialize guarded sections so parallel test threads
//! cannot contaminate the global allocation counters. For maximum determinism in CI,
//! this integration binary can also be executed with `--test-threads=1`.

#[cfg(feature = "phastft")]
use rtconv_simd::PhastFftBackend;
use rtconv_simd::{
    DirectConfig, DirectConvolver, HybridConfig, HybridConvolver, MicroFftBackend,
    PartitionedConfig, PartitionedConvolver, SimdBackend, SimdFixedFftBackend,
};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

struct CountingAllocator;

static TEST_LOCK: Mutex<()> = Mutex::new(());
static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
static DEALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    static IN_GUARD: Cell<bool> = const { Cell::new(false) };
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if IN_GUARD.try_with(Cell::get).unwrap_or(false) {
            ALLOC_COUNT.fetch_add(1, Ordering::SeqCst);
        }
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if IN_GUARD.try_with(Cell::get).unwrap_or(false) {
            DEALLOC_COUNT.fetch_add(1, Ordering::SeqCst);
        }
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn assert_zero_alloc<F: FnOnce()>(label: &str, f: F) {
    let _lock = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    ALLOC_COUNT.store(0, Ordering::SeqCst);
    DEALLOC_COUNT.store(0, Ordering::SeqCst);
    IN_GUARD.with(|flag| flag.set(true));
    f();
    IN_GUARD.with(|flag| flag.set(false));
    let allocs = ALLOC_COUNT.load(Ordering::SeqCst);
    let deallocs = DEALLOC_COUNT.load(Ordering::SeqCst);
    assert_eq!(allocs, 0, "{label}: detected {allocs} heap allocations");
    assert_eq!(
        deallocs, 0,
        "{label}: detected {deallocs} heap deallocations"
    );
}

#[test]
fn test_zero_heap_allocations_in_prepare_and_audio_callback() {
    let ir = [0.25f32; 512];
    let input = [0.5f32; 128];
    let mut output = [0.0f32; 128];

    let fft = MicroFftBackend::new();
    let backend = SimdBackend::detect_or_baseline();

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

    assert_zero_alloc("MicroFft + Direct/Partitioned/Hybrid", || {
        {
            let mut conv = DirectConvolver::prepare(&ir, d_cfg, &mut d_ws, backend).unwrap();
            let _ = conv.process_sample(0.1);
            for _ in 0..16 {
                conv.process_block(&input[..32], &mut output[..32]).unwrap();
            }
            conv.reset();
            let _ = conv.latency_samples();
            let _ = conv.capabilities();
        }
        {
            let mut conv =
                PartitionedConvolver::prepare(&ir, p_cfg, &mut p_ws, backend, fft).unwrap();
            for _ in 0..16 {
                conv.process_block(&input[..64], &mut output[..64]).unwrap();
            }
            conv.reset();
            let _ = conv.latency_samples();
            let _ = conv.capabilities();
        }
        {
            let mut conv = HybridConvolver::prepare(&ir, h_cfg, &mut h_ws, backend, fft).unwrap();
            for _ in 0..32 {
                conv.process_block(&input[..32], &mut output[..32]).unwrap();
            }
            conv.reset();
            let _ = conv.latency_samples();
            let _ = conv.capabilities();
        }
    });
}

#[test]
fn test_zero_heap_allocations_simd_fixed_and_tail_spreading() {
    let ir = [0.25f32; 1024];
    let input = [0.5f32; 128];
    let mut output = [0.0f32; 128];

    let backend = SimdBackend::detect_or_baseline();
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

    assert_zero_alloc("SimdFixedFft + Hybrid(SPREAD)", || {
        {
            let mut pconv =
                PartitionedConvolver::prepare(&ir, p_cfg, &mut p_ws, backend, fft).unwrap();
            for _ in 0..8 {
                pconv.process_block(&input, &mut output).unwrap();
            }
            pconv.reset();
        }
        {
            let mut hconv = HybridConvolver::prepare_with_tail_spreading(
                &ir, h_cfg, &mut h_ws, backend, fft, true,
            )
            .unwrap();
            for _ in 0..32 {
                hconv
                    .process_block(&input[..32], &mut output[..32])
                    .unwrap();
            }
            hconv.reset();
        }
    });
}

#[cfg(feature = "phastft")]
#[test]
fn test_zero_heap_allocations_phastft_audio_callback() {
    let ir = [0.25f32; 1024];
    let input = [0.5f32; 128];
    let mut output = [0.0f32; 128];

    let backend = SimdBackend::detect_or_baseline();
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
    let mut hconv =
        HybridConvolver::prepare_with_tail_spreading(&ir, h_cfg, &mut h_ws, backend, fft, true)
            .unwrap();

    assert_zero_alloc("PhastFftBackend audio callback", || {
        for _ in 0..32 {
            hconv
                .process_block(&input[..32], &mut output[..32])
                .unwrap();
        }
        hconv.reset();
    });
}
