//! Allocation-free real-time contract verification using a custom `GlobalAlloc`.
//!
//! Confirms zero heap allocations and zero deallocations during `prepare()`,
//! `process_sample()`, `process_block()`, `reset()`, `latency_samples()`, and
//! `capabilities()` across all three engines.

use rtconv_simd::{
    DirectConfig, DirectConvolver, HybridConfig, HybridConvolver, MicroFftBackend,
    PartitionedConfig, PartitionedConvolver, SimdBackend,
};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

struct CountingAllocator;

static GUARD_ACTIVE: AtomicBool = AtomicBool::new(false);
static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
static DEALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);

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

#[test]
fn test_zero_heap_allocations_in_prepare_and_audio_callback() {
    let ir = [0.25f32; 512];
    let input = [0.5f32; 128];
    let mut output = [0.0f32; 128];

    // Pre-allocate caller-owned workspaces outside the RT region.
    let fft = MicroFftBackend::new();
    let backend = SimdBackend::detect();

    let d_cfg = DirectConfig { max_ir_len: 512 };
    let mut d_ws = vec![0.0f32; DirectConvolver::memory_requirements(d_cfg).unwrap().total_f32];

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

    // Activate strict allocation guard.
    ALLOC_COUNT.store(0, Ordering::SeqCst);
    DEALLOC_COUNT.store(0, Ordering::SeqCst);
    GUARD_ACTIVE.store(true, Ordering::SeqCst);

    // 1. DirectConvolver
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

    // 2. PartitionedConvolver
    {
        let mut conv = PartitionedConvolver::prepare(&ir, p_cfg, &mut p_ws, backend, fft).unwrap();
        for _ in 0..16 {
            conv.process_block(&input[..64], &mut output[..64]).unwrap();
        }
        conv.reset();
        let _ = conv.latency_samples();
        let _ = conv.capabilities();
    }

    // 3. HybridConvolver
    {
        let mut conv = HybridConvolver::prepare(&ir, h_cfg, &mut h_ws, backend, fft).unwrap();
        for _ in 0..32 {
            conv.process_block(&input[..32], &mut output[..32]).unwrap();
        }
        conv.reset();
        let _ = conv.latency_samples();
        let _ = conv.capabilities();
    }

    GUARD_ACTIVE.store(false, Ordering::SeqCst);
    let allocs = ALLOC_COUNT.load(Ordering::SeqCst);
    let deallocs = DEALLOC_COUNT.load(Ordering::SeqCst);

    assert_eq!(allocs, 0, "detected {allocs} heap allocations in RT path");
    assert_eq!(
        deallocs, 0,
        "detected {deallocs} heap deallocations in RT path"
    );
}
