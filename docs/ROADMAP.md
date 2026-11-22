# Roadmap (`docs/ROADMAP.md`)

Planned post-`0.1.0` enhancements (intentionally excluded from `0.1.0` to keep the initial release surface small, audited, and stable):

1. **Immutable Shareable IR Spectra (`&[f32]` / Flash-Mapped ROM)**: Allow precomputed partition spectra to reside in read-only Flash/ROM and be shared across multiple mono/stereo channel instances, cutting RAM nearly in half for fixed speaker/cabinet IRs.
2. **FDL RAM Reduction**: Evaluate compact FDL ring-buffer layouts and optional `f16`/block-floating-point tail storage for memory-constrained MCUs.
3. **Non-Uniform Partitioning (UPOLS / Gardner)**: Multi-stage tail block sizes (e.g., 128 → 512 → 2048) for very long acoustic room IRs (`> 16,384` taps).
4. **Multi-Channel & True-Stereo Helpers**: Zero-allocation ergonomic wrappers over independent channel instances.
5. **Glitch-Free IR Crossfade**: Pre-allocated dual-engine crossfade helper for live IR switching without heap allocation.
6. **Hardware-in-the-Loop Qualification (`DEVICE_VERIFIED`)**: Execute real-time WCET and ISR jitter measurements on ARM Cortex-M7, ARM64 NEON, and RISC-V embedded boards.
