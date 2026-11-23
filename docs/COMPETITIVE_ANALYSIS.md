# Competitive Analysis (`docs/COMPETITIVE_ANALYSIS.md`)

Objective architectural comparison of pure-Rust real-time audio convolution crates based on their public repositories (no unbenchmarked performance superiority claims).

## Reference Projects Studied

- [`neodsp/fft-convolver`](https://github.com/neodsp/fft-convolver): Pure-Rust port of `HiFi-LoFi/FFTConvolver` (MIT), providing uniform `FFTConvolver` and `TwoStageFFTConvolver` (small head block + large tail block, overlap-add).
- [`slundi/rt-fft-convolver`](https://github.com/slundi/rt-fft-convolver): Pure-Rust real-time convolution crate (MIT) powered by `rustfft`, targeting guitar cabinet IRs and reverb with Mono, Stereo, and True Stereo (4-path) support and anti-denormal FTZ logic.

## Feature & Architecture Matrix

| Dimension | `rtconv-simd` (this crate) | [`neodsp/fft-convolver`](https://github.com/neodsp/fft-convolver) | [`slundi/rt-fft-convolver`](https://github.com/slundi/rt-fft-convolver) |
| :--- | :--- | :--- | :--- |
| **`#![no_std]` without `alloc`** | **Yes** (`default = []`, zero heap in `prepare` & `process`) | No (`std` / heap allocation in `init()`; separate `fft-convolver-nostd` repo exists) | No (`std` / `rustfft`) |
| **Caller-managed workspace** | **Yes** (`&mut [f32]` + checked `memory_requirements`) | No (internal `Vec` allocated in `init()`) | No (internal allocation at init) |
| **Allocation-free `process()`** | **Yes** (`GlobalAlloc` verified across all 3 FFT backends) | Yes (`process()`, `set_response()`, `reset()`) | Yes (zero heap in processing loop) |
| **FFT SIMD** | **Yes** (`SimdFixedFftBackend` `no_std` via `fearless_simd 1.2.0`; optional `PhastFftBackend`) | Via FFT dependency | Via `rustfft` SIMD |
| **Arbitrary block sizes** | `DirectConvolver` & `HybridConvolver`: arbitrary; `PartitionedConvolver`: `32/64/128` | Arbitrary via internal buffering | Partitioned / direct primitives |
| **Non-uniform partitioning** | **Yes** (`NonUniformConvolver`, up to 8 zero-latency stages; `f32` and Q1.15/BFP) | Yes (`TwoStageFFTConvolver`) | Roadmap / uniform engine |
| **FPU-less fixed-point engines** | **Yes** (Q1.15 direct, BFP FDL, BFP NUPC) | No | No |
| **FIR file loading** | **Yes** (`no_std` WAV / raw / text, zero-copy) | No (caller supplies `&[f32]`) | No |
| **Automatic engine selection** | **Yes** (`AutoConvolver`: FPU domain + cost model) | No | No |
| **Time-distributed tail spreading** | **Yes** (`HybridConvolver::set_tail_spreading(true)`, 0 added latency) | No | No |
| **Stereo / True Stereo** | `AutoStereoConvolver` (planar + interleaved); true stereo via 4 instances | Mono per instance | Built-in Mono, Stereo, True Stereo |
| **IR crossfade** | Roadmap | `set_response()` | Not primary focus |
| **Primary niche** | Embedded/MCU & deterministic RT DSP with explicit static RAM & `#![forbid(unsafe_code)]` | Desktop/plugin real-time convolution (including very long 65k+ IRs via two-stage) | Desktop guitar cab/reverb with built-in stereo routing |
