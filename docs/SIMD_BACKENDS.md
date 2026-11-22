# SIMD & FFT Backends (`docs/SIMD_BACKENDS.md`)

## Explicit Distinction of SIMD Scope

`Capabilities` exposes two independent flags so integrators never confuse CMAC vectorization with FFT vectorization:

- `Capabilities::simd_active`: `true` when `DirectConvolver` dot products and `PartitionedConvolver` FDL CMAC execute vector SIMD instructions (`false` for `SimdBackend::scalar()` or `fearless_simd::Fallback`).
- `Capabilities::fft_is_simd`: `true` only when the FFT/IFFT transform itself uses SIMD (`false` for `MicroFftBackend` and `SimdFixedFftBackend(Scalar/Fallback)`; `true` for `SimdFixedFftBackend(SSE2/AVX2/NEON/WASM)` and `PhastFftBackend`).

## Backend Architecture

1. **`MicroFftBackend`** ([`microfft 0.6.0`](https://docs.rs/microfft/0.6.0)): `#![no_std]` scalar radix-2 RFFT (`N/2` CFFT) and full `N`-point complex IFFT. Reliable baseline; zero external scratch.
2. **`SimdFixedFftBackend`** ([`fearless_simd 1.2.0`](https://docs.rs/fearless_simd/1.2.0)): `#![no_std]` split-complex SoA DIT FFT running `N/2`-point CFFT for **both** R2C and C2R (halving inverse work vs `microfft`), fused multiplication-free radix-4 stages 0–1, explicit `f32x4` (`half = 4`) and `S::f32s` (`half >= 8`) SIMD butterflies, and `const` twiddle tables.
3. **`PhastFftBackend`** ([`phastft 0.4.1`](https://docs.rs/phastft), `--features phastft`): Desktop-only SIMD R2C/C2R with pre-initialized `PlannerR2c32` and `default-features = false` (no `rayon`). Internally uses `fearless_simd 0.5.0` side-by-side with the core crate's `fearless_simd 1.2.0`.

## Disassembly Verification (`PROVEN`, `HOST_ONLY`)

Inspecting `cargo rustc --lib --release -- --emit=asm` on `x86_64-pc-windows-gnu` confirms **`96` hardware FMA (`vfmadd`/`vfmsub`/`vfnmadd`), `527` 256-bit `ymm`, and `5,251` 128-bit `xmm`** references emitted directly inside `librtconv_simd` (up from `FMA=0, YMM=0` in the WO-001 baseline).
