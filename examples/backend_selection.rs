//! Demonstrates explicit SIMD and FFT backend selection and capability inspection.

#[cfg(feature = "phastft")]
use rtconv_simd::PhastFftBackend;
use rtconv_simd::{FftBackend, FftSize, MicroFftBackend, SimdBackend, SimdFixedFftBackend};

fn main() {
    let scalar = SimdBackend::scalar();
    let baseline = SimdBackend::baseline();
    let auto = SimdBackend::detect_or_baseline();

    println!(
        "SIMD backends: scalar={} (active={}), baseline={} (active={}), auto={} (active={})",
        scalar.name(),
        scalar.is_simd_active(),
        baseline.name(),
        baseline.is_simd_active(),
        auto.name(),
        auto.is_simd_active()
    );

    let micro = MicroFftBackend::new();
    let fixed = SimdFixedFftBackend::new(auto);
    println!(
        "FFT backends:\n  - {} (simd={}, scratch_256={} B)\n  - {} (simd={}, scratch_256={} B)",
        micro.name(),
        micro.is_simd_accelerated(),
        micro.scratch_len_f32(FftSize::N256) * 4,
        fixed.name(),
        fixed.is_simd_accelerated(),
        fixed.scratch_len_f32(FftSize::N256) * 4,
    );

    #[cfg(feature = "phastft")]
    {
        let phast = PhastFftBackend::new();
        println!(
            "  - {} (simd={}, scratch_256={} B)",
            phast.name(),
            phast.is_simd_accelerated(),
            phast.scratch_len_f32(FftSize::N256) * 4,
        );
    }
}
