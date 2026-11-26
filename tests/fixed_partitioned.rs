#![cfg(feature = "fixed-fft")]

use rtconv_simd::fixed::{f64_to_q15, q15_to_f64};
use rtconv_simd::{FixedQ15PartitionedConfig, FixedQ15PartitionedConvolver};

fn ref_convolve_q15_f64(ir: &[i16], input: &[i16]) -> Vec<f64> {
    let mut out = vec![0.0f64; input.len()];
    for n in 0..input.len() {
        let mut acc = 0.0f64;
        let max_k = ir.len().min(n + 1);
        for k in 0..max_k {
            acc += q15_to_f64(ir[k]) * q15_to_f64(input[n - k]);
        }
        out[n] = acc.clamp(-1.0, 32767.0 / 32768.0);
    }
    out
}

#[test]
fn test_fixed_q15_partitioned_fdl_matches_f64_reference_for_32_64_128() {
    let mut seed = 0x004D_5678u64;
    let mut next_f64 = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((seed >> 11) as f64 / ((1u64 << 53) as f64)) * 2.0 - 1.0
    };

    let raw_ir: Vec<f64> = (0..500)
        .map(|k| next_f64() * (-0.008 * k as f64).exp())
        .collect();
    let l1: f64 = raw_ir.iter().map(|v| v.abs()).sum();
    let ir: Vec<i16> = raw_ir.iter().map(|&v| f64_to_q15(v * 0.7 / l1)).collect();
    let input: Vec<i16> = (0..1024).map(|_| f64_to_q15(next_f64() * 0.7)).collect();
    let expected = ref_convolve_q15_f64(&ir, &input);

    for &block_size in &[32usize, 64, 128] {
        let cfg = FixedQ15PartitionedConfig {
            block_size,
            max_ir_len: 512,
        };
        let req = FixedQ15PartitionedConvolver::memory_requirements(cfg).unwrap();
        let mut ws = vec![0i16; req.total_i16];
        let mut conv = FixedQ15PartitionedConvolver::prepare(&ir, cfg, &mut ws).unwrap();

        let mut out1 = vec![0i16; input.len()];
        for (ib, ob) in input
            .chunks_exact(block_size)
            .zip(out1.chunks_exact_mut(block_size))
        {
            conv.process_block(ib, ob).unwrap();
        }
        assert_eq!(conv.saturation_count(), 0);

        let mut sig = 0.0f64;
        let mut err = 0.0f64;
        let mut max_abs = 0.0f64;
        for (&r, &a) in expected.iter().zip(out1.iter()) {
            let af = q15_to_f64(a);
            let d = r - af;
            max_abs = max_abs.max(d.abs());
            sig += r * r;
            err += d * d;
        }
        let snr = 10.0 * (sig / err).log10();
        assert!(
            snr > 50.0 && max_abs < 5e-4,
            "FDL Q15 B={block_size}: snr={snr:.2} dB, max_abs={max_abs:.3e}"
        );

        // Deterministic reset
        conv.reset();
        let mut out2 = vec![0i16; input.len()];
        for (ib, ob) in input
            .chunks_exact(block_size)
            .zip(out2.chunks_exact_mut(block_size))
        {
            conv.process_block(ib, ob).unwrap();
        }
        assert_eq!(out1, out2);
    }
}

#[test]
fn test_fixed_q15_partitioned_fdl_large_blocks_256_512_1024_and_cap() {
    use rtconv_simd::ConvError;

    let mut seed = 0x2048_1111u64;
    let mut next_f64 = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((seed >> 11) as f64 / ((1u64 << 53) as f64)) * 2.0 - 1.0
    };

    let raw_ir: Vec<f64> = (0..1800)
        .map(|k| next_f64() * (-0.004 * k as f64).exp())
        .collect();
    let l1: f64 = raw_ir.iter().map(|v| v.abs()).sum();
    let ir: Vec<i16> = raw_ir.iter().map(|&v| f64_to_q15(v * 0.7 / l1)).collect();
    let input: Vec<i16> = (0..4096).map(|_| f64_to_q15(next_f64() * 0.7)).collect();
    let expected = ref_convolve_q15_f64(&ir, &input);

    for &block_size in &[256usize, 512, 1024] {
        let cfg = FixedQ15PartitionedConfig {
            block_size,
            max_ir_len: 2048,
        };
        let req = FixedQ15PartitionedConvolver::memory_requirements(cfg).unwrap();
        let mut ws = vec![0i16; req.total_i16];
        let mut conv = FixedQ15PartitionedConvolver::prepare(&ir, cfg, &mut ws).unwrap();

        let mut out1 = vec![0i16; input.len()];
        for (ib, ob) in input
            .chunks_exact(block_size)
            .zip(out1.chunks_exact_mut(block_size))
        {
            conv.process_block(ib, ob).unwrap();
        }
        assert_eq!(conv.saturation_count(), 0);

        let mut sig = 0.0f64;
        let mut err = 0.0f64;
        let mut max_abs = 0.0f64;
        for (&r, &a) in expected.iter().zip(out1.iter()) {
            let af = q15_to_f64(a);
            let d = r - af;
            max_abs = max_abs.max(d.abs());
            sig += r * r;
            err += d * d;
        }
        let snr = 10.0 * (sig / err).log10();
        println!(
            "FDL Q15 B={block_size} (N={}): snr={snr:.2} dB max_abs={max_abs:.3e}",
            2 * block_size
        );
        assert!(
            snr > 48.0 && max_abs < 1e-3,
            "FDL Q15 B={block_size}: snr={snr:.2} dB, max_abs={max_abs:.3e}"
        );
    }

    // B = 2048 (FFT = N4096) exceeds Q1.15 MAX_FIXED_FFT = 2048 and must be rejected cleanly
    assert_eq!(
        FixedQ15PartitionedConvolver::memory_requirements(FixedQ15PartitionedConfig {
            block_size: 2048,
            max_ir_len: 4096,
        }),
        Err(ConvError::UnsupportedFftSize { fft_size: 4096 })
    );
}
