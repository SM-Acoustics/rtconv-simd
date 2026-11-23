#![cfg(feature = "fixed-fft")]

use rtconv_simd::FftSize;
use rtconv_simd::fixed::{f64_to_q15, irfft_q15_bfp, q15_to_f64, rfft_q15_bfp};
use std::f64::consts::PI;

fn ref_dft_f64(x: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = x.len();
    let bins = n / 2 + 1;
    let mut re = vec![0.0f64; bins];
    let mut im = vec![0.0f64; bins];
    for k in 0..bins {
        for (t, &xt) in x.iter().enumerate() {
            let ang = 2.0 * PI * (k as f64) * (t as f64) / (n as f64);
            re[k] += xt * ang.cos();
            im[k] -= xt * ang.sin();
        }
    }
    (re, im)
}

fn snr_db(reference: &[f64], actual: &[f64]) -> f64 {
    let mut sig = 0.0f64;
    let mut err = 0.0f64;
    for (&r, &a) in reference.iter().zip(actual.iter()) {
        sig += r * r;
        let d = r - a;
        err += d * d;
    }
    if err == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (sig / err).log10()
    }
}

#[test]
fn test_fixed_fft_roundtrip_and_spectrum_accuracy_64_128_256() {
    for &size in &[FftSize::N64, FftSize::N128, FftSize::N256] {
        let n = size.len();
        let bins = size.spectrum_bins();
        let mut spec_re = vec![0i16; bins];
        let mut spec_im = vec![0i16; bins];
        let mut roundtrip = vec![0i16; n];

        // 1. Unit impulse, DC, Nyquist, integer-bin sine, off-bin sine, deterministic noise
        let mut impulse = vec![0i16; n];
        impulse[0] = f64_to_q15(0.5);
        let dc = vec![f64_to_q15(0.5); n];
        let nyquist: Vec<i16> = (0..n)
            .map(|i| {
                if i % 2 == 0 {
                    f64_to_q15(0.5)
                } else {
                    f64_to_q15(-0.5)
                }
            })
            .collect();
        let bin_sine: Vec<i16> = (0..n)
            .map(|i| f64_to_q15(0.6 * (2.0 * PI * 5.0 * i as f64 / n as f64).sin()))
            .collect();
        let off_bin: Vec<i16> = (0..n)
            .map(|i| f64_to_q15(0.6 * (2.0 * PI * 7.37 * i as f64 / n as f64).cos()))
            .collect();
        let mut seed = 0x004D_1234u64 ^ (n as u64);
        let noise: Vec<i16> = (0..n)
            .map(|_| {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                let f = ((seed >> 11) as f64 / ((1u64 << 53) as f64)) * 1.4 - 0.7;
                f64_to_q15(f)
            })
            .collect();

        for (label, sig) in [
            ("impulse", &impulse),
            ("dc", &dc),
            ("nyquist", &nyquist),
            ("bin_sine", &bin_sine),
            ("off_bin", &off_bin),
            ("noise", &noise),
        ] {
            let exp = rfft_q15_bfp(size, sig, &mut spec_re, &mut spec_im).unwrap();
            assert_eq!(spec_im[0], 0);
            assert_eq!(spec_im[bins - 1], 0);

            // Check forward spectrum against f64 DFT oracle
            let sig_f64: Vec<f64> = sig.iter().map(|&v| q15_to_f64(v)).collect();
            let (ref_re, ref_im) = ref_dft_f64(&sig_f64);
            let scale = 2.0f64.powi(exp as i32);
            let got_re: Vec<f64> = spec_re.iter().map(|&v| q15_to_f64(v) * scale).collect();
            let got_im: Vec<f64> = spec_im.iter().map(|&v| q15_to_f64(v) * scale).collect();
            let mut ref_all = ref_re;
            ref_all.extend_from_slice(&ref_im);
            let mut got_all = got_re;
            got_all.extend_from_slice(&got_im);
            let spec_snr = snr_db(&ref_all, &got_all);
            assert!(
                spec_snr > 62.0,
                "N={n} {label} forward spectrum SNR={spec_snr:.2} dB < 62 dB"
            );

            // Check IFFT roundtrip
            let sats = irfft_q15_bfp(size, &spec_re, &spec_im, exp, &mut roundtrip).unwrap();
            assert_eq!(sats, 0, "N={n} {label} unexpected saturation");
            let rt_f64: Vec<f64> = roundtrip.iter().map(|&v| q15_to_f64(v)).collect();
            let rt_snr = snr_db(&sig_f64, &rt_f64);
            assert!(
                rt_snr > 60.0,
                "N={n} {label} roundtrip SNR={rt_snr:.2} dB < 60 dB"
            );
        }
    }
}
