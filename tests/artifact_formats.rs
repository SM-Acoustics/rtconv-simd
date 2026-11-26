//! FIR artifact decoding: every WAV encoding (PCM 8/16/24/32, IEEE 32/64, EXTENSIBLE),
//! mono/stereo, raw `.bin` float auto-detection, explicit raw integers, text/CSV variants, and
//! negative cases. Fixtures are synthesized byte-for-byte from the RIFF specification.

use rtconv_simd::artifact::{ArtifactError, Container, FirArtifact, SampleEncoding};

const PCM: u16 = 1;
const FLOAT: u16 = 3;

fn coeffs(len: usize, ch: usize) -> Vec<f64> {
    // Decaying, sign-alternating, channel-distinct taps within (-1, 1).
    (0..len)
        .map(|k| {
            let env = (-(k as f64) * 0.01).exp();
            let s = if k % 2 == 0 { 1.0 } else { -1.0 };
            s * env * (0.9 - 0.3 * ch as f64) * (0.5 + 0.5 * ((k as f64) * 0.37).cos())
        })
        .collect()
}

fn encode(v: f64, enc: SampleEncoding, out: &mut Vec<u8>) {
    match enc {
        SampleEncoding::PcmU8 => {
            out.push(((v * 128.0).round().clamp(-128.0, 127.0) as i32 + 128) as u8)
        }
        SampleEncoding::PcmI16 => out.extend_from_slice(
            &((v * 32768.0).round().clamp(-32768.0, 32767.0) as i16).to_le_bytes(),
        ),
        SampleEncoding::PcmI24 => {
            let q = (v * 8_388_608.0).round().clamp(-8_388_608.0, 8_388_607.0) as i32;
            out.extend_from_slice(&q.to_le_bytes()[..3]);
        }
        SampleEncoding::PcmI32 => out.extend_from_slice(
            &((v * 2_147_483_648.0)
                .round()
                .clamp(-2_147_483_648.0, 2_147_483_647.0) as i32)
                .to_le_bytes(),
        ),
        SampleEncoding::F32 => out.extend_from_slice(&(v as f32).to_le_bytes()),
        SampleEncoding::F64 => out.extend_from_slice(&v.to_le_bytes()),
    }
}

fn interleave(chans: &[Vec<f64>], enc: SampleEncoding) -> Vec<u8> {
    let mut out = Vec::new();
    for f in 0..chans[0].len() {
        for c in chans {
            encode(c[f], enc, &mut out);
        }
    }
    out
}

fn tag_bits(enc: SampleEncoding) -> (u16, u16) {
    match enc {
        SampleEncoding::PcmU8 => (PCM, 8),
        SampleEncoding::PcmI16 => (PCM, 16),
        SampleEncoding::PcmI24 => (PCM, 24),
        SampleEncoding::PcmI32 => (PCM, 32),
        SampleEncoding::F32 => (FLOAT, 32),
        SampleEncoding::F64 => (FLOAT, 64),
    }
}

fn wav(chans: &[Vec<f64>], enc: SampleEncoding, extensible: bool, junk_chunk: bool) -> Vec<u8> {
    let (tag, bits) = tag_bits(enc);
    let nch = chans.len() as u16;
    let align = nch * bits / 8;
    let data = interleave(chans, enc);
    let mut fmt = Vec::new();
    fmt.extend_from_slice(&(if extensible { 0xFFFEu16 } else { tag }).to_le_bytes());
    fmt.extend_from_slice(&nch.to_le_bytes());
    fmt.extend_from_slice(&48_000u32.to_le_bytes());
    fmt.extend_from_slice(&(48_000u32 * align as u32).to_le_bytes());
    fmt.extend_from_slice(&align.to_le_bytes());
    fmt.extend_from_slice(&bits.to_le_bytes());
    if extensible {
        fmt.extend_from_slice(&22u16.to_le_bytes());
        fmt.extend_from_slice(&bits.to_le_bytes());
        fmt.extend_from_slice(&(if nch == 2 { 3u32 } else { 4u32 }).to_le_bytes());
        fmt.extend_from_slice(&tag.to_le_bytes());
        fmt.extend_from_slice(&[
            0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71,
        ]);
    }
    let mut body = Vec::new();
    body.extend_from_slice(b"WAVE");
    body.extend_from_slice(b"fmt ");
    body.extend_from_slice(&(fmt.len() as u32).to_le_bytes());
    body.extend_from_slice(&fmt);
    if junk_chunk {
        // Odd-sized LIST chunk exercises the RIFF pad byte rule.
        body.extend_from_slice(b"LIST");
        body.extend_from_slice(&5u32.to_le_bytes());
        body.extend_from_slice(b"INFOx\0");
    }
    body.extend_from_slice(b"data");
    body.extend_from_slice(&(data.len() as u32).to_le_bytes());
    body.extend_from_slice(&data);
    let mut out = Vec::new();
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(body.len() as u32).to_le_bytes());
    out.extend_from_slice(&body);
    out
}

fn tolerance(enc: SampleEncoding) -> f64 {
    match enc {
        SampleEncoding::PcmU8 => 1.0 / 128.0,
        SampleEncoding::PcmI16 => 1.0 / 32768.0,
        SampleEncoding::PcmI24 => 1.0 / 8_388_608.0,
        SampleEncoding::PcmI32 | SampleEncoding::F32 => 1e-7,
        SampleEncoding::F64 => 1e-12,
    }
}

const ALL: [SampleEncoding; 6] = [
    SampleEncoding::PcmU8,
    SampleEncoding::PcmI16,
    SampleEncoding::PcmI24,
    SampleEncoding::PcmI32,
    SampleEncoding::F32,
    SampleEncoding::F64,
];

#[test]
fn wav_every_encoding_mono_and_stereo_plain_and_extensible() {
    for enc in ALL {
        for nch in [1usize, 2] {
            for extensible in [false, true] {
                let chans: Vec<Vec<f64>> = (0..nch).map(|c| coeffs(257, c)).collect();
                let bytes = wav(&chans, enc, extensible, true);
                let art = FirArtifact::parse(&bytes).unwrap();
                assert_eq!(art.container(), Container::Wav);
                assert_eq!(art.encoding(), Some(enc));
                assert_eq!(art.channels(), nch);
                assert_eq!(art.frames(), 257);
                assert_eq!(art.sample_rate(), Some(48_000));
                for (c, expected) in chans.iter().enumerate() {
                    let mut out = vec![0.0f32; 257];
                    assert_eq!(art.decode_f32(c, &mut out).unwrap(), 257);
                    for (k, (&e, &o)) in expected.iter().zip(&out).enumerate() {
                        assert!(
                            (e - o as f64).abs() <= tolerance(enc) * 1.01 + 1e-7,
                            "{enc:?} ch={c} ext={extensible} k={k}: {e} vs {o}"
                        );
                    }
                    let mut q = vec![0i16; 257];
                    assert_eq!(art.decode_q15(c, 0, &mut q).unwrap(), 0);
                    for (&e, &o) in expected.iter().zip(&q) {
                        let err = (e - o as f64 / 32768.0).abs();
                        assert!(
                            err <= tolerance(enc).max(0.5 / 32768.0) + 1e-9,
                            "{enc:?} q15 err {err}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn pcm_to_q15_is_integer_exact() {
    // i16 PCM must round-trip bit-exactly; i24/i32 must round to nearest.
    let taps: Vec<i16> = vec![i16::MIN, -1, 0, 1, 12345, i16::MAX];
    let mut data = Vec::new();
    for &t in &taps {
        data.extend_from_slice(&t.to_le_bytes());
    }
    let art = FirArtifact::parse_raw(&data, SampleEncoding::PcmI16, 1).unwrap();
    let mut q = [0i16; 6];
    art.decode_q15(0, 0, &mut q).unwrap();
    assert_eq!(&q[..], &taps[..]);

    // 24-bit: 0x000080 (=128/2^23) rounds to 1 LSB of Q15 (= 256/2^23); 0x00007F rounds to 0.
    let raw24 = [0x80, 0x00, 0x00, 0x7F, 0x00, 0x00, 0x80, 0xFF, 0xFF];
    let art = FirArtifact::parse_raw(&raw24, SampleEncoding::PcmI24, 1).unwrap();
    let mut q = [0i16; 3];
    art.decode_q15(0, 0, &mut q).unwrap();
    assert_eq!(q, [1, 0, -1]);
}

#[test]
fn raw_float_auto_detection_f32_and_f64() {
    let taps = coeffs(1024, 0);
    let f32_bytes = interleave(std::slice::from_ref(&taps), SampleEncoding::F32);
    let art = FirArtifact::parse(&f32_bytes).unwrap();
    assert_eq!(art.container(), Container::Raw);
    assert_eq!(art.encoding(), Some(SampleEncoding::F32));
    assert_eq!(art.frames(), 1024);

    let f64_bytes = interleave(std::slice::from_ref(&taps), SampleEncoding::F64);
    let art = FirArtifact::parse(&f64_bytes).unwrap();
    assert_eq!(art.encoding(), Some(SampleEncoding::F64));
    assert_eq!(art.frames(), 1024);
    let mut out = vec![0.0f32; 1024];
    art.decode_f32(0, &mut out).unwrap();
    assert!((out[10] as f64 - taps[10]).abs() < 1e-7);

    // Exact-valued f64 Dirac: low mantissa words are zero.
    let mut dirac = vec![0.0f64; 64];
    dirac[3] = 1.0;
    let bytes = interleave(&[dirac], SampleEncoding::F64);
    let art = FirArtifact::parse(&bytes).unwrap();
    assert_eq!(art.encoding(), Some(SampleEncoding::F64));
    assert_eq!(art.frames(), 64);

    // Random bytes are rejected instead of silently decoded as garbage.
    let mut x = 0x1234_5678_9ABC_DEF0u64;
    let junk: Vec<u8> = (0..4096)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x >> 32) as u8 | 0x80
        })
        .collect();
    assert_eq!(
        FirArtifact::parse(&junk).unwrap_err(),
        ArtifactError::UnrecognizedBinary
    );
}

#[test]
fn raw_explicit_stereo_and_misalignment() {
    let chans = vec![coeffs(33, 0), coeffs(33, 1)];
    let bytes = interleave(&chans, SampleEncoding::PcmI32);
    let art = FirArtifact::parse_raw(&bytes, SampleEncoding::PcmI32, 2).unwrap();
    assert_eq!((art.channels(), art.frames()), (2, 33));
    let mut right = vec![0.0f32; 33];
    art.decode_f32(1, &mut right).unwrap();
    assert!((right[5] as f64 - chans[1][5]).abs() < 1e-7);
    assert!(matches!(
        FirArtifact::parse_raw(&bytes[..bytes.len() - 1], SampleEncoding::PcmI32, 2),
        Err(ArtifactError::MisalignedRaw { .. })
    ));
    assert_eq!(
        FirArtifact::parse_raw(&bytes, SampleEncoding::F32, 0).unwrap_err(),
        ArtifactError::InvalidChannelCount
    );
}

#[test]
fn text_variants() {
    let cases: &[(&str, usize, usize, f32)] = &[
        ("0.5\n-0.25\n0.125\n", 1, 3, -0.25),
        (
            "# rePhase export\n* header\n0.5\r\n-0.25\r\n1e-3\r\n",
            1,
            3,
            -0.25,
        ),
        ("0.5,0.1\n-0.25,0.2\n", 2, 2, -0.25),
        ("0.5, 0.1\n-0.25, 0.2\n", 2, 2, -0.25),
        ("0,5;0,1\n-0,25;0,2\n", 2, 2, -0.25),
        ("0,5\t0,1\n-0,25\t0,2\n", 2, 2, -0.25),
        ("0,5\n-0,25\n", 1, 2, -0.25),
    ];
    for &(text, ch, frames, second) in cases {
        let art = FirArtifact::parse(text.as_bytes()).unwrap();
        assert_eq!(art.container(), Container::Text, "{text:?}");
        assert_eq!((art.channels(), art.frames()), (ch, frames), "{text:?}");
        let mut out = vec![0.0f32; frames];
        art.decode_f32(0, &mut out).unwrap();
        assert_eq!(out[1], second, "{text:?}");
    }
    assert_eq!(
        FirArtifact::parse(b"0.5\n0.1 0.2\n").unwrap_err(),
        ArtifactError::InvalidText { line: 2 }
    );
    assert_eq!(
        FirArtifact::parse(b"0.5\n0.1x\n").unwrap_err(),
        ArtifactError::InvalidText { line: 2 }
    );
}

#[test]
fn q15_headroom_for_gain_above_unity() {
    let taps = vec![0.1f64, 1.7, -0.4];
    let bytes = interleave(std::slice::from_ref(&taps), SampleEncoding::F32);
    let art = FirArtifact::parse_raw(&bytes, SampleEncoding::F32, 1).unwrap();
    let shift = art.q15_headroom_shift(0).unwrap();
    assert_eq!(shift, 1);
    let mut q = [0i16; 3];
    assert_eq!(art.decode_q15(0, shift, &mut q).unwrap(), 0);
    assert!(((q[1] as f64) / 32768.0 * 2.0 - 1.7).abs() < 1e-4);
    // Without headroom the 1.7 tap saturates and is reported.
    assert_eq!(art.decode_q15(0, 0, &mut q).unwrap(), 1);
}

#[test]
fn negative_cases() {
    assert_eq!(FirArtifact::parse(&[]).unwrap_err(), ArtifactError::Empty);
    let good = wav(&[coeffs(16, 0)], SampleEncoding::F32, false, false);
    // Truncated before the fmt chunk body.
    assert!(FirArtifact::parse_wav(&good[..20]).is_err());
    // Unsupported 12-bit PCM.
    let mut bad = good.clone();
    bad[34] = 12;
    assert!(matches!(
        FirArtifact::parse(&bad),
        Err(ArtifactError::UnsupportedWavFormat { .. })
    ));
    // NaN sample is reported with its frame index.
    let mut nan = Vec::new();
    for v in [0.5f32, f32::NAN, 0.25] {
        nan.extend_from_slice(&v.to_le_bytes());
    }
    let art = FirArtifact::parse_raw(&nan, SampleEncoding::F32, 1).unwrap();
    let mut out = [0.0f32; 3];
    assert_eq!(
        art.decode_f32(0, &mut out).unwrap_err(),
        ArtifactError::NonFiniteSample { frame: 1 }
    );
    let art = FirArtifact::parse(&good).unwrap();
    let mut small = [0.0f32; 4];
    assert!(matches!(
        art.decode_f32(0, &mut small),
        Err(ArtifactError::OutputTooSmall { .. })
    ));
    assert!(matches!(
        art.decode_f32(1, &mut [0.0f32; 16]),
        Err(ArtifactError::ChannelOutOfRange { .. })
    ));
}
