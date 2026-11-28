//! Zero-copy, `#![no_std]`, allocation-free decoding of FIR filter artifacts.
//!
//! Accepts the file layouts produced by FIR design tools such as rePhase, REW, or a DAW
//! export, directly from a byte slice (for example `include_bytes!("filter.wav")`, a flash
//! partition, or a buffer read from an SD card):
//!
//! | Container | Encodings | Channels |
//! | :--- | :--- | :--- |
//! | RIFF/WAVE (`WAVE_FORMAT_PCM`, `WAVE_FORMAT_IEEE_FLOAT`, `WAVE_FORMAT_EXTENSIBLE`) | PCM `u8`, `i16`, `i24`, `i32`; IEEE `f32`, `f64` | any (interleaved) |
//! | Headerless raw binary (`.bin`, `.raw`, `.dbl`) | `f32`, `f64` (auto-detected) or explicit `i16`/`i24`/`i32`/`f32`/`f64` | explicit (default mono) |
//! | Text (`.txt`, `.csv`) | decimal numbers, one frame per line, columns = channels | any |
//!
//! All multi-byte values are little-endian (the RIFF and x86/ARM/RISC-V convention).
//!
//! Nothing is copied or allocated by [`FirArtifact::parse`]; samples are decoded on demand into
//! caller-provided `&mut [f32]` (floating-point engines), `&mut [i16]` (Q1.15) or `&mut [i32]`
//! (Q1.31).
//!
//! ## Integer-only fixed-point path
//! Parsing ([`FirArtifact::parse`] and friends), [`FirArtifact::q15_headroom_shift`],
//! [`FirArtifact::decode_q15`] and [`FirArtifact::decode_q31`] execute **no floating-point
//! operation** for any container: IEEE samples are decomposed from their bits and text is
//! scaled with exact integer arithmetic (`crate::numconv`), so FPU-less targets link neither
//! soft-float routines nor `core`'s decimal-to-float tables for them. Only
//! [`FirArtifact::decode_f32`] and [`FirArtifact::peak`] produce floating-point values; they are
//! linked only when called.
//!
//! Rounding into Q1.15 / Q1.31: IEEE and text sources round to nearest, ties to even
//! (IEEE Std 754-1985 §4.1, §5.4; ABNT NBR 5891:2014 §2.2–§2.4), exactly for text of any length.
//! Integer PCM sources keep their established round-half-away-from-zero rule (exact whenever
//! the destination is at least as wide as the source and no headroom shift is applied).
//!
//! ## Detection rules ([`FirArtifact::parse`])
//! 1. `RIFF....WAVE` header → WAV (format tag and `SubFormat` GUID are validated).
//! 2. Printable ASCII containing numbers → text.
//! 3. Otherwise → headerless little-endian IEEE float, choosing `f32` or `f64` by a plausibility
//!    score (finite values with audio-range magnitudes). Raw integer files carry no type
//!    information and must be opened with [`FirArtifact::parse_raw`].
//!
//! rePhase's export dialog is not formally documented; the container/encoding matrix above
//! covers the formats it is known to write (IEEE 32/64-bit and PCM 16/24/32-bit WAV, mono or
//! stereo, raw 32/64-bit float `.bin`, and text), and every path is tested against synthetic
//! fixtures generated in `tests/artifact_formats.rs`.

use core::fmt;

use crate::numconv::{Decimal, Ieee, ieee_exceeds_q15_headroom, ieee_to_fixed_rne};

/// Errors produced while parsing or decoding a FIR artifact.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArtifactError {
    /// The byte slice is empty or ends before a required header field or chunk.
    Truncated,
    /// The RIFF/WAVE structure is malformed (missing `fmt ` or `data`, bad block alignment).
    MalformedWav,
    /// The WAV format tag / bit depth combination is not supported.
    UnsupportedWavFormat {
        /// `wFormatTag` (or the `SubFormat` code for `WAVE_FORMAT_EXTENSIBLE`).
        format_tag: u16,
        /// `wBitsPerSample`.
        bits_per_sample: u16,
    },
    /// The artifact contains no samples.
    Empty,
    /// A headerless binary could not be classified as `f32` or `f64` with confidence.
    UnrecognizedBinary,
    /// The byte length is not a multiple of the frame size for the requested raw layout.
    MisalignedRaw {
        /// Bytes per frame (`channels * bytes_per_sample`).
        frame_bytes: usize,
        /// Total byte length provided.
        len: usize,
    },
    /// A text line could not be parsed, or has a different column count than the first line.
    InvalidText {
        /// 1-based line number.
        line: usize,
    },
    /// Requested channel does not exist.
    ChannelOutOfRange {
        /// Requested channel index.
        channel: usize,
        /// Channels present in the artifact.
        channels: usize,
    },
    /// Output slice is shorter than the number of frames.
    OutputTooSmall {
        /// Frames that must be written.
        required: usize,
        /// Length of the provided output slice.
        provided: usize,
    },
    /// A floating-point sample is `NaN` or infinite.
    NonFiniteSample {
        /// Frame index of the first non-finite sample.
        frame: usize,
    },
    /// Zero channels requested for a raw layout.
    InvalidChannelCount,
}

impl fmt::Display for ArtifactError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated => write!(f, "FIR artifact is truncated"),
            Self::MalformedWav => write!(f, "malformed RIFF/WAVE structure"),
            Self::UnsupportedWavFormat {
                format_tag,
                bits_per_sample,
            } => write!(
                f,
                "unsupported WAV format tag {format_tag:#06x} with {bits_per_sample} bits per sample"
            ),
            Self::Empty => write!(f, "FIR artifact contains no samples"),
            Self::UnrecognizedBinary => {
                write!(
                    f,
                    "headerless binary is not recognizable as f32 or f64 data"
                )
            }
            Self::MisalignedRaw { frame_bytes, len } => write!(
                f,
                "raw length {len} is not a multiple of the {frame_bytes}-byte frame size"
            ),
            Self::InvalidText { line } => write!(f, "invalid numeric text at line {line}"),
            Self::ChannelOutOfRange { channel, channels } => {
                write!(f, "channel {channel} out of range ({channels} channels)")
            }
            Self::OutputTooSmall { required, provided } => write!(
                f,
                "output too small: {required} frames required, {provided} provided"
            ),
            Self::NonFiniteSample { frame } => write!(f, "non-finite sample at frame {frame}"),
            Self::InvalidChannelCount => write!(f, "channel count must be at least 1"),
        }
    }
}

impl core::error::Error for ArtifactError {}

/// Sample encoding of a binary FIR artifact (always little-endian).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SampleEncoding {
    /// Unsigned 8-bit PCM with offset `128` (WAV only).
    PcmU8,
    /// Signed 16-bit PCM (`Q1.15`).
    PcmI16,
    /// Signed 24-bit PCM packed in 3 bytes (`Q1.23`).
    PcmI24,
    /// Signed 32-bit PCM (`Q1.31`).
    PcmI32,
    /// IEEE-754 binary32.
    F32,
    /// IEEE-754 binary64.
    F64,
}

impl SampleEncoding {
    /// Bytes per sample.
    #[inline]
    pub const fn bytes(self) -> usize {
        match self {
            Self::PcmU8 => 1,
            Self::PcmI16 => 2,
            Self::PcmI24 => 3,
            Self::PcmI32 | Self::F32 => 4,
            Self::F64 => 8,
        }
    }

    /// `true` for IEEE floating-point encodings.
    #[inline]
    pub const fn is_float(self) -> bool {
        matches!(self, Self::F32 | Self::F64)
    }
}

/// Container the artifact was parsed from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Container {
    /// RIFF/WAVE file.
    Wav,
    /// Headerless binary.
    Raw,
    /// Numeric text.
    Text,
}

#[derive(Clone, Copy, Debug)]
enum Payload<'a> {
    Binary(&'a [u8], SampleEncoding),
    Text(&'a str),
}

/// A parsed, borrowed view of a FIR artifact. Parsing never copies or allocates.
#[derive(Clone, Copy, Debug)]
pub struct FirArtifact<'a> {
    payload: Payload<'a>,
    container: Container,
    channels: usize,
    frames: usize,
    sample_rate: Option<u32>,
}

/// One undecoded sample, kept exact in its source representation so the fixed-point paths
/// need no FPU.
#[derive(Clone, Copy)]
enum Sample<'a> {
    /// Signed integer with `frac_bits` fractional bits (value = `v / 2^frac_bits`).
    Int { v: i32, frac_bits: u32 },
    /// IEEE 754 binary32 bit pattern.
    F32(u32),
    /// IEEE 754 binary64 bit pattern.
    F64(u64),
    /// Validated decimal text token.
    Text(Decimal<'a>),
}

const KSDATAFORMAT_TAIL: [u8; 14] = [
    0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71,
];

#[inline]
fn le_u16(b: &[u8], at: usize) -> Result<u16, ArtifactError> {
    match b.get(at..at + 2) {
        Some(s) => Ok(u16::from_le_bytes([s[0], s[1]])),
        None => Err(ArtifactError::Truncated),
    }
}

#[inline]
fn le_u32(b: &[u8], at: usize) -> Result<u32, ArtifactError> {
    match b.get(at..at + 4) {
        Some(s) => Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]])),
        None => Err(ArtifactError::Truncated),
    }
}

#[inline]
fn read_sample(bytes: &[u8], enc: SampleEncoding) -> Sample<'static> {
    match enc {
        SampleEncoding::PcmU8 => Sample::Int {
            v: bytes[0] as i32 - 128,
            frac_bits: 7,
        },
        SampleEncoding::PcmI16 => Sample::Int {
            v: i16::from_le_bytes([bytes[0], bytes[1]]) as i32,
            frac_bits: 15,
        },
        SampleEncoding::PcmI24 => Sample::Int {
            // Sign-extend the 24-bit value through the top byte of an i32.
            v: i32::from_le_bytes([0, bytes[0], bytes[1], bytes[2]]) >> 8,
            frac_bits: 23,
        },
        SampleEncoding::PcmI32 => Sample::Int {
            v: i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
            frac_bits: 31,
        },
        SampleEncoding::F32 => {
            Sample::F32(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
        }
        SampleEncoding::F64 => Sample::F64(u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ])),
    }
}

/// Floating-point value of a decimal token through `core`'s correctly rounded parser
/// (the only place text reaches floating point; used by [`FirArtifact::decode_f32`] and
/// [`FirArtifact::peak`]). The decimal comma is mapped to `.` in a 64-byte stack buffer; the
/// grammar of [`Decimal::parse`] guarantees such tokens fit and that `core` accepts the result,
/// so the `NaN` fallback (reported as a non-finite sample) is unreachable for parsed artifacts.
fn decimal_to_f64(d: &Decimal<'_>) -> f64 {
    let tok = d.token();
    let mut buf = [0u8; 64];
    let text = if tok.contains(&b',') {
        let n = tok.len().min(buf.len());
        for (dst, &src) in buf.iter_mut().zip(&tok[..n]) {
            *dst = if src == b',' { b'.' } else { src };
        }
        core::str::from_utf8(&buf[..n])
    } else {
        core::str::from_utf8(tok)
    };
    match text.map(str::parse::<f64>) {
        Ok(Ok(v)) => v,
        _ => f64::NAN,
    }
}

impl Sample<'_> {
    /// Value as `f64` (floating-point APIs only).
    #[inline]
    fn to_f64(self) -> f64 {
        match self {
            Self::Int { v, frac_bits } => v as f64 / (1u64 << frac_bits) as f64,
            Self::F32(b) => f32::from_bits(b) as f64,
            Self::F64(b) => f64::from_bits(b),
            Self::Text(d) => decimal_to_f64(&d),
        }
    }

    /// `round(x * 2^(frac - shift))` clamped to `[min, max]` with a saturation flag, integer
    /// arithmetic only. IEEE and text sources round to nearest even; integer PCM keeps the
    /// established half-away-from-zero rule. Non-finite values return `None`.
    #[inline]
    fn to_fixed(self, frac: u32, shift: u32, min: i64, max: i64) -> Option<(i64, bool)> {
        let shift = shift.min(31);
        let k = frac as i32 - shift as i32;
        match self {
            Self::Int { v, frac_bits } => {
                let v = v as i64;
                let q = if frac_bits <= frac {
                    round_shift_i64(v << (frac - frac_bits), shift)
                } else {
                    round_shift_i64(v, frac_bits - frac + shift)
                };
                let c = q.clamp(min, max);
                Some((c, c != q))
            }
            Self::F32(b) => ieee_to_fixed_rne(Ieee::from_f32_bits(b), k, min, max),
            Self::F64(b) => ieee_to_fixed_rne(Ieee::from_f64_bits(b), k, min, max),
            Self::Text(d) => d.to_fixed_rne(k, min, max),
        }
    }

    /// `true` for NaN and infinities (integer test on the encoding).
    #[inline]
    fn is_non_finite(self) -> bool {
        match self {
            Self::Int { .. } => false,
            Self::F32(b) => matches!(Ieee::from_f32_bits(b), Ieee::NonFinite { .. }),
            Self::F64(b) => matches!(Ieee::from_f64_bits(b), Ieee::NonFinite { .. }),
            Self::Text(d) => d.is_non_finite(),
        }
    }

    /// `true` when `|x| * 2^-shift` exceeds the Q1.15 maximum `32767 / 32768` (exact).
    #[inline]
    fn exceeds_q15_headroom(self, shift: u32) -> bool {
        match self {
            Self::Int { v, frac_bits } => {
                // |v| * 2^(15 - frac_bits - shift) > 32767, exact in i64.
                let mag = (v as i64).unsigned_abs();
                let e = 15 - frac_bits as i32 - shift as i32;
                let (fl, inexact) = crate::numconv::dyadic_floor(mag, e);
                fl > 32767 || (fl == 32767 && inexact)
            }
            Self::F32(b) => ieee_exceeds_q15_headroom(Ieee::from_f32_bits(b), shift),
            Self::F64(b) => ieee_exceeds_q15_headroom(Ieee::from_f64_bits(b), shift),
            Self::Text(d) => d.exceeds_q15_headroom(shift),
        }
    }
}

/// Rounds `v / 2^shift` to nearest (ties away from zero) using integer arithmetic only.
#[inline]
fn round_shift_i64(v: i64, shift: u32) -> i64 {
    if shift == 0 {
        return v;
    }
    let half = 1i64 << (shift - 1);
    if v >= 0 {
        (v + half) >> shift
    } else {
        -((-v + half) >> shift)
    }
}

// ------------------------------------------------------------------------------------------
// Text helpers
// ------------------------------------------------------------------------------------------

#[inline]
fn is_data_line(line: &str) -> bool {
    match line.trim_start().as_bytes().first() {
        Some(c) => c.is_ascii_digit() || matches!(c, b'+' | b'-' | b'.'),
        None => false,
    }
}

/// Splits a text line into numeric tokens.
///
/// Columns are separated by whitespace, `;`, or tabs. A `,` separates columns when the line
/// also contains `.` (CSV with decimal points); otherwise it is read as a decimal comma
/// (`0,125;-0,5` or `0,125 -0,5`). An integer-only CSV line such as `1,2` is therefore read as
/// the single value `1.2`.
#[inline]
fn for_each_token<'t, F: FnMut(&'t str) -> bool>(line: &'t str, mut f: F) {
    // With '.' present, numbers use decimal points, so ',' can only be a column separator.
    let comma_is_sep = line.contains('.');
    for tok in line.split(|c: char| c.is_whitespace() || c == ';' || (comma_is_sep && c == ',')) {
        if tok.is_empty() {
            continue;
        }
        if !f(tok) {
            return;
        }
    }
}

fn looks_like_text(bytes: &[u8]) -> bool {
    !bytes.is_empty()
        && bytes
            .iter()
            .all(|&b| b == b'\n' || b == b'\r' || b == b'\t' || (0x20..0x7F).contains(&b))
        && bytes.iter().any(|b| b.is_ascii_digit())
}

// ------------------------------------------------------------------------------------------
// Raw float classification
// ------------------------------------------------------------------------------------------

/// Smallest `f32` whose value is `>= 1e-15` (as `f64`): the plausibility floor of
/// [`raw_float_score`], expressed as a bit pattern so classification needs no FPU.
const F32_PLAUSIBLE_MIN_BITS: u32 = {
    let t = 1e-15f64 as f32;
    if (t as f64) < 1e-15 {
        t.to_bits() + 1
    } else {
        t.to_bits()
    }
};
/// `1e4` is exact in `f32`.
const F32_PLAUSIBLE_MAX_BITS: u32 = 1e4f32.to_bits();
const F64_PLAUSIBLE_MIN_BITS: u64 = 1e-15f64.to_bits();
const F64_PLAUSIBLE_MAX_BITS: u64 = 1e4f64.to_bits();

/// Counts samples that are implausible for a FIR filter under `enc`, plus non-zero samples.
///
/// A non-zero sample is implausible when it is non-finite or its magnitude lies outside
/// `[1e-15, 1e4]`. Positive IEEE values are ordered like their bit patterns, so the test is an
/// integer comparison of `|x|`'s bits against the bounds' bits (identical to comparing the
/// values in `f64`).
fn raw_float_score(bytes: &[u8], enc: SampleEncoding) -> (usize, usize) {
    let w = enc.bytes();
    let mut implausible = 0usize;
    let mut nonzero = 0usize;
    for chunk in bytes.chunks_exact(w) {
        let (zero, plausible) = match read_sample(chunk, enc) {
            Sample::F32(b) => {
                let a = b & 0x7FFF_FFFF;
                (
                    a == 0,
                    (F32_PLAUSIBLE_MIN_BITS..=F32_PLAUSIBLE_MAX_BITS).contains(&a),
                )
            }
            Sample::F64(b) => {
                let a = b & 0x7FFF_FFFF_FFFF_FFFF;
                (
                    a == 0,
                    (F64_PLAUSIBLE_MIN_BITS..=F64_PLAUSIBLE_MAX_BITS).contains(&a),
                )
            }
            // Only called with float encodings.
            _ => (false, false),
        };
        if zero {
            continue;
        }
        nonzero += 1;
        if !plausible {
            implausible += 1;
        }
    }
    (implausible, nonzero)
}

/// `true` when, read as `f32`, every even-indexed word is zero while odd words are not —
/// the signature of `f64` data whose low mantissa words are zero (e.g. a Dirac impulse).
fn f64_low_words_zero(bytes: &[u8]) -> bool {
    let mut any_odd = false;
    for (i, chunk) in bytes.as_chunks::<4>().0.iter().enumerate() {
        let nz = chunk.iter().any(|&b| b != 0);
        if i % 2 == 0 && nz {
            return false;
        }
        any_odd |= i % 2 == 1 && nz;
    }
    any_odd
}

impl<'a> FirArtifact<'a> {
    /// Detects the container automatically (WAV → text → raw float) and returns a view.
    pub fn parse(bytes: &'a [u8]) -> Result<Self, ArtifactError> {
        if bytes.is_empty() {
            return Err(ArtifactError::Empty);
        }
        if bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WAVE" {
            return Self::parse_wav(bytes);
        }
        if looks_like_text(bytes)
            && let Ok(text) = core::str::from_utf8(bytes)
        {
            return Self::parse_text(text);
        }
        Self::parse_raw_float_auto(bytes)
    }

    /// Parses a RIFF/WAVE file (`WAVE_FORMAT_PCM`, `WAVE_FORMAT_IEEE_FLOAT`, or
    /// `WAVE_FORMAT_EXTENSIBLE` with the PCM / IEEE_FLOAT `SubFormat` GUIDs).
    pub fn parse_wav(bytes: &'a [u8]) -> Result<Self, ArtifactError> {
        if bytes.len() < 12 {
            return Err(ArtifactError::Truncated);
        }
        if &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
            return Err(ArtifactError::MalformedWav);
        }
        // Chunk walk. A RIFF size larger than the slice is tolerated (streamed writers).
        let riff_end = (le_u32(bytes, 4)? as usize)
            .saturating_add(8)
            .min(bytes.len());
        let mut pos = 12usize;
        let mut fmt: Option<(u16, u16, u32, u16, u16)> = None; // tag, ch, rate, align, bits
        let mut data: Option<&'a [u8]> = None;
        while pos + 8 <= riff_end {
            let id = &bytes[pos..pos + 4];
            let size = le_u32(bytes, pos + 4)? as usize;
            let body_start = pos + 8;
            let body_end = body_start.saturating_add(size).min(riff_end);
            let body = &bytes[body_start..body_end];
            if id == b"fmt " {
                if body.len() < 16 {
                    return Err(ArtifactError::Truncated);
                }
                let mut tag = le_u16(body, 0)?;
                let channels = le_u16(body, 2)?;
                let rate = le_u32(body, 4)?;
                let align = le_u16(body, 12)?;
                let bits = le_u16(body, 14)?;
                if tag == 0xFFFE {
                    if body.len() < 40 {
                        return Err(ArtifactError::Truncated);
                    }
                    let sub = le_u16(body, 24)?;
                    if body[26..40] != KSDATAFORMAT_TAIL {
                        return Err(ArtifactError::UnsupportedWavFormat {
                            format_tag: 0xFFFE,
                            bits_per_sample: bits,
                        });
                    }
                    tag = sub;
                }
                fmt = Some((tag, channels, rate, align, bits));
            } else if id == b"data" {
                data = Some(body);
            }
            // Chunks are word-aligned (pad byte after odd sizes).
            pos = body_start.saturating_add(size).saturating_add(size & 1);
        }

        let (tag, channels, rate, align, bits) = fmt.ok_or(ArtifactError::MalformedWav)?;
        let data = data.ok_or(ArtifactError::MalformedWav)?;
        let encoding = match (tag, bits) {
            (1, 8) => SampleEncoding::PcmU8,
            (1, 16) => SampleEncoding::PcmI16,
            (1, 24) => SampleEncoding::PcmI24,
            (1, 32) => SampleEncoding::PcmI32,
            (3, 32) => SampleEncoding::F32,
            (3, 64) => SampleEncoding::F64,
            _ => {
                return Err(ArtifactError::UnsupportedWavFormat {
                    format_tag: tag,
                    bits_per_sample: bits,
                });
            }
        };
        let channels = channels as usize;
        if channels == 0 || align as usize != channels * encoding.bytes() {
            return Err(ArtifactError::MalformedWav);
        }
        let frames = data.len() / align as usize;
        if frames == 0 {
            return Err(ArtifactError::Empty);
        }
        Ok(Self {
            payload: Payload::Binary(&data[..frames * align as usize], encoding),
            container: Container::Wav,
            channels,
            frames,
            sample_rate: Some(rate),
        })
    }

    /// Parses a headerless little-endian binary with an explicit encoding and interleaved
    /// channel count (`1` for the mono `.bin` files written by rePhase).
    pub fn parse_raw(
        bytes: &'a [u8],
        encoding: SampleEncoding,
        channels: usize,
    ) -> Result<Self, ArtifactError> {
        if channels == 0 {
            return Err(ArtifactError::InvalidChannelCount);
        }
        if bytes.is_empty() {
            return Err(ArtifactError::Empty);
        }
        let frame_bytes = channels * encoding.bytes();
        if !bytes.len().is_multiple_of(frame_bytes) {
            return Err(ArtifactError::MisalignedRaw {
                frame_bytes,
                len: bytes.len(),
            });
        }
        Ok(Self {
            payload: Payload::Binary(bytes, encoding),
            container: Container::Raw,
            channels,
            frames: bytes.len() / frame_bytes,
            sample_rate: None,
        })
    }

    /// Classifies a headerless mono binary as little-endian `f32` or `f64`.
    ///
    /// A candidate is accepted when at most 1 % of its non-zero samples are implausible for a
    /// FIR filter (non-finite or magnitude outside `[1e-15, 1e4]`). When both candidates fit,
    /// `f64` is chosen if the `f32` reading shows zero low mantissa words (exact-valued `f64`
    /// data such as a Dirac), otherwise `f32` (the rePhase `.bin` default).
    pub fn parse_raw_float_auto(bytes: &'a [u8]) -> Result<Self, ArtifactError> {
        let fits = |enc: SampleEncoding| -> bool {
            if !bytes.len().is_multiple_of(enc.bytes()) {
                return false;
            }
            let (bad, nonzero) = raw_float_score(bytes, enc);
            nonzero > 0 && bad * 100 <= nonzero
        };
        let f32_ok = fits(SampleEncoding::F32);
        let f64_ok = fits(SampleEncoding::F64);
        let enc = match (f32_ok, f64_ok) {
            (true, true) if f64_low_words_zero(bytes) => SampleEncoding::F64,
            (true, _) => SampleEncoding::F32,
            (false, true) => SampleEncoding::F64,
            (false, false) => return Err(ArtifactError::UnrecognizedBinary),
        };
        Self::parse_raw(bytes, enc, 1)
    }

    /// Parses numeric text: one frame per line, columns are channels. Lines that do not start
    /// with a digit, sign, or `.` (headers, `#`/`*`/`%` comments) are skipped.
    pub fn parse_text(text: &'a str) -> Result<Self, ArtifactError> {
        let mut channels = 0usize;
        let mut frames = 0usize;
        for (idx, line) in text.lines().enumerate() {
            if !is_data_line(line) {
                continue;
            }
            let mut cols = 0usize;
            let mut ok = true;
            for_each_token(line, |tok| {
                if Decimal::parse(tok.as_bytes()).is_some() {
                    cols += 1;
                    true
                } else {
                    ok = false;
                    false
                }
            });
            if !ok || cols == 0 || (channels != 0 && cols != channels) {
                return Err(ArtifactError::InvalidText { line: idx + 1 });
            }
            channels = cols;
            frames += 1;
        }
        if frames == 0 {
            return Err(ArtifactError::Empty);
        }
        Ok(Self {
            payload: Payload::Text(text),
            container: Container::Text,
            channels,
            frames,
            sample_rate: None,
        })
    }

    /// Container the artifact was parsed from.
    #[inline]
    pub const fn container(&self) -> Container {
        self.container
    }

    /// Binary sample encoding (`None` for text).
    #[inline]
    pub const fn encoding(&self) -> Option<SampleEncoding> {
        match self.payload {
            Payload::Binary(_, e) => Some(e),
            Payload::Text(_) => None,
        }
    }

    /// Number of interleaved channels (1 = mono, 2 = stereo).
    #[inline]
    pub const fn channels(&self) -> usize {
        self.channels
    }

    /// Number of frames, i.e. FIR taps per channel.
    #[inline]
    pub const fn frames(&self) -> usize {
        self.frames
    }

    /// Sample rate from the WAV header (`None` for raw and text).
    #[inline]
    pub const fn sample_rate(&self) -> Option<u32> {
        self.sample_rate
    }

    fn check_channel(&self, channel: usize) -> Result<(), ArtifactError> {
        if channel >= self.channels {
            Err(ArtifactError::ChannelOutOfRange {
                channel,
                channels: self.channels,
            })
        } else {
            Ok(())
        }
    }

    /// Visits every sample of `channel` in frame order.
    fn visit<F: FnMut(usize, Sample<'a>) -> Result<(), ArtifactError>>(
        &self,
        channel: usize,
        mut f: F,
    ) -> Result<(), ArtifactError> {
        self.check_channel(channel)?;
        match self.payload {
            Payload::Binary(bytes, enc) => {
                let w = enc.bytes();
                let stride = w * self.channels;
                for (frame, chunk) in bytes.chunks_exact(stride).enumerate() {
                    f(
                        frame,
                        read_sample(&chunk[channel * w..channel * w + w], enc),
                    )?;
                }
            }
            Payload::Text(text) => {
                let mut frame = 0usize;
                for (idx, line) in text.lines().enumerate() {
                    if !is_data_line(line) {
                        continue;
                    }
                    let mut col = 0usize;
                    let mut value = None;
                    for_each_token(line, |tok| {
                        if col == channel {
                            value = Decimal::parse(tok.as_bytes());
                            return false;
                        }
                        col += 1;
                        true
                    });
                    let v = value.ok_or(ArtifactError::InvalidText { line: idx + 1 })?;
                    f(frame, Sample::Text(v))?;
                    frame += 1;
                }
            }
        }
        Ok(())
    }

    /// Largest absolute sample value of `channel` (full scale = `1.0`).
    pub fn peak(&self, channel: usize) -> Result<f64, ArtifactError> {
        let mut peak = 0.0f64;
        self.visit(channel, |frame, s| {
            let v = s.to_f64();
            if !v.is_finite() {
                return Err(ArtifactError::NonFiniteSample { frame });
            }
            let a = v.abs();
            if a > peak {
                peak = a;
            }
            Ok(())
        })?;
        Ok(peak)
    }

    /// Decodes `channel` into `out[..frames()]` as `f32` (full scale = `1.0`), verbatim:
    /// no normalization, windowing, or gain change. Returns the number of frames written.
    pub fn decode_f32(&self, channel: usize, out: &mut [f32]) -> Result<usize, ArtifactError> {
        self.check_channel(channel)?;
        if out.len() < self.frames {
            return Err(ArtifactError::OutputTooSmall {
                required: self.frames,
                provided: out.len(),
            });
        }
        self.visit(channel, |frame, s| {
            let v = match s {
                // f32 → f32 and short PCM → f32 are exact; avoid double rounding via f64.
                Sample::Int { v, frac_bits } if frac_bits <= 23 => {
                    v as f32 / (1u32 << frac_bits) as f32
                }
                Sample::F32(b) => f32::from_bits(b),
                other => other.to_f64() as f32,
            };
            if !v.is_finite() {
                return Err(ArtifactError::NonFiniteSample { frame });
            }
            out[frame] = v;
            Ok(())
        })?;
        Ok(self.frames)
    }

    /// Smallest right shift `s` such that `peak(channel) * 2^-s` fits Q1.15 without clipping
    /// (`<= 32767/32768`, or exactly `-1.0` for PCM), capped at `31`. Integer PCM always returns
    /// `0`. Computed with exact integer comparisons (no FPU) for every container; text uses the
    /// exact decimal value.
    pub fn q15_headroom_shift(&self, channel: usize) -> Result<u32, ArtifactError> {
        if let Payload::Binary(_, enc) = self.payload
            && !enc.is_float()
        {
            self.check_channel(channel)?;
            return Ok(0);
        }
        let mut s = 0u32;
        self.visit(channel, |frame, smp| {
            if smp.is_non_finite() {
                return Err(ArtifactError::NonFiniteSample { frame });
            }
            while s < 31 && smp.exceeds_q15_headroom(s) {
                s += 1;
            }
            Ok(())
        })?;
        Ok(s)
    }

    /// Visits `channel` as Q1.31 (`round(x * 2^-shift * 2^31)`, saturating), returning the
    /// number of saturated samples. Integer arithmetic only, for every container (see the
    /// module documentation for the rounding rule).
    pub(crate) fn for_each_q31<F: FnMut(usize, i32)>(
        &self,
        channel: usize,
        shift: u32,
        mut f: F,
    ) -> Result<usize, ArtifactError> {
        let mut saturated = 0usize;
        self.visit(channel, |frame, s| {
            let (v, sat) = s
                .to_fixed(31, shift, i32::MIN as i64, i32::MAX as i64)
                .ok_or(ArtifactError::NonFiniteSample { frame })?;
            if sat {
                saturated += 1;
            }
            f(frame, v as i32);
            Ok(())
        })?;
        Ok(saturated)
    }

    /// Decodes `channel` into Q1.31 (`out[..frames()]`) as `round(x * 2^-shift * 2^31)` with
    /// saturation, returning the number of saturated samples. Preserves the full precision of
    /// 24/32-bit PCM and rounds `f32` / `f64` / text exactly (ties to even) with integer
    /// arithmetic only.
    pub fn decode_q31(
        &self,
        channel: usize,
        shift: u32,
        out: &mut [i32],
    ) -> Result<usize, ArtifactError> {
        self.check_channel(channel)?;
        if out.len() < self.frames {
            return Err(ArtifactError::OutputTooSmall {
                required: self.frames,
                provided: out.len(),
            });
        }
        self.for_each_q31(channel, shift, |frame, v| out[frame] = v)
    }

    /// Decodes `channel` into Q1.15 (`out[..frames()]`) as `round(x * 2^-shift * 32768)` with
    /// saturation, returning the number of saturated samples. Every container is decoded with
    /// integer arithmetic only (no FPU required); use [`Self::q15_headroom_shift`] to pick a
    /// `shift` that avoids saturation for filters with gain above 0 dBFS.
    pub fn decode_q15(
        &self,
        channel: usize,
        shift: u32,
        out: &mut [i16],
    ) -> Result<usize, ArtifactError> {
        self.check_channel(channel)?;
        if out.len() < self.frames {
            return Err(ArtifactError::OutputTooSmall {
                required: self.frames,
                provided: out.len(),
            });
        }
        let mut saturated = 0usize;
        self.visit(channel, |frame, s| {
            let (v, sat) = s
                .to_fixed(15, shift, i16::MIN as i64, i16::MAX as i64)
                .ok_or(ArtifactError::NonFiniteSample { frame })?;
            if sat {
                saturated += 1;
            }
            out[frame] = v as i16;
            Ok(())
        })?;
        Ok(saturated)
    }
}
