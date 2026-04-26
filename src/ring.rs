//! Polynomial ring definition.
//!
//! A [`Ring`] bundles the immutable data every polynomial operation needs:
//! number of variables, monomial ordering, and the precomputed bitmasks 
//! used by [`crate::monomial::MonoTerm`]'s mul and cmp routines. Rings are 
//! shared between threads via `Arc<Ring>` (see `~/project/docs/rust-bba-port-plan.md` 
//! §6.1); the type is `Send + Sync` because it holds only immutable data.
//!
//! This bootstrap fixes two representation parameters:
//!
//! * **Ordering**: [`MonoOrder::DegRevLex`] only.
//! * **Bits per variable**: 7 + 1 guard. Variables are packed 8 bits per
//!   slot in a 4 × u64 (32-byte) block; bits 0-6 of each variable byte
//!   hold the exponent (max value 127), and bit 7 is reserved as an
//!   overflow guard so a Singular-style divmask check on the result of
//!   a packed-word add can detect per-byte overflow in O(words) ops.
//!   See `~/ark_gb/docs/design-decisions.md` ADR-005. The total degree
//!   byte (byte 31) keeps the full 8 bits since it is rewritten cleanly
//!   after every mul rather than incremented, so it needs no guard.
//!
//! Future widths (wider per-variable packing, dynamic ring widening
//! mirroring Singular's `kStratChangeTailRing`) are listed as deferred
//! enhancements in ADR-005.

use std::marker::PhantomData;
use crate::field::Field;
use crate::ordering::MonoOrder;

/// Bits used to store each variable's exponent in the packed monomial.
/// Eight bytes per slot, of which 7 hold the exponent and 1 is the
/// overflow guard.
pub const BITS_PER_VAR: u8 = 8;

/// Maximum value a single variable's exponent may take. Bit 7 of each
/// variable byte is the overflow guard (see ADR-005), so the usable
/// range is [0, 127]. Per ADR-018, ring construction is responsible
/// for ensuring no bba-step product exceeds this bound; release-build
/// [`crate::monomial::MonoTerm::mul`] does not check.
pub const MAX_VAR_EXP: u32 = 0x7F;

/// Maximum number of variables supported by the 8-bit packing.
///
/// One 8-bit byte is reserved at the front of the packed representation
/// for total degree, leaving 31 bytes of a 256-bit (four-word)
/// exponent block for variables. The port plan aims at 25-variable
/// staging workloads, so 31 gives comfortable headroom.
pub const MAX_VARS: u32 = 31;

/// An immutable polynomial ring.
///
/// Construct via [`Ring::new`]. Share via `Arc<Ring>`. Never mutated
/// after construction; every method takes `&self`.
#[derive(Debug, Clone)]
pub struct Ring<F: Field + Copy + Send + Sync> {
    /// Number of variables. `1 ≤ nvars ≤ MAX_VARS`.
    nvars: u32,
    /// MonoTerm ordering. Currently always [`MonoOrder::DegRevLex`].
    ordering: MonoOrder,
    /// Phantom data for the coefficient field type.
    _marker: PhantomData<F>,
    /// Per-word overflow guard mask: bit 7 set in each variable byte
    /// slot, 0 elsewhere (top "total-degree" byte and any unused
    /// low bytes). Used by `MonoTerm::assert_canonical` and by
    /// `MonoTerm::mul`'s `debug_assert!` invariant (ADR-018). Release
    /// builds of `mul` no longer consult this mask — matching
    /// Singular's PDEBUG-gated check. See ADR-005 / ADR-018 in
    /// `~/ark_gb/docs/design-decisions.md`.
    overflow_mask: [u64; 4],
    /// Per-word XOR mask used to flip the degrevlex tie-break direction
    /// at compare time: `0x7F` in each variable byte slot, `0x00` in
    /// the total-degree byte and unused low bytes. XOR'd into packed
    /// words before the lex compare in `MonoTerm::cmp_degrevlex`. With
    /// direct exponent storage (ADR-005), the variable byte direction
    /// has to be flipped to encode "smaller exponent at the
    /// largest-index differing variable wins"; the top byte is left
    /// untouched so larger total degree wins directly.
    cmp_flip_mask: [u64; 4],
}

impl<F: Field + Copy + Send + Sync> Ring<F> {
    /// Construct a new ring.
    ///
    /// Returns `None` if `nvars` is out of range (`0` or `> MAX_VARS`)
    /// or if the caller passes an unsupported ordering. Today only
    /// `DegRevLex` is supported.
    ///
    /// **Caller contract (ADR-018, mirroring Singular's `rComplete`):**
    /// the caller must ensure that every `MonoTerm::mul` product
    /// arising in the intended computation stays within
    /// [`MAX_VAR_EXP`] (= 127) per variable and within `u32::MAX`
    /// in total degree. Release builds of [`crate::monomial::MonoTerm::mul`]
    /// do not check this; violating the contract produces silent
    /// exponent corruption (matching Singular's release-mode
    /// `p_ExpVectorAdd` at
    /// `~/Singular/libpolys/polys/monomials/p_polys.h:1432`). Debug
    /// builds catch the violation via `debug_assert!`. If a future
    /// FFI caller admits rings whose bba-step products could
    /// overflow, the dispatch filter (in Singular-ark_gb's
    /// `ark_gb-dispatch.lib`) must tighten to exclude them before
    /// the ring reaches this constructor.
    pub fn new(nvars: u32, ordering: MonoOrder) -> Option<Self> {
        if nvars == 0 || nvars > MAX_VARS {
            return None;
        }
        // Validate the ordering against nvars.
        match ordering {
            MonoOrder::DegRevLex => {}
            MonoOrder::Elim { split } => {
                if split > nvars {
                    return None;
                }
            }
        }
        let (overflow_mask, cmp_flip_mask) = compute_packing_masks(nvars);
        Some(Self {
            nvars,
            ordering,
            _marker: PhantomData,
            overflow_mask,
            cmp_flip_mask,
        })
    }

    /// Number of variables.
    #[inline]
    pub fn nvars(&self) -> u32 {
        self.nvars
    }

    /// MonoTerm ordering.
    #[inline]
    pub fn ordering(&self) -> MonoOrder {
        self.ordering
    }

    /// Per-word overflow guard mask. See struct docstring.
    #[inline]
    pub fn overflow_mask(&self) -> &[u64; 4] {
        &self.overflow_mask
    }

    /// Per-word degrevlex compare flip mask. See struct docstring.
    #[inline]
    pub fn cmp_flip_mask(&self) -> &[u64; 4] {
        &self.cmp_flip_mask
    }
}

impl<F: Field + Copy + Send + Sync> PartialEq for Ring<F> {
    fn eq(&self, other: &Self) -> bool {
        // The masks are a pure function of nvars + ordering, so we don't
        // need to compare them explicitly.
        self.nvars == other.nvars && self.ordering == other.ordering
    }
}
impl<F: Field + Copy + Send + Sync> Eq for Ring<F> {}

/// Compute the packing masks for a ring with the given number of
/// variables. The variable bytes occupy positions `[31 - nvars, 30]`
/// of the 32-byte packed block (byte 31 = total-degree, low bytes
/// always zero). See `monomial::byte_index_for_var`.
fn compute_packing_masks(nvars: u32) -> ([u64; 4], [u64; 4]) {
    let n = nvars as usize;
    let mut overflow = [0u64; 4];
    let mut flip = [0u64; 4];
    let first_var_byte = 31 - n; // (4*8 - 1) - n
    let last_var_byte = 30;
    for byte_idx in first_var_byte..=last_var_byte {
        let word = byte_idx / 8;
        let shift = ((byte_idx % 8) * 8) as u32;
        overflow[word] |= 0x80u64 << shift;
        flip[word] |= 0x7Fu64 << shift;
    }
    (overflow, flip)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bls12_381::Fr;

    #[test]
    fn constructs_valid_ring() {
        let r = Ring::<Fr>::new(5, MonoOrder::DegRevLex).unwrap();
        assert_eq!(r.nvars(), 5);
    }

    #[test]
    fn rejects_out_of_range_nvars() {
        assert!(Ring::<Fr>::new(0, MonoOrder::DegRevLex).is_none());
        assert!(Ring::<Fr>::new(MAX_VARS + 1, MonoOrder::DegRevLex).is_none());
        assert!(Ring::<Fr>::new(MAX_VARS, MonoOrder::DegRevLex).is_some());
    }

    #[test]
    fn packing_masks_cover_variable_bytes_only() {
        // For nvars = 3: variable bytes are 28, 29, 30 (in word 3,
        // shifts 32, 40, 48). Top byte 31 (shift 56, total-degree)
        // and all of words 0..3 should be zero.
        let r = Ring::<Fr>::new(3, MonoOrder::DegRevLex).unwrap();
        let ovf = r.overflow_mask();
        let flip = r.cmp_flip_mask();
        assert_eq!(ovf, &[0, 0, 0, 0x0080808000000000]);
        assert_eq!(flip, &[0, 0, 0, 0x007F7F7F00000000]);

        // For nvars = 25: variable bytes 6..=30. Crosses word
        // boundaries at byte 8, 16, 24. Top byte (shift 56) excluded.
        let r = Ring::<Fr>::new(25, MonoOrder::DegRevLex).unwrap();
        let ovf = r.overflow_mask();
        // Word 0: bytes 6, 7 → shifts 48, 56.
        assert_eq!(ovf[0], 0x8080_0000_0000_0000);
        // Word 1, 2: all eight bytes have the guard.
        assert_eq!(ovf[1], 0x8080_8080_8080_8080);
        assert_eq!(ovf[2], 0x8080_8080_8080_8080);
        // Word 3: bytes 24..=30 (shifts 0..=48) get the guard;
        //         byte 31 (shift 56, total-deg) does NOT.
        assert_eq!(ovf[3], 0x0080_8080_8080_8080);
    }

    #[test]
    fn packing_masks_have_no_overlap_with_unused_bytes() {
        // For nvars = 1: only byte 30. Verify no other byte set.
        let r = Ring::<Fr>::new(1, MonoOrder::DegRevLex).unwrap();
        let ovf = r.overflow_mask();
        assert_eq!(ovf, &[0, 0, 0, 0x0080_0000_0000_0000]);
    }
}
