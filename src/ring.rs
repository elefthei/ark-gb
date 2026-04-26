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
//! * **Ordering**: [`crate::ordering::DegRevLex`] only.
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

use crate::field::Field;
use crate::monomial::Monomial;
use crate::ordering::MonoOrder;
use std::cmp::Ordering;
use std::marker::PhantomData;
use std::ops::Deref;

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

/// Order-independent payload of a polynomial ring.
///
/// Contains the number of variables and the precomputed bitmasks.
/// This is what monomial operations need — they do not need the
/// ordering.
#[derive(Debug, Clone)]
pub struct RingData<F: Field + Copy + Send + Sync> {
    /// Number of variables. `1 ≤ nvars ≤ MAX_VARS`.
    nvars: u32,
    /// Phantom data for the coefficient field type.
    _marker: PhantomData<F>,
    /// Per-word overflow guard mask.
    overflow_mask: [u64; 4],
    /// Per-word XOR mask used to flip the degrevlex tie-break direction.
    cmp_flip_mask: [u64; 4],
}

impl<F: Field + Copy + Send + Sync> RingData<F> {
    /// Number of variables.
    #[inline]
    pub fn nvars(&self) -> u32 {
        self.nvars
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

/// An immutable polynomial ring, parametric in the monomial ordering.
///
/// Construct via [`Ring::new`]. Share via `Arc<Ring>`. Never mutated
/// after construction; every method takes `&self`.
///
/// `Deref<Target = RingData<F>>` allows transparent access to
/// `nvars()`, `overflow_mask()`, etc.
#[derive(Debug, Clone)]
pub struct Ring<F: Field + Copy + Send + Sync, O: MonoOrder> {
    /// Order-independent payload.
    data: RingData<F>,
    /// MonoTerm ordering.
    order: O,
}

impl<F: Field + Copy + Send + Sync, O: MonoOrder> Ring<F, O> {
    /// Construct a new ring.
    ///
    /// Returns `None` if `nvars` is out of range (`0` or `> MAX_VARS`)
    /// or if the ordering fails validation against `nvars`.
    pub fn new(nvars: u32, order: O) -> Option<Self> {
        if nvars == 0 || nvars > MAX_VARS {
            return None;
        }
        if !order.validate(nvars) {
            return None;
        }
        let (overflow_mask, cmp_flip_mask) = compute_packing_masks(nvars);
        Some(Self {
            data: RingData {
                nvars,
                _marker: PhantomData,
                overflow_mask,
                cmp_flip_mask,
            },
            order,
        })
    }

    /// The monomial ordering.
    #[inline]
    pub fn order(&self) -> &O {
        &self.order
    }

    /// Compare two monomials under this ring's ordering.
    #[inline]
    pub fn cmp<M: Monomial<F>>(&self, a: &M, b: &M) -> Ordering {
        self.order.cmp(a, b, &self.data)
    }
}

impl<F: Field + Copy + Send + Sync, O: MonoOrder> Deref for Ring<F, O> {
    type Target = RingData<F>;

    #[inline]
    fn deref(&self) -> &RingData<F> {
        &self.data
    }
}

impl<F: Field + Copy + Send + Sync, O: MonoOrder> PartialEq for Ring<F, O> {
    fn eq(&self, other: &Self) -> bool {
        self.data.nvars == other.data.nvars && self.order == other.order
    }
}
impl<F: Field + Copy + Send + Sync, O: MonoOrder> Eq for Ring<F, O> {}

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
    use crate::ordering::DegRevLex;
    use ark_bls12_381::Fr;

    #[test]
    fn constructs_valid_ring() {
        let r = Ring::<Fr, DegRevLex>::new(5, DegRevLex).unwrap();
        assert_eq!(r.nvars(), 5);
    }

    #[test]
    fn rejects_out_of_range_nvars() {
        assert!(Ring::<Fr, DegRevLex>::new(0, DegRevLex).is_none());
        assert!(Ring::<Fr, DegRevLex>::new(MAX_VARS + 1, DegRevLex).is_none());
        assert!(Ring::<Fr, DegRevLex>::new(MAX_VARS, DegRevLex).is_some());
    }

    #[test]
    fn packing_masks_cover_variable_bytes_only() {
        let r = Ring::<Fr, DegRevLex>::new(3, DegRevLex).unwrap();
        let ovf = r.overflow_mask();
        let flip = r.cmp_flip_mask();
        assert_eq!(ovf, &[0, 0, 0, 0x0080808000000000]);
        assert_eq!(flip, &[0, 0, 0, 0x007F7F7F00000000]);

        let r = Ring::<Fr, DegRevLex>::new(25, DegRevLex).unwrap();
        let ovf = r.overflow_mask();
        assert_eq!(ovf[0], 0x8080_0000_0000_0000);
        assert_eq!(ovf[1], 0x8080_8080_8080_8080);
        assert_eq!(ovf[2], 0x8080_8080_8080_8080);
        assert_eq!(ovf[3], 0x0080_8080_8080_8080);
    }

    #[test]
    fn packing_masks_have_no_overlap_with_unused_bytes() {
        let r = Ring::<Fr, DegRevLex>::new(1, DegRevLex).unwrap();
        let ovf = r.overflow_mask();
        assert_eq!(ovf, &[0, 0, 0, 0x0080_0000_0000_0000]);
    }
}
