//! Polynomial ring definition.
//!
//! A [`Ring`] bundles the immutable data every polynomial operation needs:
//! number of variables and the precomputed bitmasks used by
//! [`crate::monomial::MonoTerm`]'s mul and cmp routines.

use crate::field::Field;
use std::marker::PhantomData;

/// Bits used to store each variable's exponent in the packed monomial.
pub const BITS_PER_VAR: u8 = 8;

/// Maximum value a single variable's exponent may take.
pub const MAX_VAR_EXP: u32 = 0x7F;

/// Maximum number of variables supported by the 8-bit packing.
pub const MAX_VARS: u32 = 31;

/// An immutable polynomial ring.
///
/// Construct via [`Ring::new`]. Share via `Arc<Ring>`. Never mutated
/// after construction; every method takes `&self`.
#[derive(Debug, Clone)]
pub struct Ring<F: Field + Copy + Send + Sync> {
    /// Number of variables. `1 ≤ nvars ≤ MAX_VARS`.
    nvars: u32,
    /// Per-word overflow guard mask.
    overflow_mask: [u64; 4],
    /// Per-word XOR mask used to flip the degrevlex tie-break direction.
    cmp_flip_mask: [u64; 4],
    /// Phantom data for the coefficient field type.
    _marker: PhantomData<F>,
}

impl<F: Field + Copy + Send + Sync> Ring<F> {
    /// Construct a new ring.
    ///
    /// Returns `None` if `nvars` is out of range (`0` or `> MAX_VARS`).
    pub fn new(nvars: u32) -> Option<Self> {
        if nvars == 0 || nvars > MAX_VARS {
            return None;
        }
        let (overflow_mask, cmp_flip_mask) = compute_packing_masks(nvars);
        Some(Self {
            nvars,
            overflow_mask,
            cmp_flip_mask,
            _marker: PhantomData,
        })
    }

    /// Number of variables.
    #[inline]
    pub fn nvars(&self) -> u32 {
        self.nvars
    }

    /// Per-word overflow guard mask.
    #[inline]
    pub fn overflow_mask(&self) -> &[u64; 4] {
        &self.overflow_mask
    }

    /// Per-word degrevlex compare flip mask.
    #[inline]
    pub fn cmp_flip_mask(&self) -> &[u64; 4] {
        &self.cmp_flip_mask
    }
}

impl<F: Field + Copy + Send + Sync> PartialEq for Ring<F> {
    fn eq(&self, other: &Self) -> bool {
        self.nvars == other.nvars
    }
}
impl<F: Field + Copy + Send + Sync> Eq for Ring<F> {}

/// Compute the packing masks for a ring with the given number of
/// variables.
fn compute_packing_masks(nvars: u32) -> ([u64; 4], [u64; 4]) {
    let n = nvars as usize;
    let mut overflow = [0u64; 4];
    let mut flip = [0u64; 4];
    let first_var_byte = 31 - n;
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
        let r = Ring::<Fr>::new(5).unwrap();
        assert_eq!(r.nvars(), 5);
    }

    #[test]
    fn rejects_out_of_range_nvars() {
        assert!(Ring::<Fr>::new(0).is_none());
        assert!(Ring::<Fr>::new(MAX_VARS + 1).is_none());
        assert!(Ring::<Fr>::new(MAX_VARS).is_some());
    }

    #[test]
    fn packing_masks_cover_variable_bytes_only() {
        let r = Ring::<Fr>::new(3).unwrap();
        let ovf = r.overflow_mask();
        let flip = r.cmp_flip_mask();
        assert_eq!(ovf, &[0, 0, 0, 0x0080808000000000]);
        assert_eq!(flip, &[0, 0, 0, 0x007F7F7F00000000]);

        let r = Ring::<Fr>::new(25).unwrap();
        let ovf = r.overflow_mask();
        assert_eq!(ovf[0], 0x8080_0000_0000_0000);
        assert_eq!(ovf[1], 0x8080_8080_8080_8080);
        assert_eq!(ovf[2], 0x8080_8080_8080_8080);
        assert_eq!(ovf[3], 0x0080_8080_8080_8080);
    }

    #[test]
    fn packing_masks_have_no_overlap_with_unused_bytes() {
        let r = Ring::<Fr>::new(1).unwrap();
        let ovf = r.overflow_mask();
        assert_eq!(ovf, &[0, 0, 0, 0x0080_0000_0000_0000]);
    }
}
