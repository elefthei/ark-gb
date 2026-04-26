//! Packed-exponent monomials and the `Monomial<F>` trait.
//!
//! See original module docs for the packing layout.

use crate::field::Field;
use crate::ring::{BITS_PER_VAR, Ring};
use std::cmp::Ordering;
use std::hash::Hash;

/// Number of u64 words in the packed exponent block.
pub const WORDS_PER_MONO: usize = 4;

const _BITS_PER_VAR_IS_8: () = assert!(BITS_PER_VAR == 8);

/// Const flip mask for degrevlex on the full 31-variable space.
const DEGREVLEX_FLIP: [u64; 4] = [
    0x7F7F_7F7F_7F7F_7F7F,
    0x7F7F_7F7F_7F7F_7F7F,
    0x7F7F_7F7F_7F7F_7F7F,
    0x007F_7F7F_7F7F_7F7F,
];

/// The `Monomial<F>` trait: order-aware monomial abstraction.
///
/// Implementors provide an `Ord` impl that pins the monomial order.
pub trait Monomial<F: Field + Copy + Send + Sync>:
    Sized + Copy + Send + Sync + std::fmt::Debug + PartialEq + Eq + std::hash::Hash + Ord
{
    /// The identity monomial (all exponents zero).
    fn one(ring: &Ring<F>) -> Self;
    /// Build a monomial from an exponent slice of length `ring.nvars()`.
    fn from_exponents(ring: &Ring<F>, exps: &[u32]) -> Option<Self>;
    /// Exponent of variable `i`. Returns `None` if `i >= ring.nvars()`.
    fn exponent(&self, ring: &Ring<F>, i: u32) -> Option<u32>;
    /// Copy the exponent vector into a `Vec<u32>`.
    fn exponents(&self, ring: &Ring<F>) -> Vec<u32>;
    /// Total degree.
    fn total_deg(&self, ring: &Ring<F>) -> u32;
    /// Multiply two monomials.
    fn mul(&self, other: &Self, ring: &Ring<F>) -> Self;
    /// `true` iff `self | other` (each `e_i(self) ≤ e_i(other)`).
    fn divides(&self, other: &Self, ring: &Ring<F>) -> bool;
    /// Divide. Precondition `other.divides(self)`; returns `None` otherwise.
    fn div(&self, other: &Self, ring: &Ring<F>) -> Option<Self>;
    /// Componentwise maximum (least common multiple of monomials).
    fn lcm(&self, other: &Self, ring: &Ring<F>) -> Self;
    /// Access the underlying `MonoTerm`.
    fn as_mono_term(&self) -> &MonoTerm;

    /// Short exponent vector (ring-independent).
    #[inline]
    fn sev(&self) -> u64 {
        self.as_mono_term().sev()
    }

    /// Cached total degree (ring-independent).
    #[inline]
    fn raw_total_deg(&self) -> u32 {
        self.as_mono_term().total_deg()
    }
}

/// Packed-exponent monomial. See module documentation for layout.
///
/// `MonoTerm` is a pure data carrier — no `Ord` impl. The monomial
/// order is pinned by the newtype wrappers (`GrevLexTerm`,
/// `OddElimTerm`) via their `Ord` impls.
#[derive(Clone, Copy, Debug)]
pub struct MonoTerm {
    /// Four u64 words; word 3 is most significant.
    packed: [u64; WORDS_PER_MONO],
    /// Short exponent vector.
    sev: u64,
    /// True total degree, uncapped.
    total_deg: u32,
    /// Component index. Always 0 today.
    component: u16,
    /// Number of variables in the parent ring (cached for ringless comparisons).
    nvars: u16,
}

impl MonoTerm {
    // ----- Construction -----

    /// Build a monomial from an exponent slice of length `ring.nvars()`.
    pub fn from_exponents<F: Field + Copy + Send + Sync>(
        ring: &Ring<F>,
        exps: &[u32],
    ) -> Option<Self> {
        let n = ring.nvars() as usize;
        if exps.len() != n {
            return None;
        }
        for &e in exps {
            if e > crate::ring::MAX_VAR_EXP {
                return None;
            }
        }

        let mut packed = [0u64; WORDS_PER_MONO];
        let mut total: u64 = 0;
        let mut sev: u64 = 0;

        for (i, &e) in exps.iter().enumerate() {
            total += e as u64;
            if e > 0 {
                sev |= 1u64 << (i % 64);
            }
            let byte_idx = byte_index_for_var(n, i);
            let (word, shift) = split_byte_index(byte_idx);
            packed[word] |= (e as u64) << shift;
        }

        let capped = total.min(u8::MAX as u64);
        packed[WORDS_PER_MONO - 1] |= capped << 56;

        if total > u32::MAX as u64 {
            return None;
        }

        Some(Self {
            packed,
            sev,
            total_deg: total as u32,
            component: 0,
            nvars: n as u16,
        })
    }

    /// The identity monomial (all exponents zero).
    pub fn one<F: Field + Copy + Send + Sync>(ring: &Ring<F>) -> Self {
        let zeros = vec![0u32; ring.nvars() as usize];
        Self::from_exponents(ring, &zeros).expect("identity monomial fits trivially")
    }

    // ----- Accessors -----

    /// Short exponent vector.
    #[inline]
    pub fn sev(&self) -> u64 {
        self.sev
    }

    /// Total degree (uncapped).
    #[inline]
    pub fn total_deg(&self) -> u32 {
        self.total_deg
    }

    /// Component index. Always 0 in this bootstrap.
    #[inline]
    pub fn component(&self) -> u32 {
        self.component as u32
    }

    /// Borrow the packed exponent block (4 × u64 = 32 bytes).
    #[inline]
    pub fn packed(&self) -> &[u64; 4] {
        &self.packed
    }

    /// Exponent of variable `i`. Returns `None` if `i >= ring.nvars()`.
    pub fn exponent<F: Field + Copy + Send + Sync>(&self, ring: &Ring<F>, i: u32) -> Option<u32> {
        if i >= ring.nvars() {
            return None;
        }
        Some(self.exponent_raw(ring.nvars() as usize, i as usize))
    }

    #[inline]
    fn exponent_raw(&self, nvars: usize, i: usize) -> u32 {
        let byte_idx = byte_index_for_var(nvars, i);
        let (word, shift) = split_byte_index(byte_idx);
        ((self.packed[word] >> shift) & 0x7F) as u32
    }

    /// Copy the exponent vector into a `Vec<u32>`.
    pub fn exponents<F: Field + Copy + Send + Sync>(&self, ring: &Ring<F>) -> Vec<u32> {
        let n = ring.nvars() as usize;
        (0..n).map(|i| self.exponent_raw(n, i)).collect()
    }

    // ----- Arithmetic -----

    /// Multiply two monomials.
    pub fn mul<F: Field + Copy + Send + Sync>(&self, other: &Self, ring: &Ring<F>) -> Self {
        const _: () = assert!(WORDS_PER_MONO == 4);

        let mut packed: [u64; WORDS_PER_MONO] = [
            self.packed[0].wrapping_add(other.packed[0]),
            self.packed[1].wrapping_add(other.packed[1]),
            self.packed[2].wrapping_add(other.packed[2]),
            self.packed[3].wrapping_add(other.packed[3]),
        ];

        if cfg!(debug_assertions) {
            let m = ring.overflow_mask();
            let ovf =
                (packed[0] & m[0]) | (packed[1] & m[1]) | (packed[2] & m[2]) | (packed[3] & m[3]);
            debug_assert_eq!(
                ovf, 0,
                "MonoTerm::mul overflow: per-byte exponent > 127 (ADR-018 contract: \
                 caller's ring construction must guarantee no bba-step product overflows)"
            );
            debug_assert!(
                self.total_deg.checked_add(other.total_deg).is_some(),
                "MonoTerm::mul total-degree u32 overflow (ADR-018 contract)"
            );
        }

        let total = self.total_deg.wrapping_add(other.total_deg);
        let capped = (total as u64).min(u8::MAX as u64);
        packed[WORDS_PER_MONO - 1] =
            (packed[WORDS_PER_MONO - 1] & !(0xFFu64 << 56)) | (capped << 56);

        let sev = self.sev | other.sev;

        Self {
            packed,
            sev,
            total_deg: total,
            component: 0,
            nvars: self.nvars,
        }
    }

    /// `true` iff `self | other` (each `e_i(self) ≤ e_i(other)`).
    pub fn divides<F: Field + Copy + Send + Sync>(&self, other: &Self, ring: &Ring<F>) -> bool {
        let n = ring.nvars() as usize;
        let first_var_byte = (WORDS_PER_MONO * 8 - 1) - n;
        let last_var_byte = WORDS_PER_MONO * 8 - 2;
        for byte_idx in first_var_byte..=last_var_byte {
            let (word, shift) = split_byte_index(byte_idx);
            let ea = (self.packed[word] >> shift) & 0x7F;
            let eb = (other.packed[word] >> shift) & 0x7F;
            if ea > eb {
                return false;
            }
        }
        true
    }

    /// Divide. Precondition `other.divides(self)`; returns `None` otherwise.
    pub fn div<F: Field + Copy + Send + Sync>(&self, other: &Self, ring: &Ring<F>) -> Option<Self> {
        let n = ring.nvars() as usize;
        let first_var_byte = (WORDS_PER_MONO * 8 - 1) - n;
        let last_var_byte = WORDS_PER_MONO * 8 - 2;
        let mut packed = [0u64; WORDS_PER_MONO];
        let mut sev: u64 = 0;
        for byte_idx in first_var_byte..=last_var_byte {
            let (word, shift) = split_byte_index(byte_idx);
            let ea = (self.packed[word] >> shift) & 0x7F;
            let eb = (other.packed[word] >> shift) & 0x7F;
            if eb > ea {
                return None;
            }
            let new_e = ea - eb;
            packed[word] |= new_e << shift;
            if new_e > 0 {
                let var_i = byte_idx + n - (WORDS_PER_MONO * 8 - 1);
                sev |= 1u64 << (var_i % 64);
            }
        }
        if other.total_deg > self.total_deg {
            return None;
        }
        let total = self.total_deg - other.total_deg;
        let capped = (total as u64).min(u8::MAX as u64);
        packed[WORDS_PER_MONO - 1] |= capped << 56;
        Some(Self {
            packed,
            sev,
            total_deg: total,
            component: 0,
            nvars: self.nvars,
        })
    }

    /// Componentwise maximum (least common multiple of monomials).
    pub fn lcm<F: Field + Copy + Send + Sync>(&self, other: &Self, ring: &Ring<F>) -> Self {
        let n = ring.nvars() as usize;
        let mut exps = vec![0u32; n];
        for (i, slot) in exps.iter_mut().enumerate() {
            *slot = self.exponent_raw(n, i).max(other.exponent_raw(n, i));
        }
        Self::from_exponents(ring, &exps).expect("lcm per-var exponents ≤ MAX_VAR_EXP")
    }

    // ----- Ordering helpers (pub(crate)) -----

    /// Degrevlex comparison using the CONST flip mask.
    pub(crate) fn cmp_degrevlex_packed(&self, other: &Self) -> Ordering {
        let a_cap = (self.packed[WORDS_PER_MONO - 1] >> 56) & 0xFF;
        let b_cap = (other.packed[WORDS_PER_MONO - 1] >> 56) & 0xFF;
        let saturated = a_cap == u8::MAX as u64 || b_cap == u8::MAX as u64;

        if saturated {
            match self.total_deg.cmp(&other.total_deg) {
                Ordering::Equal => {}
                ord => return ord,
            }
            for i in (0..WORDS_PER_MONO - 1).rev() {
                let av = self.packed[i] ^ DEGREVLEX_FLIP[i];
                let bv = other.packed[i] ^ DEGREVLEX_FLIP[i];
                match av.cmp(&bv) {
                    Ordering::Equal => {}
                    ord => return ord,
                }
            }
            let lo_mask = (1u64 << 56) - 1;
            let av_top =
                (self.packed[WORDS_PER_MONO - 1] ^ DEGREVLEX_FLIP[WORDS_PER_MONO - 1]) & lo_mask;
            let bv_top =
                (other.packed[WORDS_PER_MONO - 1] ^ DEGREVLEX_FLIP[WORDS_PER_MONO - 1]) & lo_mask;
            return av_top.cmp(&bv_top);
        }

        for i in (0..WORDS_PER_MONO).rev() {
            let av = self.packed[i] ^ DEGREVLEX_FLIP[i];
            let bv = other.packed[i] ^ DEGREVLEX_FLIP[i];
            match av.cmp(&bv) {
                Ordering::Equal => {}
                ord => return ord,
            }
        }
        Ordering::Equal
    }

    /// Degrevlex comparison using the ring's dynamic flip mask.
    #[allow(dead_code)]
    pub(crate) fn cmp_degrevlex<F: Field + Copy + Send + Sync>(
        &self,
        other: &Self,
        ring: &Ring<F>,
    ) -> Ordering {
        let a_cap = (self.packed[WORDS_PER_MONO - 1] >> 56) & 0xFF;
        let b_cap = (other.packed[WORDS_PER_MONO - 1] >> 56) & 0xFF;
        let saturated = a_cap == u8::MAX as u64 || b_cap == u8::MAX as u64;
        let mask = ring.cmp_flip_mask();

        if saturated {
            match self.total_deg.cmp(&other.total_deg) {
                Ordering::Equal => {}
                ord => return ord,
            }
            for i in (0..WORDS_PER_MONO - 1).rev() {
                let av = self.packed[i] ^ mask[i];
                let bv = other.packed[i] ^ mask[i];
                match av.cmp(&bv) {
                    Ordering::Equal => {}
                    ord => return ord,
                }
            }
            let lo_mask = (1u64 << 56) - 1;
            let av_top = (self.packed[WORDS_PER_MONO - 1] ^ mask[WORDS_PER_MONO - 1]) & lo_mask;
            let bv_top = (other.packed[WORDS_PER_MONO - 1] ^ mask[WORDS_PER_MONO - 1]) & lo_mask;
            return av_top.cmp(&bv_top);
        }

        for i in (0..WORDS_PER_MONO).rev() {
            let av = self.packed[i] ^ mask[i];
            let bv = other.packed[i] ^ mask[i];
            match av.cmp(&bv) {
                Ordering::Equal => {}
                ord => return ord,
            }
        }
        Ordering::Equal
    }

    /// Block elimination comparison with a closure deciding which vars are in the elim block.
    pub(crate) fn cmp_elim_packed(
        &self,
        other: &Self,
        eliminate: impl Fn(usize) -> bool,
    ) -> Ordering {
        let n = self.nvars as usize;
        let mut wa: u32 = 0;
        let mut wb: u32 = 0;
        for i in 0..n {
            if eliminate(i) {
                let byte_idx = byte_index_for_var(n, i);
                let (word, shift) = split_byte_index(byte_idx);
                let ea = ((self.packed[word] >> shift) & 0x7F) as u32;
                let eb = ((other.packed[word] >> shift) & 0x7F) as u32;
                wa += ea;
                wb += eb;
            }
        }
        match wa.cmp(&wb) {
            Ordering::Equal => {}
            ord => return ord,
        }
        self.cmp_degrevlex_packed(other)
    }

    // ----- Invariants -----

    /// Panic if any internal invariant is violated.
    pub fn assert_canonical<F: Field + Copy + Send + Sync>(&self, ring: &Ring<F>) {
        let n = ring.nvars() as usize;
        let mut total: u64 = 0;
        let mut sev: u64 = 0;

        for i in 0..n {
            let e = self.exponent_raw(n, i);
            assert!(
                e <= crate::ring::MAX_VAR_EXP,
                "exponent {e} at var {i} exceeds 7-bit limit ({})",
                crate::ring::MAX_VAR_EXP
            );
            total += e as u64;
            if e > 0 {
                sev |= 1u64 << (i % 64);
            }
        }

        for word in 0..WORDS_PER_MONO {
            assert_eq!(
                self.packed[word] & ring.overflow_mask()[word],
                0,
                "overflow guard bit set in word {word}"
            );
        }

        assert!(total <= u32::MAX as u64, "total degree overflows u32");
        assert_eq!(total as u32, self.total_deg, "total_deg cache mismatch");
        assert_eq!(sev, self.sev, "sev cache mismatch");

        let expected_cap = total.min(u8::MAX as u64);
        let cap = (self.packed[WORDS_PER_MONO - 1] >> 56) & 0xFF;
        assert_eq!(cap, expected_cap, "top-byte total-degree cap mismatch");

        let expected = Self::from_exponents(ring, &self.exponents(ring))
            .expect("re-canonicalising from our own exponents must succeed");
        assert_eq!(
            self.packed, expected.packed,
            "packed differs from canonical"
        );

        assert_eq!(self.component, 0, "non-zero component not yet supported");
    }
}

impl PartialEq for MonoTerm {
    fn eq(&self, other: &Self) -> bool {
        self.packed == other.packed && self.component == other.component
    }
}
impl Eq for MonoTerm {}

impl Hash for MonoTerm {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.packed.hash(state);
        self.component.hash(state);
    }
}

// ----- Newtypes -----

/// Graded reverse lexicographic order.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GrevLexTerm(pub(crate) MonoTerm);

/// Elimination order: odd-indexed variables form the elimination block.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OddElimTerm(pub(crate) MonoTerm);

impl From<MonoTerm> for GrevLexTerm {
    fn from(t: MonoTerm) -> Self {
        GrevLexTerm(t)
    }
}
impl From<MonoTerm> for OddElimTerm {
    fn from(t: MonoTerm) -> Self {
        OddElimTerm(t)
    }
}

impl Ord for GrevLexTerm {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.cmp_degrevlex_packed(&other.0)
    }
}
impl PartialOrd for GrevLexTerm {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}

impl Ord for OddElimTerm {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.cmp_elim_packed(&other.0, |idx| idx % 2 == 1)
    }
}
impl PartialOrd for OddElimTerm {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}

// ----- Monomial<F> impls -----

impl<F: Field + Copy + Send + Sync> Monomial<F> for GrevLexTerm {
    #[inline]
    fn one(ring: &Ring<F>) -> Self {
        Self(MonoTerm::one(ring))
    }
    #[inline]
    fn from_exponents(ring: &Ring<F>, exps: &[u32]) -> Option<Self> {
        MonoTerm::from_exponents(ring, exps).map(Self)
    }
    #[inline]
    fn exponent(&self, ring: &Ring<F>, i: u32) -> Option<u32> {
        self.0.exponent(ring, i)
    }
    #[inline]
    fn exponents(&self, ring: &Ring<F>) -> Vec<u32> {
        self.0.exponents(ring)
    }
    #[inline]
    fn total_deg(&self, _ring: &Ring<F>) -> u32 {
        self.0.total_deg
    }
    #[inline]
    fn mul(&self, other: &Self, ring: &Ring<F>) -> Self {
        Self(self.0.mul(&other.0, ring))
    }
    #[inline]
    fn divides(&self, other: &Self, ring: &Ring<F>) -> bool {
        self.0.divides(&other.0, ring)
    }
    #[inline]
    fn div(&self, other: &Self, ring: &Ring<F>) -> Option<Self> {
        self.0.div(&other.0, ring).map(Self)
    }
    #[inline]
    fn lcm(&self, other: &Self, ring: &Ring<F>) -> Self {
        Self(self.0.lcm(&other.0, ring))
    }
    #[inline]
    fn as_mono_term(&self) -> &MonoTerm {
        &self.0
    }
}

impl<F: Field + Copy + Send + Sync> Monomial<F> for OddElimTerm {
    #[inline]
    fn one(ring: &Ring<F>) -> Self {
        Self(MonoTerm::one(ring))
    }
    #[inline]
    fn from_exponents(ring: &Ring<F>, exps: &[u32]) -> Option<Self> {
        MonoTerm::from_exponents(ring, exps).map(Self)
    }
    #[inline]
    fn exponent(&self, ring: &Ring<F>, i: u32) -> Option<u32> {
        self.0.exponent(ring, i)
    }
    #[inline]
    fn exponents(&self, ring: &Ring<F>) -> Vec<u32> {
        self.0.exponents(ring)
    }
    #[inline]
    fn total_deg(&self, _ring: &Ring<F>) -> u32 {
        self.0.total_deg
    }
    #[inline]
    fn mul(&self, other: &Self, ring: &Ring<F>) -> Self {
        Self(self.0.mul(&other.0, ring))
    }
    #[inline]
    fn divides(&self, other: &Self, ring: &Ring<F>) -> bool {
        self.0.divides(&other.0, ring)
    }
    #[inline]
    fn div(&self, other: &Self, ring: &Ring<F>) -> Option<Self> {
        self.0.div(&other.0, ring).map(Self)
    }
    #[inline]
    fn lcm(&self, other: &Self, ring: &Ring<F>) -> Self {
        Self(self.0.lcm(&other.0, ring))
    }
    #[inline]
    fn as_mono_term(&self) -> &MonoTerm {
        &self.0
    }
}

// ----- packing helpers -----

/// Byte index of variable `i` in the 32-byte packed block.
#[inline]
fn byte_index_for_var(nvars: usize, i: usize) -> usize {
    debug_assert!(i < nvars);
    debug_assert!(nvars < WORDS_PER_MONO * 8);
    i + (WORDS_PER_MONO * 8 - 1) - nvars
}

/// Split a byte index in `[0, 32)` into `(word_idx, bit_shift)`.
#[inline]
fn split_byte_index(byte_idx: usize) -> (usize, u32) {
    debug_assert!(byte_idx < WORDS_PER_MONO * 8);
    let word = byte_idx / 8;
    let shift = ((byte_idx % 8) * 8) as u32;
    (word, shift)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bls12_381::Fr;

    fn mk_ring(nvars: u32) -> Ring<Fr> {
        Ring::<Fr>::new(nvars).unwrap()
    }

    #[test]
    fn round_trip_exponents() {
        let r = mk_ring(5);
        let exps = vec![0u32, 3, 7, 0, 12];
        let m = MonoTerm::from_exponents(&r, &exps).unwrap();
        assert_eq!(m.exponents(&r), exps);
        assert_eq!(m.total_deg(), 22);
        m.assert_canonical(&r);
    }

    #[test]
    fn one_is_canonical() {
        let r = mk_ring(7);
        let one = MonoTerm::one(&r);
        assert_eq!(one.total_deg(), 0);
        assert_eq!(one.sev(), 0);
        one.assert_canonical(&r);
    }

    #[test]
    fn from_exponents_rejects_above_max_var_exp() {
        let r = mk_ring(3);
        assert!(MonoTerm::from_exponents(&r, &[127, 0, 0]).is_some());
        assert!(MonoTerm::from_exponents(&r, &[128, 0, 0]).is_none());
        assert!(MonoTerm::from_exponents(&r, &[0, 200, 0]).is_none());
        assert!(MonoTerm::from_exponents(&r, &[0, 0, 255]).is_none());
    }

    #[test]
    fn mul_within_budget_succeeds() {
        let r = mk_ring(4);
        let a = MonoTerm::from_exponents(&r, &[63, 0, 0, 0]).unwrap();
        let b = MonoTerm::from_exponents(&r, &[64, 0, 0, 0]).unwrap();
        let p = a.mul(&b, &r);
        p.assert_canonical(&r);
        assert_eq!(p.exponent(&r, 0).unwrap(), 127);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "MonoTerm::mul overflow")]
    fn mul_debug_asserts_on_per_byte_overflow() {
        let r = mk_ring(4);
        let a = MonoTerm::from_exponents(&r, &[1, 100, 0, 0]).unwrap();
        let b = MonoTerm::from_exponents(&r, &[1, 50, 0, 0]).unwrap();
        let _ = a.mul(&b, &r);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "MonoTerm::mul overflow")]
    fn mul_debug_asserts_on_exact_guard_bit_trip() {
        let r = mk_ring(4);
        let a = MonoTerm::from_exponents(&r, &[64, 0, 0, 0]).unwrap();
        let b = MonoTerm::from_exponents(&r, &[64, 0, 0, 0]).unwrap();
        let _ = a.mul(&b, &r);
    }

    #[test]
    fn mul_no_carry_propagation_between_neighbouring_bytes() {
        let r = mk_ring(5);
        let a = MonoTerm::from_exponents(&r, &[60, 70, 80, 90, 100]).unwrap();
        let b = MonoTerm::from_exponents(&r, &[60, 50, 40, 30, 20]).unwrap();
        let p = a.mul(&b, &r);
        p.assert_canonical(&r);
        for i in 0..5 {
            assert_eq!(p.exponent(&r, i).unwrap(), 120);
        }
        assert_eq!(p.total_deg(), 600);
    }

    #[test]
    fn sev_matches_nonzero_vars() {
        let r = mk_ring(10);
        let m = MonoTerm::from_exponents(&r, &[0, 2, 0, 0, 5, 0, 0, 1, 0, 0]).unwrap();
        let expected = (1u64 << 1) | (1u64 << 4) | (1u64 << 7);
        assert_eq!(m.sev(), expected);
    }

    #[test]
    fn divides_is_componentwise_le() {
        let r = mk_ring(3);
        let a = MonoTerm::from_exponents(&r, &[1, 2, 3]).unwrap();
        let b = MonoTerm::from_exponents(&r, &[2, 2, 4]).unwrap();
        assert!(a.divides(&b, &r));
        assert!(!b.divides(&a, &r));
    }

    #[test]
    fn div_after_mul_roundtrip() {
        let r = mk_ring(4);
        let a = MonoTerm::from_exponents(&r, &[1, 2, 0, 5]).unwrap();
        let b = MonoTerm::from_exponents(&r, &[3, 0, 4, 1]).unwrap();
        let p = a.mul(&b, &r);
        let back = p.div(&b, &r).unwrap();
        assert_eq!(back, a);
    }

    #[test]
    fn lcm_is_max_componentwise() {
        let r = mk_ring(3);
        let a = MonoTerm::from_exponents(&r, &[1, 5, 3]).unwrap();
        let b = MonoTerm::from_exponents(&r, &[4, 2, 3]).unwrap();
        let l = a.lcm(&b, &r);
        assert_eq!(l.exponents(&r), vec![4, 5, 3]);
    }

    #[test]
    fn degrevlex_cmp_basic() {
        let r = mk_ring(3);
        let x2 = GrevLexTerm::from_exponents(&r, &[2, 0, 0]).unwrap();
        let xy = GrevLexTerm::from_exponents(&r, &[1, 1, 0]).unwrap();
        let y2 = GrevLexTerm::from_exponents(&r, &[0, 2, 0]).unwrap();
        let xz = GrevLexTerm::from_exponents(&r, &[1, 0, 1]).unwrap();
        let yz = GrevLexTerm::from_exponents(&r, &[0, 1, 1]).unwrap();
        let z2 = GrevLexTerm::from_exponents(&r, &[0, 0, 2]).unwrap();
        assert_eq!(x2.cmp(&xy), Ordering::Greater);
        assert_eq!(xy.cmp(&y2), Ordering::Greater);
        assert_eq!(y2.cmp(&xz), Ordering::Greater);
        assert_eq!(xz.cmp(&yz), Ordering::Greater);
        assert_eq!(yz.cmp(&z2), Ordering::Greater);
    }

    #[test]
    fn degrevlex_cmp_by_total_deg() {
        let r = mk_ring(3);
        let a = GrevLexTerm::from_exponents(&r, &[3, 0, 0]).unwrap();
        let b = GrevLexTerm::from_exponents(&r, &[0, 0, 2]).unwrap();
        assert_eq!(a.cmp(&b), Ordering::Greater);
    }

    #[test]
    fn degrevlex_cmp_equal() {
        let r = mk_ring(4);
        let a = GrevLexTerm::from_exponents(&r, &[1, 2, 3, 4]).unwrap();
        let b = GrevLexTerm::from_exponents(&r, &[1, 2, 3, 4]).unwrap();
        assert_eq!(a.cmp(&b), Ordering::Equal);
    }

    #[test]
    fn large_total_deg_cap_still_orders_correctly() {
        let r = mk_ring(3);
        let a = GrevLexTerm::from_exponents(&r, &[127, 50, 127]).unwrap();
        let b = GrevLexTerm::from_exponents(&r, &[50, 127, 127]).unwrap();
        assert_eq!(a.cmp(&b), Ordering::Greater);
    }

    #[test]
    fn degrevlex_tiebreak_on_last_variable() {
        let r = mk_ring(3);
        let xy2 = GrevLexTerm::from_exponents(&r, &[1, 2, 0]).unwrap();
        let y3 = GrevLexTerm::from_exponents(&r, &[0, 3, 0]).unwrap();
        let xyz = GrevLexTerm::from_exponents(&r, &[1, 1, 1]).unwrap();
        let y2z = GrevLexTerm::from_exponents(&r, &[0, 2, 1]).unwrap();
        let xz2 = GrevLexTerm::from_exponents(&r, &[1, 0, 2]).unwrap();
        let yz2 = GrevLexTerm::from_exponents(&r, &[0, 1, 2]).unwrap();
        let z3 = GrevLexTerm::from_exponents(&r, &[0, 0, 3]).unwrap();
        let sequence = [&xy2, &y3, &xyz, &y2z, &xz2, &yz2, &z3];
        for w in sequence.windows(2) {
            assert_eq!(w[0].cmp(w[1]), Ordering::Greater);
        }
    }

    #[test]
    fn cmp_degrevlex_packed_agrees_with_ring_based() {
        let r = mk_ring(3);
        let pairs: Vec<([u32; 3], [u32; 3])> = vec![
            ([0, 0, 0], [1, 0, 0]),
            ([1, 0, 0], [0, 1, 0]),
            ([2, 0, 0], [1, 1, 0]),
            ([1, 1, 0], [0, 2, 0]),
            ([1, 0, 1], [0, 1, 1]),
            ([3, 0, 0], [0, 0, 2]),
            ([127, 50, 127], [50, 127, 127]),
        ];
        for (a_exp, b_exp) in &pairs {
            let a = MonoTerm::from_exponents(&r, a_exp).unwrap();
            let b = MonoTerm::from_exponents(&r, b_exp).unwrap();
            let packed = a.cmp_degrevlex_packed(&b);
            let ring_based = a.cmp_degrevlex(&b, &r);
            assert_eq!(
                packed, ring_based,
                "disagreement for {:?} vs {:?}",
                a_exp, b_exp
            );
        }
    }
}
