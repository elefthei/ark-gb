//! Monomial orderings as a trait.

use crate::field::Field;
use crate::monomial::Monomial;
use crate::ring::RingData;
use std::cmp::Ordering;
use std::fmt::Debug;

/// A monomial ordering.
pub trait MonoOrder: Sized + Copy + Send + Sync + Debug + PartialEq + Eq {
    /// Compare two monomials under this ordering.
    fn cmp<F, M>(&self, a: &M, b: &M, ring: &RingData<F>) -> Ordering
    where
        F: Field + Copy + Send + Sync,
        M: Monomial<F>;

    /// Validate that this order is compatible with a ring of `nvars`.
    fn validate(&self, nvars: u32) -> bool {
        let _ = nvars;
        true
    }
}

/// Graded reverse lexicographic order.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct DegRevLex;

/// Block elimination order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Elim {
    /// Number of variables in the elimination block.
    pub split: u32,
}

impl MonoOrder for DegRevLex {
    #[inline]
    fn cmp<F, M>(&self, a: &M, b: &M, ring: &RingData<F>) -> Ordering
    where
        F: Field + Copy + Send + Sync,
        M: Monomial<F>,
    {
        a.cmp_degrevlex(b, ring)
    }
}

impl MonoOrder for Elim {
    #[inline]
    fn cmp<F, M>(&self, a: &M, b: &M, ring: &RingData<F>) -> Ordering
    where
        F: Field + Copy + Send + Sync,
        M: Monomial<F>,
    {
        a.cmp_elim(b, ring, self.split as usize)
    }

    fn validate(&self, nvars: u32) -> bool {
        self.split <= nvars
    }
}
