//! MonoTerm orderings.

/// The monomial ordering of a [`Ring`](crate::ring::Ring).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MonoOrder {
    /// Graded reverse lexicographic order: compare total degree first
    /// (larger = greater); break ties by the leftmost differing variable,
    /// where the *smaller* exponent on that variable wins.
    DegRevLex,
    /// Block elimination order with the first `split` variables in the
    /// elimination block.
    ///
    /// Concretely: monomial `m1 > m2` iff
    /// 1. the sum of exponents in variables `0 .. split` (the "block
    ///    weight") is strictly greater for `m1`, OR
    /// 2. the block weights are equal and `m1 > m2` under degrevlex on
    ///    the full variable set.
    ///
    /// Equivalent to Singular's `(a(1,..,1,0,..,0), dp)` block weight
    /// order. With `split = 0` it degenerates to plain degrevlex; with
    /// `split = nvars` it eliminates everything (effectively still
    /// degrevlex with a redundant tie-break). The standard use is
    /// computing an elimination ideal: a Gröbner basis under
    /// `Elim { split: k }` intersected with `F[x_k, ..., x_{n-1}]`
    /// equals a Gröbner basis of the ideal restricted to those
    /// variables.
    Elim {
        /// Number of variables in the elimination block (the leading
        /// variables `0 .. split`). Must satisfy `split <= nvars` at
        /// [`Ring::new`](crate::ring::Ring::new).
        split: u32,
    },
}
