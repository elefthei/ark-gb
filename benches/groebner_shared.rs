//! Shared Cyclic-n / Katsura-n polynomial generators.
//!
//! Used by the Criterion bench (`benches/groebner.rs`) and the
//! correctness regression suite (`tests/groebner_correctness.rs`,
//! `tests/groebner_sage.rs`).
//!
//! Ported from zippel's `benches/groebner_shared.rs`, which itself is a
//! translation of Singular's `polylib.lib` (`proc cyclic`, `proc katsura`,
//! `proc kat_var`). The arithmetic is reformulated for ark-gb's
//! [`Poly<F>`] / [`Ring<F>`] abstractions.

#![allow(dead_code)]

use std::sync::Arc;

use ark_bls12_381::Fr;
use ark_ff::One;
use ark_gb::monomial::Monomial;
use ark_gb::ordering::MonoOrder;
use ark_gb::poly::Poly;
use ark_gb::ring::Ring;

// ---------------------------------------------------------------------------
// Bench sizes — shared between benches and the correctness suite.
//
// Sizes are the largest n for which a single Buchberger run terminates in
// under a few minutes on a developer workstation (release build, BLS12-381
// scalar field). Going one step higher in any family takes much longer.
//
//   * Katsura-6 grevlex did not complete in 10 min (probed in zippel).
//   * Cyclic-6 / Cyclic-5 lex are intractable for our implementation.
// ---------------------------------------------------------------------------

/// Katsura-n under DegRevLex.
pub const KATSURA_GREVLEX_SIZES: &[usize] = &[3, 4, 5];
/// Katsura-n under block elimination order (split = n/2).
pub const KATSURA_ELIM_SIZES: &[usize] = &[3, 4, 5];
/// Cyclic-n under DegRevLex / Elim.
pub const CYCLIC_SIZES: &[usize] = &[4, 5];

// ---------------------------------------------------------------------------
// Ring + monomial constructors.
// ---------------------------------------------------------------------------

pub fn grevlex_ring(nvars: usize) -> Arc<Ring<Fr>> {
    Arc::new(
        Ring::<Fr>::new(nvars as u32, MonoOrder::DegRevLex)
            .expect("nvars within ark-gb monomial limits"),
    )
}

pub fn elim_ring(nvars: usize) -> Arc<Ring<Fr>> {
    let split = (nvars / 2) as u32;
    Arc::new(
        Ring::<Fr>::new(nvars as u32, MonoOrder::Elim { split })
            .expect("nvars within ark-gb monomial limits"),
    )
}

fn unit_mono(ring: &Ring<Fr>) -> Monomial {
    let nvars = ring.nvars() as usize;
    Monomial::from_exponents(ring, &vec![0u32; nvars]).unwrap()
}

fn var_mono(ring: &Ring<Fr>, i: usize) -> Monomial {
    let nvars = ring.nvars() as usize;
    let mut e = vec![0u32; nvars];
    e[i] = 1;
    Monomial::from_exponents(ring, &e).unwrap()
}

// ---------------------------------------------------------------------------
// Polynomial helpers — minimal calculus over `Poly<F>`.
// ---------------------------------------------------------------------------

/// Single-variable polynomial `x_i` (coefficient 1).
pub fn var_poly(ring: &Ring<Fr>, i: usize) -> Poly<Fr> {
    Poly::from_terms(ring, vec![(Fr::one(), var_mono(ring, i))])
}

/// Constant polynomial `1`.
pub fn one_poly(ring: &Ring<Fr>) -> Poly<Fr> {
    Poly::from_terms(ring, vec![(Fr::one(), unit_mono(ring))])
}

/// Iterated product `f_0 * f_1 * ... * f_{k-1}`. Empty product is `1`.
pub fn product_poly<I>(ring: &Ring<Fr>, factors: I) -> Poly<Fr>
where
    I: IntoIterator<Item = Poly<Fr>>,
{
    let mut acc = one_poly(ring);
    for f in factors {
        acc = acc.mul(&f, ring);
    }
    acc
}

// ---------------------------------------------------------------------------
// Cyclic-n — Singular polylib.lib `proc cyclic(int n)` translation.
// ---------------------------------------------------------------------------
//
//   ideal m = maxideal(1);
//   m = m[1..n], m[1..n];
//   for (j = 0; j <= n-2; j++) {
//     t = 0;
//     for (i = 1; i <= n; i++) { t = t + product(m, i..i+j); }
//     s = s + t;
//   }
//   s = s, product(m, 1..n) - 1;
//
// The `m[1..n], m[1..n]` doubling lets `product(m, i..i+j)` wrap; we
// implement the wrap with `% n`.

pub fn cyclic_polys(ring: &Ring<Fr>) -> Vec<Poly<Fr>> {
    let n = ring.nvars() as usize;
    assert!(n >= 1, "Cyclic-n requires n >= 1");

    let mut polys = Vec::with_capacity(n);
    for j in 0..n.saturating_sub(1) {
        let mut t = Poly::<Fr>::zero();
        for i in 0..n {
            let factors = (0..=j).map(|k| var_poly(ring, (i + k) % n));
            let prod = product_poly(ring, factors);
            t = t.add(&prod, ring);
        }
        polys.push(t);
    }
    // Final: x_0 * x_1 * ... * x_{n-1} - 1.
    let full = product_poly(ring, (0..n).map(|i| var_poly(ring, i)));
    polys.push(full.sub(&one_poly(ring), ring));
    polys
}

// ---------------------------------------------------------------------------
// Katsura-n — Singular polylib.lib `proc katsura` + `kat_var`.
// ---------------------------------------------------------------------------
//
// Singular takes integer argument n_arg; internally: n = n_arg - 1.
// `kat_var(i, n)`: if |i| <= n, returns var(|i|+1) (1-indexed), else 0.
//   s[1] = -1 + sum_{i=-n..=n} kat_var(i, n)
//   for (i = 0; i < n; i++) {
//     s[i+2] = -kat_var(i, n) + sum_{j=-n..=n} kat_var(j, n) * kat_var(i-j, n)
//   }
//
// Matches Sage's `sage.rings.ideal.Katsura(R, n_arg)`: `n_arg` variables
// produce `n_arg` generators with Singular-internal `n = n_arg - 1`.

pub fn katsura_polys(ring: &Ring<Fr>) -> Vec<Poly<Fr>> {
    let n_arg = ring.nvars() as usize;
    assert!(n_arg >= 1, "Katsura-n requires at least one variable");
    let n = (n_arg - 1) as isize;

    let kat_var = |i: isize| -> Option<Poly<Fr>> {
        let ai = i.unsigned_abs();
        if (ai as isize) <= n {
            Some(var_poly(ring, ai))
        } else {
            None
        }
    };

    let mut polys = Vec::with_capacity(n_arg);

    // Linear: -1 + sum_{i=-n..=n} kat_var(i, n)
    let mut lin = Poly::<Fr>::zero();
    for i in -n..=n {
        if let Some(v) = kat_var(i) {
            lin = lin.add(&v, ring);
        }
    }
    lin = lin.sub(&one_poly(ring), ring);
    polys.push(lin);

    // Quadratic: for i = 0..n:
    //   -kat_var(i, n) + sum_{j=-n..=n} kat_var(j, n) * kat_var(i-j, n)
    for i in 0..n {
        let mut q = Poly::<Fr>::zero();
        for j in -n..=n {
            if let (Some(a), Some(b)) = (kat_var(j), kat_var(i - j)) {
                let prod = a.mul(&b, ring);
                q = q.add(&prod, ring);
            }
        }
        if let Some(v) = kat_var(i) {
            q = q.sub(&v, ring);
        }
        polys.push(q);
    }
    polys
}

// ---------------------------------------------------------------------------
// Convenience: build (ring, polys) for a given family + size.
// ---------------------------------------------------------------------------

pub fn cyclic_input(n: usize, ring: &Arc<Ring<Fr>>) -> Vec<Poly<Fr>> {
    assert_eq!(ring.nvars() as usize, n);
    cyclic_polys(ring)
}

pub fn katsura_input(n: usize, ring: &Arc<Ring<Fr>>) -> Vec<Poly<Fr>> {
    assert_eq!(ring.nvars() as usize, n);
    katsura_polys(ring)
}
