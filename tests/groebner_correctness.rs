//! Regression-detection suite for `compute_gb`.
//!
//! Runs on the same problem instances as the Criterion bench
//! (`benches/groebner.rs`), reusing the builders from
//! `benches/groebner_shared.rs`. Five layers of checks per
//! (family, n, order):
//!
//! 1. **Ideal inclusion + S-pair closure** — via
//!    [`ark_gb::validate::is_groebner_basis`] (Buchberger's iff).
//! 2. **Shuffle invariance** — reordering input generators
//!    (deterministic seed) yields the same reduced GB.
//! 3. **Standard-monomial count** — for these 0-dim ideals, the count
//!    is independent of monomial order, and equals `2^n` for Katsura-n
//!    and `n!` for Cyclic-n on small cases.
//! 4. **Cross-order dim invariance** — `DegRevLex` and `Elim` agree on
//!    the standard-monomial count.
//! 5. **Pinned reduced GBs** — Katsura-3 and Cyclic-3 under DegRevLex,
//!    with literal coefficients generated once via sympy
//!    (`sympy.polys.groebner.groebner(..., order='grevlex')`) and pasted
//!    here. This catches numeric regressions in the reducer.
//!
//! Tests are deliberately scoped to the small Katsura-3 / Cyclic-3
//! cases so the suite runs quickly and is suitable for `cargo test` on
//! every commit. Heavy cases (Katsura-4,5 / Cyclic-4) are covered by
//! the Criterion bench; regressions on larger sizes therefore surface
//! there, not here.
//!
//! ### Determinism
//!
//! ark-gb's parallel path is permutation-equivalent but not
//! bit-for-bit deterministic across thread counts. Shuffle-invariance
//! tests therefore drive `ark_gb::bba::compute_gb_serial` directly so
//! they don't depend on `ARK_GB_THREADS`.

use std::sync::Arc;

use ark_bls12_381::Fr;
use ark_ff::Field as ArkField;
use ark_gb::bba::compute_gb_serial;
use ark_gb::compute_gb;
use ark_gb::monomial::Monomial;
use ark_gb::poly::Poly;
use ark_gb::ring::Ring;
use ark_gb::validate::is_groebner_basis;

#[path = "../benches/groebner_shared.rs"]
mod shared;
use shared::{
    cyclic_polys, elim_ring, grevlex_ring, katsura_polys, one_poly, var_poly,
};

// ---------------------------------------------------------------------------
// Helpers.
// ---------------------------------------------------------------------------

/// Tiny deterministic shuffle (Fisher–Yates with a splitmix64 stream)
/// so the test is reproducible without pulling in `rand`.
fn deterministic_shuffle<X: Clone>(xs: &[X], seed: u64) -> Vec<X> {
    let mut out: Vec<X> = xs.to_vec();
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xDEAD_BEEF_DEAD_BEEF;
    let mut next = || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    for i in (1..out.len()).rev() {
        let j = (next() as usize) % (i + 1);
        out.swap(i, j);
    }
    out
}

fn assert_proper(label: &str, gb: &[Poly<Fr>], ring: &Ring<Fr>) {
    assert!(!gb.is_empty(), "[{label}] empty reduced GB");
    let nvars = ring.nvars() as usize;
    let zero_exps = vec![0u32; nvars];
    let unit = Monomial::from_exponents(ring, &zero_exps).unwrap();
    let unit_seen = gb.iter().any(|p| {
        let terms: Vec<_> = p.iter().collect();
        terms.len() == 1 && *terms[0].1 == unit
    });
    assert!(
        !unit_seen,
        "[{label}] reduced GB contains a constant — ideal is the whole ring"
    );
}

/// Count standard monomials of `gb` (= `dim_F(R/I)` when `I` is 0-dim).
/// Returns `None` if any variable lacks a pure-power leading monomial
/// in `gb`, meaning the ideal is positive-dimensional.
fn standard_monomial_count(ring: &Ring<Fr>, gb: &[Poly<Fr>]) -> Option<usize> {
    let nvars = ring.nvars() as usize;
    let lm_exps: Vec<Vec<u32>> = gb
        .iter()
        .filter_map(|p| p.leading().map(|(_, m)| m.exponents(ring)))
        .collect();

    // For each variable, smallest pure-power exponent appearing as a
    // leading monomial. Bounding box for std monomials: per var [0, d_v).
    let bounds: Vec<u32> = (0..nvars)
        .map(|i| {
            lm_exps
                .iter()
                .filter_map(|e| {
                    let nonzero: Vec<usize> = (0..nvars).filter(|k| e[*k] > 0).collect();
                    if nonzero.len() == 1 && nonzero[0] == i {
                        Some(e[i])
                    } else {
                        None
                    }
                })
                .min()
        })
        .collect::<Option<Vec<_>>>()?;

    fn divides(a: &[u32], b: &[u32]) -> bool {
        a.iter().zip(b).all(|(x, y)| x <= y)
    }

    let mut idx = vec![0u32; nvars];
    let mut count = 0usize;
    loop {
        let candidate: Vec<u32> = idx.clone();
        if !lm_exps.iter().any(|lm| divides(lm, &candidate)) {
            count += 1;
        }

        // Increment idx within bounding box.
        let mut i = 0;
        loop {
            if i == nvars {
                return Some(count);
            }
            idx[i] += 1;
            if idx[i] < bounds[i] {
                break;
            }
            idx[i] = 0;
            i += 1;
        }
    }
}

/// Ideal inclusion + S-pair closure (Layers 1a+1b) via Buchberger's iff.
fn assert_is_gb(label: &str, ring: &Arc<Ring<Fr>>, input: &[Poly<Fr>], gb: &[Poly<Fr>]) {
    if let Err(err) = is_groebner_basis(ring, input, gb) {
        panic!("[{label}] compute_gb output failed Buchberger's criterion: {err:?}");
    }
}

// ---------------------------------------------------------------------------
// Self-checks parametrized over (family, n, order).
// ---------------------------------------------------------------------------

fn run_self_checks(
    label: &str,
    ring: &Arc<Ring<Fr>>,
    input: &[Poly<Fr>],
    expected_dim: Option<usize>,
) -> Vec<Poly<Fr>> {
    let gb = compute_gb_serial(Arc::clone(ring), input.to_vec());
    assert_proper(label, &gb, ring);

    // Layers 1a + 1b — Buchberger's iff.
    assert_is_gb(label, ring, input, &gb);

    // Layer 2e — input-shuffle invariance.
    let shuffled = deterministic_shuffle(input, 0x00C0_FFEE_u64);
    let gb_shuf = compute_gb_serial(Arc::clone(ring), shuffled);
    assert_eq!(
        gb, gb_shuf,
        "[{label}] reduced GB depends on input order"
    );

    // Layer 2f — standard-monomial count.
    let dim = standard_monomial_count(ring, &gb);
    if let Some(exp) = expected_dim {
        assert_eq!(
            dim,
            Some(exp),
            "[{label}] standard-monomial count mismatch (got {dim:?}, expected Some({exp}))"
        );
    }
    gb
}

// ---------------------------------------------------------------------------
// Default-run cases — fast.
// ---------------------------------------------------------------------------

#[test]
fn katsura_3_grevlex_self_checks() {
    let ring = grevlex_ring(3);
    let inputs = katsura_polys(&ring);
    run_self_checks("katsura/3/grevlex", &ring, &inputs, Some(1 << 2));
}

#[test]
fn katsura_3_elim_self_checks() {
    let ring = elim_ring(3);
    let inputs = katsura_polys(&ring);
    run_self_checks("katsura/3/elim", &ring, &inputs, Some(1 << 2));
}

#[test]
fn katsura_3_dim_order_invariant() {
    let ring_g = grevlex_ring(3);
    let ring_e = elim_ring(3);
    let gb_g = compute_gb(Arc::clone(&ring_g), katsura_polys(&ring_g));
    let gb_e = compute_gb(Arc::clone(&ring_e), katsura_polys(&ring_e));
    let d_g = standard_monomial_count(&ring_g, &gb_g);
    let d_e = standard_monomial_count(&ring_e, &gb_e);
    assert_eq!(
        d_g, d_e,
        "katsura/3 standard-monomial count differs between grevlex and elim (got {d_g:?} vs {d_e:?})"
    );
    assert_eq!(d_g, Some(1 << 2), "katsura/3 should be 0-dim with 4 std mons");
}

#[test]
fn cyclic_3_grevlex_self_checks() {
    let ring = grevlex_ring(3);
    let inputs = cyclic_polys(&ring);
    run_self_checks("cyclic/3/grevlex", &ring, &inputs, None);
}

#[test]
fn cyclic_3_elim_self_checks() {
    let ring = elim_ring(3);
    let inputs = cyclic_polys(&ring);
    run_self_checks("cyclic/3/elim", &ring, &inputs, None);
}

#[test]
fn cyclic_3_dim_order_invariant() {
    let ring_g = grevlex_ring(3);
    let ring_e = elim_ring(3);
    let gb_g = compute_gb(Arc::clone(&ring_g), cyclic_polys(&ring_g));
    let gb_e = compute_gb(Arc::clone(&ring_e), cyclic_polys(&ring_e));
    let d_g = standard_monomial_count(&ring_g, &gb_g);
    let d_e = standard_monomial_count(&ring_e, &gb_e);
    assert_eq!(
        d_g, d_e,
        "cyclic/3 standard-monomial count differs between grevlex and elim ({d_g:?} vs {d_e:?})"
    );
    // Cyclic-3 is 0-dim with 6 finite roots over an algebraically
    // closed field (well-known); pin alongside the order-invariance
    // check.
    assert_eq!(d_g, Some(6), "cyclic/3 should be 0-dim with 6 std mons");
}

// ---------------------------------------------------------------------------
// Layer 3g — sympy-pinned reduced GBs.
//
// Reference values were generated once with sympy:
//
//   sympy.polys.groebner.groebner(I, *gens, order='grevlex')
//
// where the generators are k0 > k1 > k2 (Katsura-3) or c0 > c1 > c2
// (Cyclic-3). Pasted here as integer-rational literals
// `(numerator, denominator, [(var_idx, power), ...])`.
// ---------------------------------------------------------------------------

type PolyLit<'a> = &'a [(i64, u64, &'a [(usize, usize)])];
type BasisLit<'a> = &'a [PolyLit<'a>];

fn fr_from_rational(num: i64, den: u64) -> Fr {
    let mag = Fr::from(num.unsigned_abs());
    let n = if num < 0 { -mag } else { mag };
    let d = Fr::from(den);
    n * d.inverse().expect("denominator must be non-zero")
}

fn poly_from_lit(ring: &Ring<Fr>, lit: PolyLit<'_>) -> Poly<Fr> {
    let mut out = Poly::<Fr>::zero();
    for &(num, den, mono) in lit {
        let c = fr_from_rational(num, den);
        // Build c * prod(var_poly(vi)^pow).
        let mut term = Poly::from_terms(
            ring,
            vec![(c, Monomial::from_exponents(ring, &vec![0u32; ring.nvars() as usize]).unwrap())],
        );
        for &(vi, pow) in mono {
            for _ in 0..pow {
                let v = var_poly(ring, vi);
                term = term.mul(&v, ring);
            }
        }
        out = out.add(&term, ring);
    }
    out
}

fn pinned_basis(ring: &Ring<Fr>, lits: BasisLit<'_>) -> Vec<Poly<Fr>> {
    lits.iter().map(|p| poly_from_lit(ring, p)).collect()
}

/// On a true GB, `compute_gb_serial` is idempotent up to monicization
/// and sort. We canonicalize the pinned reference by re-running it
/// through the pipeline, then byte-compare with our own GB.
fn assert_matches_pin(
    label: &str,
    ring: &Arc<Ring<Fr>>,
    our_gb: &[Poly<Fr>],
    pinned: Vec<Poly<Fr>>,
) {
    let pinned_canon = compute_gb_serial(Arc::clone(ring), pinned);
    assert_eq!(
        our_gb.len(),
        pinned_canon.len(),
        "[{label}] reduced-GB length differs from sympy reference ({} vs {})",
        our_gb.len(),
        pinned_canon.len()
    );
    assert_eq!(
        our_gb, &pinned_canon[..],
        "[{label}] reduced GB doesn't match sympy reference"
    );
}

// --- Pinned literals (generated by sympy, see header) -----------------------

const KATSURA_3_GB: BasisLit<'static> = &[
    &[
        (7, 1, &[(1, 1)]),
        (210, 1, &[(2, 3)]),
        (-79, 1, &[(2, 2)]),
        (3, 1, &[(2, 1)]),
    ],
    &[
        (5, 1, &[(1, 2)]),
        (-1, 1, &[(1, 1)]),
        (-3, 1, &[(2, 2)]),
        (1, 1, &[(2, 1)]),
    ],
    &[
        (10, 1, &[(1, 1), (2, 1)]),
        (-1, 1, &[(1, 1)]),
        (12, 1, &[(2, 2)]),
        (-4, 1, &[(2, 1)]),
    ],
    &[
        (1, 1, &[(0, 1)]),
        (2, 1, &[(1, 1)]),
        (2, 1, &[(2, 1)]),
        (-1, 1, &[]),
    ],
];

const CYCLIC_3_GB: BasisLit<'static> = &[
    &[(1, 1, &[(2, 3)]), (-1, 1, &[])],
    &[
        (1, 1, &[(1, 2)]),
        (1, 1, &[(1, 1), (2, 1)]),
        (1, 1, &[(2, 2)]),
    ],
    &[(1, 1, &[(0, 1)]), (1, 1, &[(1, 1)]), (1, 1, &[(2, 1)])],
];

// --- Pinned tests -----------------------------------------------------------

#[test]
fn katsura_3_grevlex_pinned() {
    let ring = grevlex_ring(3);
    let our_gb = compute_gb_serial(Arc::clone(&ring), katsura_polys(&ring));
    let pinned = pinned_basis(&ring, KATSURA_3_GB);
    assert_matches_pin("katsura/3/grevlex/pinned", &ring, &our_gb, pinned);
}

#[test]
fn cyclic_3_grevlex_pinned() {
    let ring = grevlex_ring(3);
    let our_gb = compute_gb_serial(Arc::clone(&ring), cyclic_polys(&ring));
    let pinned = pinned_basis(&ring, CYCLIC_3_GB);
    assert_matches_pin("cyclic/3/grevlex/pinned", &ring, &our_gb, pinned);
}

// ---------------------------------------------------------------------------
// Trivial sanity — the helpers themselves.
// ---------------------------------------------------------------------------

#[test]
fn helpers_one_poly_is_unit() {
    let ring = grevlex_ring(3);
    let one = one_poly(&ring);
    let v0 = var_poly(&ring, 0);
    let prod = v0.mul(&one, &ring);
    assert_eq!(prod, v0, "1 * x_0 must equal x_0");
    let _ = Fr::ONE;
}
