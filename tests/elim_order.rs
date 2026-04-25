//! Tests for `MonoOrder::Elim` (block elimination order).
//!
//! The block weight order with `split = k` is designed so that a
//! Gröbner basis under that order, intersected with the variables
//! `x_k, ..., x_{n-1}`, equals a Gröbner basis of the elimination
//! ideal `I ∩ F[x_k, ..., x_{n-1}]`.
//!
//! We exercise the smallest non-trivial case `[xy - 1, x + y]` in
//! `Fr[x, y]` with `split = 1` (eliminating `x`). The elimination
//! ideal is `(y² + 1)`, so the GB must contain a polynomial whose
//! leading term lives entirely in `y` and whose support, after
//! reduction, is `{y², 1}`.

use std::sync::Arc;

use ark_bls12_381::Fr;
use ark_ff::One;
use ark_gb::compute_gb;
use ark_gb::monomial::Monomial;
use ark_gb::ordering::MonoOrder;
use ark_gb::poly::Poly;
use ark_gb::ring::Ring;

fn mono(ring: &Ring<Fr>, exps: &[u32]) -> Monomial {
    Monomial::from_exponents(ring, exps).unwrap()
}

#[test]
fn elim_xy_minus_1_x_plus_y() {
    // Ring: Fr[x, y] with x = var 0 (the eliminated variable) and
    // y = var 1.
    let ring = Arc::new(Ring::<Fr>::new(2, MonoOrder::Elim { split: 1 }).unwrap());

    // f1 = x*y - 1
    let f1 = Poly::from_terms(
        &ring,
        vec![
            (Fr::one(), mono(&ring, &[1, 1])),
            (-Fr::one(), mono(&ring, &[0, 0])),
        ],
    );
    // f2 = x + y
    let f2 = Poly::from_terms(
        &ring,
        vec![
            (Fr::one(), mono(&ring, &[1, 0])),
            (Fr::one(), mono(&ring, &[0, 1])),
        ],
    );

    let gb = compute_gb(ring.clone(), vec![f1, f2]);

    // Look for a polynomial entirely in y (no x). The elimination
    // ideal of `(xy - 1, x + y)` is `(y² + 1)`, so we must find a
    // generator whose every term has `x`-exponent 0.
    let mut found_pure_y = false;
    for (i, p) in gb.iter().enumerate() {
        let exps = p.iter().map(|(_, m)| m.exponents(&ring)).collect::<Vec<_>>();
        let pure_y = !exps.is_empty() && exps.iter().all(|e| e[0] == 0);
        if pure_y {
            found_pure_y = true;
            // Sanity: the support should be {y^2, 1} (in either order
            // depending on normalization). We just require that y^2
            // appears.
            assert!(
                exps.iter().any(|e| e[0] == 0 && e[1] == 2),
                "GB[{i}] is x-free but missing y^2 term: {exps:?}"
            );
            // And a constant term must appear (it's `y^2 + 1` up to
            // a unit).
            assert!(
                exps.iter().any(|e| e[0] == 0 && e[1] == 0),
                "GB[{i}] is x-free but missing constant term: {exps:?}"
            );
        }
    }
    assert!(
        found_pure_y,
        "Elim order failed to produce an x-free generator; GB had {} polys",
        gb.len()
    );
}

#[test]
fn elim_with_split_zero_matches_degrevlex() {
    // Sanity: Elim { split: 0 } is degrevlex on the full block, so
    // the resulting GB should be identical (up to permutation /
    // normalisation) to the DegRevLex GB on the same input.
    let ring_e = Arc::new(Ring::<Fr>::new(2, MonoOrder::Elim { split: 0 }).unwrap());
    let ring_d = Arc::new(Ring::<Fr>::new(2, MonoOrder::DegRevLex).unwrap());

    // Input: a single non-trivial pair.
    let make_input = |r: &Ring<Fr>| {
        vec![
            Poly::from_terms(
                r,
                vec![
                    (Fr::one(), mono(r, &[2, 0])),
                    (-Fr::one(), mono(r, &[0, 1])),
                ],
            ),
            Poly::from_terms(
                r,
                vec![
                    (Fr::one(), mono(r, &[1, 1])),
                    (-Fr::one(), mono(r, &[0, 0])),
                ],
            ),
        ]
    };
    let gb_e = compute_gb(ring_e.clone(), make_input(&ring_e));
    let gb_d = compute_gb(ring_d.clone(), make_input(&ring_d));
    assert_eq!(
        gb_e.len(),
        gb_d.len(),
        "Elim {{ split: 0 }} must produce same-size GB as DegRevLex"
    );
}

#[test]
fn elim_constructor_rejects_oversized_split() {
    assert!(Ring::<Fr>::new(3, MonoOrder::Elim { split: 4 }).is_none());
    assert!(Ring::<Fr>::new(3, MonoOrder::Elim { split: 3 }).is_some());
    assert!(Ring::<Fr>::new(3, MonoOrder::Elim { split: 0 }).is_some());
}
