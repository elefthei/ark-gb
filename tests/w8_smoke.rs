//! Smoke tests proving the const-generic-W refactor monomorphises
//! at non-default widths. With `W=8`, `max_vars = W*8 - 1 = 63`, so
//! we can build rings with var counts that are unreachable at the
//! default `W=4` (cap 31).

use std::sync::Arc;

use ark_bls12_381::Fr;
use ark_ff::One;
use ark_gb::compute_gb;
use ark_gb::monomial::{GrevLexTerm, MonoTerm, Monomial};
use ark_gb::poly::Poly;
use ark_gb::ring::Ring;

fn mono<const W: usize>(
    ring: &Ring<Fr, W>,
    exps: &[u32],
) -> GrevLexTerm<W> {
    GrevLexTerm::from(MonoTerm::<W>::from_exponents(ring, exps).unwrap())
}

/// `Ring::<Fr, 8>` accepts up to 63 variables, vs. 31 at default
/// `W=4`. Construct one with 40 vars (impossible at W=4) and run
/// `compute_gb` on `[x_0 - x_39, x_0 - 1]`. The basis must collapse
/// to the unit ideal.
#[test]
fn w8_unit_ideal_with_var_index_above_31() {
    const W: usize = 8;
    let ring = Arc::new(Ring::<Fr, W>::new(40).unwrap());

    // f1 = x_0 - x_39
    let mut e1 = vec![0u32; 40];
    e1[0] = 1;
    let mut e1b = vec![0u32; 40];
    e1b[39] = 1;
    let f1 = Poly::<Fr, GrevLexTerm<W>, W>::from_terms(
        &ring,
        vec![
            (Fr::one(), mono(&ring, &e1)),
            (-Fr::one(), mono(&ring, &e1b)),
        ],
    );

    // f2 = x_0 - x_39 - 1  (so f1 - f2 = 1)
    let zeros = vec![0u32; 40];
    let f2 = Poly::<Fr, GrevLexTerm<W>, W>::from_terms(
        &ring,
        vec![
            (Fr::one(), mono(&ring, &e1)),
            (-Fr::one(), mono(&ring, &e1b)),
            (-Fr::one(), mono(&ring, &zeros)),
        ],
    );

    let gb = compute_gb(ring.clone(), vec![f1, f2]);
    assert_eq!(gb.len(), 1, "GB must collapse to the unit ideal");
    let terms: Vec<_> = gb[0].iter().collect();
    assert_eq!(terms.len(), 1);
    assert_eq!(terms[0].0, Fr::one());
    assert_eq!(terms[0].1.exponents(&ring), zeros);
}

/// Tiny W=8 GB matches the W=4 result on a problem that fits both:
/// `[x + y, x - y]` over `Fr[x, y]`, leading monomials `{x, y}`.
#[test]
fn w8_matches_w4_on_shared_problem() {
    const W: usize = 8;
    let ring = Arc::new(Ring::<Fr, W>::new(2).unwrap());
    let f1 = Poly::<Fr, GrevLexTerm<W>, W>::from_terms(
        &ring,
        vec![
            (Fr::one(), mono(&ring, &[1, 0])),
            (Fr::one(), mono(&ring, &[0, 1])),
        ],
    );
    let f2 = Poly::<Fr, GrevLexTerm<W>, W>::from_terms(
        &ring,
        vec![
            (Fr::one(), mono(&ring, &[1, 0])),
            (-Fr::one(), mono(&ring, &[0, 1])),
        ],
    );
    let gb = compute_gb(ring.clone(), vec![f1, f2]);
    assert_eq!(gb.len(), 2);
    let mut lm: Vec<Vec<u32>> = gb
        .iter()
        .map(|g| g.leading().unwrap().1.exponents(&ring))
        .collect();
    lm.sort();
    assert_eq!(lm, vec![vec![0, 1], vec![1, 0]]);
}
