//! Cross-check the ported Katsura / Cyclic generators against Sage's
//! published `sage.rings.ideal.Katsura` / `sage.rings.ideal.Cyclic`
//! small-n examples. See `benches/groebner_shared.rs` for the generator
//! code (translated from Singular's polylib.lib).

use ark_bls12_381::Fr;
use ark_gb::monomial::MonoTerm;
use ark_gb::poly::Poly;
use ark_gb::ring::Ring;

#[path = "../benches/groebner_shared.rs"]
mod shared;
use shared::{cyclic_polys, grevlex_ring, katsura_polys, var_poly};

/// Build a polynomial from `(coeff, [(var_index, power), ...])` terms.
fn grev_poly(ring: &Ring<Fr>, terms: &[(i64, &[(usize, usize)])]) -> Poly<Fr> {
    let nvars = ring.nvars() as usize;
    let unit = MonoTerm::from_exponents(ring, &vec![0u32; nvars]).unwrap();
    let mut out = Poly::<Fr>::zero();
    for (coeff, mono) in terms {
        let mag = Fr::from(coeff.unsigned_abs());
        let c = if *coeff < 0 { -mag } else { mag };
        let mut term = Poly::from_terms(ring, vec![(c, unit)]);
        for &(vi, power) in *mono {
            for _ in 0..power {
                let v = var_poly(ring, vi);
                term = term.mul(&v, ring);
            }
        }
        out = out.add(&term, ring);
    }
    out
}

/// Sage `Katsura(P, 3)` with `P = (x, y, z)`:
///   `(x + 2y + 2z - 1, x² + 2y² + 2z² - x, 2xy + 2yz - y)`
#[test]
fn katsura_3_matches_sage() {
    let ring = grevlex_ring(3);
    let got = katsura_polys(&ring);
    assert_eq!(got.len(), 3);

    let lin = grev_poly(
        &ring,
        &[(1, &[(0, 1)]), (2, &[(1, 1)]), (2, &[(2, 1)]), (-1, &[])],
    );
    let q0 = grev_poly(
        &ring,
        &[
            (1, &[(0, 2)]),
            (2, &[(1, 2)]),
            (2, &[(2, 2)]),
            (-1, &[(0, 1)]),
        ],
    );
    let q1 = grev_poly(
        &ring,
        &[
            (2, &[(0, 1), (1, 1)]),
            (2, &[(1, 1), (2, 1)]),
            (-1, &[(1, 1)]),
        ],
    );

    assert_eq!(got[0], lin);
    assert_eq!(got[1], q0);
    assert_eq!(got[2], q1);
}

/// Sage `Cyclic(P, 3)` with `P = (x, y, z)`:
///   `(x + y + z, xy + xz + yz, xyz - 1)`
#[test]
fn cyclic_3_matches_sage() {
    let ring = grevlex_ring(3);
    let got = cyclic_polys(&ring);
    assert_eq!(got.len(), 3);

    let deg1 = grev_poly(&ring, &[(1, &[(0, 1)]), (1, &[(1, 1)]), (1, &[(2, 1)])]);
    let deg2 = grev_poly(
        &ring,
        &[
            (1, &[(0, 1), (1, 1)]),
            (1, &[(1, 1), (2, 1)]),
            (1, &[(2, 1), (0, 1)]),
        ],
    );
    let deg3 = grev_poly(&ring, &[(1, &[(0, 1), (1, 1), (2, 1)]), (-1, &[])]);

    assert_eq!(got[0], deg1);
    assert_eq!(got[1], deg2);
    assert_eq!(got[2], deg3);
}

/// Smoke test: generators produce `n` polynomials for `n` variables
/// (matches Sage / Singular convention).
#[test]
fn generator_counts() {
    for n in 3..=5 {
        let ring = grevlex_ring(n);
        assert_eq!(katsura_polys(&ring).len(), n);
    }
    for n in 4..=5 {
        let ring = grevlex_ring(n);
        assert_eq!(cyclic_polys(&ring).len(), n);
    }
}
