//! Fixture tests: `ark_gb::compute_gb` vs pre-computed Singular
//! reference output.
//!
//! TODO: needs regeneration over Fr via Sage/Singular. The fixture
//! files in `tests/fixtures/*.gb.txt` were generated for Z/32003 and
//! are not directly usable now that ark-gb is generic over
//! `ark_ff::Field`. All tests in this file are `#[ignore]`d until
//! the fixtures are regenerated against the BLS12-381 scalar field.

use std::sync::Arc;

use ark_bls12_381::Fr;
use ark_ff::{One, Zero};
use ark_gb::compute_gb;
use ark_gb::monomial::Monomial;
use ark_gb::ordering::MonoOrder;
use ark_gb::poly::Poly;
use ark_gb::ring::Ring;

/// Parser state for a single polynomial line. The parser scans left
/// to right emitting `(coeff, Monomial)` pairs, then hands them to
/// `Poly::from_terms`.
struct LineParser<'a> {
    src: &'a [u8],
    pos: usize,
    ring: &'a Ring<Fr>,
    var_names: &'a [&'a str],
}

impl<'a> LineParser<'a> {
    fn new(src: &'a str, ring: &'a Ring<Fr>, var_names: &'a [&'a str]) -> Self {
        Self {
            src: src.as_bytes(),
            pos: 0,
            ring,
            var_names,
        }
    }

    fn peek(&self) -> Option<u8> {
        self.src.get(self.pos).copied()
    }

    fn bump(&mut self) -> Option<u8> {
        let b = self.peek()?;
        self.pos += 1;
        Some(b)
    }

    fn skip_whitespace(&mut self) {
        while let Some(b) = self.peek() {
            if b == b' ' || b == b'\t' {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    /// Try to read an unsigned integer. Returns 0 terms consumed if
    /// the next character isn't a digit.
    fn read_uint(&mut self) -> Option<u64> {
        let start = self.pos;
        while let Some(b) = self.peek() {
            if b.is_ascii_digit() {
                self.pos += 1;
            } else {
                break;
            }
        }
        if self.pos == start {
            return None;
        }
        let s = std::str::from_utf8(&self.src[start..self.pos]).ok()?;
        s.parse::<u64>().ok()
    }

    /// Try to match `var_names[i]` at the current position; return
    /// `Some(i)` and advance on success, else return `None`.
    fn match_var(&mut self) -> Option<usize> {
        // Longest-match rule: among names sharing a prefix, pick the
        // longest. Here the name list has no common prefixes (e.g.
        // `u0,u1,u2,u3`), but the longest-match rule still matters
        // for name sets like `a, ab` (hypothetical). We implement it
        // defensively.
        let mut best: Option<(usize, usize)> = None; // (var_idx, len)
        for (i, &name) in self.var_names.iter().enumerate() {
            let bytes = name.as_bytes();
            if self.src[self.pos..].starts_with(bytes)
                && best.map(|(_, l)| bytes.len() > l).unwrap_or(true)
            {
                best = Some((i, bytes.len()));
            }
        }
        let (idx, len) = best?;
        self.pos += len;
        Some(idx)
    }

    /// Read one term. Returns `None` at end of string or if a sign
    /// with no following term. Honours a leading `+`/`-` sign on
    /// each term; a positive sign at the start of the polynomial is
    /// implicit.
    ///
    /// Returns `(sign, coeff_opt, exponents)` where `sign` is +1 or
    /// -1, `coeff_opt` is the literal integer (or `None` if none was
    /// written, which means 1), and `exponents` is the per-variable
    /// exponent vector.
    fn read_term(&mut self, first: bool) -> Option<(i64, Option<u64>, Vec<u32>)> {
        self.skip_whitespace();
        let sign = match self.peek() {
            Some(b'+') => {
                self.bump();
                self.skip_whitespace();
                1i64
            }
            Some(b'-') => {
                self.bump();
                self.skip_whitespace();
                -1i64
            }
            Some(_) if first => 1,
            None => return None,
            Some(_) => 1,
        };

        // Possible leading integer coefficient.
        let coeff = self.read_uint();

        // Zero or more variable factors, each with optional exponent.
        let nvars = self.var_names.len();
        let mut exps = vec![0u32; nvars];
        let mut saw_var = false;
        loop {
            // Optional `*` separator before a variable factor.
            self.skip_whitespace();
            if self.peek() == Some(b'*') {
                self.bump();
                self.skip_whitespace();
            }
            let Some(var_idx) = self.match_var() else {
                break;
            };
            saw_var = true;
            // Optional exponent: either `^N` or a bare integer.
            self.skip_whitespace();
            let exp = if self.peek() == Some(b'^') {
                self.bump();
                self.skip_whitespace();
                self.read_uint().expect("expected integer after ^") as u32
            } else if let Some(b) = self.peek() {
                if b.is_ascii_digit() {
                    self.read_uint().expect("peeked digit") as u32
                } else {
                    1
                }
            } else {
                1
            };
            exps[var_idx] += exp;
        }

        // No term content at all: return None to signal EOL.
        if coeff.is_none() && !saw_var {
            return None;
        }

        Some((sign, coeff, exps))
    }

    fn parse(mut self) -> Poly<Fr> {
        let mut terms: Vec<(Fr, Monomial)> = Vec::new();
        let mut first = true;
        loop {
            match self.read_term(first) {
                None => break,
                Some((sign, coeff_opt, exps)) => {
                    first = false;
                    let mag = coeff_opt.unwrap_or(1);
                    let mag_fr = Fr::from(mag);
                    let signed = if sign < 0 { -mag_fr } else { mag_fr };
                    if !signed.is_zero() {
                        let m = Monomial::from_exponents(self.ring, &exps).unwrap();
                        terms.push((signed, m));
                    }
                }
            }
        }
        Poly::from_terms(self.ring, terms)
    }
}

fn parse_fixture(text: &str, ring: &Ring<Fr>, var_names: &[&str]) -> Vec<Poly<Fr>> {
    text.lines()
        .filter_map(|line| {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(LineParser::new(trimmed, ring, var_names).parse())
            }
        })
        .collect()
}

fn sort_gb_ascending(mut gb: Vec<Poly<Fr>>, ring: &Ring<Fr>) -> Vec<Poly<Fr>> {
    gb.sort_by(|a, b| {
        let la = a.leading().expect("nonzero").1;
        let lb = b.leading().expect("nonzero").1;
        la.cmp(lb, ring)
    });
    gb
}

fn mk_ring(nvars: u32) -> Arc<Ring<Fr>> {
    Arc::new(Ring::<Fr>::new(nvars, MonoOrder::DegRevLex).unwrap())
}

fn mono(r: &Ring<Fr>, e: &[u32]) -> Monomial {
    Monomial::from_exponents(r, e).unwrap()
}

#[test]
#[ignore = "needs regeneration over Fr via Sage/Singular"]
fn cyclic3_matches_singular_fixture() {
    let r = mk_ring(3);
    let f1 = Poly::from_terms(
        &r,
        vec![
            (Fr::one(), mono(&r, &[1, 0, 0])),
            (Fr::one(), mono(&r, &[0, 1, 0])),
            (Fr::one(), mono(&r, &[0, 0, 1])),
        ],
    );
    let f2 = Poly::from_terms(
        &r,
        vec![
            (Fr::one(), mono(&r, &[1, 1, 0])),
            (Fr::one(), mono(&r, &[0, 1, 1])),
            (Fr::one(), mono(&r, &[1, 0, 1])),
        ],
    );
    let f3 = Poly::from_terms(
        &r,
        vec![(Fr::one(), mono(&r, &[1, 1, 1])), (-Fr::one(), mono(&r, &[0, 0, 0]))],
    );

    let got = compute_gb(Arc::clone(&r), vec![f1, f2, f3]);
    let text = include_str!("fixtures/cyclic-3.gb.txt");
    let expected = sort_gb_ascending(parse_fixture(text, &r, &["x", "y", "z"]), &r);
    assert_eq!(
        got.len(),
        expected.len(),
        "cyclic-3 basis size: got {}, expected {}",
        got.len(),
        expected.len()
    );
    assert_eq!(got, expected, "cyclic-3 fixture mismatch");
}

#[test]
#[ignore = "needs regeneration over Fr via Sage/Singular"]
fn cyclic4_matches_singular_fixture() {
    let r = mk_ring(4);
    let m = |e: &[u32]| mono(&r, e);

    // cyclic-4 over (a, b, c, d):
    let f1 = Poly::from_terms(
        &r,
        vec![
            (Fr::one(), m(&[1, 0, 0, 0])),
            (Fr::one(), m(&[0, 1, 0, 0])),
            (Fr::one(), m(&[0, 0, 1, 0])),
            (Fr::one(), m(&[0, 0, 0, 1])),
        ],
    );
    // ab + bc + cd + da
    let f2 = Poly::from_terms(
        &r,
        vec![
            (Fr::one(), m(&[1, 1, 0, 0])),
            (Fr::one(), m(&[0, 1, 1, 0])),
            (Fr::one(), m(&[0, 0, 1, 1])),
            (Fr::one(), m(&[1, 0, 0, 1])),
        ],
    );
    // abc + bcd + cda + dab
    let f3 = Poly::from_terms(
        &r,
        vec![
            (Fr::one(), m(&[1, 1, 1, 0])),
            (Fr::one(), m(&[0, 1, 1, 1])),
            (Fr::one(), m(&[1, 0, 1, 1])),
            (Fr::one(), m(&[1, 1, 0, 1])),
        ],
    );
    // abcd - 1
    let f4 = Poly::from_terms(&r, vec![(Fr::one(), m(&[1, 1, 1, 1])), (-Fr::one(), m(&[0, 0, 0, 0]))]);

    let got = compute_gb(Arc::clone(&r), vec![f1, f2, f3, f4]);
    let text = include_str!("fixtures/cyclic-4.gb.txt");
    let expected = sort_gb_ascending(parse_fixture(text, &r, &["a", "b", "c", "d"]), &r);
    assert_eq!(
        got.len(),
        expected.len(),
        "cyclic-4 basis size: got {}, expected {}",
        got.len(),
        expected.len()
    );
    assert_eq!(got, expected, "cyclic-4 fixture mismatch");
}

#[test]
#[ignore = "needs regeneration over Fr via Sage/Singular"]
fn cyclic5_matches_singular_fixture() {
    let r = mk_ring(5);
    let m = |e: &[u32]| mono(&r, e);

    // cyclic-5 generators (standard form).
    let f1 = Poly::from_terms(
        &r,
        vec![
            (Fr::one(), m(&[1, 0, 0, 0, 0])),
            (Fr::one(), m(&[0, 1, 0, 0, 0])),
            (Fr::one(), m(&[0, 0, 1, 0, 0])),
            (Fr::one(), m(&[0, 0, 0, 1, 0])),
            (Fr::one(), m(&[0, 0, 0, 0, 1])),
        ],
    );
    // ab + bc + cd + de + ea
    let f2 = Poly::from_terms(
        &r,
        vec![
            (Fr::one(), m(&[1, 1, 0, 0, 0])),
            (Fr::one(), m(&[0, 1, 1, 0, 0])),
            (Fr::one(), m(&[0, 0, 1, 1, 0])),
            (Fr::one(), m(&[0, 0, 0, 1, 1])),
            (Fr::one(), m(&[1, 0, 0, 0, 1])),
        ],
    );
    // abc + bcd + cde + dea + eab
    let f3 = Poly::from_terms(
        &r,
        vec![
            (Fr::one(), m(&[1, 1, 1, 0, 0])),
            (Fr::one(), m(&[0, 1, 1, 1, 0])),
            (Fr::one(), m(&[0, 0, 1, 1, 1])),
            (Fr::one(), m(&[1, 0, 0, 1, 1])),
            (Fr::one(), m(&[1, 1, 0, 0, 1])),
        ],
    );
    // abcd + bcde + cdea + deab + eabc
    let f4 = Poly::from_terms(
        &r,
        vec![
            (Fr::one(), m(&[1, 1, 1, 1, 0])),
            (Fr::one(), m(&[0, 1, 1, 1, 1])),
            (Fr::one(), m(&[1, 0, 1, 1, 1])),
            (Fr::one(), m(&[1, 1, 0, 1, 1])),
            (Fr::one(), m(&[1, 1, 1, 0, 1])),
        ],
    );
    // abcde - 1
    let f5 = Poly::from_terms(
        &r,
        vec![(Fr::one(), m(&[1, 1, 1, 1, 1])), (-Fr::one(), m(&[0, 0, 0, 0, 0]))],
    );

    let got = compute_gb(Arc::clone(&r), vec![f1, f2, f3, f4, f5]);
    let text = include_str!("fixtures/cyclic-5.gb.txt");
    let expected = sort_gb_ascending(parse_fixture(text, &r, &["a", "b", "c", "d", "e"]), &r);
    assert_eq!(
        got.len(),
        expected.len(),
        "cyclic-5 basis size: got {}, expected {}",
        got.len(),
        expected.len()
    );
    assert_eq!(got, expected, "cyclic-5 fixture mismatch");
}

#[test]
#[ignore = "needs regeneration over Fr via Sage/Singular"]
fn katsura3_matches_singular_fixture() {
    // 4-variable system: u0, u1, u2, u3.
    let r = mk_ring(4);
    let m = |e: &[u32]| mono(&r, e);

    // u0 + 2*u1 + 2*u2 + 2*u3 - 1
    let f1 = Poly::from_terms(
        &r,
        vec![
            (Fr::one(), m(&[1, 0, 0, 0])),
            (Fr::from(2u64), m(&[0, 1, 0, 0])),
            (Fr::from(2u64), m(&[0, 0, 1, 0])),
            (Fr::from(2u64), m(&[0, 0, 0, 1])),
            (-Fr::one(), m(&[0, 0, 0, 0])),
        ],
    );
    // u0^2 + 2*u1^2 + 2*u2^2 + 2*u3^2 - u0
    let f2 = Poly::from_terms(
        &r,
        vec![
            (Fr::one(), m(&[2, 0, 0, 0])),
            (Fr::from(2u64), m(&[0, 2, 0, 0])),
            (Fr::from(2u64), m(&[0, 0, 2, 0])),
            (Fr::from(2u64), m(&[0, 0, 0, 2])),
            (-Fr::one(), m(&[1, 0, 0, 0])),
        ],
    );
    // 2*u0*u1 + 2*u1*u2 + 2*u2*u3 - u1
    let f3 = Poly::from_terms(
        &r,
        vec![
            (Fr::from(2u64), m(&[1, 1, 0, 0])),
            (Fr::from(2u64), m(&[0, 1, 1, 0])),
            (Fr::from(2u64), m(&[0, 0, 1, 1])),
            (-Fr::one(), m(&[0, 1, 0, 0])),
        ],
    );
    // 2*u0*u2 + 2*u1^2 - u2 + 2*u1*u3
    let f4 = Poly::from_terms(
        &r,
        vec![
            (Fr::from(2u64), m(&[1, 0, 1, 0])),
            (Fr::from(2u64), m(&[0, 2, 0, 0])),
            (-Fr::one(), m(&[0, 0, 1, 0])),
            (Fr::from(2u64), m(&[0, 1, 0, 1])),
        ],
    );

    let got = compute_gb(Arc::clone(&r), vec![f1, f2, f3, f4]);
    let text = include_str!("fixtures/katsura-3.gb.txt");
    let expected = sort_gb_ascending(parse_fixture(text, &r, &["u0", "u1", "u2", "u3"]), &r);
    assert_eq!(
        got.len(),
        expected.len(),
        "katsura-3 basis size: got {}, expected {}",
        got.len(),
        expected.len()
    );
    assert_eq!(got, expected, "katsura-3 fixture mismatch");
}

#[test]
#[ignore = "needs regeneration over Fr via Sage/Singular"]
fn parser_round_trips_monomial_forms() {
    // Smoke-test for the parser: Singular emits both `x2` and `x^2`
    // depending on the variable-name format. Both must parse to the
    // same polynomial.
    let r = mk_ring(3);
    let a = LineParser::new("x2y+3xy2-z3+1", &r, &["x", "y", "z"]).parse();
    let b = LineParser::new("x^2*y+3*x*y^2-z^3+1", &r, &["x", "y", "z"]).parse();
    assert_eq!(a, b);
    // Explicit canonical form.
    let m = |e: &[u32]| Monomial::from_exponents(&r, e).unwrap();
    let expected = Poly::from_terms(
        &r,
        vec![
            (Fr::one(), m(&[2, 1, 0])),
            (Fr::from(3u64), m(&[1, 2, 0])),
            (-Fr::one(), m(&[0, 0, 3])),
            (Fr::one(), m(&[0, 0, 0])),
        ],
    );
    assert_eq!(a, expected);
}
