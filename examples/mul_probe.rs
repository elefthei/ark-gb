use ark_bls12_381::Fr;
use ark_gb::monomial::Monomial;
use ark_gb::ordering::MonoOrder;
use ark_gb::ring::Ring;

#[inline(never)]
#[unsafe(no_mangle)]
pub extern "C" fn probe_mul(a: &Monomial, b: &Monomial, ring: &Ring<Fr>, out: &mut Monomial) {
    *out = a.mul(b, ring);
}

fn main() {
    let ring = Ring::<Fr>::new(25, MonoOrder::DegRevLex).unwrap();
    let a = Monomial::from_exponents(&ring, &[1u32; 25]).unwrap();
    let b = Monomial::from_exponents(&ring, &[2u32; 25]).unwrap();
    let mut out = Monomial::one(&ring);
    probe_mul(&a, &b, &ring, &mut out);
    eprintln!("{:?}", out.exponents(&ring));
}
