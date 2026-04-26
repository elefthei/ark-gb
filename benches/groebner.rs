//! Criterion benchmarks for `ark_gb::compute_gb` (the BBA Buchberger
//! implementation) on the canonical Katsura-n / Cyclic-n families used
//! by SymbolicData and the wider CAS ecosystem (Singular, Maple, Magma).
//!
//! ark-gb's `compute_gb` always returns the **reduced** Gröbner basis
//! (`tail_reduce_all` runs internally), so there is no separate "raw"
//! variant to bench — published CAS timings against the reduced GB are
//! the right comparison point and `compute_gb` is what they correspond
//! to.
//!
//! # Source of generators
//!
//! The Katsura / Cyclic generators (in `groebner_shared.rs`) are direct
//! ports of Singular `polylib.lib` procedures that Sage, Macaulay2, and
//! the SD tools (`sdsage`) all invoke via `singular.katsura(n)` /
//! `singular.cyclic(n)`. Correctness is cross-checked against Sage's
//! published small-n examples (`tests/groebner_sage.rs`) and via
//! Buchberger's iff property (`tests/groebner_correctness.rs`).
//!
//! Run with: `cargo bench --bench groebner`
//! Filter  : `cargo bench --bench groebner gb_katsura_grevlex/3`

use std::sync::Arc;
use std::time::Duration;

use ark_bls12_381::Fr;
use ark_gb::bba::compute_gb_serial;
use ark_gb::ordering::MonoOrder;
use ark_gb::poly::Poly;
use ark_gb::ring::Ring;
use ark_gb::validate::is_groebner_basis;
use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};

#[path = "groebner_shared.rs"]
mod shared;
use shared::{
    CYCLIC_SIZES, KATSURA_ELIM_SIZES, KATSURA_GREVLEX_SIZES, cyclic_polys, elim_ring, grevlex_ring,
    katsura_polys,
};

// ---------------------------------------------------------------------------
// Criterion knobs.
// ---------------------------------------------------------------------------

const SAMPLE_SIZE: usize = 10;
const MEASUREMENT_TIME: Duration = Duration::from_secs(30);

// ---------------------------------------------------------------------------
// Generic helpers.
// ---------------------------------------------------------------------------

/// Time `compute_gb_serial` (the deterministic serial path) on each `n`
/// in `sizes`. We use `iter_batched` so the per-iter input clone is
/// excluded from the measurement.
fn bench_compute_gb<O: MonoOrder + 'static, RB, IB>(
    c: &mut Criterion,
    group_name: &str,
    sizes: &[usize],
    ring_builder: RB,
    input_builder: IB,
) where
    RB: Fn(usize) -> Arc<Ring<Fr, O>>,
    IB: Fn(&Ring<Fr, O>) -> Vec<Poly<Fr>>,
{
    let mut group = c.benchmark_group(group_name);
    group.sample_size(SAMPLE_SIZE);
    group.measurement_time(MEASUREMENT_TIME);
    for &n in sizes {
        let ring = ring_builder(n);
        let input = input_builder(&ring);
        group.bench_with_input(
            BenchmarkId::from_parameter(n),
            &(ring, input),
            |b, (ring, input)| {
                b.iter_batched(
                    || (Arc::clone(ring), input.clone()),
                    |(r, i)| compute_gb_serial(r, i),
                    BatchSize::SmallInput,
                )
            },
        );
    }
    group.finish();
}

/// Time `is_groebner_basis(input, gb)` for each `n` in `sizes`.
/// Pre-computes the reduced GB once outside the iter loop; each iter
/// reduces every input generator + every S-poly modulo that basis
/// (Buchberger's iff check).
fn bench_validation<O: MonoOrder + 'static, RB, IB>(
    c: &mut Criterion,
    group_name: &str,
    sizes: &[usize],
    ring_builder: RB,
    input_builder: IB,
) where
    RB: Fn(usize) -> Arc<Ring<Fr, O>>,
    IB: Fn(&Ring<Fr, O>) -> Vec<Poly<Fr>>,
{
    let mut group = c.benchmark_group(group_name);
    group.sample_size(SAMPLE_SIZE);
    group.measurement_time(MEASUREMENT_TIME);
    for &n in sizes {
        let ring = ring_builder(n);
        let input = input_builder(&ring);
        let gb = compute_gb_serial(Arc::clone(&ring), input.clone());
        group.bench_with_input(
            BenchmarkId::from_parameter(n),
            &(ring, input, gb),
            |b, (ring, input, gb)| b.iter(|| is_groebner_basis(ring, input, gb).is_ok()),
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Bench entry points — one line of wiring per (family × order).
// ---------------------------------------------------------------------------

fn bench_gb_katsura_elim(c: &mut Criterion) {
    bench_compute_gb(
        c,
        "gb_katsura_elim",
        KATSURA_ELIM_SIZES,
        elim_ring,
        katsura_polys,
    );
}

fn bench_gb_katsura_grevlex(c: &mut Criterion) {
    bench_compute_gb(
        c,
        "gb_katsura_grevlex",
        KATSURA_GREVLEX_SIZES,
        grevlex_ring,
        katsura_polys,
    );
}

fn bench_gb_cyclic_elim(c: &mut Criterion) {
    bench_compute_gb(c, "gb_cyclic_elim", CYCLIC_SIZES, elim_ring, cyclic_polys);
}

fn bench_gb_cyclic_grevlex(c: &mut Criterion) {
    bench_compute_gb(
        c,
        "gb_cyclic_grevlex",
        CYCLIC_SIZES,
        grevlex_ring,
        cyclic_polys,
    );
}

fn bench_gb_validate_katsura(c: &mut Criterion) {
    bench_validation(
        c,
        "gb_validate_katsura",
        KATSURA_GREVLEX_SIZES,
        grevlex_ring,
        katsura_polys,
    );
}

fn bench_gb_validate_cyclic(c: &mut Criterion) {
    bench_validation(
        c,
        "gb_validate_cyclic",
        CYCLIC_SIZES,
        grevlex_ring,
        cyclic_polys,
    );
}

criterion_group!(
    benches,
    bench_gb_katsura_elim,
    bench_gb_katsura_grevlex,
    bench_gb_cyclic_elim,
    bench_gb_cyclic_grevlex,
    bench_gb_validate_katsura,
    bench_gb_validate_cyclic,
);
criterion_main!(benches);
