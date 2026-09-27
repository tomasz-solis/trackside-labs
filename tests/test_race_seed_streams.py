"""Regression test: per-sim RNG streams must not collide across base seeds.

``default_rng(base_seed + sim_idx)`` made seed 43 sim i reuse seed 42 sim i+1's
stream, collapsing the seed-to-seed noise floor. ``_sim_rng`` uses a
``SeedSequence`` over the ``(base_seed, sim_idx)`` pair instead.
"""

from __future__ import annotations

from src.predictors.baseline.race.race_simulation import _sim_rng


def test_sim_rng_streams_disjoint_across_base_seeds():
    n_sims = 300
    draws_42 = {i: _sim_rng(42, i).random(50).tobytes() for i in range(n_sims)}
    draws_43 = {i: _sim_rng(43, i).random(50).tobytes() for i in range(n_sims)}

    # Old bug: seed 43 sim i produced the same stream as seed 42 sim i+1.
    for i in range(n_sims - 1):
        assert draws_43[i] != draws_42[i + 1]

    # No draw sequence should overlap across the two seed sets at all.
    assert set(draws_42.values()).isdisjoint(draws_43.values())


def test_sim_rng_deterministic_for_same_inputs():
    assert _sim_rng(42, 7).random(20).tobytes() == _sim_rng(42, 7).random(20).tobytes()
