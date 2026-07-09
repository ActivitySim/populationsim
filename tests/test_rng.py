"""Determinism guards for the random number generator.

Cross-platform regression tests are relaxed to a zone-level tolerance (see
``tests.assert_expanded_close``) because the MILP solvers produce different but
equally valid integer solutions per platform. That relaxation deliberately does
*not* cover the RNG: seeded numpy ``RandomState`` (Mersenne Twister) is
deterministic given a seed, on every platform. These tests pin that down so a
regression that breaks seeding/reproducibility is caught immediately, and the
end-to-end double-run test confirms the whole pipeline is reproducible on a
single machine.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from populationsim.core import inject, pipeline, tracing
from populationsim.core.random import Random


def test_rng_sequence_is_stable():
    """The package RNG produces a fixed, known sequence for a fixed base seed.

    This exercises the exact surfaces the pipeline relies on:
    ``get_external_rng`` (used by ``expand_households`` to choose seed
    households) and the per-step ``get_global_rng``.
    """
    # External (step-independent) rng -- the generator used at
    # populationsim/steps/expand_households.py to make repeatable hh choices.
    rng = Random()
    rng.set_base_seed(0)
    external = rng.get_external_rng("expand_households").rand(5)
    np.testing.assert_allclose(
        external,
        [0.588287535536, 0.330113270395, 0.678519979423, 0.662137545976, 0.984482317067],
        rtol=0,
        atol=1e-12,
    )

    # Per-step global rng -- reseeded to [base_seed, step_seed] at begin_step.
    rng = Random()
    rng.set_base_seed(0)
    rng.begin_step("initial_seed_balancing")
    global_seq = rng.get_global_rng().rand(5)
    np.testing.assert_allclose(
        global_seq,
        [0.110703758714, 0.27713119323, 0.534727114207, 0.229387720433, 0.258099309877],
        rtol=0,
        atol=1e-12,
    )


def test_rng_seed_is_repeatable_and_seed_sensitive():
    """Same seed -> same stream; different base seed -> different stream."""
    def draw(seed):
        r = Random()
        r.set_base_seed(seed)
        return r.get_external_rng("expand_households").rand(3)

    np.testing.assert_array_equal(draw(0), draw(0))  # repeatable
    assert not np.allclose(draw(0), draw(1))  # seed-sensitive


def _setup_test_steps_injectables():
    """Mirror the injectable setup used by tests/test_steps.py::setup_function."""
    example_dir = Path(__file__).parent.parent / "examples"
    example_configs_dir = example_dir / "example_test" / "configs"
    configs_dir = Path(__file__).parent / "configs"
    output_dir = Path(__file__).parent / "output"
    data_dir = example_dir / "example_test" / "data"

    inject.reinject_decorated_tables()
    inject.add_injectable("configs_dir", [configs_dir, example_configs_dir])
    inject.add_injectable("output_dir", output_dir)
    inject.add_injectable("data_dir", data_dir)
    inject.clear_cache()
    tracing.config_logger()


def teardown_function(func):
    if pipeline.is_open():
        pipeline.close_pipeline()
    inject.clear_cache()
    inject.reinject_decorated_tables()


def test_pipeline_is_reproducible():
    """Running the same pipeline twice on one machine yields identical output.

    This is a *same-machine* determinism guard, so a bit-for-bit ``.equals()``
    comparison is valid (the cross-platform solver drift that motivates the
    relaxed comparator does not apply within a single machine).
    """
    _MODELS = [
        "input_pre_processor",
        "setup_data_structures",
        "initial_seed_balancing",
        "meta_control_factoring",
        "final_seed_balancing",
        "integerize_final_seed_weights",
        "sub_balancing.geography=TRACT",
        "sub_balancing.geography=TAZ",
        "expand_households",
    ]

    def run_once():
        _setup_test_steps_injectables()
        pipeline.run(models=_MODELS, resume_after=None)
        result = pipeline.get_table("expanded_household_ids").copy()
        pipeline.close_pipeline()
        inject.clear_cache()
        inject.reinject_decorated_tables()
        return result

    first = run_once()
    second = run_once()

    assert isinstance(first, pd.DataFrame)
    assert not first.empty
    assert first.equals(second)
