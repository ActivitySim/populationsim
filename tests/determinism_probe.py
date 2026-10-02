"""Subprocess runner used by the end-to-end determinism regression test."""

import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from populationsim.core import config, inject, pipeline
from populationsim.integerizing import reproducibility
from populationsim.integerizing import simul_integerizer, single_integerizer


epsilon = float(sys.argv[1])
seed = int(sys.argv[2])
rng = np.random.default_rng(seed)
original_quantize = reproducibility.quantize_weights
quantize_calls = 0
perturbed_values = 0


def perturb_then_quantize(weights, quantum):
    global perturbed_values, quantize_calls

    quantize_calls += 1
    values = np.asarray(weights, dtype=np.float64)
    perturbed = values * (1.0 + epsilon * rng.standard_normal(values.shape))
    perturbed_values += np.count_nonzero(perturbed != values)
    values = perturbed
    return original_quantize(values, quantum)


single_integerizer.quantize_weights = perturb_then_quantize
simul_integerizer.quantize_weights = perturb_then_quantize

logging.disable(logging.CRITICAL)
root = Path(__file__).resolve().parents[1]
inject.reinject_decorated_tables()
inject.add_injectable(
    "configs_dir",
    [root / "tests" / "configs", root / "examples" / "example_test" / "configs"],
)
inject.add_injectable("output_dir", root / "tests" / "output")
inject.add_injectable("data_dir", root / "examples" / "example_test" / "data")
inject.clear_cache()
config.override_setting("INTEGERIZER_QUANTUM", 1e-6)

pipeline.run(
    models=[
        "input_pre_processor",
        "setup_data_structures",
        "initial_seed_balancing",
        "meta_control_factoring",
        "final_seed_balancing",
        "integerize_final_seed_weights",
        "sub_balancing.geography=TRACT",
        "sub_balancing.geography=TAZ",
        "expand_households",
    ],
    resume_after=None,
)

households = pipeline.get_table("expanded_household_ids")
result = {
    "hash": hashlib.sha256(
        pd.util.hash_pandas_object(households, index=True).values.tobytes()
    ).hexdigest(),
    "rows": len(households),
    "zone_counts": households.groupby("TAZ").size().to_dict(),
    "quantize_calls": quantize_calls,
    "perturbed_values": int(perturbed_values),
}
pipeline.close_pipeline()
print("DETERMINISM_RESULT " + json.dumps(result, sort_keys=True))
