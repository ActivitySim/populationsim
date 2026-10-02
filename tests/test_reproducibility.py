import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from populationsim.integerizing.reproducibility import quantize_weights
from populationsim.integerizing.smart_round import smart_round


EXPECTED_QUANTIZED_HASH = (
    "b325ce29b82162222a814524369f92d838cdd92c5dcecd241433f82c55b27391"
)


def test_quantize_weights_is_disabled_by_default():
    weights = np.array([0.300000000001, 1.999999999999])
    np.testing.assert_array_equal(quantize_weights(weights, None), weights)


def test_quantize_weights_canonicalizes_near_ties():
    a = np.array([0.300000000001, 0.299999999999])
    b = np.array([0.300000000002, 0.299999999998])
    np.testing.assert_array_equal(
        quantize_weights(a, 1e-6), quantize_weights(b, 1e-6)
    )


def test_quantize_weights_preserves_positive_eligibility():
    quantized = quantize_weights(np.array([0.0, 1e-12, 2e-12]), 1e-6)
    assert quantized[0] == 0
    assert quantized[1] == quantized[2] > 0


@pytest.mark.parametrize("quantum", [True, -1, np.inf, np.nan, "invalid"])
def test_quantize_weights_rejects_invalid_quantum(quantum):
    with pytest.raises((TypeError, ValueError), match="INTEGERIZER_QUANTUM"):
        quantize_weights([0.5], quantum)


def test_smart_round_uses_position_to_break_exact_ties():
    rounded = smart_round(
        int_weights=np.zeros(4, dtype=int),
        resid_weights=np.full(4, 0.5),
        target_sum=2,
        tie_break_by_position=True,
    )
    np.testing.assert_array_equal(rounded, [1, 1, 0, 0])


def test_smart_round_preserves_historical_default():
    rounded = smart_round(
        int_weights=np.zeros(4, dtype=int),
        resid_weights=np.full(4, 0.5),
        target_sum=2,
    )
    np.testing.assert_array_equal(rounded, [0, 0, 1, 1])


def test_quantized_pipeline_resists_boundary_perturbation():
    # EXPECTED_QUANTIZED_HASH is a fixed literal, so the CI matrix doubles as a
    # cross-platform check: every OS asserts the same fingerprint. Agreement
    # there confirms quantization also makes CBC's own platform-dependent tie
    # choice converge, not just our pre-solver rounding.
    root = Path(__file__).resolve().parents[1]
    probe = Path(__file__).with_name("determinism_probe.py")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root)

    def run_probe(epsilon, seed):
        completed = subprocess.run(
            [sys.executable, str(probe), str(epsilon), str(seed)],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
            check=True,
        )
        result_line = next(
            line
            for line in completed.stdout.splitlines()
            if line.startswith("DETERMINISM_RESULT ")
        )
        return json.loads(result_line.removeprefix("DETERMINISM_RESULT "))

    scenarios = [(0, 1)] + [
        (epsilon, seed) for epsilon in (1e-12, 1e-9) for seed in (1, 2, 3)
    ]
    results = [(epsilon, run_probe(epsilon, seed)) for epsilon, seed in scenarios]

    for epsilon, result in results:
        assert result["quantize_calls"] > 0
        if epsilon == 0:
            assert result["perturbed_values"] == 0
        else:
            assert result["perturbed_values"] > 0
        assert result["rows"] == 1500
        assert result["hash"] == EXPECTED_QUANTIZED_HASH

    canonical_outputs = {
        (result["hash"], tuple(sorted(result["zone_counts"].items())))
        for _, result in results
    }
    assert len(canonical_outputs) == 1
