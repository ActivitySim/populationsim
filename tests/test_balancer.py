# PopulationSim
# See full license in LICENSE.txt.

import numpy as np
import pandas as pd
import numpy.testing as npt
import pytest
import time

from populationsim.balancer import ListBalancer
from populationsim.balancer import np_balancer
from populationsim.balancer import DEFAULT_MAX_ITERATIONS, MIN_CONTROL_VALUE


@pytest.mark.parametrize("use_numba", [True, False])
def test_Konduri(use_numba):

    # example from Konduri et al. "Enhanced Synthetic Population Generator..."
    # Journal of the Transportation Research Board, No. 2563

    # rows are elements for which factors are calculated, columns are constraints to be satisfied
    incidence_table = pd.DataFrame(
        {
            "hh_1": [1, 1, 1, 0, 0, 0, 0, 0],
            "hh_2": [0, 0, 0, 1, 1, 1, 1, 1],
            "p1": [1, 1, 2, 1, 0, 1, 2, 1],
            "p2": [1, 0, 1, 0, 2, 1, 1, 1],
            "p3": [1, 1, 0, 2, 1, 0, 2, 0],
        }
    )

    # one weight per row in incidence table
    initial_weights = np.asanyarray([1, 1, 1, 1, 1, 1, 1, 1])

    # column totals which the final weighted incidence table sums must satisfy
    control_totals = [35, 65, 91, 65, 104]

    # one for every column in incidence_table
    control_importance_weights = [100000] * len(control_totals)

    balancer = ListBalancer(
        incidence_table=incidence_table,
        initial_weights=initial_weights,
        control_totals=control_totals,
        control_importance_weights=control_importance_weights,
        lb_weights=0,
        ub_weights=30,
        master_control_index=None,
        max_iterations=DEFAULT_MAX_ITERATIONS,
        use_numba=use_numba,
    )

    status, weights, controls = balancer.balance()

    weighted_sum = [
        round((incidence_table.loc[:, c] * weights.final).sum(), 2)
        for c in controls.index
    ]

    published_final_weights = [1.36, 25.66, 7.98, 27.79, 18.45, 8.64, 1.47, 8.64]
    published_weighted_sum = [
        round((incidence_table.loc[:, c] * published_final_weights).sum(), 2)
        for c in controls.index
    ]
    npt.assert_almost_equal(weighted_sum, published_weighted_sum, decimal=1)

    npt.assert_almost_equal(weighted_sum, controls["control"], decimal=1)
    assert status["converged"]


def test_balancer_compare_numba_vs_py():
    np.random.seed(42)

    sample_count = 1000
    control_count = 100
    iterations = DEFAULT_MAX_ITERATIONS

    incidence = np.random.rand(control_count, sample_count)
    weights_initial = np.ones(sample_count)
    weights_lower_bound = MIN_CONTROL_VALUE
    weights_upper_bound = 1 / MIN_CONTROL_VALUE
    controls_constraint = np.random.uniform(1000, 5000, control_count)
    controls_importance = np.random.uniform(0.5, 2.0, control_count)
    master_control_index = -1

    # --- Run Python version ---
    start_py = time.perf_counter()
    w_py, r_py, s_py = np_balancer(
        sample_count,
        control_count,
        master_control_index,
        incidence,
        weights_initial,
        weights_lower_bound,
        weights_upper_bound,
        controls_constraint,
        controls_importance,
        iterations,
        use_numba=False,
    )
    duration_py = time.perf_counter() - start_py
    print(
        f"\nPython: {duration_py:.4f}s, Iter: {s_py['iter']}, Converged: {s_py['converged']}"
    )

    # Warm up Numba to exclude compilation time from the first run
    np_balancer(
        sample_count,
        control_count,
        master_control_index,
        incidence,
        weights_initial,
        weights_lower_bound,
        weights_upper_bound,
        controls_constraint,
        controls_importance,
        2,
        use_numba=True,
    )

    # --- Run Numba version ---
    start_numba = time.perf_counter()
    w_numba, r_numba, s_numba = np_balancer(
        sample_count,
        control_count,
        master_control_index,
        incidence,
        weights_initial,
        weights_lower_bound,
        weights_upper_bound,
        controls_constraint,
        controls_importance,
        iterations,
        use_numba=True,
    )
    duration_numba = time.perf_counter() - start_numba
    print(
        f"Numba: {duration_numba:.4f}s, Iter: {s_numba['iter']}, Converged: {s_numba['converged']}"
    )

    print(f"Speedup: {(duration_py / duration_numba):.2f}x")

    # --- Assertions ---
    assert np.allclose(w_numba, w_py, rtol=1e-4, atol=1e-4), "Weights mismatch"
    assert s_numba["converged"], "Numba version did not converge"
    assert s_py["converged"], "Python version did not converge"
    assert duration_numba < duration_py * 0.5, "Numba version not at least 2x faster"
