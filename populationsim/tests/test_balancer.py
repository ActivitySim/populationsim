# PopulationSim
# See full license in LICENSE.txt.

import pandas as pd

import numpy.testing as npt
import pytest

from ..balancer import ListBalancer


def test_Konduri():

    # example from Konduri et al. "Enhanced Synthetic Population Generator..."
    # Journal of the Transportation Research Board, No. 2563

    # rows are elements for which factors are calculated, columns are constraints to be satisfied
    incidence_table = pd.DataFrame({
        'hh_1': [1, 1, 1, 0, 0, 0, 0, 0],
        'hh_2': [0, 0, 0, 1, 1, 1, 1, 1],
        'p1': [1, 1, 2, 1, 0, 1, 2, 1],
        'p2': [1, 0, 1, 0, 2, 1, 1, 1],
        'p3': [1, 1, 0, 2, 1, 0, 2, 0],
    })

    # one weight per row in incidence table
    initial_weights = [1, 1, 1, 1, 1, 1, 1, 1]

    # column totals which the final weighted incidence table sums must satisfy
    control_totals = [35, 65, 91, 65, 104]

    # one for every column in incidence_table
    control_importance_weights = 100000

    balancer = ListBalancer(
        incidence_table=incidence_table,
        initial_weights=initial_weights,
        control_totals=control_totals,
        control_importance_weights=control_importance_weights,
        master_control_index=None
        )

    status = balancer.balance()

    weights = balancer.weights
    controls = balancer.controls

    weighted_sum = \
        [round((incidence_table.ix[:, c] * weights.final).sum(), 2) for c in controls.index]

    published_final_weights = [1.36, 25.66, 7.98, 27.79, 18.45, 8.64, 1.47, 8.64]
    published_weighted_sum = [
        round((incidence_table.ix[:, c] * published_final_weights).sum(), 2)
        for c in controls.index]
    npt.assert_almost_equal(weighted_sum, published_weighted_sum, decimal=1)

    npt.assert_almost_equal(weighted_sum, controls.constraint, decimal=1)
    assert status['converged']
