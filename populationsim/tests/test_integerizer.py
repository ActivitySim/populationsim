# PopulationSim
# See full license in LICENSE.txt.

import pandas as pd

import numpy.testing as npt
import pytest

from ..integerizer import do_integerizing


def test_integerizer():

    # rows are elements for which factors are calculated, columns are constraints to be satisfied
    incidence_table = pd.DataFrame({
        'hh_1': [1, 1, 1, 0, 0, 0, 0, 0],
        'hh_2': [0, 0, 0, 1, 1, 1, 1, 1],
        'p1': [1, 1, 2, 1, 0, 1, 2, 1],
        'p2': [1, 0, 1, 0, 2, 1, 1, 1],
        'p3': [1, 1, 0, 2, 1, 0, 2, 0],
    })

    household_based_controls = [True, True, True, False, False, False]
    master_control_index = 0

    # one weight per row in incidence table
    initial_weights = [1, 1, 1, 1, 1, 1, 1, 1]

    # column totals which the final weighted incidence table sums must satisfy
    control_totals = [35, 65, 91, 65, 104]

    control_importance_weights = 100000

    final_weights = \
        [1.362893, 25.658290, 7.978812, 27.789651, 18.451021, 8.641589, 1.476104, 8.641589]

    relaxation_factors = \
        [0.999999, 1.000000, 0.999999, 1.000042, 1.000047, 1.000036]

    zot = do_integerizing(
        label='label',
        id=42,
        incidence_table=incidence_table,
        control_totals=control_totals,
        control_importance_weights=control_importance_weights,
        initial_weights=initial_weights,
        final_weights=final_weights,
        relaxation_factors=relaxation_factors,
        household_based_controls=household_based_controls,
        total_households_control_index=master_control_index,
        debug_control_set=True)
