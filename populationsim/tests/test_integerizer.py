# PopulationSim
# See full license in LICENSE.txt.

import os
import numpy as np
import pandas as pd
import orca

from ..integerizer import do_integerizing


def test_integerizer():

    # rows are elements for which factors are calculated, columns are constraints to be satisfied
    incidence_table = pd.DataFrame({
        'num_hh': [1, 1, 1, 1, 1, 1, 1, 1],
        'hh_1': [1, 1, 1, 0, 0, 0, 0, 0],
        'hh_2': [0, 0, 0, 1, 1, 1, 1, 1],
        'p1': [1, 1, 2, 1, 0, 1, 2, 1],
        'p2': [1, 0, 1, 0, 2, 1, 1, 1],
        'p3': [1, 1, 0, 2, 1, 0, 2, 0],
        'float_weights':
            [1.362893, 25.658290, 7.978812, 27.789651, 18.451021, 8.641589, 1.476104, 8.641589]
    })

    control_cols = ['num_hh', 'hh_1', 'hh_2', 'p1', 'p2', 'p3']

    control_spec = pd.DataFrame(
        {
            'seed_table':
                ['households', 'households', 'households', 'persons', 'persons', 'persons'],
            'target': control_cols,
            'importance': [10000000, 1000, 1000, 1000, 1000, 1000]
        }
    )

    # column totals which the final weighted incidence table sums must satisfy
    control_totals = pd.Series([100, 35, 65, 91, 65, 104], index=control_spec.target.values)

    zot = do_integerizing(
        trace_label='label',
        control_spec=control_spec,
        control_totals=control_totals,
        incidence_table=incidence_table[control_cols],
        float_weights=incidence_table['float_weights'],
        total_hh_control_col='num_hh'
    )
