
# PopulationSim
# See full license in LICENSE.txt.

import os
import numpy as np
import pandas as pd

from activitysim.core import inject

from populationsim import integerizer


def test_integerizer():

    configs_dir = os.path.join(os.path.dirname(__file__), 'configs')
    inject.add_injectable("configs_dir", configs_dir)

    # data_dir = os.path.join(os.path.dirname(__file__), 'data')
    # inject.add_injectable("data_dir", data_dir)
    #
    # output_dir = os.path.join(os.path.dirname(__file__), 'output')
    # inject.add_injectable("output_dir", output_dir)

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

    integerized_weights, status = integerizer.do_integerizing(
        trace_label='label',
        control_spec=control_spec,
        control_totals=control_totals,
        incidence_table=incidence_table[control_cols],
        float_weights=incidence_table['float_weights'],
        total_hh_control_col='num_hh'
    )

    print("do_integerizing status", status)
    print("sum", integerized_weights.sum())
    print("do_integerizing integerized_weights\n", integerized_weights)

    assert integerized_weights.sum() == 100

    # exact outcome is variable from version to version of ortools integerizer!
    # ortools cbc
    # assert (integerized_weights.values == [
    #      1, 26, 8, 28, 18, 8, 2, 9,
    # ]).all()
