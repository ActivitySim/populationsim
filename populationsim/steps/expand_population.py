# PopulationSim
# See full license in LICENSE.txt.

import logging

import orca
import pandas as pd
import numpy as np

from activitysim.core import pipeline

from populationsim.util import setting
from helper import get_control_table
from helper import get_weight_table

from helper import weight_table_name

logger = logging.getLogger(__name__)

GROUP_BY_INCIDENCE_SIGNATURE = setting('GROUP_BY_INCIDENCE_SIGNATURE')


@orca.step()
def expand_population():

    geographies = setting('geographies')
    household_id_col = setting('household_id_col')

    low_geography = geographies[-1]

    # only one we really need is low_geography
    seed_geography = setting('seed_geography')
    geography_cols = geographies[geographies.index(seed_geography):]

    weights = get_weight_table(low_geography, sparse=True)
    weights = weights[geography_cols + [household_id_col, 'integer_weight']]

    # - expand weights table by integer_weight, so there is one row per desired hh
    weight_cols = weights.columns.values
    weights_np = np.repeat(weights.as_matrix(), weights.integer_weight.values, axis=0)
    expanded_weights = pd.DataFrame(data=weights_np, columns=weight_cols)

    if GROUP_BY_INCIDENCE_SIGNATURE:

        # the household_id_col is really the group_id
        expanded_weights.rename(columns={household_id_col: 'group_id'}, inplace=True)

        # the original incidence table with one row per hh, with index hh_id
        ungrouped_incidence_table = pipeline.get_table('ungrouped_incidence_table')[[household_id_col, 'group_id', 'sample_weight']]

        grouper = ungrouped_incidence_table.groupby('group_id')

        # - probs approach
        HH_IDS = 0
        HH_PROBS = 1
        group_hh_probs = [0] * len(grouper)

        for group_id, df in grouper:
            hh_ids = list(df[household_id_col])
            probs = list(df.sample_weight / df.sample_weight.sum())
            group_hh_probs[group_id] = [hh_ids, probs]

        # now make a hh_id choice for each group_id in expanded_weights
        choose = lambda group_id : np.random.choice(group_hh_probs[group_id][HH_IDS], p=group_hh_probs[group_id][HH_PROBS])
        expanded_weights[household_id_col] = expanded_weights.group_id.apply(choose, convert_dtype=True,)

        #del expanded_weights['group_id']
        #del expanded_weights['integer_weight']

    orca.add_table('expanded_household_ids', expanded_weights)

