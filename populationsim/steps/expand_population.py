# PopulationSim
# See full license in LICENSE.txt.

import logging

import pandas as pd
import numpy as np

from activitysim.core import pipeline
from activitysim.core import inject

from populationsim.util import setting
from helper import get_control_table
from helper import get_weight_table

from helper import weight_table_name

logger = logging.getLogger(__name__)


@inject.step()
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

    if setting('GROUP_BY_INCIDENCE_SIGNATURE'):

        # the household_id_col is really the group_id
        expanded_weights.rename(columns={household_id_col: 'group_id'}, inplace=True)

        # the original incidence table with one row per hh, with index hh_id
        household_groups = pipeline.get_table('household_groups')
        household_groups = household_groups[[household_id_col, 'group_id', 'sample_weight']]

        # for each group, lists of hh_ids and their sample_weights (as relative probabiliities)
        # [ [ [<group_0_hh_id_list>], [<group_0_hh_prob_list>] ],
        #   [ [<group_1_hh_id_list>], [<group_1_hh_prob_list>] ], ... ]
        HH_IDS = 0
        HH_PROBS = 1
        grouper = household_groups.groupby('group_id')
        group_hh_probs = [0] * len(grouper)
        for group_id, df in grouper:
            hh_ids = list(df[household_id_col])
            probs = list(df.sample_weight / df.sample_weight.sum())
            group_hh_probs[group_id] = [hh_ids, probs]

        # now make a hh_id choice for each group_id in expanded_weights
        def chooser(group_id):
            hh_ids = group_hh_probs[group_id][HH_IDS]
            hh_probs = group_hh_probs[group_id][HH_PROBS]
            return np.random.choice(hh_ids, p=hh_probs)
        expanded_weights[household_id_col] = \
            expanded_weights.group_id.apply(chooser, convert_dtype=True,)

        # FIXME - omit in production?
        del expanded_weights['group_id']
        del expanded_weights['integer_weight']

    append = inject.get_step_arg('append', False)
    replace = inject.get_step_arg('replace', False)
    assert not (append and replace), "can't specify both append and replace for expand_population"

    if append or replace:
        t = inject.get_table('expanded_household_ids').to_frame()
        print "\nprior taz_hh_counts\n", t.groupby('TAZ').size()
        if replace:
            # FIXME - should really get from crosswalk table?
            low_ids_to_replace = expanded_weights[low_geography].unique()
            t = t[~t[low_geography].isin(low_ids_to_replace)]
        expanded_weights = pd.concat([t, expanded_weights], ignore_index=True)

    inject.add_table('expanded_household_ids', expanded_weights)
