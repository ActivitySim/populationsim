
# PopulationSim
# See full license in LICENSE.txt.

import logging

import pandas as pd
import numpy as np

from activitysim.core import pipeline
from activitysim.core import inject

from activitysim.core.config import setting
from .helper import get_control_table
from .helper import get_weight_table

from .helper import weight_table_name

logger = logging.getLogger(__name__)


@inject.step()
def expand_households():
    """
    Create a complete expanded synthetic household list with their assigned geographic zone ids.

    This is the skeleton synthetic household id list with no household or person attributes,
    one row per household, with geography columns and seed household table household_id.

    Creates pipeline table expanded_household_ids
    """

    if setting('NO_INTEGERIZATION_EVER', False):
        logger.warning("skipping expand_households: NO_INTEGERIZATION_EVER")
        inject.add_table('expanded_household_ids', pd.DataFrame())
        return

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
    weights_np = np.repeat(weights.values, weights.integer_weight.values, axis=0)
    expanded_weights = pd.DataFrame(data=weights_np, columns=weight_cols)

    if setting('GROUP_BY_INCIDENCE_SIGNATURE'):

        # get these in a repeatable order so np.random.choice behaves the same regardless of weight table order
        # i.e. which could vary depending on whether we ran single or multi process due to apportioned/coalesce
        expanded_weights = expanded_weights.sort_values(geography_cols + [household_id_col])

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

        # get a repeatable random number sequence generator for consistent choice results
        prng = pipeline.get_rn_generator().get_external_rng('expand_households')

        # now make a hh_id choice for each group_id in expanded_weights
        def chooser(group_id):
            hh_ids = group_hh_probs[group_id][HH_IDS]
            hh_probs = group_hh_probs[group_id][HH_PROBS]
            return prng.choice(hh_ids, p=hh_probs)
        expanded_weights[household_id_col] = \
            expanded_weights.group_id.apply(chooser, convert_dtype=True,)

        # FIXME - omit in production?
        del expanded_weights['group_id']
        del expanded_weights['integer_weight']

    append = inject.get_step_arg('append', False)
    replace = inject.get_step_arg('replace', False)
    assert not (append and replace), "can't specify both append and replace for expand_households"

    if append or replace:
        t = inject.get_table('expanded_household_ids').to_frame()
        prev_hhs = len(t.index)
        added_hhs = len(expanded_weights.index)

        if replace:
            # FIXME - should really get from crosswalk table?
            low_ids_to_replace = expanded_weights[low_geography].unique()
            t = t[~t[low_geography].isin(low_ids_to_replace)]

        expanded_weights = pd.concat([t, expanded_weights], ignore_index=True)

        dropped_hhs = prev_hhs - len(t.index)
        final_hhs = len(expanded_weights.index)
        op = 'append' if append else 'replace'
        logger.info("expand_households op: %s prev hh count %s dropped %s added %s final %s" %
                    (op, prev_hhs, dropped_hhs, added_hhs, final_hhs))

    # sort this so results will be consistent whether single or multiprocessing, GROUP_BY_INCIDENCE_SIGNATURE, etc...
    expanded_weights = expanded_weights.sort_values(geography_cols + [household_id_col])

    repop = inject.get_step_arg('repop', default=False)
    inject.add_table('expanded_household_ids', expanded_weights, replace=repop)
