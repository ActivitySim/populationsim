# PopulationSim
# See full license in LICENSE.txt.

import logging
import os

import orca
import pandas as pd
import numpy as np

from activitysim.core import assign
from ..balancer import ListBalancer

logger = logging.getLogger(__name__)


def dump_table(table_name, table):

    print "\n%s\n" % table_name, table


@orca.step()
def meta_control_factoring(settings, control_spec, incidence_table, seed_controls, meta_controls):

    incidence_df = incidence_table.to_frame()
    seed_controls_df = seed_controls.to_frame()
    meta_controls_df = meta_controls.to_frame()

    geography_settings = settings.get('geography_settings')
    seed_col = geography_settings['seed'].get('id_column')
    meta_col = geography_settings['meta'].get('id_column')

    meta_controls_spec = control_spec[control_spec.geography == 'meta']
    meta_control_targets = meta_controls_spec['target']

    # weights of meta targets at hh (incidence table) level
    hh_level_weights = incidence_df[[seed_col, meta_col]].copy()
    for target in meta_control_targets:
        hh_level_weights[target] = incidence_df[target] * incidence_df['seed_weight']

    # weights of meta targets at seed level
    factored_seed_weights = hh_level_weights.groupby([seed_col, meta_col], as_index=False).sum()
    factored_seed_weights.set_index(seed_col, inplace=True)
    dump_table("factored_seed_weights", factored_seed_weights)

    # weights of meta targets summed from seed level to  meta level
    factored_meta_weights = factored_seed_weights.groupby(meta_col, as_index=True).sum()
    dump_table("factored_meta_weights", factored_meta_weights)

    # only the meta level controls from meta_controls table
    meta_controls_df = meta_controls_df[meta_control_targets]
    dump_table("meta_controls_df", meta_controls_df)

    # compute the scaling factors to be applied to the seed-level totals:
    meta_factors = pd.DataFrame(index=meta_controls_df.index)
    for target in meta_control_targets:
        meta_factors[target] = meta_controls_df[target] / factored_meta_weights[target]
    dump_table("meta_factors", meta_factors)

    # compute seed-level controls from meta-level controls
    seed_level_meta_controls = pd.DataFrame(index=factored_seed_weights.index)
    for target in meta_control_targets:
        #  meta level scaling_factor for this meta_control
        scaling_factor = factored_seed_weights[meta_col].map(meta_factors[target])
        # scale the seed_level_meta_controls by meta_level scaling_factor
        seed_level_meta_controls[target] = factored_seed_weights[target] * scaling_factor
        # FIXME - not clear why we need to round to int?
        seed_level_meta_controls[target] = seed_level_meta_controls[target].round().astype(int)
    dump_table("seed_level_meta_controls", seed_level_meta_controls)

    # create final balancing controls
    # add newly created meta-to-seed level to the existing set of seed level controls
    final_seed_controls = pd.concat([seed_controls_df, seed_level_meta_controls], axis=1)
    dump_table("final_seed_controls", final_seed_controls)

    orca.add_table('final_seed_controls', final_seed_controls)
