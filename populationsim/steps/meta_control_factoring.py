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
def meta_control_factoring(settings, geo_cross_walk, control_spec, incidence_table, seed_controls):

    incidence_df = incidence_table.to_frame()
    seed_controls_df = seed_controls.to_frame()

    geographies = settings.get('geographies')
    seed_col = geographies['seed'].get('id_column')
    meta_col = geographies['meta'].get('id_column')

    meta_controls_spec = control_spec[control_spec.geography == 'meta']
    meta_control_targets = meta_controls_spec['target']
    meta_control_fields = meta_controls_spec['control_field'].tolist()

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

    # meta_controls table
    control_data_table_name = geographies['meta'].get('control_data_table')
    control_data_df = orca.get_table(control_data_table_name).to_frame()
    meta_controls = control_data_df[[meta_col] + meta_control_fields]
    meta_controls.set_index(meta_col, inplace=True)
    dump_table("meta_controls", meta_controls)

    # compute the scaling factors to be applied to the seed-level totals:
    meta_factors = pd.DataFrame(index=meta_controls.index)
    for target, control_field in zip(meta_control_targets, meta_control_fields):
        meta_factors[target] = meta_controls[control_field] / factored_meta_weights[target]
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
