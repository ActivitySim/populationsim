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

    print "\ntable_name\n", table


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
    seed_level_weights = hh_level_weights.groupby([seed_col, meta_col], as_index=False).sum()
    seed_level_weights.set_index(seed_col, inplace=True)
    dump_table("seed_level_weights", seed_level_weights)

    # weights of meta targets at meta level
    meta_level_weights = seed_level_weights.groupby(meta_col, as_index=True).sum()
    dump_table("meta_level_weights", meta_level_weights)

    # meta level controls
    geography = geographies['meta']
    control_data_table_name = geography['control_data_table']
    control_data_df = orca.get_table(control_data_table_name).to_frame()
    meta_controls = control_data_df[[meta_col] + meta_control_fields]
    meta_controls.set_index(meta_col, inplace=True)
    dump_table("meta_controls", meta_controls)

    meta_factors = pd.DataFrame(index=meta_controls.index)
    for target, control_field in zip(meta_control_targets, meta_control_fields):
        meta_factors[target] = meta_controls[control_field] / meta_level_weights[target]
    dump_table("meta_factors", meta_factors)

    new_meta_controls = pd.DataFrame(index=seed_level_weights.index)
    for target in meta_control_targets:
        new_meta_controls[target] = seed_level_weights[target] * seed_level_weights[meta_col].map(meta_factors[target])
        new_meta_controls[target] = new_meta_controls[target].round().astype(int)
    dump_table("new_meta_controls", new_meta_controls)

    dump_table("nseed_controls_df", seed_controls_df)

    final_seed_controls = pd.concat([seed_controls_df, new_meta_controls], axis=1)
    dump_table("final_seed_controls", final_seed_controls)

    orca.add_table('final_seed_controls', final_seed_controls)
