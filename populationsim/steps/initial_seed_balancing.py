# PopulationSim
# See full license in LICENSE.txt.

import logging
import os

import orca
import pandas as pd
import numpy as np

from ..balancer import ListBalancer


logger = logging.getLogger(__name__)


def seed_balancer(seed_control_spec, seed_id, seed_col, master_control_col,
                  incidence_df, seed_controls_df):

    # slice incidence rows for this seed geography
    incidence_df = incidence_df[incidence_df[seed_col] == seed_id]

    # initial hh weights
    initial_weights = incidence_df['initial_weight']

    # incidence table should only have control columns
    incidence_df = incidence_df[seed_control_spec.target]

    control_totals = seed_controls_df.loc[seed_id].values

    control_importance_weights = seed_control_spec.importance

    # determine master_control_index if specified in settings
    master_control_index = None
    if master_control_col:
        if master_control_col not in incidence_df.columns:
            print incidence_df.columns
            raise RuntimeError("total_hh_control column '%s' not found in incidence table"
                               % master_control_col)
        master_control_index = incidence_df.columns.get_loc(master_control_col)

    balancer = ListBalancer(
        incidence_table=incidence_df,
        initial_weights=initial_weights,
        control_totals=control_totals,
        control_importance_weights=control_importance_weights,
        master_control_index=master_control_index
    )

    return balancer


@orca.step()
def initial_seed_balancing(settings, geo_cross_walk, control_spec,
                           incidence_table, seed_controls):

    geo_cross_walk_df = geo_cross_walk.to_frame()
    incidence_df = incidence_table.to_frame()
    seed_controls_df = seed_controls.to_frame()

    geographies = settings.get('geographies')
    seed_col = geographies['seed'].get('id_column')

    # only want control_spec rows for sub_geographies
    sub_geographies = settings['lower_level_geographies']
    seed_control_spec = control_spec[control_spec['geography'].isin(sub_geographies)]

    # determine master_control_index if specified in settings
    master_control_col = settings.get('total_hh_control', None)

    # run balancer for each seed geography
    weight_list = []
    seed_ids = geo_cross_walk_df[seed_col].unique()
    for seed_id in seed_ids:

        logger.info("initial_seed_balancing seed id %s" % seed_id)

        balancer = seed_balancer(
            seed_control_spec,
            seed_id,
            seed_col,
            master_control_col,
            incidence_df=incidence_df,
            seed_controls_df=seed_controls_df)

        # balancer.dump()
        status = balancer.balance()

        # FIXME - what to do if it fails to converge?
        logger.info("seed_balancer status: %s" % status)

        weight_list.append(balancer.weights['final'])

    # bulk concat all seed level results
    weights = pd.concat(weight_list)

    orca.add_column('incidence_table', 'seed_weight', weights)
