# PopulationSim
# See full license in LICENSE.txt.

import logging
import os

import orca
import pandas as pd
import numpy as np

from ..balancer import seed_balancer

logger = logging.getLogger(__name__)


# def seed_balancer(seed_control_spec, seed_id, seed_col, master_control_col,
#                   incidence_df, seed_controls_df):
#
#     # slice incidence rows for this seed geography
#     incidence_df = incidence_df[incidence_df[seed_col] == seed_id]
#
#     # initial hh weights
#     initial_weights = incidence_df['initial_weight']
#
#     # incidence table should only have control columns
#     incidence_df = incidence_df[seed_control_spec.target]
#
#     control_totals = seed_controls_df.loc[seed_id].values
#
#     control_importance_weights = seed_control_spec.importance
#
#     # determine master_control_index if specified in settings
#     master_control_index = None
#     if master_control_col:
#         if master_control_col not in incidence_df.columns:
#             print incidence_df.columns
#             raise RuntimeError("total_hh_control column '%s' not found in incidence table"
#                                % master_control_col)
#         master_control_index = incidence_df.columns.get_loc(master_control_col)
#
#     balancer = ListBalancer(
#         incidence_table=incidence_df,
#         initial_weights=initial_weights,
#         control_totals=control_totals,
#         control_importance_weights=control_importance_weights,
#         master_control_index=master_control_index
#     )
#
#     return balancer


@orca.step()
def initial_seed_balancing(settings, geo_cross_walk, control_spec,
                           incidence_table, seed_controls):

    geo_cross_walk_df = geo_cross_walk.to_frame()
    incidence_df = incidence_table.to_frame()
    seed_controls_df = seed_controls.to_frame()

    seed_col = settings.get('geography_settings')['seed'].get('id_column')

    # only want control_spec rows for sub_geographies
    geographies = settings['geographies']
    sub_geographies = geographies[geographies.index('seed')+1:]
    seed_control_spec = control_spec[control_spec['geography'].isin(sub_geographies)]

    # determine master_control_index if specified in settings
    total_hh_control_col = settings.get('total_hh_control')

    max_expansion_factor = settings.get('max_expansion_factor', None)

    # run balancer for each seed geography
    weight_list = []
    seed_ids = geo_cross_walk_df[seed_col].unique()
    for seed_id in seed_ids:

        logger.info("initial_seed_balancing seed id %s" % seed_id)

        balancer = seed_balancer(
            seed_control_spec=seed_control_spec,
            seed_id=seed_id,
            seed_col=seed_col,
            total_hh_control_col=total_hh_control_col,
            max_expansion_factor=max_expansion_factor,
            incidence_df=incidence_df,
            seed_controls_df=seed_controls_df)

        # print "balancer.initial_weights\n", balancer.initial_weights
        # print "balancer.ub_weights\n", balancer.ub_weights
        # assert False

        # balancer.dump()
        status = balancer.balance()

        logger.info("seed_balancer status: %s" % status)
        if not status['converged']:
            raise RuntimeError("initial_seed_balancing for seed_id %s did not converge" % seed_id)

        weight_list.append(balancer.weights['final'])

        # print "balancer.weights\n", balancer.weights
        # print "balancer.controls\n", balancer.controls
        # assert False

    # bulk concat all seed level results
    weights = pd.concat(weight_list)

    orca.add_column('incidence_table', 'seed_weight', weights)
