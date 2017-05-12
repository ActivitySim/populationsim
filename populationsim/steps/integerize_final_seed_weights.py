# PopulationSim
# See full license in LICENSE.txt.

import logging
import os

import orca
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@orca.step()
def integerize_final_seed_weights(settings, geo_cross_walk, seed_controls, incidence_table):

    geo_cross_walk_df = geo_cross_walk.to_frame()
    incidence_df = incidence_table.to_frame()
    seed_controls_df = seed_controls.to_frame()

    seed_col = settings.get('geography_settings')['seed'].get('id_column')

    # determine master_control_index if specified in settings
    total_hh_control_col = settings.get('total_hh_control')

    # run balancer for each seed geography
    weight_list = []
    seed_ids = geo_cross_walk_df[seed_col].unique()
    for seed_id in seed_ids:

        logger.info("integerize_final_seed_weights seed id %s" % seed_id)

        incidence = incidence_df[incidence_df[seed_col] == seed_id]
        integer_weights = incidence['final_seed_weight'].round()

        # controls = pd.DataFrame({'control': seed_controls_df.loc[seed_id]})
        # controls['integer_weight'] \
        #     = [(incidence.ix[:, c] * integer_weights).sum() for c in controls.index]
        # print "controls\n", controls
        # print "integer_weights\n", integer_weights

        weight_list.append(integer_weights)

    # bulk concat all seed level results
    integer_seed_weights = pd.concat(weight_list)

    orca.add_column('incidence_table', 'integer_seed_weight', integer_seed_weights)
