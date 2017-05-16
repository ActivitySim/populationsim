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

    # run balancer for each seed geography
    weight_list = []
    seed_ids = geo_cross_walk_df[seed_col].unique()
    for seed_id in seed_ids:

        logger.info("integerize_final_seed_weights seed id %s" % seed_id)

        incidence = incidence_df[incidence_df[seed_col] == seed_id]

        # FIXME - for now we just round
        integer_weights = incidence['final_seed_weight'].round()
        weight_list.append(integer_weights)

    # bulk concat all seed level results
    integer_seed_weights = pd.concat(weight_list)

    orca.add_column('incidence_table', 'integer_seed_weight', integer_seed_weights)
