# PopulationSim
# See full license in LICENSE.txt.

import logging
import os

import orca
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def bucket_round(weights):

    rounded = []
    residue = 0.0
    for w in weights.tolist():
        ACCUMULATE_RESIDUES = False
        # max prevents -0.5 from rounding to -1
        r = max(int(round(w + residue)), 0)
        if ACCUMULATE_RESIDUES:
            residue += w - r
        else:
            residue = w - r
        rounded.append(r)

    rounded = pd.Series(rounded, index=weights.index)
    return rounded


@orca.step()
def integerize_sub_weights(settings, geo_cross_walk, seed_controls, incidence_table):

    geographies = settings.get('geographies')
    geography_settings = settings.get('geography_settings')

    geo_cross_walk_df = geo_cross_walk.to_frame()
    incidence_df = incidence_table.to_frame()

    seed_col = geography_settings['seed'].get('id_column')

    lowest_geography = geographies[-1]
    low_col = geography_settings[lowest_geography].get('id_column')

    results_df = orca.get_table('sub_results').to_frame()

    # run balancer for each seed geography
    zone_ids = geo_cross_walk_df[low_col].unique()

    for zone_id in zone_ids:
        logger.info("integerize_final_seed_weights zone_id %s" % zone_id)

        zone_row_map = results_df[low_col] == zone_id
        zone_weights = results_df[zone_row_map]

        weights = zone_weights['final_weight']

        rounded_weights = weights.round().astype(int)
        results_df.ix[zone_row_map, 'rounded_weights'] = rounded_weights

        bucket_rounded_weights = bucket_round(weights)
        results_df.ix[zone_row_map, 'bucket_rounded_weights'] = bucket_rounded_weights

    orca.add_column('sub_results', 'rounded_weights', results_df['rounded_weights'])
    orca.add_column('sub_results', 'bucket_rounded_weights', results_df['bucket_rounded_weights'])

    # print "rounded_weights\n", results_df['rounded_weights']
    # print "sum rounded_weights\n", results_df['rounded_weights'].sum()
    #
    # print "bucket_rounded_weights\n", results_df['bucket_rounded_weights']
    # print "sum bucket_rounded_weights\n", results_df['bucket_rounded_weights'].sum()
