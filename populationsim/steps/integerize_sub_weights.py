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
        # max prevents -0.5 from rounding to -1
        r = max(int(round(w + residue)), 0)
        residue = w - r
        rounded.append(r)

        #print "w %s r %s residue %s" % (w, r, residue)

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

    # run balancer for each seed geography
    seed_ids = geo_cross_walk_df[seed_col].unique()
    for seed_id in seed_ids:
        logger.info("integerize_final_seed_weights seed id %s" % seed_id)

        sub_balanced_weights_table_name = 'sub_weights_%s' % seed_id


        sub_balanced_weights = orca.get_table(sub_balanced_weights_table_name).to_frame()

        rounded_weights = pd.DataFrame(index=sub_balanced_weights.index)
        bucket_rounded_weights = pd.DataFrame(index=sub_balanced_weights.index)

        low_geography_ids = geo_cross_walk_df.loc[geo_cross_walk_df[seed_col] == seed_id, low_col].tolist()
        for low_id in low_geography_ids:

            weight_col = '%s_%s' % (low_col, low_id)

            weights = sub_balanced_weights[weight_col]

            rounded_weights[weight_col] = weights.round()

            bucket_rounded_weights[weight_col] = bucket_round(weights)

            # print "weights\n", weights
            # print "sum weights", weights.sum()
            # print "bucket_rounded_weights\n", bucket_rounded_weights[weight_col]
            # print "sum bucket_rounded_weights\n", bucket_rounded_weights[weight_col].sum()

            # print "weight_cols", weight_col
            # print "weights\n", weights
            # print "\n"

        rounded_weights_table_name = 'rounded_sub_weights_%s' % seed_id
        orca.add_table(rounded_weights_table_name, rounded_weights)

        bucket_rounded_sub_weights_table_name = 'bucket_rounded_sub_weights_%s' % seed_id
        orca.add_table(bucket_rounded_sub_weights_table_name, bucket_rounded_weights)
