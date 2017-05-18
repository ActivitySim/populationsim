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


def integerize_weights(geography, settings, geo_cross_walk_df, results_df):

    geography_settings = settings.get('geography_settings')

    id_col = geography_settings[geography].get('id_column')

    # all zone_ids in geography
    zone_ids = geo_cross_walk_df[id_col].unique()

    logger.info("integerize_final_seed_weights geography %s" % (geography,))

    for zone_id in zone_ids:
        logger.info("integerize_final_seed_weights geography %s zone_id %s" % (geography, zone_id,))

        zone_row_map = results_df[id_col] == zone_id
        zone_weights = results_df[zone_row_map]

        weights = zone_weights['final_weight']

        rounded_weights = weights.round().astype(int)
        results_df.ix[zone_row_map, 'rounded_weights'] = rounded_weights

        bucket_rounded_weights = bucket_round(weights)
        results_df.ix[zone_row_map, 'bucket_rounded_weights'] = bucket_rounded_weights


@orca.step()
def integerize_sub_weights(settings, sub_results, geo_cross_walk):

    geographies = settings.get('geographies')
    sub_geographies = geographies[geographies.index('seed') + 1:]

    geo_cross_walk_df = geo_cross_walk.to_frame()
    results_df = sub_results.to_frame()

    for geography in sub_geographies:
        integerize_weights(geography, settings, geo_cross_walk_df, results_df)

    rounded_weights = results_df['rounded_weights'].astype(int)
    orca.add_column('sub_results', 'rounded_weights', rounded_weights)

    bucket_rounded_weights = results_df['bucket_rounded_weights'].astype(int)
    orca.add_column('sub_results', 'bucket_rounded_weights', bucket_rounded_weights)
