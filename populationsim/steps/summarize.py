# PopulationSim
# See full license in LICENSE.txt.

import logging
import os

import orca
import pandas as pd
import numpy as np


logger = logging.getLogger(__name__)


def summarize_geography(geography, weight_col,
                        settings, geo_cross_walk_df, results_df, incidence_df):

    geography_settings = settings.get('geography_settings')
    geo_col = geography_settings[geography].get('id_column')

    # controls_table for current geography level
    control_table_name = geography_settings[geography].get('controls_table')
    controls_df = orca.get_table(control_table_name).to_frame()
    control_names = controls_df.columns.tolist()

    # run balancer for each seed geography
    zone_ids = geo_cross_walk_df[geo_col].unique()

    results = []
    controls = []
    for zone_id in zone_ids:
        logger.info("integerize_final_seed_weights zone_id %s" % zone_id)

        zone_controls = controls_df.loc[zone_id].tolist()
        controls.append(zone_controls)

        zone_row_map = results_df[geo_col] == zone_id
        zone_weights = results_df[zone_row_map]

        incidence = incidence_df.loc[zone_weights.hh_id]

        weights = zone_weights[weight_col].tolist()
        x = [(incidence[c] * weights).sum() for c in control_names]
        results.append(x)

    controls_df = pd.DataFrame(
        data=np.asanyarray(controls),
        columns=['%s_control' % c for c in control_names],
        index=zone_ids
    )

    summary_df = pd.DataFrame(
        data=np.asanyarray(results),
        columns=['%s_result' % c for c in control_names],
        index=zone_ids
    )
    summary_df = pd.concat([controls_df, summary_df], axis=1)

    summary_cols = summary_df.columns.tolist()

    summary_df['geography'] = geo_col
    summary_df['id'] = summary_df.index
    summary_df.index = summary_df['geography'] + '_' + summary_df['id'].astype(str)
    summary_df = summary_df[['geography', 'id'] + summary_cols]

    return summary_df


@orca.step()
def summarize(settings, geo_cross_walk, control_spec, incidence_table):

    geographies = settings.get('geographies')

    sub_geographies = geographies[geographies.index('seed') + 1:]
    low_geography = geographies[-1]

    geo_cross_walk_df = geo_cross_walk.to_frame()
    incidence_df = incidence_table.to_frame()
    results_df = orca.get_table('sub_results').to_frame()

    for geography in sub_geographies:
        summary_df = summarize_geography('mid', 'final_weight',
                                         settings, geo_cross_walk_df, results_df, incidence_df)
        orca.add_table('summarize_final_weight_%s' % geography, summary_df)

    summary_df = summarize_geography(low_geography, 'rounded_weights',
                                     settings, geo_cross_walk_df, results_df, incidence_df)
    orca.add_table('summarize_rounded_weights', summary_df)

    summary_df = summarize_geography(low_geography, 'bucket_rounded_weights',
                                     settings, geo_cross_walk_df, results_df, incidence_df)

    orca.add_table('summarize_bucket_rounded_weights', summary_df)
