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

#
# def summarize_seeds(seed_col, seed_controls_df, aggegrate_weights, incidence_df):
#
#     results = []
#     controls = []
#     control_names = seed_controls_df.columns
#
#     seed_ids = aggegrate_weights[seed_col].unique()
#     for seed_id in seed_ids:
#
#         print "\nSEED ZONE %s" % seed_id
#         incidence = incidence_df[ incidence_df[seed_col] == seed_id]
#         weights = aggegrate_weights[incidence_df[seed_col] == seed_id]
#
#         seed_controls = seed_controls_df.loc[seed_id].tolist()
#         controls.append(seed_controls)
#
#         seed_results = [(incidence[c] * aggegrate_weights['integer_weight']).sum() for c in control_names]
#         results.append(seed_results)
#
#     controls_df = pd.DataFrame(
#         data=np.asanyarray(controls),
#         columns=['%s_control' % c for c in control_names],
#         index=seed_ids
#     )
#
#     summary_df = pd.DataFrame(
#         data=np.asanyarray(results),
#         columns=['%s_result' % c for c in control_names],
#         index=seed_ids
#     )
#     summary_df = pd.concat([controls_df, summary_df], axis=1)
#
#     summary_df['id'] = summary_df.index
#
#
#     print "summary_df\n", summary_df
#
#     return summary_df


@orca.step()
def summarize(settings, geo_cross_walk, incidence_table, seed_controls):

    geography_settings = settings.get('geography_settings')
    geographies = settings.get('geographies')

    seed_col = geography_settings['seed'].get('id_column')

    sub_geography = geographies[geographies.index('seed') + 1]
    sub_col = geography_settings[sub_geography].get('id_column')

    geo_cross_walk_df = geo_cross_walk.to_frame()
    incidence_df = incidence_table.to_frame()
    seed_controls_df = seed_controls.to_frame()

    mid_weights_df = orca.get_table('mid_weights').to_frame()

    # aggregate to seed level
    hh_id_col = incidence_df.index.name
    aggegrate_weights = mid_weights_df.groupby([seed_col, hh_id_col], as_index=False).sum()
    del aggegrate_weights[sub_col]
    aggegrate_weights.set_index(hh_id_col, inplace=True)

    for geography in ['seed', 'mid']:
        df = summarize_geography(geography, 'integer_weight', settings, geo_cross_walk_df, mid_weights_df, incidence_df)
        orca.add_table('%s_summary' % (geography,), df)

