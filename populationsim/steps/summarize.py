# PopulationSim
# See full license in LICENSE.txt.

import logging

import orca
import pandas as pd
import numpy as np

from helper import get_control_table

logger = logging.getLogger(__name__)


def summarize_geography(geography, weight_col,
                        settings, crosswalk_df, results_df, incidence_df):

    geography_settings = settings.get('geography_settings')

    # controls_table for current geography level
    controls_df = get_control_table(geography)
    control_names = controls_df.columns.tolist()

    zone_ids = crosswalk_df[geography].unique()

    results = []
    controls = []
    for zone_id in zone_ids:
        logger.info("summarize_geography %s %s" % (geography, zone_id))

        zone_controls = controls_df.loc[zone_id].tolist()
        controls.append(zone_controls)

        zone_row_map = results_df[geography] == zone_id
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

    dif_df = pd.DataFrame(
        data=np.asanyarray(results) - np.asanyarray(controls),
        columns=['%s_diff' % c for c in control_names],
        index=zone_ids
    )

    summary_df = pd.concat([controls_df, summary_df, dif_df], axis=1)

    summary_cols = summary_df.columns.tolist()

    summary_df['geography'] = geography
    summary_df['id'] = summary_df.index
    summary_df.index = summary_df['geography'] + '_' + summary_df['id'].astype(str)
    summary_df = summary_df[['geography', 'id'] + summary_cols]

    return summary_df


@orca.step()
def summarize(settings, crosswalk, incidence_table):

    crosswalk_df = crosswalk.to_frame()
    incidence_df = incidence_table.to_frame()

    geographies = settings.get('geographies')
    seed_geography = settings.get('seed_geography')
    sub_geography = geographies[geographies.index(seed_geography) + 1]

    seed_controls_df = get_control_table(seed_geography)


    # aggregate to seed level
    weights_df = orca.get_table('%s_weights' % sub_geography).to_frame()
    hh_id_col = incidence_df.index.name
    aggegrate_weights = weights_df.groupby([seed_geography, hh_id_col], as_index=False).sum()
    del aggegrate_weights[sub_geography]
    aggegrate_weights.set_index(hh_id_col, inplace=True)
    print aggegrate_weights

    weights_df = orca.get_table('%s_weights' % sub_geography).to_frame()

    for geography in [seed_geography, sub_geography]:
        df = summarize_geography(geography, 'integer_weight', settings, crosswalk_df, weights_df, incidence_df)
        orca.add_table('%s_summary' % (geography,), df)

