# PopulationSim
# See full license in LICENSE.txt.

import logging
import os

import orca
import pandas as pd
import numpy as np

from integerizer import do_integerizing

DUMP = True

logger = logging.getLogger(__name__)


def do_sequential_integerizing(
        incidence_df,
        sub_weights, sub_controls,
        control_spec, total_hh_control_col,
        sub_control_zones,
        sub_geography):

    # integerize the sub_zone weights
    integer_weights_list = []
    for zone_id, zone_name in sub_control_zones.iteritems():

        logger.info("sequential_multi_integerize zone_id %s zone_name %s" % (zone_id, zone_name))

        weights = sub_weights[zone_name]

        integer_weights, status = do_integerizing(
            label=sub_geography,
            id=zone_id,
            control_spec=control_spec,
            control_totals=sub_controls.loc[zone_id],
            incidence_table=incidence_df[control_spec.target],
            float_weights=weights,
            total_hh_control_col=total_hh_control_col
        )

        zone_weights_df = pd.DataFrame(index=range(0, len(integer_weights.index)))
        zone_weights_df[weights.index.name] = weights.index
        zone_weights_df[sub_geography] = zone_id
        zone_weights_df['balanced_weight'] = weights.values
        zone_weights_df['integer_weight'] = integer_weights.astype(int).values

        integer_weights_list.append(zone_weights_df)

    integer_weights_df = pd.concat(integer_weights_list)
    return integer_weights_df


