# PopulationSim
# See full license in LICENSE.txt.

import logging
import os

import orca
import pandas as pd
import numpy as np

from integerizer import do_integerizing
from integerizer import STATUS_SUCCESS

logger = logging.getLogger(__name__)


def do_sequential_integerizing(
        incidence_df,
        sub_weights, sub_controls,
        control_spec, total_hh_control_col,
        sub_control_zones,
        sub_geography,
        parent_geography,
        parent_id):

    # integerize the sub_zone weights
    integer_weights_list = []
    for zone_id, zone_name in sub_control_zones.iteritems():

        logger.info("sequential_multi_integerize zone_id %s zone_name %s" % (zone_id, zone_name))

        weights = sub_weights[zone_name]

        trace_label = "%s_%s_%s_%s" % (parent_geography, parent_id, sub_geography, zone_id)

        integer_weights, status = do_integerizing(
            trace_label=trace_label,
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


def do_round_infeasible_subzones(
        incidence_df,
        sub_weights, sub_controls,
        control_spec, total_hh_control_col,
        sub_control_zones,
        sub_geography,
        parent_geography,
        parent_id):

    # integerize the sub_zone weights
    integer_weights_list = []
    zone_list = []
    for zone_id, zone_name in sub_control_zones.iteritems():

        logger.info("sequential_multi_integerize zone_id %s zone_name %s" % (zone_id, zone_name))

        weights = sub_weights[zone_name]

        trace_label = "%s_%s_%s_%s" % (parent_geography, parent_id, sub_geography, zone_id)

        integer_weights, status = do_integerizing(
            trace_label=trace_label,
            control_spec=control_spec,
            control_totals=sub_controls.loc[zone_id],
            incidence_table=incidence_df[control_spec.target],
            float_weights=weights,
            total_hh_control_col=total_hh_control_col
        )

        if status in STATUS_SUCCESS:
            continue

        zone_weights_df = pd.DataFrame(index=range(0, len(integer_weights.index)))
        zone_weights_df[weights.index.name] = weights.index
        zone_weights_df[sub_geography] = zone_id
        zone_weights_df['balanced_weight'] = weights.values
        zone_weights_df['integer_weight'] = integer_weights.astype(int).values

        integer_weights_list.append(zone_weights_df)
        zone_list.append(zone_id)

    feasible_subzones = sub_control_zones.loc[ ~sub_control_zones.index.isin(zone_list) ]
    infeasible_subzones = sub_control_zones.loc[ sub_control_zones.index.isin(zone_list) ]
    integer_weights_df = pd.concat(integer_weights_list)
    return feasible_subzones, infeasible_subzones, integer_weights_df


