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
        trace_label,
        incidence_df,
        sub_weights, sub_controls,
        control_spec, total_hh_control_col,
        sub_control_zones,
        sub_geography,
        combine_results=True):
    """

    note: this method returns different results depending on the value of combine_results

    Parameters
    ----------
    incidence_df : pandas.Dataframe
        full incidence_df for all hh samples in seed zone
    sub_zone_weights : pandas.DataFame
        balanced subzone household sample weights to integerize
    sub_controls_df : pandas.Dataframe
        sub_geography controls (one row per zone indexed by sub_zone id)
    control_spec : pandas.Dataframe
        full control spec with columns 'target', 'seed_table', 'importance', ...
    total_hh_control_col : str
        name of total_hh column (so we can preferentially match this control)
    sub_geography : str
        subzone geography name (e.g. 'TAZ')
    sub_control_zones : pandas.Series
        series mapping zone_id (index) to zone label (value)
        for use in sub_controls_df column names
    combine_results : bool
        return all results in a single frame or return infeasible rounded results separately?
    Returns
    -------

    For combined results:

        integerized_weights_df : pandas.DataFrame
            canonical form weight table, with columns for 'balanced_weight', 'integer_weight'
            plus columns for household id, and sub_geography zone ids

    for segregated results:

        integerized_zone_ids : array(int)
            zone_ids of feasible (integerized) zones
        rounded_zone_ids : array(int)
            zone_ids of infeasible (rounded) zones
        integerized_weights_df : pandas.DataFrame or None if all zones infeasible
            integerized weights for feasible zones
        rounded_weights_df : pandas.DataFrame or None if all zones feasible
            rounded weights for infeasible aones

    Results dataframes are canonical form weight table,
    with columns for 'balanced_weight', 'integer_weight'
    plus columns for household id, and sub_geography zone ids

    """
    integerized_weights_list = []
    rounded_weights_list = []
    integerized_zone_ids = []
    rounded_zone_ids = []
    for zone_id, zone_name in sub_control_zones.iteritems():

        logger.info("sequential_integerizing zone_id %s zone_name %s" % (zone_id, zone_name))

        weights = sub_weights[zone_name]

        sub_trace_label = "%s_%s_%s" % (trace_label, sub_geography, zone_id)

        integer_weights, status = do_integerizing(
            trace_label=sub_trace_label,
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

        if status in STATUS_SUCCESS:
            integerized_weights_list.append(zone_weights_df)
            integerized_zone_ids.append(zone_id)
        else:
            rounded_weights_list.append(zone_weights_df)
            rounded_zone_ids.append(zone_id)

    if combine_results:
        integerized_weights_df = pd.concat(integerized_weights_list + rounded_weights_list)
        return integerized_weights_df

    integerized_weights_df = pd.concat(integerized_weights_list) if integerized_zone_ids else None
    rounded_weights_df = pd.concat(rounded_weights_list) if rounded_zone_ids else None

    return integerized_zone_ids, rounded_zone_ids, integerized_weights_df, rounded_weights_df
