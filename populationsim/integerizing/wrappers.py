import logging
import numpy as np
import pandas as pd

from populationsim.core import config
from populationsim.integerizing.single_integerizer import Integerizer
from populationsim.integerizing.simul_integerizer import SimulIntegerizer

logger = logging.getLogger(__name__)

STATUS_OPTIMAL = "OPTIMAL"
STATUS_FEASIBLE = "FEASIBLE"
STATUS_SUCCESS = [STATUS_OPTIMAL, STATUS_FEASIBLE]


def try_simul_integerizing(
    trace_label,
    incidence_df,
    sub_weights,
    sub_controls_df,
    sub_geography,
    control_spec,
    total_hh_control_col,
    sub_control_zones,
):
    """
    Attempt simultaneous integerization and return integerized weights if successful

    Parameters
    ----------
    incidence_df
    sub_weights
    sub_controls_df
    sub_geography
    control_spec
    total_hh_control_col
    sub_control_zones

    Returns
    -------
    status : str
        str value of integerizer status from STATUS_TEXT dict
        integerization was successful if status in STATUS_SUCCESS list

    integerized_weights_df : pandas.DataFrame or None
        canonical form weight table, with columns for 'balanced_weight', 'integer_weight'
        or None if integerization failed
    """

    zero_weight_rows = sub_weights.sum(axis=1) == 0

    if zero_weight_rows.any():
        logger.info(
            "omitting %s zero weight rows out of %s"
            % (zero_weight_rows.sum(), len(incidence_df.index))
        )

    integerizer = SimulIntegerizer(
        incidence_df[~zero_weight_rows],
        sub_weights[~zero_weight_rows],
        sub_controls_df,
        control_spec,
        total_hh_control_col,
        trace_label,
    )

    status = integerizer.integerize()

    if status not in STATUS_SUCCESS:
        return status, None

    if zero_weight_rows.any():
        # restore zero_weight_rows to integerized_weights
        logger.info("restoring %s zero weight rows" % zero_weight_rows.sum())
        integerized_weights = pd.DataFrame(
            data=np.zeros(sub_weights.shape, dtype=np.int64),
            columns=sub_weights.columns,
            index=sub_weights.index,
        )
        integerized_weights.update(integerizer.integerized_weights)
    else:
        integerized_weights = integerizer.integerized_weights

    integerized_weights_df = reshape_result(
        sub_weights, integerized_weights, sub_geography, sub_control_zones
    )

    logger.debug("SimulIntegerizer status %s" % (status,))

    return status, integerized_weights_df


def reshape_result(
    float_weights, integerized_weights, sub_geography, sub_control_zones
):
    """
    Reshape results into unstacked form - (same as that returned by sequential integerizer)
    with columns for 'balanced_weight', 'integer_weight'
    plus columns for household id, and sub_geography zone ids

    Parameters
    ----------
    float_weights : pandas.DataFrame
        dataframe with one row per sample hh and one column per sub_zone
    integerized_weights : pandas.DataFrame
        dataframe with one row per sample hh and one column per sub_zone
    sub_geography : str
        name of sub_geography for result column name
    sub_control_zones : pandas.Series
        series mapping zone_id (index) to zone label (value)

    Returns
    -------
    integer_weights_df : pandas.DataFrame
        canonical form weight table, with columns for 'balanced_weight', 'integer_weight'
        plus columns for household id, and sub_geography zone ids
    """

    # integerize the sub_zone weights
    integer_weights_list = []
    for zone_id, zone_name in list(sub_control_zones.items()):

        weights = float_weights[zone_name]

        zone_weights_df = pd.DataFrame(
            index=list(range(0, len(integerized_weights.index)))
        )
        zone_weights_df[weights.index.name] = float_weights.index
        zone_weights_df[sub_geography] = zone_id
        zone_weights_df["balanced_weight"] = float_weights[zone_name].values
        zone_weights_df["integer_weight"] = (
            integerized_weights[zone_name].astype(int).values
        )

        integer_weights_list.append(zone_weights_df)

    integer_weights_df = pd.concat(integer_weights_list)

    return integer_weights_df


def do_integerizing(
    trace_label,
    control_spec,
    control_totals,
    incidence_table,
    float_weights,
    total_hh_control_col,
):
    """

    Parameters
    ----------
    trace_label : str
        trace label indicating geography zone being integerized (e.g. PUMA_600)
    control_spec : pandas.Dataframe
        full control spec with columns 'target', 'seed_table', 'importance', ...
    control_totals : pandas.Series
        control totals explicitly specified for this zone
    incidence_table : pandas.Dataframe
    float_weights : pandas.Series
        balanced float weights to integerize
    total_hh_control_col : str
        name of total_hh column (preferentially constrain to match this control)

    Returns
    -------
    integerized_weights : pandas.Series
    status : str
        as defined in integerizer.STATUS_TEXT and STATUS_SUCCESS
    """

    # incidence table should only have control columns
    incidence_table = incidence_table[control_spec.target]

    if total_hh_control_col not in incidence_table.columns:
        raise RuntimeError(
            "total_hh_control column '%s' not found in incidence table"
            % total_hh_control_col
        )

    zero_weight_rows = float_weights == 0
    if zero_weight_rows.any():
        logger.debug(
            "omitting %s zero weight rows out of %s"
            % (zero_weight_rows.sum(), len(incidence_table.index))
        )
        incidence_table = incidence_table[~zero_weight_rows]
        float_weights = float_weights[~zero_weight_rows]

    total_hh_control_value = control_totals[total_hh_control_col]

    status = None
    if config.setting("INTEGERIZE_WITH_BACKSTOPPED_CONTROLS") and len(
        control_totals
    ) < len(incidence_table.columns):

        ##########################################
        # - backstopped control_totals
        # Use balanced float weights to establish target values for all control values
        # note: this more frequently results in infeasible solver results
        ##########################################

        relaxed_control_totals = np.round(
            np.dot(np.asanyarray(float_weights), incidence_table.values)
        )
        relaxed_control_totals = pd.Series(
            relaxed_control_totals, index=incidence_table.columns.values
        )

        # if the incidence table has only one record, then the final integer weights
        # should be just an array with 1 element equal to the total number of households;
        assert len(incidence_table.index) > 1

        integerizer = Integerizer(
            incidence_table=incidence_table,
            control_importance_weights=control_spec.importance,
            float_weights=float_weights,
            relaxed_control_totals=relaxed_control_totals,
            total_hh_control_value=total_hh_control_value,
            total_hh_control_index=incidence_table.columns.get_loc(
                total_hh_control_col
            ),
            control_is_hh_based=control_spec["seed_table"] == "households",
            trace_label="backstopped_%s" % trace_label,
        )

        # otherwise, solve for the integer weights using the Mixed Integer Programming solver.
        status = integerizer.integerize()

        logger.debug(
            "Integerizer status for backstopped %s: %s" % (trace_label, status)
        )

    # if we either tried backstopped controls or failed, or never tried at all
    if status not in STATUS_SUCCESS:

        ##########################################
        # - unbackstopped partial control_totals
        # Use balanced weights to establish control totals only for explicitly specified controls
        # note: this usually results in feasible solver results, except for some single hh zones
        ##########################################

        balanced_control_cols = control_totals.index
        incidence_table = incidence_table[balanced_control_cols]
        control_spec = control_spec[control_spec.target.isin(balanced_control_cols)]

        relaxed_control_totals = np.round(
            np.dot(np.asanyarray(float_weights), incidence_table.values)
        )
        relaxed_control_totals = pd.Series(
            relaxed_control_totals, index=incidence_table.columns.values
        )

        integerizer = Integerizer(
            incidence_table=incidence_table,
            control_importance_weights=control_spec.importance,
            float_weights=float_weights,
            relaxed_control_totals=relaxed_control_totals,
            total_hh_control_value=total_hh_control_value,
            total_hh_control_index=incidence_table.columns.get_loc(
                total_hh_control_col
            ),
            control_is_hh_based=control_spec["seed_table"] == "households",
            trace_label=trace_label,
        )

        status = integerizer.integerize()

        logger.debug(
            "Integerizer status for unbackstopped %s: %s" % (trace_label, status)
        )

    if status not in STATUS_SUCCESS:
        logger.error(
            "Integerizer failed for %s status %s. "
            "Returning smart-rounded original weights" % (trace_label, status)
        )
    elif status != "OPTIMAL":
        logger.warning(
            "Integerizer status non-optimal for %s status %s." % (trace_label, status)
        )

    integerized_weights = pd.Series(0, index=zero_weight_rows.index)
    integerized_weights.update(integerizer.weights["integerized_weight"])
    return integerized_weights, status


def do_simul_integerizing(
    trace_label,
    incidence_df,
    sub_weights,
    sub_controls_df,
    control_spec,
    total_hh_control_col,
    sub_geography,
    sub_control_zones,
):
    """

    Wrapper around simultaneous integerizer to handle solver failure for infeasible subzones.

    Simultaneous integerize balanced float sub_weights,
    If simultaneous integerization fails, integerize serially to identify infeasible subzones,
    remove and smart_round infeasible subzones, and try simultaneous integerization again.
    (That ought to succeed, but if not, then fall back to all sequential integerization)
    Finally combine all results into a single result dataframe.

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
        index is zone id and value is zone label (e.g. TAZ_101)
        for use in sub_controls_df column names

    Returns
    -------
    integer_weights_df : pandas.DataFrame
        canonical form weight table, with columns for 'balanced_weight', 'integer_weight'
        plus columns for household id, and sub_geography zone ids
    """

    # try simultaneous integerization of all subzones
    status, integerized_weights_df = try_simul_integerizing(
        trace_label,
        incidence_df,
        sub_weights,
        sub_controls_df,
        sub_geography,
        control_spec,
        total_hh_control_col,
        sub_control_zones,
    )

    if status in STATUS_SUCCESS:
        logger.info(
            "do_simul_integerizing succeeded for %s status %s. " % (trace_label, status)
        )
        return integerized_weights_df

    logger.warning(
        "do_simul_integerizing failed for %s status %s. " % (trace_label, status)
    )

    # if simultaneous integerization failed, sequentially integerize to detect infeasible subzones
    # infeasible zones will be smart rounded and returned in rounded_weights_df
    (
        feasible_zone_ids,
        rounded_zone_ids,
        sequentially_integerized_weights_df,
        rounded_weights_df,
    ) = do_sequential_integerizing(
        trace_label,
        incidence_df,
        sub_weights,
        sub_controls_df,
        control_spec,
        total_hh_control_col,
        sub_control_zones,
        sub_geography,
        combine_results=False,
    )

    if len(feasible_zone_ids) == 0:
        # if all subzones are infeasible, then we don't have any feasible zones to try
        # so the best we can do is return rounded_weights_df
        logger.warning(
            "do_sequential_integerizing failed for all subzones %s. " % trace_label
        )
        logger.info(
            "do_simul_integerizing returning smart rounded weights for %s."
            % trace_label
        )
        return rounded_weights_df

    if len(rounded_zone_ids) == 0:
        # if all subzones are feasible, then there are no zones to remove in order to retry
        # so the best we can do is return sequentially_integerized_weights_df
        logger.warning(
            "do_simul_integerizing failed but found no infeasible sub zones %s. "
            % trace_label
        )
        logger.info(
            "do_simul_integerizing falling back to sequential integerizing for %s."
            % trace_label
        )
        return sequentially_integerized_weights_df

    if len(feasible_zone_ids) == 1:
        # if only one zone is feasible, not much point in simul_integerizing it
        # so the best we can do is return do_sequential_integerizing combined results
        logger.warning(
            "do_simul_integerizing failed but found no infeasible sub zones %s. "
            % trace_label
        )
        return pd.concat([sequentially_integerized_weights_df, rounded_weights_df])

    # - remove the infeasible subzones and retry simul_integerizing

    sub_controls_df = sub_controls_df.loc[feasible_zone_ids]
    sub_control_zones = sub_control_zones.loc[
        sub_control_zones.index.isin(feasible_zone_ids)
    ]
    sub_weights = sub_weights[sub_control_zones]

    logger.info(
        "do_simul_integerizing %s infeasable subzones for %s. "
        % (len(rounded_zone_ids), trace_label)
    )

    status, integerized_weights_df = try_simul_integerizing(
        "retry_%s" % trace_label,
        incidence_df,
        sub_weights,
        sub_controls_df,
        sub_geography,
        control_spec,
        total_hh_control_col,
        sub_control_zones,
    )

    if status in STATUS_SUCCESS:
        # we successfully simul_integerized the sequentially feasible sub zones, so we can
        # return the simul_integerized results along with the rounded_weights for the infeasibles
        logger.info(
            "do_simul_integerizing retry succeeded for %s status %s. "
            % (trace_label, status)
        )
        return pd.concat([integerized_weights_df, rounded_weights_df])

    # haven't seen this happen, but I suppose it could...
    logger.error(
        "do_simul_integerizing retry failed for %s status %s." % (trace_label, status)
    )
    logger.info(
        "do_simul_integerizing falling back to sequential integerizing for %s."
        % trace_label
    )

    # nothing to do but return do_sequential_integerizing combined results
    return pd.concat([sequentially_integerized_weights_df, rounded_weights_df])


def do_sequential_integerizing(
    trace_label,
    incidence_df,
    sub_weights,
    sub_controls_df,
    control_spec,
    total_hh_control_col,
    sub_control_zones,
    sub_geography,
    combine_results=True,
):
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
    for zone_id, zone_name in list(sub_control_zones.items()):

        logger.info(
            "sequential_integerizing zone_id %s zone_name %s" % (zone_id, zone_name)
        )

        weights = sub_weights[zone_name]

        sub_trace_label = "%s_%s_%s" % (trace_label, sub_geography, zone_id)

        integer_weights, status = do_integerizing(
            trace_label=sub_trace_label,
            control_spec=control_spec,
            control_totals=sub_controls_df.loc[zone_id],
            incidence_table=incidence_df[control_spec.target],
            float_weights=weights,
            total_hh_control_col=total_hh_control_col,
        )

        zone_weights_df = pd.DataFrame(index=list(range(0, len(integer_weights.index))))
        zone_weights_df[weights.index.name] = weights.index
        zone_weights_df[sub_geography] = zone_id
        zone_weights_df["balanced_weight"] = weights.values
        zone_weights_df["integer_weight"] = integer_weights.astype(int).values

        if status in STATUS_SUCCESS:
            integerized_weights_list.append(zone_weights_df)
            integerized_zone_ids.append(zone_id)
        else:
            rounded_weights_list.append(zone_weights_df)
            rounded_zone_ids.append(zone_id)

    if combine_results:
        integerized_weights_df = pd.concat(
            integerized_weights_list + rounded_weights_list
        )
        return integerized_weights_df

    integerized_weights_df = (
        pd.concat(integerized_weights_list) if integerized_zone_ids else None
    )
    rounded_weights_df = pd.concat(rounded_weights_list) if rounded_zone_ids else None

    return (
        integerized_zone_ids,
        rounded_zone_ids,
        integerized_weights_df,
        rounded_weights_df,
    )


def do_no_integerizing(sub_weights, sub_control_zones, sub_geography, **kwargs):
    """
    Return a dataframe with the sub_weights as integerized weights
    without any integerization, just to satisfy the interface.
    Parameters
    ----------
    sub_weights : pandas.DataFrame
        balanced subzone household sample weights to integerize
    sub_controls_df : pandas.Dataframe
        sub_geography controls (one row per zone indexed by sub_zone id)
    sub_geography : str
        subzone geography name (e.g. 'TAZ')
    Returns
    -------
    integerized_weights_df : pandas.DataFrame
        canonical form weight table, with columns for 'balanced_weight', 'integer_weight'
        plus columns for household id, and sub_geography zone ids
    """
    integerized_weights_list = []
    rounded_weights_list = []
    integerized_zone_ids = []

    for zone_id, zone_name in sub_control_zones.items():

        logger.info(
            "sequential_integerizing zone_id %s zone_name %s" % (zone_id, zone_name)
        )

        weights = sub_weights[zone_name]

        zone_weights_df = pd.DataFrame(index=range(0, len(weights.index)))
        zone_weights_df[weights.index.name] = weights.index
        zone_weights_df[sub_geography] = zone_id
        zone_weights_df["balanced_weight"] = weights.values

        integerized_weights_list.append(zone_weights_df)
        integerized_zone_ids.append(zone_id)

    integerized_weights_df = pd.concat(integerized_weights_list + rounded_weights_list)
    return integerized_weights_df
