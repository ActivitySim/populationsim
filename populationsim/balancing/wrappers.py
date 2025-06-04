import numpy as np
import pandas as pd
from populationsim.core.config import setting
from populationsim.balancing.single_balancer import ListBalancer
from populationsim.balancing.simul_balancer import SimultaneousListBalancer
from populationsim.balancing.constants import DEFAULT_MAX_ITERATIONS


def do_balancing(
    control_spec,
    total_hh_control_col,
    max_expansion_factor,
    min_expansion_factor,
    absolute_upper_bound,
    absolute_lower_bound,
    incidence_df,
    control_totals,
    initial_weights,
    use_hard_constraints,
    use_numba,
    numba_precision,
):

    # incidence table should only have control columns
    incidence_df = incidence_df[control_spec.target]

    # master_control_index is total_hh_control_col
    if total_hh_control_col not in incidence_df.columns:
        raise RuntimeError(
            "total_hh_control column '%s' not found in incidence table"
            % total_hh_control_col
        )
    total_hh_control_index = incidence_df.columns.get_loc(total_hh_control_col)

    # control_totals series rows and incidence_df columns should be aligned
    assert total_hh_control_index == control_totals.index.get_loc(total_hh_control_col)

    control_totals = control_totals.values

    control_importance_weights = control_spec.importance

    if min_expansion_factor:

        # number_of_households in this seed geograpy as specified in seed_controls
        number_of_households = control_totals[total_hh_control_index]

        total_weights = initial_weights.sum()
        lb_ratio = (
            min_expansion_factor * float(number_of_households) / float(total_weights)
        )

        # Added hard limit of min_expansion_factor value that would otherwise drift
        # due to the float(number_of_households) / float(total_weights) calculation
        if use_hard_constraints:
            lb_ratio = np.clip(lb_ratio, a_min=min_expansion_factor, a_max=None)

        lb_weights = initial_weights * lb_ratio

        if absolute_lower_bound:
            lb_weights = lb_weights.clip(lower=absolute_lower_bound)
        else:
            lb_weights = lb_weights.clip(lower=0)

    elif absolute_lower_bound:
        lb_weights = initial_weights.clip(lower=absolute_lower_bound)

    else:
        lb_weights = None

    if max_expansion_factor:

        # number_of_households in this seed geograpy as specified in seed_controlss
        number_of_households = control_totals[total_hh_control_index]

        total_weights = initial_weights.sum()
        ub_ratio = (
            max_expansion_factor * float(number_of_households) / float(total_weights)
        )

        # Added hard limit of max_expansion_factor value that would otherwise drift
        # due to the float(number_of_households) / float(total_weights) calculation
        if use_hard_constraints:
            ub_ratio = np.clip(ub_ratio, a_max=max_expansion_factor, a_min=None)

        ub_weights = initial_weights * ub_ratio

        if absolute_upper_bound:
            ub_weights = (
                ub_weights.round().clip(upper=absolute_upper_bound, lower=1).astype(int)
            )
        else:
            ub_weights = ub_weights.round().clip(lower=1).astype(int)

    elif absolute_upper_bound:
        ub_weights = (
            ub_weights.round().clip(upper=absolute_upper_bound, lower=1).astype(int)
        )

    else:
        ub_weights = None

    max_iterations = setting(
        "MAX_BALANCE_ITERATIONS_SEQUENTIAL", DEFAULT_MAX_ITERATIONS
    )

    balancer = ListBalancer(
        incidence_table=incidence_df,
        initial_weights=initial_weights,
        control_totals=control_totals,
        control_importance_weights=control_importance_weights,
        lb_weights=lb_weights,
        ub_weights=ub_weights,
        master_control_index=total_hh_control_index,
        max_iterations=max_iterations,
        use_numba=use_numba,
        numba_precision=numba_precision,
    )

    status, weights, controls = balancer.balance()

    return status, weights, controls


def do_simul_balancing(
    incidence_df,
    parent_weights,
    sub_controls_df,
    control_spec,
    total_hh_control_col,
    parent_geography,
    parent_id,
    sub_geographies,
    sub_control_zones,
    use_numba,
    numba_precision,
):
    """

    Parameters
    ----------
    incidence_df : pandas.Dataframe
        full incidence_df for all hh samples in seed zone
    parent_weights : pandas.Series
        parent zone balanced (possibly integerized) aggregate target weights
    sub_controls_df : pandas.Dataframe
        sub_geography controls (one row per zone indexed by sub_zone id)
    control_spec : pandas.Dataframe
        full control spec with columns 'target', 'seed_table', 'importance', ...
    total_hh_control_col : str
        name of total_hh column (so we can preferentially match this control)
    parent_geography : str
        parent geography zone name
    parent_id : int
        parent geography zone id
    sub_geographies : list(str)
        list of subgeographies in descending order
    sub_control_zones : pandas.Series
        index is zone id and value is zone label (e.g. TAZ_101)
        for use in sub_controls_df column names

    Returns
    -------
    sub_zone_weights : pandas.DataFrame
        balanced subzone household float sample weights
    """

    sub_control_spec = control_spec[control_spec["geography"].isin(sub_geographies)]

    assert (sub_controls_df.columns.values == sub_control_spec.target.values).all()

    # controls - organized in legible form
    controls = pd.DataFrame({"name": sub_control_spec.target})
    controls["importance"] = sub_control_spec.importance
    controls["total"] = sub_controls_df.sum(axis=0).values

    # Perform in bulk rather than slow loop over pandas
    new_controls = sub_controls_df.loc[sub_control_zones.index].transpose()
    new_controls.columns = sub_control_zones.values

    if len(set(controls.columns).intersection(new_controls.columns)) > 0:
        controls.update(new_controls)
    else:
        controls = controls.merge(new_controls, left_on="name", right_index=True)

    # for zone, zone_name in sub_control_zones.items():
    #     controls[zone_name] = sub_controls_df.loc[zone].values

    # incidence table should only have control columns
    sub_incidence_df = incidence_df[sub_control_spec.target]

    balancer = SimultaneousListBalancer(
        incidence_table=sub_incidence_df,
        parent_weights=parent_weights,
        controls=controls,
        sub_control_zones=sub_control_zones,
        total_hh_control_col=total_hh_control_col,
        use_numba=use_numba,
        numba_precision=numba_precision,
    )

    status = balancer.balance()

    return balancer.sub_zone_weights, status
