# PopulationSim
# See full license in LICENSE.txt.

import numpy as np
from populationsim.integerizing.constants import (
    STATUS_OPTIMAL,
    STATUS_FEASIBLE,
    STATUS_SUCCESS,
)


def _add_variables(model, upper_bounds, objective, names):
    """Append continuous variables with lower bound 0 to an MPModelProto."""
    add = model.variable.add
    for ub, obj, name in zip(
        upper_bounds.tolist(), objective.tolist(), names, strict=True
    ):
        add(lower_bound=0.0, upper_bound=ub, objective_coefficient=obj, name=name)


def _add_constraint(model, lower_bound, upper_bound, var_index, coefficient):
    """Append ``lower_bound <= sum(coefficient * x[var_index]) <= upper_bound``."""
    model.constraint.add(
        lower_bound=float(lower_bound),
        upper_bound=float(upper_bound),
        var_index=var_index.tolist(),
        coefficient=coefficient.tolist(),
    )


def _load_solver(name, model):
    """
    Load an MPModelProto into a new CBC solver.

    Variable names are kept and constraints get the same automatic names as
    when they are created with pywraplp calls, so the solver holds exactly
    the model the per-element construction builds.
    """
    from ortools.linear_solver import pywraplp

    solver = pywraplp.Solver(name, pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
    error = solver.LoadModelFromProtoKeepNames(model)
    if error:
        raise RuntimeError("LoadModelFromProto failed: %s" % error)
    return solver


def _solve(solver, num_x, timeout_in_seconds):
    """Solve; return (result_status, values of the first num_x variables)."""
    from ortools.linear_solver import linear_solver_pb2

    solver.set_time_limit(timeout_in_seconds * 1000)
    solver.EnableOutput()

    result_status = solver.Solve()

    response = linear_solver_pb2.MPSolutionResponse()
    solver.FillSolutionResponseProto(response)
    values = np.array(response.variable_value[:num_x], dtype=np.float64)
    return result_status, values


def _single_model(
    incidence,
    resid_weights,
    log_resid_weights,
    control_importance_weights,
    total_hh_control_index,
    lp_right_hand_side,
    relax_ge_upper_bound,
    hh_constraint_ge_bound,
):
    """Build the single-integerizer LP as an MPModelProto (see np_integerizer_ortools)."""
    from ortools.linear_solver import linear_solver_pb2

    control_count, sample_count = incidence.shape
    model = linear_solver_pb2.MPModelProto(maximize=True)

    # - binary x variables, capped at 0 where the residual weight is zero
    # - objective: maximize log residual weight; the tiny epsilon * hh term
    #   breaks ties and is added in log_resid_weights' precision
    epsilon = 1e-6
    tie_break = (epsilon * np.arange(sample_count)).astype(log_resid_weights.dtype)
    _add_variables(
        model,
        1.0 - (resid_weights == 0.0),
        log_resid_weights + tie_break,
        ["x_%d" % hh for hh in range(sample_count)],
    )

    # - relaxation variables, penalized by control importance
    #   (none for the total households control)
    relax_le = {}
    relax_ge = {}
    for c in range(control_count):
        if c == total_hh_control_index:
            continue
        relax_le[c] = len(model.variable)
        relax_ge[c] = relax_le[c] + 1
        _add_variables(
            model,
            np.array([lp_right_hand_side[c], relax_ge_upper_bound[c]]),
            -np.array([control_importance_weights[c]] * 2),
            ["relax_le_%d" % c, "relax_ge_%d" % c],
        )

    # - inequality constraints, one pair per control
    for c in relax_le:
        hh = np.flatnonzero(incidence[c])
        coef = incidence[c, hh]
        _add_constraint(
            model,
            0,
            lp_right_hand_side[c],
            np.append(hh, relax_le[c]),
            np.append(coef, -1.0),
        )
        _add_constraint(
            model,
            lp_right_hand_side[c],
            hh_constraint_ge_bound[c],
            np.append(hh, relax_ge[c]),
            np.append(coef, 1.0),
        )

    # - equality constraint for the total households control
    total_hh = lp_right_hand_side[total_hh_control_index]
    _add_constraint(
        model, total_hh, total_hh, np.arange(sample_count), np.ones(sample_count)
    )

    return model


def _simul_model(
    sub_int_weights,
    parent_countrol_importance,
    parent_relax_ge_upper_bound,
    sub_control_importance,
    sub_float_weights,
    sub_resid_weights,
    lp_right_hand_side,
    parent_hh_constraint_ge_bound,
    sub_incidence,
    parent_incidence,
    total_hh_right_hand_side,
    relax_ge_upper_bound,
    parent_lp_right_hand_side,
    hh_constraint_ge_bound,
    parent_resid_weights,
    total_hh_sub_control_index,
    total_hh_parent_control_index,
):
    """Build the simul-integerizer LP as an MPModelProto (see np_simul_integerizer_ortools)."""
    from ortools.linear_solver import linear_solver_pb2

    sample_count, sub_control_count = sub_incidence.shape
    _, parent_control_count = parent_incidence.shape
    sub_zone_count, _ = sub_float_weights.shape
    model = linear_solver_pb2.MPModelProto(maximize=True)

    LOG_OVERFLOW = -725
    log_resid_weights = np.log(np.maximum(sub_resid_weights, np.exp(LOG_OVERFLOW)))
    assert not np.isnan(log_resid_weights).any()

    log_parent_resid_weights = np.log(
        np.maximum(parent_resid_weights, np.exp(LOG_OVERFLOW))
    )
    assert not np.isnan(log_parent_resid_weights).any()

    # - resid weight variables x[z, hh] at index z * sample_count + hh;
    #   x_max is 1.0 unless resid_weights is zero, in which case x is fixed at 0.0
    x_max = (~(sub_float_weights == sub_int_weights)).astype(float)
    _add_variables(
        model,
        x_max.ravel(),
        (log_resid_weights + log_parent_resid_weights).ravel(),
        [
            "x[%s,%s]" % (z, hh)
            for z in range(sub_zone_count)
            for hh in range(sample_count)
        ],
    )

    # - relaxation variables, penalized by control importance
    sub_controls = [
        c for c in range(sub_control_count) if c != total_hh_sub_control_index
    ]
    relax_le = {}
    for z in range(sub_zone_count):
        for c in sub_controls:
            relax_le[z, c] = len(model.variable)
            _add_variables(
                model,
                np.array([lp_right_hand_side[z, c], relax_ge_upper_bound[z, c]]),
                -np.array([sub_control_importance[c]] * 2),
                ["relax_le[%s,%s]" % (z, c), "relax_ge[%s,%s]" % (z, c)],
            )

    parent_relax_le = {}
    for c in range(parent_control_count):
        parent_relax_le[c] = len(model.variable)
        _add_variables(
            model,
            np.array([parent_lp_right_hand_side[c], parent_relax_ge_upper_bound[c]]),
            -np.array([parent_countrol_importance[c]] * 2),
            ["parent_relax_le[%s]" % c, "parent_relax_ge[%s]" % c],
        )

    # - sub inequality constraints
    sub_hh = {c: np.flatnonzero(sub_incidence[:, c]) for c in sub_controls}
    for z in range(sub_zone_count):
        for c in sub_controls:
            hh = sub_hh[c]
            x = z * sample_count + hh
            coef = sub_incidence[hh, c]
            _add_constraint(
                model,
                0,
                lp_right_hand_side[z, c],
                np.append(x, relax_le[z, c]),
                np.append(coef, -1.0),
            )
            _add_constraint(
                model,
                lp_right_hand_side[z, c],
                hh_constraint_ge_bound[z, c],
                np.append(x, relax_le[z, c] + 1),
                np.append(coef, 1.0),
            )

    # - equality constraint for the total households control
    for z in range(sub_zone_count):
        total_hh = total_hh_right_hand_side[z]
        _add_constraint(
            model,
            total_hh,
            total_hh,
            z * sample_count + np.arange(sample_count),
            np.ones(sample_count),
        )

    # - parent inequality constraints, summed over all sub zones
    zone_offsets = np.arange(sub_zone_count)[:, None] * sample_count
    for c in range(parent_control_count):
        if c == total_hh_parent_control_index:
            continue
        hh = np.flatnonzero(parent_incidence[:, c])
        x = (zone_offsets + hh).ravel()
        coef = np.tile(parent_incidence[hh, c], sub_zone_count)
        _add_constraint(
            model,
            0,
            parent_lp_right_hand_side[c],
            np.append(x, parent_relax_le[c]),
            np.append(coef, -1.0),
        )
        _add_constraint(
            model,
            parent_lp_right_hand_side[c],
            parent_hh_constraint_ge_bound[c],
            np.append(x, parent_relax_le[c] + 1),
            np.append(coef, 1.0),
        )

    return model


def np_integerizer_ortools(
    incidence,
    resid_weights,
    log_resid_weights,
    control_importance_weights,
    total_hh_control_index,
    lp_right_hand_side,
    relax_ge_upper_bound,
    hh_constraint_ge_bound,
    timeout_in_seconds,
):
    """
    ortools single-integerizer function taking numpy data types and conforming to a
    standard function signature that allows it to be swapped interchangeably with alternate
    LP implementations.


    Parameters
    ----------
    incidence : numpy.ndarray(control_count, sample_count) float
    resid_weights : numpy.ndarray(sample_count,) float
    log_resid_weights : numpy.ndarray(sample_count,) float
    control_importance_weights : numpy.ndarray(control_count,) float
    total_hh_control_index : int
    lp_right_hand_side : numpy.ndarray(control_count,) float
    relax_ge_upper_bound : numpy.ndarray(control_count,) float
    hh_constraint_ge_bound : numpy.ndarray(control_count,) float
    timeout_in_seconds : int

    Returns
    -------
    resid_weights_out : numpy.ndarray(sample_count,)
    status_text : str
    """

    from ortools.linear_solver import pywraplp

    STATUS_TEXT = {
        pywraplp.Solver.OPTIMAL: "OPTIMAL",
        pywraplp.Solver.FEASIBLE: "FEASIBLE",
        pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
        pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
        pywraplp.Solver.ABNORMAL: "ABNORMAL",
        pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
    }

    sample_count = incidence.shape[1]

    model = _single_model(
        incidence,
        resid_weights,
        log_resid_weights,
        control_importance_weights,
        total_hh_control_index,
        lp_right_hand_side,
        relax_ge_upper_bound,
        hh_constraint_ge_bound,
    )
    solver = _load_solver("IntegerizeCbc", model)
    del model  # free the proto before solving; the solver has its own copy
    result_status, x = _solve(solver, sample_count, timeout_in_seconds)

    status_text = STATUS_TEXT[result_status]

    if status_text in STATUS_SUCCESS:
        resid_weights_out = x
    else:
        resid_weights_out = resid_weights

    return resid_weights_out, status_text


def np_simul_integerizer_ortools(
    sub_int_weights,
    parent_countrol_importance,
    parent_relax_ge_upper_bound,
    sub_control_importance,
    sub_float_weights,
    sub_resid_weights,
    lp_right_hand_side,
    parent_hh_constraint_ge_bound,
    sub_incidence,
    parent_incidence,
    total_hh_right_hand_side,
    relax_ge_upper_bound,
    parent_lp_right_hand_side,
    hh_constraint_ge_bound,
    parent_resid_weights,
    total_hh_sub_control_index,
    total_hh_parent_control_index,
    timeout_in_seconds,
):
    """
    ortools-based siuml-integerizer function taking numpy data types and conforming to a
    standard function signature that allows it to be swapped interchangeably with alternate
    LP implementations.

    Parameters
    ----------
    sub_int_weights : numpy.ndarray(sub_zone_count, sample_count) int
    parent_countrol_importance : numpy.ndarray(parent_control_count,) float
    parent_relax_ge_upper_bound : numpy.ndarray(parent_control_count,) float
    sub_control_importance : numpy.ndarray(sub_control_count,) float
    sub_float_weights : numpy.ndarray(sub_zone_count, sample_count) float
    sub_resid_weights : numpy.ndarray(sub_zone_count, sample_count) float
    lp_right_hand_side : numpy.ndarray(sub_zone_count, sub_control_count) float
    parent_hh_constraint_ge_bound : numpy.ndarray(parent_control_count,) float
    sub_incidence : numpy.ndarray(sample_count, sub_control_count) float
    parent_incidence : numpy.ndarray(sample_count, parent_control_count) float
    total_hh_right_hand_side : numpy.ndarray(sub_zone_count,) float
    relax_ge_upper_bound : numpy.ndarray(sub_zone_count, sub_control_count) float
    parent_lp_right_hand_side : numpy.ndarray(parent_control_count,) float
    hh_constraint_ge_bound : numpy.ndarray(sub_zone_count, sub_control_count) float
    parent_resid_weights : numpy.ndarray(sample_count,) float
    total_hh_sub_control_index : int
    total_hh_parent_control_index : int
    timeout_in_seconds : int

    Returns
    -------
    resid_weights_out : numpy.ndarray of float
        residual weights in range [0..1] as solved,
        or, in case of failure, sub_resid_weights unchanged
    status_text : string
        STATUS_OPTIMAL, STATUS_FEASIBLE in case of success, or a solver-specific failure status
    """

    from ortools.linear_solver import pywraplp

    STATUS_TEXT = {
        pywraplp.Solver.OPTIMAL: STATUS_OPTIMAL,
        pywraplp.Solver.FEASIBLE: STATUS_FEASIBLE,
        pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
        pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
        pywraplp.Solver.ABNORMAL: "ABNORMAL",
        pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
    }

    # setting indexes to -1 prevents creation of hh_controls relaxation variables
    # setting hh_control importance to zero eliminates them from the objective function
    # the latter approach is used by the cvx version
    # total_hh_sub_control_index = -1
    sub_control_importance[total_hh_sub_control_index] = 0

    # FIXME total_hh_parent_control_index should not exist???
    if total_hh_parent_control_index > 0:
        parent_countrol_importance[total_hh_parent_control_index] = 0

    model = _simul_model(
        sub_int_weights,
        parent_countrol_importance,
        parent_relax_ge_upper_bound,
        sub_control_importance,
        sub_float_weights,
        sub_resid_weights,
        lp_right_hand_side,
        parent_hh_constraint_ge_bound,
        sub_incidence,
        parent_incidence,
        total_hh_right_hand_side,
        relax_ge_upper_bound,
        parent_lp_right_hand_side,
        hh_constraint_ge_bound,
        parent_resid_weights,
        total_hh_sub_control_index,
        total_hh_parent_control_index,
    )
    solver = _load_solver("SimulIntegerizeCbc", model)
    del model  # free the proto before solving; the solver has its own copy
    result_status, x = _solve(solver, sub_float_weights.size, timeout_in_seconds)

    status_text = STATUS_TEXT[result_status]

    if status_text in STATUS_SUCCESS:
        resid_weights_out = x.reshape(sub_resid_weights.shape)
    else:
        resid_weights_out = sub_resid_weights

    return resid_weights_out, status_text
