# PopulationSim
# See full license in LICENSE.txt.

# from builtins import str
from builtins import range
import logging
import os

import numpy as np

STATUS_OPTIMAL = 'OPTIMAL'
STATUS_FEASIBLE = 'FEASIBLE'
STATUS_SUCCESS = [STATUS_OPTIMAL, STATUS_FEASIBLE]


def np_integerizer_ortools(
        incidence,
        resid_weights,
        log_resid_weights,
        control_importance_weights,
        total_hh_control_index,
        lp_right_hand_side,
        relax_ge_upper_bound,
        hh_constraint_ge_bound):
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

    Returns
    -------
    resid_weights_out : numpy.ndarray(sample_count,)
    status_text : str
    """

    from ortools.linear_solver import pywraplp

    STATUS_TEXT = {
        pywraplp.Solver.OPTIMAL: 'OPTIMAL',
        pywraplp.Solver.FEASIBLE: 'FEASIBLE',
        pywraplp.Solver.INFEASIBLE: 'INFEASIBLE',
        pywraplp.Solver.UNBOUNDED: 'UNBOUNDED',
        pywraplp.Solver.ABNORMAL: 'ABNORMAL',
        pywraplp.Solver.NOT_SOLVED: 'NOT_SOLVED',
    }
    CBC_TIMEOUT_IN_SECONDS = 60

    control_count, sample_count = incidence.shape

    # - Instantiate a mixed-integer solver
    solver = pywraplp.Solver('IntegerizeCbc', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

    # - Create binary integer variables
    x = [[]] * sample_count
    for hh in range(0, sample_count):
        # max_x == 0.0 if float_weights is an int, otherwise 1.0
        max_x = 1.0 - (resid_weights[hh] == 0.0)
        x[hh] = solver.NumVar(0.0, max_x, 'x_' + str(hh))

    # - Create positive continuous constraint relaxation variables
    relax_le = [[]] * control_count
    relax_ge = [[]] * control_count
    for c in range(0, control_count):
        # no relaxation for total households control
        if c != total_hh_control_index:
            relax_le[c] = solver.NumVar(0.0, lp_right_hand_side[c], 'relax_le_' + str(c))
            relax_ge[c] = solver.NumVar(0.0, relax_ge_upper_bound[c], 'relax_ge_' + str(c))

    # - Set objective function coefficients
    # use negative for objective and positive for relaxation penalties since solver is minimizing
    # objective = solver.Objective()
    # for hh in range(sample_count):
    #     objective.SetCoefficient(x[hh], -1.0 * log_resid_weights[hh])
    # for c in range(control_count):
    #     if c != total_hh_control_index:
    #         objective.SetCoefficient(relax_le[c], control_importance_weights[c])
    #         objective.SetCoefficient(relax_ge[c], control_importance_weights[c])

    z = solver.Sum(x[hh] * log_resid_weights[hh]
                   for hh in range(sample_count)) - \
        solver.Sum(relax_le[c] * control_importance_weights[c]
                   for c in range(control_count) if c != total_hh_control_index) - \
        solver.Sum(relax_ge[c] * control_importance_weights[c]
                   for c in range(control_count) if c != total_hh_control_index)

    objective = solver.Maximize(z)

    # - inequality constraints
    hh_constraint_ge = [[]] * control_count
    hh_constraint_le = [[]] * control_count
    for c in range(0, control_count):
        # don't add inequality constraints for total households control
        if c == total_hh_control_index:
            continue
        # add the lower bound relaxation inequality constraint
        hh_constraint_le[c] = solver.Constraint(0, lp_right_hand_side[c])
        for hh in range(0, sample_count):
            hh_constraint_le[c].SetCoefficient(x[hh], incidence[c, hh])
            hh_constraint_le[c].SetCoefficient(relax_le[c], -1.0)

        # add the upper bound relaxation inequality constraint
        hh_constraint_ge[c] = solver.Constraint(lp_right_hand_side[c], hh_constraint_ge_bound[c])
        for hh in range(0, sample_count):
            hh_constraint_ge[c].SetCoefficient(x[hh], incidence[c, hh])
            hh_constraint_ge[c].SetCoefficient(relax_ge[c], 1.0)

    # using Add and Sum is easier to read but a lot slower
    # for c in range(control_count):
    #     if c == total_hh_control_index:
    #         continue
    #     solver.Add(solver.Sum(x[hh]*incidence[c, hh] for hh in range(sample_count)) - relax_le[c]
    #                >= 0)
    #     solver.Add(solver.Sum(x[hh]*incidence[c, hh] for hh in range(sample_count)) - relax_le[c]
    #                <= lp_right_hand_side[c])
    #     solver.Add(solver.Sum(x[hh]*incidence[c, hh] for hh in range(sample_count)) + relax_ge[c]
    #                >= lp_right_hand_side[c])
    #     solver.Add(solver.Sum(x[hh]*incidence[c, hh] for hh in range(sample_count)) + relax_ge[c]
    #                <= hh_constraint_ge_bound[c])

    # - equality constraint for the total households control
    total_hh_constraint = lp_right_hand_side[total_hh_control_index]
    constraint_eq = solver.Constraint(total_hh_constraint, total_hh_constraint)
    for hh in range(0, sample_count):
        constraint_eq.SetCoefficient(x[hh], 1.0)

    solver.set_time_limit(CBC_TIMEOUT_IN_SECONDS * 1000)

    solver.EnableOutput()

    result_status = solver.Solve()

    status_text = STATUS_TEXT[result_status]

    if status_text in STATUS_SUCCESS:
        resid_weights_out = np.asanyarray([x.solution_value() for x in x]).astype(np.float64)
    else:
        resid_weights_out = resid_weights

    return resid_weights_out, status_text


def np_simul_integerizer_ortools(
        sub_int_weights,
        parent_countrol_importance,
        parent_relax_ge_upper_bound,
        sub_countrol_importance,
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
        total_hh_parent_control_index):

    """
    ortools-based siuml-integerizer function taking numpy data types and conforming to a
    standard function signature that allows it to be swapped interchangeably with alternate
    LP implementations.

    Parameters
    ----------
    sub_int_weights : numpy.ndarray(sub_zone_count, sample_count) int
    parent_countrol_importance : numpy.ndarray(parent_control_count,) float
    parent_relax_ge_upper_bound : numpy.ndarray(parent_control_count,) float
    sub_countrol_importance : numpy.ndarray(sub_control_count,) float
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
        pywraplp.Solver.INFEASIBLE: 'INFEASIBLE',
        pywraplp.Solver.UNBOUNDED: 'UNBOUNDED',
        pywraplp.Solver.ABNORMAL: 'ABNORMAL',
        pywraplp.Solver.NOT_SOLVED: 'NOT_SOLVED',
    }
    CBC_TIMEOUT_IN_SECONDS = 60

    sample_count, sub_control_count = sub_incidence.shape
    _, parent_control_count = parent_incidence.shape
    sub_zone_count, _ = sub_float_weights.shape

    # setting indexes to -1 prevents creation of hh_controls relaxation variables
    # setting hh_control importance to zero eliminates them from the objective function
    # the latter approach is used by the cvx version
    # total_hh_sub_control_index = -1
    sub_countrol_importance[total_hh_sub_control_index] = 0

    # FIXME total_hh_parent_control_index should not exist???
    if total_hh_parent_control_index > 0:
        parent_countrol_importance[total_hh_parent_control_index] = 0

    # - Instantiate a mixed-integer solver
    solver = pywraplp.Solver('SimulIntegerizeCbc', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
    solver.EnableOutput()
    solver.set_time_limit(CBC_TIMEOUT_IN_SECONDS * 1000)

    # constraints = [
    #     x >= 0.0,
    #     x <= x_max,
    #
    #     relax_le >= 0.0,
    #     relax_le <= lp_right_hand_side,
    #     relax_ge >= 0.0,
    #     relax_ge <= relax_ge_upper_bound,
    #
    #     parent_relax_le >= 0.0,
    #     parent_relax_le <= parent_lp_right_hand_side,
    #     parent_relax_ge >= 0.0,
    #     parent_relax_ge <= parent_relax_ge_upper_bound,
    # ]

    # x_max is 1.0 unless resid_weights is zero, in which case constrain x to 0.0
    x_max = (~(sub_float_weights == sub_int_weights)).astype(float)

    # - Create resid weight variables
    x = {}
    for z in range(sub_zone_count):
        for hh in range(sample_count):
            x[z, hh] = solver.NumVar(0.0, x_max[z, hh], 'x[%s,%s]' % (z, hh))

    # - Create positive continuous constraint relaxation variables
    relax_le = {}
    relax_ge = {}
    for z in range(sub_zone_count):
        for c in range(sub_control_count):
            # no relaxation for total households control
            if c == total_hh_sub_control_index:
                continue
            relax_le[z, c] = \
                solver.NumVar(0.0, lp_right_hand_side[z, c], 'relax_le[%s,%s]' % (z, c))
            relax_ge[z, c] = \
                solver.NumVar(0.0, relax_ge_upper_bound[z, c], 'relax_ge[%s,%s]' % (z, c))

    parent_relax_le = {}
    parent_relax_ge = {}
    for c in range(parent_control_count):
        parent_relax_le[c] = \
            solver.NumVar(0.0, parent_lp_right_hand_side[c], 'parent_relax_le[%s]' % c)
        parent_relax_ge[c] = \
            solver.NumVar(0.0, parent_relax_ge_upper_bound[c], 'parent_relax_ge[%s]' % c)

    LOG_OVERFLOW = -725
    log_resid_weights = np.log(np.maximum(sub_resid_weights, np.exp(LOG_OVERFLOW)))
    assert not np.isnan(log_resid_weights).any()

    log_parent_resid_weights = \
        np.log(np.maximum(parent_resid_weights, np.exp(LOG_OVERFLOW)))
    assert not np.isnan(log_parent_resid_weights).any()

    # objective = cvx.Maximize(
    #     cvx.sum_entries(cvx.mul_elemwise(log_resid_weights, cvx.vec(x))) +
    #     cvx.sum_entries(cvx.mul_elemwise(log_parent_resid_weights, cvx.vec(cvx.sum_entries(x, axis=0)))) -  # nopep8
    #     cvx.sum_entries(relax_le * sub_countrol_importance) -
    #     cvx.sum_entries(relax_ge * sub_countrol_importance) -
    #     cvx.sum_entries(cvx.mul_elemwise(parent_countrol_importance, parent_relax_le)) -
    #     cvx.sum_entries(cvx.mul_elemwise(parent_countrol_importance, parent_relax_ge))
    # )

    z = solver.Sum(x[z, hh] * log_resid_weights[z, hh]
                   for z in range(sub_zone_count)
                   for hh in range(sample_count)) + \
        solver.Sum(x[z, hh] * log_parent_resid_weights[hh]
                   for hh in range(sample_count)
                   for z in range(sub_zone_count)) - \
        solver.Sum(relax_le[z, c] * sub_countrol_importance[c]
                   for z in range(sub_zone_count)
                   for c in range(sub_control_count) if c != total_hh_sub_control_index) - \
        solver.Sum(relax_ge[z, c] * sub_countrol_importance[c]
                   for z in range(sub_zone_count)
                   for c in range(sub_control_count) if c != total_hh_sub_control_index) - \
        solver.Sum(parent_relax_le[c] * parent_countrol_importance[c]
                   for c in range(parent_control_count)) - \
        solver.Sum(parent_relax_ge[c] * parent_countrol_importance[c]
                   for c in range(parent_control_count))

    objective = solver.Maximize(z)

    # constraints = [
    #     # - sub inequality constraints
    #     (x * sub_incidence) - relax_le >= 0,
    #     (x * sub_incidence) - relax_le <= lp_right_hand_side,
    #     (x * sub_incidence) + relax_ge >= lp_right_hand_side,
    #     (x * sub_incidence) + relax_ge <= hh_constraint_ge_bound,
    # ]

    # - sub inequality constraints
    sub_constraint_ge = {}
    sub_constraint_le = {}
    for z in range(sub_zone_count):
        for c in range(sub_control_count):

            # don't add inequality constraints for total households control
            if c == total_hh_sub_control_index:
                continue

            sub_constraint_le[z, c] = \
                solver.Constraint(0, lp_right_hand_side[z, c])
            for hh in range(sample_count):
                sub_constraint_le[z, c].SetCoefficient(x[z, hh], sub_incidence[hh, c])
                sub_constraint_le[z, c].SetCoefficient(relax_le[z, c], -1.0)

            sub_constraint_ge[z, c] = \
                solver.Constraint(lp_right_hand_side[z, c], hh_constraint_ge_bound[z, c])
            for hh in range(sample_count):
                sub_constraint_ge[z, c].SetCoefficient(x[z, hh], sub_incidence[hh, c])
                sub_constraint_ge[z, c].SetCoefficient(relax_ge[z, c], 1.0)

    # constraints = [
    #     # - equality constraint for the total households control
    #     cvx.sum_entries(x, axis=1) == total_hh_right_hand_side,
    # ]

    # - equality constraint for the total households control
    constraint_eq = {}
    for z in range(sub_zone_count):
        total_hh_constraint = total_hh_right_hand_side[z]

        constraint_eq[z] = solver.Constraint(total_hh_constraint, total_hh_constraint)
        for hh in range(sample_count):
            constraint_eq[z].SetCoefficient(x[z, hh], 1.0)

    # constraints = [
    #     cvx.vec(cvx.sum_entries(x, axis=0) * parent_incidence) - parent_relax_le >= 0,                              # nopep8
    #     cvx.vec(cvx.sum_entries(x, axis=0) * parent_incidence) - parent_relax_le <= parent_lp_right_hand_side,      # nopep8
    #     cvx.vec(cvx.sum_entries(x, axis=0) * parent_incidence) + parent_relax_ge >= parent_lp_right_hand_side,      # nopep8
    #     cvx.vec(cvx.sum_entries(x, axis=0) * parent_incidence) + parent_relax_ge <= parent_hh_constraint_ge_bound,  # nopep8
    # ]
    # - sub inequality constraints
    parent_constraint_le = {}
    parent_constraint_ge = {}
    for c in range(parent_control_count):

        if c == total_hh_parent_control_index:
            continue

        parent_constraint_le[c] = \
            solver.Constraint(0, parent_lp_right_hand_side[c])
        parent_constraint_ge[c] = \
            solver.Constraint(parent_lp_right_hand_side[c], parent_hh_constraint_ge_bound[c])

        for z in range(sub_zone_count):
            for hh in range(sample_count):
                parent_constraint_le[c].SetCoefficient(x[z, hh], parent_incidence[hh, c])
                parent_constraint_le[c].SetCoefficient(parent_relax_le[c], -1.0)

                parent_constraint_ge[c].SetCoefficient(x[z, hh], parent_incidence[hh, c])
                parent_constraint_ge[c].SetCoefficient(parent_relax_ge[c], 1.0)

    result_status = solver.Solve()

    status_text = STATUS_TEXT[result_status]

    if status_text in STATUS_SUCCESS:
        resid_weights_out = np.zeros(sub_resid_weights.shape)

        for z in range(sub_zone_count):
            for hh in range(sample_count):
                resid_weights_out[z, hh] = x[z, hh].solution_value()

        resid_weights_out = resid_weights_out.astype(np.float64)
    else:
        resid_weights_out = sub_resid_weights

    return resid_weights_out, status_text
