
# PopulationSim
# See full license in LICENSE.txt.

import logging

import numpy as np
from activitysim.core.config import setting

logger = logging.getLogger(__name__)

STATUS_OPTIMAL = 'OPTIMAL'
STATUS_FEASIBLE = 'FEASIBLE'
STATUS_SUCCESS = [STATUS_OPTIMAL, STATUS_FEASIBLE]

# 'CBC', 'GLPK_MI', 'ECOS_BB'
CVX_SOLVER = 'GLPK_MI'


def np_integerizer_cvx(
        incidence,
        resid_weights,
        log_resid_weights,
        control_importance_weights,
        total_hh_control_index,
        lp_right_hand_side,
        relax_ge_upper_bound,
        hh_constraint_ge_bound):
    """
    cvx-based single-integerizer function taking numpy data types and conforming to a
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

    import cvxpy as cvx

    STATUS_TEXT = {
        cvx.OPTIMAL: STATUS_OPTIMAL,
        cvx.INFEASIBLE: 'INFEASIBLE',
        cvx.UNBOUNDED: 'UNBOUNDED',
        cvx.OPTIMAL_INACCURATE: STATUS_FEASIBLE,
        cvx.INFEASIBLE_INACCURATE: 'INFEASIBLE_INACCURATE',
        cvx.UNBOUNDED_INACCURATE: 'UNBOUNDED_INACCURATE',
        None: 'FAILED'
    }
    CVX_MAX_ITERS = 300

    incidence = incidence.T
    sample_count, control_count = incidence.shape

    # - Decision variables for optimization
    x = cvx.Variable(1, sample_count)

    # - Create positive continuous constraint relaxation variables
    relax_le = cvx.Variable(control_count)
    relax_ge = cvx.Variable(control_count)

    # FIXME - could ignore as handled by constraint?
    control_importance_weights[total_hh_control_index] = 0

    # - Set objective

    objective = cvx.Maximize(
        cvx.sum_entries(cvx.mul_elemwise(log_resid_weights, cvx.vec(x))) -
        cvx.sum_entries(cvx.mul_elemwise(control_importance_weights, relax_le)) -
        cvx.sum_entries(cvx.mul_elemwise(control_importance_weights, relax_ge))
    )

    total_hh_constraint = lp_right_hand_side[total_hh_control_index]

    # 1.0 unless resid_weights is zero
    max_x = (~(resid_weights == 0.0)).astype(float).reshape((1, -1))

    constraints = [
        # - inequality constraints
        cvx.vec(x * incidence) - relax_le >= 0,
        cvx.vec(x * incidence) - relax_le <= lp_right_hand_side,
        cvx.vec(x * incidence) + relax_ge >= lp_right_hand_side,
        cvx.vec(x * incidence) + relax_ge <= hh_constraint_ge_bound,

        x >= 0.0,
        x <= max_x,

        relax_le >= 0.0,
        relax_le <= lp_right_hand_side,

        relax_ge >= 0.0,
        relax_ge <= relax_ge_upper_bound,

        # - equality constraint for the total households control
        cvx.sum_entries(x) == total_hh_constraint,
    ]

    prob = cvx.Problem(objective, constraints)

    assert CVX_SOLVER in cvx.installed_solvers(), \
        "CVX Solver '%s' not in installed solvers %s." % (CVX_SOLVER, cvx.installed_solvers())
    logger.info("integerizing with '%s' solver." % CVX_SOLVER)

    try:
        prob.solve(solver=CVX_SOLVER, verbose=True, max_iters=CVX_MAX_ITERS)
    except cvx.SolverError:
        logging.exception(
            'Solver error encountered in weight discretization. Weights will be rounded.')

    status_text = STATUS_TEXT[prob.status]

    if status_text in STATUS_SUCCESS:
        assert x.value is not None
        resid_weights_out = np.asarray(x.value)[0]
    else:
        assert x.value is None
        resid_weights_out = resid_weights

    return resid_weights_out, status_text


def np_simul_integerizer_cvx(
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
        total_hh_sub_control_index):
    """
    cvx-based siuml-integerizer function taking numpy data types and conforming to a
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

    Returns
    -------
    resid_weights_out : numpy.ndarray of float
        residual weights in range [0..1] as solved,
        or, in case of failure, sub_resid_weights unchanged
    status_text : string
        STATUS_OPTIMAL, STATUS_FEASIBLE in case of success, or a solver-specific failure status
    """

    import cvxpy as cvx

    STATUS_TEXT = {
        cvx.OPTIMAL: 'OPTIMAL',
        cvx.INFEASIBLE: 'INFEASIBLE',
        cvx.UNBOUNDED: 'UNBOUNDED',
        cvx.OPTIMAL_INACCURATE: 'FEASIBLE',  # for compatability with ortools
        cvx.INFEASIBLE_INACCURATE: 'INFEASIBLE_INACCURATE',
        cvx.UNBOUNDED_INACCURATE: 'UNBOUNDED_INACCURATE',
        None: 'FAILED'
    }
    CVX_MAX_ITERS = 1000

    sample_count, sub_control_count = sub_incidence.shape
    _, parent_control_count = parent_incidence.shape
    sub_zone_count, _ = sub_float_weights.shape

    # - Decision variables for optimization
    x = cvx.Variable(sub_zone_count, sample_count)

    # x range is 0.0 to 1.0 unless resid_weights is zero, in which case constrain x to 0.0
    x_max = (~(sub_float_weights == sub_int_weights)).astype(float)

    # - Create positive continuous constraint relaxation variables
    relax_le = cvx.Variable(sub_zone_count, sub_control_count)
    relax_ge = cvx.Variable(sub_zone_count, sub_control_count)

    parent_relax_le = cvx.Variable(parent_control_count)
    parent_relax_ge = cvx.Variable(parent_control_count)

    # - Set objective

    # can probably ignore as handled by constraint
    sub_countrol_importance[total_hh_sub_control_index] = 0

    LOG_OVERFLOW = -725
    log_resid_weights = np.log(np.maximum(sub_resid_weights, np.exp(LOG_OVERFLOW))).flatten('F')
    assert not np.isnan(log_resid_weights).any()

    log_parent_resid_weights = \
        np.log(np.maximum(parent_resid_weights, np.exp(LOG_OVERFLOW))).flatten('F')
    assert not np.isnan(log_parent_resid_weights).any()

    # subzone and parent objective and relaxation penalties
    # note: cvxpy overloads * so * in following is matrix multiplication
    objective = cvx.Maximize(
        cvx.sum_entries(cvx.mul_elemwise(log_resid_weights, cvx.vec(x))) +
        cvx.sum_entries(cvx.mul_elemwise(log_parent_resid_weights, cvx.vec(cvx.sum_entries(x, axis=0)))) -  # nopep8
        cvx.sum_entries(relax_le * sub_countrol_importance) -
        cvx.sum_entries(relax_ge * sub_countrol_importance) -
        cvx.sum_entries(cvx.mul_elemwise(parent_countrol_importance, parent_relax_le)) -
        cvx.sum_entries(cvx.mul_elemwise(parent_countrol_importance, parent_relax_ge))
    )

    constraints = [
        (x * sub_incidence) - relax_le >= 0,
        (x * sub_incidence) - relax_le <= lp_right_hand_side,
        (x * sub_incidence) + relax_ge >= lp_right_hand_side,
        (x * sub_incidence) + relax_ge <= hh_constraint_ge_bound,

        x >= 0.0,
        x <= x_max,

        relax_le >= 0.0,
        relax_le <= lp_right_hand_side,

        relax_ge >= 0.0,
        relax_ge <= relax_ge_upper_bound,

        # - equality constraint for the total households control
        cvx.sum_entries(x, axis=1) == total_hh_right_hand_side,

        cvx.vec(cvx.sum_entries(x, axis=0) * parent_incidence) - parent_relax_le >= 0,                              # nopep8
        cvx.vec(cvx.sum_entries(x, axis=0) * parent_incidence) - parent_relax_le <= parent_lp_right_hand_side,      # nopep8
        cvx.vec(cvx.sum_entries(x, axis=0) * parent_incidence) + parent_relax_ge >= parent_lp_right_hand_side,      # nopep8
        cvx.vec(cvx.sum_entries(x, axis=0) * parent_incidence) + parent_relax_ge <= parent_hh_constraint_ge_bound,  # nopep8

        parent_relax_le >= 0.0,
        parent_relax_le <= parent_lp_right_hand_side,

        parent_relax_ge >= 0.0,
        parent_relax_ge <= parent_relax_ge_upper_bound,
    ]

    prob = cvx.Problem(objective, constraints)

    assert CVX_SOLVER in cvx.installed_solvers(), \
        "CVX Solver '%s' not in installed solvers %s." % (
        CVX_SOLVER, cvx.installed_solvers())
    logger.info("simul_integerizing with '%s' solver." % CVX_SOLVER)

    try:
        prob.solve(solver=CVX_SOLVER, verbose=True, max_iters=CVX_MAX_ITERS)
    except cvx.SolverError as e:
        logging.warning('Solver error in SimulIntegerizer: %s' % e)

    # if we got a result
    if np.any(x.value):
        resid_weights_out = np.asarray(x.value)
    else:
        resid_weights_out = sub_resid_weights

    status_text = STATUS_TEXT[prob.status]

    return resid_weights_out, status_text
