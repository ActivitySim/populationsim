# PopulationSim
# See full license in LICENSE.txt.

import logging

import numpy as np
import pandas as pd
from util import setting

USE_CVX = setting('USE_CVX')
INTEGERIZE_WITH_BACKSTOPPED_CONTROLS = setting('INTEGERIZE_WITH_BACKSTOPPED_CONTROLS')


if USE_CVX:
    import cylp
    import cvxpy as cvx

    CVX_STATUS = {
        cvx.OPTIMAL: 'OPTIMAL',
        cvx.INFEASIBLE: 'INFEASIBLE',
        cvx.UNBOUNDED: 'UNBOUNDED',
        cvx.OPTIMAL_INACCURATE: 'OPTIMAL_INACCURATE',
        cvx.INFEASIBLE_INACCURATE: 'INFEASIBLE_INACCURATE',
        cvx.UNBOUNDED_INACCURATE: 'UNBOUNDED_INACCURATE',
        None: 'FAILED'

    }

else:
    from ortools.linear_solver import pywraplp

    CBC_STATUS = {
        pywraplp.Solver.OPTIMAL: 'OPTIMAL',
        pywraplp.Solver.FEASIBLE: 'FEASIBLE',
        pywraplp.Solver.INFEASIBLE: 'INFEASIBLE',
        pywraplp.Solver.UNBOUNDED: 'UNBOUNDED',
        pywraplp.Solver.ABNORMAL: 'ABNORMAL',
        pywraplp.Solver.NOT_SOLVED: 'NOT_SOLVED',
    }

STATUS_SUCCESS = ['OPTIMAL', 'FEASIBLE', 'OPTIMAL_INACCURATE']

logger = logging.getLogger(__name__)


def smart_round(int_weights, resid_weights, total_household_control):
    assert len(int_weights) == len(resid_weights)

    int_shortfall = total_household_control - int_weights.sum()
    int_shortfall = np.clip(int_shortfall, 0, len(resid_weights))

    # indices of the int_shortfall highest resid_weights
    i = np.argsort(resid_weights)[-int_shortfall:]

    rounded_weights = np.copy(int_weights)
    rounded_weights[i] += 1

    return rounded_weights


class Integerizer(object):
    def __init__(self,
                 control_totals,
                 incidence_table,
                 control_importance_weights,
                 float_weights,
                 relaxed_control_totals,
                 total_hh_control_index,
                 control_is_hh_based):

        self.control_totals = control_totals
        self.incidence_table = incidence_table
        self.control_importance_weights = control_importance_weights
        self.float_weights = float_weights
        self.relaxed_control_totals = relaxed_control_totals
        self.total_hh_control_index = total_hh_control_index
        self.control_is_hh_based = control_is_hh_based
        self.timeout_in_seconds = 60

    def integerize(self, try_harder=False):

        sample_count = len(self.incidence_table.index)
        control_count = len(self.incidence_table.columns)

        incidence = self.incidence_table.as_matrix().transpose().astype(np.float64)
        float_weights = np.asanyarray(self.float_weights).astype(np.float64)
        control_totals = np.asanyarray(self.control_totals).astype(np.int)
        relaxed_control_totals = np.asanyarray(self.relaxed_control_totals).astype(np.float64)
        control_is_hh_based = np.asanyarray(self.control_is_hh_based).astype(np.int)
        control_importance_weights = np.asanyarray(self.control_importance_weights).astype(
            np.float64)

        assert len(float_weights) == sample_count
        assert len(control_totals) == control_count
        assert len(relaxed_control_totals) == control_count
        assert len(control_is_hh_based) == control_count
        assert len(self.incidence_table.columns) == control_count


        if USE_CVX:
            int_weights, resid_weights, status = np_integerizer_cvx(
                incidence=incidence,
                float_weights=float_weights,
                control_importance_weights=control_importance_weights,
                control_totals=control_totals,
                relaxed_control_totals=relaxed_control_totals,
                total_hh_control_index=self.total_hh_control_index,
                control_is_hh_based=control_is_hh_based,
                try_harder=try_harder
            )
        else:
            int_weights, resid_weights, status \
                = np_integerizer_cbc(
                sample_count=sample_count,
                control_count=control_count,
                incidence=incidence,
                float_weights=float_weights,
                control_importance_weights=control_importance_weights,
                control_totals=control_totals,
                relaxed_control_totals=relaxed_control_totals,
                total_hh_control_index=self.total_hh_control_index,
                control_is_hh_based=control_is_hh_based,
                timeout_in_seconds=self.timeout_in_seconds,
                try_harder=try_harder
            )

        total_household_control = control_totals[self.total_hh_control_index]

        integerized_weights = smart_round(int_weights, resid_weights, total_household_control)

        self.weights = pd.DataFrame(index=self.incidence_table.index)
        self.weights['integerized_weight'] = integerized_weights

        delta = (integerized_weights != np.round(float_weights)).sum()
        logger.debug("Integerizer: %s out of %s different from round" % (delta, len(float_weights)))


        return status


def np_integerizer_cvx(incidence,
                       float_weights,
                       control_importance_weights,
                       control_totals,
                       relaxed_control_totals,
                       total_hh_control_index,
                       control_is_hh_based,
                       try_harder):
    assert not np.isnan(incidence).any()
    assert not np.isnan(float_weights).any()

    if (float_weights == 0).any():
        # not sure this matters...
        logger.warn("np_integerizer_cvx: %s zero weights" % ((float_weights == 0).sum(),))

    incidence = incidence.T
    # float_weights = np.matrix(float_weights)

    sample_count, control_count = incidence.shape

    int_weights = float_weights.astype(int)
    resid_weights = float_weights % 1.0

    # resid_control_totals - control totals of resid_weights
    resid_control_totals = np.dot(resid_weights, incidence)

    # - lp_right_hand_side - relaxed_control_shortfall
    lp_right_hand_side = np.round(relaxed_control_totals) - np.dot(int_weights, incidence)
    lp_right_hand_side = np.maximum(lp_right_hand_side, 0.0)

    # - create the inequality constraint upper bounds
    max_incidence_value = np.amax(incidence, axis=0)
    assert (max_incidence_value[control_is_hh_based] == 1).all()
    num_households = relaxed_control_totals[total_hh_control_index]
    relax_ge_upper_bound = np.maximum(max_incidence_value * num_households - lp_right_hand_side, 0)

    # - Decision variables for optimization
    x = cvx.Variable(1, sample_count)

    # - Create positive continuous constraint relaxation variables
    relax_le = cvx.Variable(control_count)
    relax_ge = cvx.Variable(control_count)

    # FIXME - ignore as handled by constraint?
    # control_importance_weights[total_hh_control_index] = 0

    # - Set objective

    LOG_OVERFLOW = -725
    log_resid_weights = np.log(np.maximum(resid_weights, np.exp(LOG_OVERFLOW)))
    assert not np.isnan(log_resid_weights).any()

    # control_importance_weights = [999]*control_count
    # control_importance_weights[total_hh_control_index] = 2000
    objective = cvx.Maximize(
        cvx.sum_entries(cvx.mul_elemwise(log_resid_weights, cvx.vec(x))) -
        cvx.sum_entries(cvx.mul_elemwise(control_importance_weights, relax_le)) -
        cvx.sum_entries(cvx.mul_elemwise(control_importance_weights, relax_ge))
    )

    total_hh_right_hand_side = lp_right_hand_side[total_hh_control_index]

    hh_constraint_ge_bound = np.maximum(control_totals * max_incidence_value, lp_right_hand_side)

    constraints = [
        # any reason we can't try to match controls?
        # cvx.vec(x*incidence) <= resid_control_totals + relax_ge,
        # cvx.vec(x*incidence) >= resid_control_totals - relax_le,
        # this is what popsyn3 does - try to match rounded relaxed controls
        cvx.vec(x * incidence) - relax_le >= 0,
        cvx.vec(x * incidence) - relax_le <= lp_right_hand_side,
        cvx.vec(x * incidence) + relax_ge >= lp_right_hand_side,
        cvx.vec(x * incidence) + relax_ge <= hh_constraint_ge_bound,
        ###
        #
        x >= 0.0,
        x <= 1.0,
        # y
        relax_le >= 0.0,
        relax_le <= lp_right_hand_side,
        # z
        relax_ge >= 0.0,
        relax_ge <= relax_ge_upper_bound,
        cvx.sum_entries(x) >= total_hh_right_hand_side,
        cvx.sum_entries(x) <= total_hh_right_hand_side
    ]

    prob = cvx.Problem(objective, constraints)

    try:
        # - solver list: http://www.cvxpy.org/en/latest/tutorial/advanced/
        # cvx.installed_solvers(): ['ECOS_BB', 'SCS', 'ECOS', 'LS']
        # ['CBC', 'CVXOPT', 'ECOS_BB', 'GLPK_MI', 'SCS', 'ECOS', 'GLPK', 'LS']
        # prob.solve(solver=cvx.ECOS, verbose=True)

        prob.solve(solver=cvx.CBC, max_iters=10, verbose=try_harder)

    except cvx.SolverError:
        logging.exception(
            'Solver error encountered in weight discretization. Weights will be rounded.')

    if np.any(x.value):
        resid_weights_out = np.asarray(x.value)[0]
    else:
        resid_weights_out = resid_weights

    return int_weights, resid_weights_out, CVX_STATUS[prob.status]


def np_integerizer_cbc(sample_count,
                       control_count,
                       incidence,
                       float_weights,
                       control_importance_weights,
                       control_totals,
                       relaxed_control_totals,
                       total_hh_control_index,
                       control_is_hh_based,
                       timeout_in_seconds,
                       try_harder):

    # - Instantiate a mixed-integer solver
    solver = pywraplp.Solver('IntegerizeCbc', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

    int_weights = np.trunc(float_weights)
    resid_weights = float_weights % 1.0

    # Create binary integer variables
    x = [[]] * sample_count
    for hh in range(0, sample_count):
        # if final_weights is an int
        if float_weights[hh] == int_weights[hh]:
            # FIXME - or should we slice them out beforehand?
            x[hh] = solver.NumVar(0.0, 0.0, 'x_' + str(hh))
        else:
            x[hh] = solver.NumVar(0.0, 1.0, 'x_' + str(hh))

    lp_right_hand_side = [0] * control_count

    # lp_right_hand_side
    relaxed_control_totals = np.round(relaxed_control_totals)
    for c in range(0, control_count):
        weighted_incidence = (int_weights * incidence[c]).sum()
        lp_right_hand_side[c] = relaxed_control_totals[c] - weighted_incidence
    lp_right_hand_side = np.maximum(lp_right_hand_side, 0.0)

    # max_incidence_value of each control
    max_incidence_value = np.amax(incidence, axis=1)

    # - create the inequality constraint upper bounds
    num_households = relaxed_control_totals[total_hh_control_index]
    relax_ge_upper_bound = [0] * control_count
    for c in range(0, control_count):
        if control_is_hh_based[c]:
            # for household controls only
            relax_ge_upper_bound[c] = max(num_households - lp_right_hand_side[c], 0)
        else:
            # for person controls only
            relax_ge_upper_bound[c] = max(
                max_incidence_value[c] * num_households - lp_right_hand_side[c], 0)

    # - Create positive continuous constraint relaxation variables
    relax_le = [[]] * control_count
    relax_ge = [[]] * control_count
    for c in range(0, control_count):
        # no relaxation for total households control
        if c != total_hh_control_index:
            relax_le[c] = solver.NumVar(0.0, lp_right_hand_side[c], 'relax_le_' + str(c))
            relax_ge[c] = solver.NumVar(0.0, relax_ge_upper_bound[c], 'relax_ge_' + str(c))

    # - Set objective: min sum{c(n)*x(n)} + 999*y(i) - 999*z(i)}
    objective = solver.Objective()
    # use negative for coefficients since solver is minimizing
    # avoid overflow
    PENALTY = 700
    objective_function_coefficients = -1.0 * np.log(resid_weights)
    objective_function_coefficients[(resid_weights <= np.exp(-PENALTY))] = PENALTY

    for hh in range(0, sample_count):
        objective.SetCoefficient(x[hh], objective_function_coefficients[hh])

    for c in range(0, control_count):
        if c != total_hh_control_index:
            objective.SetCoefficient(relax_le[c], control_importance_weights[c])
            objective.SetCoefficient(relax_ge[c], control_importance_weights[c])

    # - inequality constraints
    hh_constraint_ge = [[]] * control_count
    hh_constraint_le = [[]] * control_count
    hh_constraint_ge_bound = np.maximum(control_totals * max_incidence_value, lp_right_hand_side)
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

    # - add an equality constraint for the total households control
    total_hh_constraint = lp_right_hand_side[total_hh_control_index]
    constraint_eq = solver.Constraint(total_hh_constraint, total_hh_constraint)
    for hh in range(0, sample_count):
        constraint_eq.SetCoefficient(x[hh], incidence[total_hh_control_index, hh])

    solver.set_time_limit(timeout_in_seconds * 1000)

    solver.EnableOutput()
    result_status = solver.Solve()

    resid_weights_out = np.asanyarray(map(lambda x: x.solution_value(), x)).astype(np.float64)

    int_weights = int_weights.astype(int)

    return int_weights, resid_weights_out, CBC_STATUS[result_status]


def do_integerizing(
        label,
        id,
        control_spec,
        control_totals,
        incidence_table,
        float_weights,
        total_hh_control_col):


    zero_weight_rows = float_weights == 0
    if zero_weight_rows.any():
        logger.debug("omitting %s zero weight rows out of %s" % (
        zero_weight_rows.sum(), len(incidence_table.index)))
        incidence_table = incidence_table[~zero_weight_rows]
        float_weights = float_weights[~zero_weight_rows]

    status = None
    if INTEGERIZE_WITH_BACKSTOPPED_CONTROLS and len(control_totals) < len(incidence_table.columns):

        ##########################################
        # - backstopped control_totals
        ##########################################

        relaxed_control_totals = np.round(np.dot(np.asanyarray(float_weights), incidence_table.as_matrix()))
        relaxed_control_totals = pd.Series(relaxed_control_totals, index=incidence_table.columns.values)

        backstopped_control_totals = relaxed_control_totals.copy()
        backstopped_control_totals.update(control_totals)

        # print "control_totals\n", control_totals
        # print "relaxed_control_totals\n", relaxed_control_totals
        # print "backstopped_control_totals\n", backstopped_control_totals

        # master_control_index is column index in incidence table of total_hh_control_col
        if total_hh_control_col not in incidence_table.columns:
            raise RuntimeError("total_hh_control column '%s' not found in incidence table"
                               % total_hh_control_col)

        # if the incidence table has only one record, then the final integer weights
        # should be just an array with 1 element equal to the total number of households;
        assert len(incidence_table.index) > 1

        integerizer = Integerizer(
            control_totals=backstopped_control_totals,
            incidence_table=incidence_table,
            control_importance_weights=control_spec.importance,
            float_weights=float_weights,
            relaxed_control_totals=relaxed_control_totals,
            total_hh_control_index=incidence_table.columns.get_loc(total_hh_control_col),
            control_is_hh_based=control_spec['seed_table'] == 'households'
        )

        # otherwise, solve for the integer weights using the Mixed Integer Programming solver.
        status = integerizer.integerize()

        logger.debug("Integerizer status for backdropped %s %s: %s" % (label, id, status))

    # if we either tried backstopped controls or failed, or never tried at all
    if status not in STATUS_SUCCESS:

        ##########################################
        # - unbackstopped partial control_totals
        ##########################################

        balanced_control_cols = control_totals.index
        incidence_table = incidence_table[balanced_control_cols]
        control_spec = control_spec[control_spec.target.isin(balanced_control_cols)]

        relaxed_control_totals = np.round(np.dot(np.asanyarray(float_weights), incidence_table.as_matrix()))
        relaxed_control_totals = pd.Series(relaxed_control_totals, index=incidence_table.columns.values)

        integerizer = Integerizer(
            control_totals=control_totals,
            incidence_table=incidence_table,
            control_importance_weights=control_spec.importance,
            float_weights=float_weights,
            relaxed_control_totals=relaxed_control_totals,
            total_hh_control_index=incidence_table.columns.get_loc(total_hh_control_col),
            control_is_hh_based=control_spec['seed_table'] == 'households'
        )

        status = integerizer.integerize()

        logger.debug("Integerizer status for unbackstopped control_totals %s %s: %s" % (label, id, status))


    if status not in STATUS_SUCCESS:
        logger.error("Integerizer failed for %s %s. Returning smart-rounded original weights" % (label, id,))

    integerized_weights = pd.Series(0.0, index=zero_weight_rows.index)
    integerized_weights.update(integerizer.weights['integerized_weight'])
    return integerized_weights, status
