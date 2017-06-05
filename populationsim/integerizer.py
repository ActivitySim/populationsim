# PopulationSim
# See full license in LICENSE.txt.

import logging

import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp
import cvxpy as cvx


STATUS_SUCCESS = ['OPTIMAL', 'FEASIBLE', 'OPTIMAL_INACCURATE']

CVX_STATUS = {
    cvx.OPTIMAL: 'OPTIMAL',
    cvx.INFEASIBLE: 'INFEASIBLE',
    cvx.UNBOUNDED: 'UNBOUNDED',
    cvx.OPTIMAL_INACCURATE: 'OPTIMAL_INACCURATE',
    cvx.INFEASIBLE_INACCURATE: 'INFEASIBLE_INACCURATE',
    cvx.UNBOUNDED_INACCURATE: 'UNBOUNDED_INACCURATE'

}

CBC_STATUS = {
    pywraplp.Solver.OPTIMAL: 'OPTIMAL',
    pywraplp.Solver.FEASIBLE: 'FEASIBLE',
    pywraplp.Solver.INFEASIBLE: 'INFEASIBLE',
    pywraplp.Solver.UNBOUNDED: 'UNBOUNDED',
    pywraplp.Solver.ABNORMAL: 'ABNORMAL',
    pywraplp.Solver.NOT_SOLVED: 'NOT_SOLVED',
}

logger = logging.getLogger(__name__)


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

    def integerize(self):

        sample_count = len(self.incidence_table.index)
        control_count = len(self.incidence_table.columns)

        incidence = self.incidence_table.as_matrix().transpose()
        float_weights = np.asanyarray(self.float_weights).astype(np.float64)
        control_totals = np.asanyarray(self.control_totals).astype(np.int)
        relaxed_control_totals = np.asanyarray(self.relaxed_control_totals).astype(np.float64)
        control_is_hh_based = np.asanyarray(self.control_is_hh_based).astype(np.int)
        control_importance_weights = np.asanyarray(self.control_importance_weights).astype(np.float64)

        assert len(float_weights) == sample_count
        assert len(control_totals) == control_count
        assert len(relaxed_control_totals) == control_count
        assert len(control_is_hh_based) == control_count
        assert len(self.incidence_table.columns) == control_count


        integerized_weights, status \
            = np_integerizer_cvx(
            sample_count=sample_count,
            control_count=control_count,
            incidence=incidence,
            float_weights=float_weights,
            control_importance_weights=control_importance_weights,
            control_totals=control_totals,
            relaxed_control_totals=relaxed_control_totals,
            total_hh_control_index=self.total_hh_control_index,
            control_is_hh_based=control_is_hh_based,
            timeout_in_seconds=self.timeout_in_seconds
        )

        self.weights = pd.DataFrame(index=self.incidence_table.index)
        self.weights['integerized_weight'] = integerized_weights

        return status


def np_integerizer_cvx(sample_count,
                       control_count,
                       incidence,
                       float_weights,
                       control_importance_weights,
                       control_totals,
                       relaxed_control_totals,
                       total_hh_control_index,
                       control_is_hh_based,
                       timeout_in_seconds):

    assert not np.isnan(incidence).any()
    assert not np.isnan(float_weights).any()
    assert not (float_weights == 0).any()

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
    num_households = relaxed_control_totals[total_hh_control_index]
    relax_ge_upper_bound = np.maximum(max_incidence_value * num_households - lp_right_hand_side, 0)

    # - Decision variables for optimization
    x = cvx.Variable(1, sample_count)

    # - Create positive continuous constraint relaxation variables
    # no relaxation for total households control
    relax_le = cvx.Variable(control_count)
    relax_ge = cvx.Variable(control_count)

    # FIXME - ignore as handled by constraint?
    #control_importance_weights[total_hh_control_index] = 0

    # - Set objective
    log_resid_weights = np.log(resid_weights)
    log_resid_weights[(resid_weights <= np.exp( -999 ))] = -999
    assert not np.isnan(log_resid_weights).any()

    # objective = cvx.Maximize(
    #     cvx.sum_entries(
    #         log_resid_weights * x -
    #         (control_importance_weights) * relax_le -
    #         (control_importance_weights) * relax_ge
    #     )
    # )

    # objective = cvx.Maximize(
    #     cvx.sum_entries(
    #         cvx.sum_entries(cvx.mul_elemwise(log_resid_weights, x), axis=1) -
    #         (control_importance_weights) * cvx.sum_entries(relax_le, axis=1) -
    #         (control_importance_weights) * cvx.sum_entries(relax_ge, axis=1)
    #     )
    # )

    objective = cvx.Maximize(
        cvx.sum_entries(cvx.mul_elemwise(log_resid_weights, cvx.vec(x))) -
        #cvx.sum_entries(cvx.mul_elemwise(control_importance_weights, relax_le)) -
        #cvx.sum_entries(cvx.mul_elemwise(control_importance_weights, relax_ge))
        (100.) * cvx.sum_entries(relax_le) -
        (100.) * cvx.sum_entries(relax_ge)
    )

    constraints = [
        # inequaity constraints
        cvx.vec(x*incidence) <= resid_control_totals + relax_ge,
        cvx.vec(x*incidence) >= resid_control_totals - relax_le,
        x >= 0,
        x <= 1.0,
        relax_le >= 0,
        #relax_le <= lp_right_hand_side,
        relax_ge >= 0,
        #relax_ge <= relax_ge_upper_bound,
        #x*incidence[total_hh_control_index] == resid_control_totals[total_hh_control_index]
    ]

    prob = cvx.Problem(objective, constraints)

    try:
        prob.solve(verbose=True)

    except cvx.SolverError:
        logging.exception(
            'Solver error encountered in weight discretization. Weights will be rounded.')

    if np.any(x.value):
        resid_weights_out = np.asarray(x.value)[0]
    else:
        resid_weights_out = resid_weights

    # Make results binary
    resid_weights_out = np.array(resid_weights_out > 0.5).astype(int)
    weights_out = int_weights + resid_weights_out

    return weights_out, CVX_STATUS[prob.status]

def np_integerizer_cbc(sample_count,
                       control_count,
                       incidence,
                       float_weights,
                       control_importance_weights,
                       control_totals,
                       relaxed_control_totals,
                       total_hh_control_index,
                       control_is_hh_based,
                       timeout_in_seconds):

    # - Instantiate a mixed-integer solver
    solver = pywraplp.Solver('IntegerizeCbc', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
    # Create binary integer variables
    x = [[]] * sample_count
    for hh in range(0, sample_count):
        # if final_weights is an int
        if float_weights[hh] == int(float_weights[hh]):
            # FIXME - or should we slice them out beforehand?
            x[hh] = solver.NumVar(0.0, 0.0, 'x_' + str(hh))
        else:
            x[hh] = solver.NumVar(0.0, 1.0, 'x_' + str(hh))

    lp_right_hand_side = [0] * control_count

    # lp_right_hand_side
    relaxed_control_totals = np.round(relaxed_control_totals)
    for c in range(0, control_count):
        weighted_incidence = (np.trunc(float_weights) * incidence[c]).sum()
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
            relax_ge_upper_bound[c] = max(max_incidence_value[c] * num_households - lp_right_hand_side[c], 0)

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
    resid_weights = float_weights % 1.0
    # use negative for coefficients since solver is minimizing
    PENALTY = 999
    objective_function_coefficients = -1.0 * np.log(resid_weights)
    objective_function_coefficients[(resid_weights <= np.exp( -PENALTY ))] = PENALTY

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
            hh_constraint_le[c].SetCoefficient(relax_ge[c], -1.0)
        #logger.debug("Set hh_constraint_le to %s, %s" % (0, lp_right_hand_side[c]))
        # add the upper bound relaxation inequality constraint
        hh_constraint_ge[c] = solver.Constraint(lp_right_hand_side[c], hh_constraint_ge_bound[c])
        for hh in range(0, sample_count):
            hh_constraint_ge[c].SetCoefficient(x[hh], incidence[c, hh])
            hh_constraint_ge[c].SetCoefficient(relax_le[c], 1.0)
        #logger.debug("Set hh_constraint_ge to %s, %s" % (lp_right_hand_side[c], hh_constraint_ge_bound[c]))

    # - add an equality constraint for the total households control
    total_hh_constraint = lp_right_hand_side[total_hh_control_index]
    constraint_eq = solver.Constraint(total_hh_constraint, total_hh_constraint)
    for hh in range(0, sample_count):
        constraint_eq.SetCoefficient(x[hh], incidence[total_hh_control_index, hh])
    logger.debug("Set total_hh_control constraint to %s" % total_hh_constraint)

    solver.set_time_limit(timeout_in_seconds * 1000)

    result_status = solver.Solve()

    continuous_solution = np.asanyarray(map(lambda x: x.solution_value(), x)).astype(np.float64)
    lp_solution = np.round(continuous_solution)

    integerized_weights = np.trunc(float_weights) + lp_solution

    # print "final_weights", final_weights
    # print "continuous_solution", continuous_solution
    # print "lp_solution        ", lp_solution
    # print "integerized_weights", integerized_weights
    # print "simple rounding    ", np.round(final_weights)

    logger.debug("Solver result_status = %s" % result_status)
    logger.debug("Optimal objective value = %s" % solver.Objective().Value())
    logger.debug('Number of variables = %s' % solver.NumVariables())
    logger.debug('Number of constraints = %s' % solver.NumConstraints())

    # for variable in x:
    #     print('%s = %d' % (variable.name(), variable.solution_value()))
    #
    # for variable in y:
    #     if variable:
    #         print('%s = %d' % (variable.name(), variable.solution_value()))
    #
    # for variable in z:
    #     if variable:
    #         print('%s = %d' % (variable.name(), variable.solution_value()))

    return integerized_weights, CBC_STATUS[result_status]


def do_integerizing(
        label,
        id,
        control_spec,
        control_totals,
        incidence_table,
        float_weights,
        total_hh_control_col):
    """

    Parameters
    ----------
    label
    id
    control_spec
    control_totals
    incidence_table
    balanced_weights
    total_hh_control_col

    Returns
    -------

    """

    TRY_BACKSTOPPED_CONTROLS = True
    if TRY_BACKSTOPPED_CONTROLS:
        imputed_control_totals = np.round(np.dot(np.asanyarray(float_weights), incidence_table.as_matrix()))
        relaxed_control_totals = pd.Series(imputed_control_totals, index=incidence_table.columns.values)

        backstopped_control_totals = pd.Series(imputed_control_totals, index=incidence_table.columns.values)
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

        if status in STATUS_SUCCESS:
            return integerizer.weights['integerized_weight'], status

        logger.warn("Integerizer did not find feasible solution for backstopped %s %s: %s"
                    % (label, id, status))

    balanced_control_cols = control_totals.index
    incidence_table = incidence_table[balanced_control_cols]
    control_spec = control_spec[ control_spec.target.isin(balanced_control_cols) ]

    imputed_control_totals = np.round(np.dot(np.asanyarray(float_weights), incidence_table.as_matrix()))
    relaxed_control_totals = pd.Series(imputed_control_totals, index=incidence_table.columns.values)

    integerizer = Integerizer(
        control_totals=control_totals,
        incidence_table=incidence_table,
        control_importance_weights=control_spec.importance,
        float_weights=float_weights,
        relaxed_control_totals=relaxed_control_totals,
        total_hh_control_index=incidence_table.columns.get_loc(total_hh_control_col),
        control_is_hh_based=control_spec['seed_table'] == 'households'
        )

    # otherwise, solve for the integer weights using the Mixed Integer Programming solver.
    status = integerizer.integerize()

    if status not in STATUS_SUCCESS:
        logger.warn("Integerizer did not find optimal solution for %s %s: %s"
                    % (label, id, status))

    return integerizer.weights['integerized_weight'], status
