# PopulationSim
# See full license in LICENSE.txt.

import logging

import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp

SOLVER_STATUS_STRINGS = {
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
                 final_weights,
                 relaxation_factors,
                 total_hh_control_index,
                 control_is_hh_based):

        self.control_totals = control_totals
        self.incidence_table = incidence_table
        self.control_importance_weights = control_importance_weights
        self.final_weights = final_weights
        self.relaxation_factors = relaxation_factors
        self.total_hh_control_index = total_hh_control_index
        self.control_is_hh_based = control_is_hh_based
        self.timeout_in_seconds = 60

    def integerize(self):

        sample_count = len(self.incidence_table.index)
        control_count = len(self.incidence_table.columns)

        incidence = self.incidence_table.as_matrix().transpose()
        final_weights = np.asanyarray(self.final_weights).astype(np.float64)
        control_totals = np.asanyarray(self.control_totals).astype(np.int)
        relaxation_factors = np.asanyarray(self.relaxation_factors).astype(np.float64)
        control_is_hh_based = np.asanyarray(self.control_is_hh_based).astype(np.int)
        control_importance_weights = np.asanyarray(self.control_importance_weights).astype(np.float64)

        assert len(final_weights) == sample_count
        assert len(control_totals) == control_count
        assert len(relaxation_factors) == control_count
        assert len(control_is_hh_based) == control_count
        assert len(self.incidence_table.columns) == control_count
        # print self.incidence_table
        # print incidence

        integerized_weights, status \
            = np_integerizer_cbc(
            sample_count=sample_count,
            control_count=control_count,
            incidence=incidence,
            final_weights=final_weights,
            control_importance_weights=control_importance_weights,
            control_totals=control_totals,
            relaxation_factors=relaxation_factors,
            total_hh_control_index=self.total_hh_control_index,
            control_is_hh_based=control_is_hh_based,
            timeout_in_seconds=self.timeout_in_seconds
        )

        self.weights = pd.DataFrame(index=self.incidence_table.index)
        self.weights['integerized_weight'] = integerized_weights

        return status



def np_integerizer_cbc(sample_count,
                       control_count,
                       incidence,
                       final_weights,
                       control_importance_weights,
                       control_totals,
                       relaxation_factors,
                       total_hh_control_index,
                       control_is_hh_based,
                       timeout_in_seconds):

    # Instantiate a mixed-integer solver
    solver = pywraplp.Solver('IntegerizeCbc', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
    #solver = pywraplp.Solver('IntegerizeGlop', pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)

    # Create binary integer variables
    x = [[]] * sample_count
    for hh in range(0, sample_count):
        # if final_weights is an int
        if final_weights[hh] == int(final_weights[hh]):
            # FIXME - or should we slice them out beforehand?
            x[hh] = solver.NumVar(0.0, 0.0, 'x_' + str(hh))
        else:
            x[hh] = solver.NumVar(0.0, 1.0, 'x_' + str(hh))

    lp_right_hand_side = [0] * control_count


    # lp_right_hand_side
    relaxed_control_total = np.round(control_totals * relaxation_factors)
    # FIXME - any reason we can't use unrelaxed control total?
    relaxed_control_total = control_totals

    for c in range(0, control_count):
        weighted_incidence = (np.trunc(final_weights) * incidence[c]).sum()
        lp_right_hand_side[c] = relaxed_control_total[c] - weighted_incidence
    lp_right_hand_side = np.maximum(lp_right_hand_side, 0.0)

    # max_incidence_value of each control
    max_incidence_value = np.amax(incidence, axis=1)

    # create the inequality constraint upper bounds
    num_households = round(control_totals[total_hh_control_index] * relaxation_factors[total_hh_control_index])
    relax_ge_upper_bound = [0] * control_count
    for c in range(0, control_count):
        if control_is_hh_based[c]:
            # for household controls only
            relax_ge_upper_bound[c] = max(num_households - lp_right_hand_side[c], 0)
        else:
            # for person controls only
            relax_ge_upper_bound[c] = max(max_incidence_value[c] * num_households - lp_right_hand_side[c], 0)

    # Create positive continuous constraint relaxation variables
    relax_le = [[]] * control_count
    relax_ge = [[]] * control_count
    for c in range(0, control_count):
        # no relaxation for total households control
        if c != total_hh_control_index:
            relax_le[c] = solver.NumVar(0.0, lp_right_hand_side[c], 'relax_le_' + str(c))
            relax_ge[c] = solver.NumVar(0.0, relax_ge_upper_bound[c], 'relax_ge_' + str(c))

    # Set objective: min sum{c(n)*x(n)} + 999*y(i) - 999*z(i)}
    objective = solver.Objective()
    resid_weights = final_weights % 1.0
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

    # inequality constraints
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

    # add an equality constraint for the total households control
    total_hh_constraint = lp_right_hand_side[total_hh_control_index]
    constraint_eq = solver.Constraint(total_hh_constraint, total_hh_constraint)
    for hh in range(0, sample_count):
        constraint_eq.SetCoefficient(x[hh], incidence[total_hh_control_index, hh])
    logger.debug("Set total_hh_control constraint to %s" % total_hh_constraint)

    solver.set_time_limit(timeout_in_seconds * 1000)

    result_status = solver.Solve()

    continuous_solution = np.asanyarray(map(lambda x: x.solution_value(), x)).astype(np.float64)
    lp_solution = np.round(continuous_solution)

    integerized_weights = np.trunc(final_weights) + lp_solution

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

    return integerized_weights, result_status

def do_integerizing(
        label,
        id,
        control_spec,
        control_totals,
        incidence_table,
        final_weights,
        relaxation_factors,
        total_hh_control_col):
    """

    Parameters
    ----------
    label : str
    id : int
    control_totals : int array
    incidence_table : DataFrame
    control_importance_weights : Series int
    final_weights : Series, index hh_id
    relaxation_factors : Series, index control column names
    total_hh_control_col : str

    Returns
    -------

    """


    # master_control_index is column index in incidence table of total_hh_control_col
    if total_hh_control_col not in incidence_table.columns:
        #print incidence_table.columns
        raise RuntimeError("total_hh_control column '%s' not found in incidence table"
                           % total_hh_control_col)
    total_hh_control_index = incidence_table.columns.get_loc(total_hh_control_col)

    control_importance_weights = control_spec.importance

    control_is_hh_based = control_spec['seed_table'] == 'households'

    # if the incidence table has only one record, then the final integer weights
    # should be just an array with 1 element equal to the total number of households;
    assert len(incidence_table.index) > 1

    # print "################################# do_integerizing"
    # print "label", label
    # print "id", id
    # print "control_spec\n", control_spec
    # print "control_totals\n", control_totals
    # print "incidence_table\n", incidence_table
    # print "control_importance_weights\n", control_importance_weights
    # print "final_weights\n", final_weights
    # print "relaxation_factors\n", relaxation_factors
    # print "total_hh_control_col\n", total_hh_control_col
    # print "total_hh_control_index\n", total_hh_control_index
    # print "control_is_hh_based\n", control_is_hh_based

    integerizer = Integerizer(
        control_totals,
        incidence_table,
        control_importance_weights,
        final_weights,
        relaxation_factors,
        total_hh_control_index,
        control_is_hh_based
        )

    # otherwise, solve for the integer weights using the Mixed Integer Programming solver.
    status = integerizer.integerize()

    if status != pywraplp.Solver.OPTIMAL:
        logger.warn("Integerizer did not find optimal solution for %s %s: %s %s"
                    % (label, id, status, SOLVER_STATUS_STRINGS[status]))

    return integerizer.weights['integerized_weight'], status
