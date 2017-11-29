# PopulationSim
# See full license in LICENSE.txt.

import logging
import os

import numpy as np
import pandas as pd

from util import setting

from activitysim.core.tracing import print_elapsed_time

from .integerizer import smart_round
from .integerizer import use_cvxpy
from .sequential_integerizer import do_sequential_integerizing


STATUS_SUCCESS = ['OPTIMAL', 'OPTIMAL_INACCURATE']
CVX_MAX_ITERS = 1000

# CVX_SOLVER = 'CBC'
CVX_SOLVER = 'GLPK_MI'
# CVX_SOLVER = 'ECOS_BB'

logger = logging.getLogger(__name__)


def log_settings():
    logger.info("have_simul_integerizer: %s" % have_simul_integerizer())
    logger.info("use_simul_integerizer: %s" % use_simul_integerizer())


def have_simul_integerizer():

    #return use_cvxpy()
    return True


def use_simul_integerizer():

    # use_simul_integerizer it if we can it unless told not to
    _use_simul_integerizer = setting('USE_SIMUL_INTEGERIZER', have_simul_integerizer())

    if _use_simul_integerizer:
        assert have_simul_integerizer()

    return _use_simul_integerizer


class SimulIntegerizer(object):

    def __init__(self,
                 incidence_df,
                 sub_weights, sub_controls_df,
                 control_spec, total_hh_control_col,
                 trace_label=''
                 ):

        sample_count = len(sub_weights.index)
        sub_zone_count = len(sub_weights.columns)

        assert len(sub_weights.index) == sample_count
        assert len(incidence_df.index) == sample_count
        assert len(sub_controls_df.index) == sub_zone_count
        assert len(sub_weights.columns) == sub_zone_count
        assert total_hh_control_col in sub_controls_df.columns
        assert total_hh_control_col in incidence_df.columns
        assert (incidence_df.columns == control_spec.target).all()

        self.incidence_df = incidence_df
        self.sub_weights = sub_weights
        self.total_hh_control_col = total_hh_control_col

        # control spec rows and control_df columns should be same and in same order
        sub_countrol_cols = list(sub_controls_df.columns)
        sub_control_spec = control_spec[control_spec.target.isin(sub_countrol_cols)]
        self.sub_controls_df = sub_controls_df[sub_control_spec.target.values]
        self.sub_countrol_importance = sub_control_spec.importance

        # only care about parent control columns NOT in sub_controls
        # control spec rows and control_df columns should be same and in same order
        parent_control_spec = control_spec[~control_spec.target.isin(self.sub_controls_df.columns)]
        self.parent_countrol_cols = parent_control_spec.target.values
        self.parent_countrol_importance = parent_control_spec.importance

        self.trace_label = trace_label

    def regress(self, float_weights, resid_weights, integerized_weights):

        #return

        data_file_path = "./regress/simul_integerize_%s.csv" % self.trace_label

        WRITE_REGRESS = not os.path.exists(data_file_path)

        # one column of resid_weights per sub_zone
        regress = pd.DataFrame(data=resid_weights.T,
                               columns=self.sub_weights.columns,
                               index=self.incidence_df.index)

        if WRITE_REGRESS:
            regress.to_csv(data_file_path, index=True)
        else:
            previous = pd.read_csv(data_file_path, comment='#')
            previous.set_index('hh_id', inplace=True)

            # print self.trace_label
            # print "\nregress\n", regress
            # print "\nprevious\n", previous

            digits = 6
            dif = 0
            for z in previous.columns:
                dif += (previous[z].round(digits) != regress[z].round(digits)).sum()
            if dif:
                new_data_file_path = "./regress/simul_integerize_%s_new.csv" % self.trace_label
                regress.to_csv(new_data_file_path, index=True)
                assert False, "regression error %s resid weights do not match, check %s" % \
                              (dif, new_data_file_path)

    def integerize(self):

        # - subzone

        total_hh_sub_control_index = \
            self.sub_controls_df.columns.get_loc(self.total_hh_control_col),

        total_hh_parent_control_index = \
            self.sub_controls_df.columns.get_loc(self.total_hh_control_col)

        sub_incidence = self.incidence_df[self.sub_controls_df.columns]
        sub_incidence = sub_incidence.as_matrix().astype(np.float64)

        sub_float_weights = self.sub_weights.as_matrix().transpose().astype(np.float64)
        sub_int_weights = sub_float_weights.astype(int)
        sub_resid_weights = sub_float_weights % 1.0

        # print "sub_float_weights\n", sub_float_weights
        # print "sub_int_weights\n", sub_int_weights
        # print "sub_resid_weights\n", sub_resid_weights

        sub_control_totals = np.asanyarray(self.sub_controls_df).astype(np.int)
        sub_countrol_importance = np.asanyarray(self.sub_countrol_importance).astype(np.float64)

        relaxed_sub_control_totals = np.dot(sub_float_weights, sub_incidence)

        # lp_right_hand_side
        lp_right_hand_side = \
            np.round(relaxed_sub_control_totals) - np.dot(sub_int_weights, sub_incidence)
        lp_right_hand_side = np.maximum(lp_right_hand_side, 0.0)

        # inequality constraint upper bounds
        sub_num_households = relaxed_sub_control_totals[:, total_hh_sub_control_index]
        sub_max_control_values = np.amax(sub_incidence, axis=0) * sub_num_households
        relax_ge_upper_bound = np.maximum(sub_max_control_values - lp_right_hand_side, 0)
        hh_constraint_ge_bound = np.maximum(sub_max_control_values, lp_right_hand_side)

        # equality constraint for the total households control
        total_hh_right_hand_side = lp_right_hand_side[:, total_hh_sub_control_index]

        # - parent
        parent_incidence = self.incidence_df[self.parent_countrol_cols]
        parent_incidence = parent_incidence.as_matrix().astype(np.float64)

        # note:
        # sum(sub_int_weights) might be different from parent_float_weights.astype(int)
        # parent_resid_weights might be > 1.0, OK as we are using in objective, not for rounding
        parent_float_weights = np.sum(sub_float_weights, axis=0)
        parent_int_weights = np.sum(sub_int_weights, axis=0)
        parent_resid_weights = np.sum(sub_resid_weights, axis=0)

        # print "parent_float_weights\n", parent_float_weights
        # print "parent_int_weights\n", parent_int_weights
        # print "parent_resid_weights\n", parent_resid_weights

        # - parent control totals based on sub_zone balanced weights
        relaxed_parent_control_totals = np.dot(parent_float_weights, parent_incidence)

        parent_countrol_importance = \
            np.asanyarray(self.parent_countrol_importance).astype(np.float64)

        parent_lp_right_hand_side = \
            np.round(relaxed_parent_control_totals) - np.dot(parent_int_weights, parent_incidence)
        parent_lp_right_hand_side = np.maximum(parent_lp_right_hand_side, 0.0)

        # - create the inequality constraint upper bounds
        parent_num_households = np.sum(sub_num_households)

        parent_max_possible_control_values = \
            np.amax(parent_incidence, axis=0) * parent_num_households
        parent_relax_ge_upper_bound = \
            np.maximum(parent_max_possible_control_values - parent_lp_right_hand_side, 0)
        parent_hh_constraint_ge_bound = \
            np.maximum(parent_max_possible_control_values, parent_lp_right_hand_side)

        # how could this not be the case?
        assert (parent_hh_constraint_ge_bound == parent_max_possible_control_values).all()

        if use_cvxpy():
            integerizer_func = np_integerize_cvx
        else:
            integerizer_func = np_integerize_ortools

        resid_weights_out, status_text = \
            integerizer_func(sub_int_weights,
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
                             total_hh_parent_control_index)

        print "\n\nxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx %s %s\n" % (self.trace_label, status_text)
        print "resid_weights_out\n", resid_weights_out

        # smart round resid_weights_out for each sub_zone
        total_household_controls = sub_control_totals[:, total_hh_sub_control_index].flatten()
        integerized_weights = np.empty_like(sub_int_weights)

        sub_zone_count = len(self.sub_weights.columns)
        for i in range(sub_zone_count):
            integerized_weights[i] = \
                smart_round(sub_int_weights[i], resid_weights_out[i], total_household_controls[i])

        # integerized_weights df: one column of integerized weights per sub_zone
        self.integerized_weights = pd.DataFrame(data=integerized_weights.T,
                                                columns=self.sub_weights.columns,
                                                index=self.incidence_df.index)

        self.regress(sub_float_weights, resid_weights_out, integerized_weights)

        return status_text


def np_integerize_cvx(sub_int_weights,
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


def np_integerize_ortools(sub_int_weights,
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

    sample_count, sub_control_count = sub_incidence.shape
    _, parent_control_count = parent_incidence.shape
    sub_zone_count, _ = sub_float_weights.shape

    # print "sub_zone_count", sub_zone_count
    # print "sample_count", sample_count
    # print "sub_control_count", sub_control_count
    # print "parent_control_count", parent_control_count
    # print "sub_incidence.shape", sub_incidence.shape
    # print "sub_float_weights.shape", sub_float_weights.shape
    # print "lp_right_hand_side.shape", lp_right_hand_side.shape
    # print "relax_ge_upper_bound.shape", relax_ge_upper_bound.shape
    # print "hh_constraint_ge_bound.shape", hh_constraint_ge_bound.shape
    # print "total_hh_right_hand_side.shape", total_hh_right_hand_side.shape
    # print "parent_lp_right_hand_side.shape", parent_lp_right_hand_side.shape

    #bug
    #total_hh_sub_control_index = total_hh_parent_control_index = -1

    # - Instantiate a mixed-integer solver
    solver = pywraplp.Solver('IntegerizeCbc', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

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
            relax_le[z, c] = solver.NumVar(0.0, lp_right_hand_side[z, c], 'relax_le[%s,%s]' % (z, c))
            relax_ge[z, c] = solver.NumVar(0.0, relax_ge_upper_bound[z, c], 'relax_ge[%s,%s]' % (z, c))

    parent_relax_le = {}
    parent_relax_ge = {}
    for c in range(parent_control_count):
        parent_relax_le[c] = solver.NumVar(0.0, parent_lp_right_hand_side[c], 'parent_relax_le[%s]' % c)
        parent_relax_ge[c] = solver.NumVar(0.0, parent_relax_ge_upper_bound[c], 'parent_relax_ge[%s]' % c)

    LOG_OVERFLOW = -725
    log_resid_weights = np.log(np.maximum(sub_resid_weights, np.exp(LOG_OVERFLOW))) #.flatten('F')
    assert not np.isnan(log_resid_weights).any()

    log_parent_resid_weights = \
        np.log(np.maximum(parent_resid_weights, np.exp(LOG_OVERFLOW))) #.flatten('F')
    assert not np.isnan(log_parent_resid_weights).any()


    # objective = cvx.Maximize(
    #     cvx.sum_entries(cvx.mul_elemwise(log_resid_weights, cvx.vec(x))) +
    #     cvx.sum_entries(cvx.mul_elemwise(log_parent_resid_weights, cvx.vec(cvx.sum_entries(x, axis=0)))) -  # nopep8
    #     cvx.sum_entries(relax_le * sub_countrol_importance) -
    #     cvx.sum_entries(relax_ge * sub_countrol_importance) -
    #     cvx.sum_entries(cvx.mul_elemwise(parent_countrol_importance, parent_relax_le)) -
    #     cvx.sum_entries(cvx.mul_elemwise(parent_countrol_importance, parent_relax_ge))
    # )

    # solver.Sum(solver.Sum(x[z, hh] for hh in range(sample_count)) * log_parent_resid_weights[hh]
    #            for z in range(sub_zone_count)) - \

    z = solver.Sum(x[z, hh] * log_resid_weights[z, hh]
                   for z in range(sub_zone_count)
                   for hh in range(sample_count)) + \
        solver.Sum(x[z, hh] * log_parent_resid_weights[hh]
                   for hh in range(sample_count)
                   for z in range(sub_zone_count)) - \
        solver.Sum(relax_le[z, c] * sub_countrol_importance[c]
                   for z in range(sub_zone_count)
                   for c in range(sub_control_count)) - \
        solver.Sum(relax_ge[z, c] * sub_countrol_importance[c]
                   for z in range(sub_zone_count)
                   for c in range(sub_control_count)) - \
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
    #
    #     # - equality constraint for the total households control
    #     cvx.sum_entries(x, axis=1) == total_hh_right_hand_side,
    #
    #     cvx.vec(cvx.sum_entries(x, axis=0) * parent_incidence) - parent_relax_le >= 0,                              # nopep8
    #     cvx.vec(cvx.sum_entries(x, axis=0) * parent_incidence) - parent_relax_le <= parent_lp_right_hand_side,      # nopep8
    #     cvx.vec(cvx.sum_entries(x, axis=0) * parent_incidence) + parent_relax_ge >= parent_lp_right_hand_side,      # nopep8
    #     cvx.vec(cvx.sum_entries(x, axis=0) * parent_incidence) + parent_relax_ge <= parent_hh_constraint_ge_bound,  # nopep8
    # ]

    # - sub inequality constraints
    sub_constraint_ge = {}
    sub_constraint_le = {}
    for z in range(sub_zone_count):
        for c in range(sub_control_count):

            # don't add inequality constraints for total households control
            if c == total_hh_sub_control_index:
                continue

            sub_constraint_le[z, c] = solver.Constraint(0, lp_right_hand_side[z, c])
            for hh in range(sample_count):
                sub_constraint_le[z, c].SetCoefficient(x[z, hh], sub_incidence[hh, c])
                sub_constraint_le[z, c].SetCoefficient(relax_le[z, c], -1.0)

            sub_constraint_ge[z, c] = solver.Constraint(lp_right_hand_side[z, c], hh_constraint_ge_bound[z, c])
            for hh in range(sample_count):
                sub_constraint_ge[z, c].SetCoefficient(x[z, hh], sub_incidence[hh, c])
                sub_constraint_ge[z, c].SetCoefficient(relax_ge[z, c], 1.0)

    # - equality constraint for the total households control
    constraint_eq = {}
    for z in range(sub_zone_count):
        total_hh_constraint = total_hh_right_hand_side[z][0]

        constraint_eq[z] = solver.Constraint(total_hh_constraint, total_hh_constraint)
        for hh in range(sample_count):
            constraint_eq[z].SetCoefficient(x[z, hh], 1.0)

    # - sub inequality constraints
    parent_constraint_le = {}
    parent_constraint_ge = {}
    for c in range(parent_control_count):

        if c == total_hh_parent_control_index:
            continue

        parent_constraint_le[c] = solver.Constraint(0, parent_lp_right_hand_side[c])
        parent_constraint_ge[c] = solver.Constraint(parent_lp_right_hand_side[c], parent_hh_constraint_ge_bound[c])

        for z in range(sub_zone_count):
            for hh in range(sample_count):
                parent_constraint_le[c].SetCoefficient(x[z, hh], parent_incidence[hh, c])
                parent_constraint_le[c].SetCoefficient(relax_le[z, c], -1.0)

                parent_constraint_ge[c].SetCoefficient(x[z, hh], parent_incidence[hh, c])
                parent_constraint_ge[c].SetCoefficient(relax_le[z, c], 1.0)


    #solver.set_time_limit(CBC_TIMEOUT_IN_SECONDS * 1000)

    solver.EnableOutput()

    t0 = print_elapsed_time()
    result_status = solver.Solve()
    t0 = print_elapsed_time("solver.Solve", t0)

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


def try_simul_integerizing(
        trace_label,
        incidence_df,
        sub_weights, sub_controls_df,
        sub_geography,
        control_spec, total_hh_control_col,
        sub_control_zones):
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
        logger.info("omitting %s zero weight rows out of %s"
                    % (zero_weight_rows.sum(), len(incidence_df.index)))

    integerizer = SimulIntegerizer(
        incidence_df[~zero_weight_rows],
        sub_weights[~zero_weight_rows],
        sub_controls_df,
        control_spec,
        total_hh_control_col,
        trace_label
    )

    status = integerizer.integerize()

    if status not in STATUS_SUCCESS:
        return status, None

    if zero_weight_rows.any():
        # restore zero_weight_rows to integerized_weights
        logger.info("restoring %s zero weight rows" % zero_weight_rows.sum())
        integerized_weights = \
            pd.DataFrame(data=np.zeros(
                sub_weights.shape, dtype=np.int),
                columns=sub_weights.columns,
                index=sub_weights.index)
        integerized_weights.update(integerizer.integerized_weights)
    else:
        integerized_weights = integerizer.integerized_weights

    integerized_weights_df = \
        reshape_result(sub_weights, integerized_weights, sub_geography, sub_control_zones)

    logger.debug("SimulIntegerizer status %s" % (status,))

    return status, integerized_weights_df


def reshape_result(float_weights, integerized_weights, sub_geography, sub_control_zones):
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
    for zone_id, zone_name in sub_control_zones.iteritems():

        weights = float_weights[zone_name]

        zone_weights_df = pd.DataFrame(index=range(0, len(integerized_weights.index)))
        zone_weights_df[weights.index.name] = float_weights.index
        zone_weights_df[sub_geography] = zone_id
        zone_weights_df['balanced_weight'] = float_weights[zone_name].values
        zone_weights_df['integer_weight'] = integerized_weights[zone_name].astype(int).values

        integer_weights_list.append(zone_weights_df)

    integer_weights_df = pd.concat(integer_weights_list)

    return integer_weights_df


def do_simul_integerizing(
        trace_label,
        incidence_df,
        sub_weights,
        sub_controls_df,
        control_spec,
        total_hh_control_col,
        sub_geography,
        sub_control_zones):
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
    status,  integerized_weights_df = try_simul_integerizing(
        trace_label,
        incidence_df,
        sub_weights, sub_controls_df,
        sub_geography,
        control_spec, total_hh_control_col,
        sub_control_zones)

    if status in STATUS_SUCCESS:
        logger.info("do_simul_integerizing succeeded for %s status %s. " % (trace_label, status))
        return integerized_weights_df

    logger.warn("do_simul_integerizing failed for %s status %s. " % (trace_label, status))

    # if simultaneous integerization failed, sequentially integerize to detect infeasible subzones
    # infeasible zones will be smart rounded and returned in rounded_weights_df
    feasible_zone_ids, rounded_zone_ids, sequentially_integerized_weights_df, rounded_weights_df = \
        do_sequential_integerizing(
            trace_label,
            incidence_df,
            sub_weights, sub_controls_df,
            control_spec, total_hh_control_col,
            sub_control_zones,
            sub_geography,
            combine_results=False)

    if len(feasible_zone_ids) == 0:
        # if all subzones are infeasible, then we don't have any feasible zones to try
        # so the best we can do is return rounded_weights_df
        logger.warn("do_sequential_integerizing failed for all subzones %s. " % trace_label)
        logger.info("do_simul_integerizing returning smart rounded weights for %s."
                    % trace_label)
        return rounded_weights_df

    if len(rounded_zone_ids) == 0:
        # if all subzones are feasible, then there are no zones to remove in order to retry
        # so the best we can do is return sequentially_integerized_weights_df
        logger.warn("do_simul_integerizing failed but found no infeasible sub zones %s. "
                    % trace_label)
        logger.info("do_simul_integerizing falling back to sequential integerizing for %s."
                    % trace_label)
        return sequentially_integerized_weights_df

    if len(feasible_zone_ids) == 1:
        # if only one zone is feasible, not much point in simul_integerizing it
        # so the best we can do is return do_sequential_integerizing combined results
        logger.warn("do_simul_integerizing failed but found no infeasible sub zones %s. "
                    % trace_label)
        return pd.concat([sequentially_integerized_weights_df, rounded_weights_df])

    # - remove the infeasible subzones and retry simul_integerizing

    sub_controls_df = sub_controls_df.loc[feasible_zone_ids]
    sub_control_zones = sub_control_zones.loc[sub_control_zones.index.isin(feasible_zone_ids)]
    sub_weights = sub_weights[sub_control_zones]

    logger.info("do_simul_integerizing %s infeasable subzones for %s. "
                % (len(rounded_zone_ids), trace_label))

    status, integerized_weights_df = try_simul_integerizing(
        "retry_%s" % trace_label,
        incidence_df,
        sub_weights, sub_controls_df,
        sub_geography,
        control_spec, total_hh_control_col,
        sub_control_zones)

    if status in STATUS_SUCCESS:
        # we successfully simul_integerized the sequentially feasible sub zones, so we can
        # return the simul_integerized results along with the rounded_weights for the infeasibles
        logger.info("do_simul_integerizing retry succeeded for %s status %s. "
                    % (trace_label, status))
        return pd.concat([integerized_weights_df, rounded_weights_df])

    # haven't seen this happen, but I suppose it could...
    logger.error("do_simul_integerizing retry failed for %s status %s." % (trace_label, status))
    logger.info("do_simul_integerizing falling back to sequential integerizing for %s."
                % trace_label)

    # nothing to do but return do_sequential_integerizing combined results
    return pd.concat([sequentially_integerized_weights_df, rounded_weights_df])
