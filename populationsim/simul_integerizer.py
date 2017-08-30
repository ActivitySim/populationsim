# PopulationSim
# See full license in LICENSE.txt.

import logging

import numpy as np
import pandas as pd

import cylp
import cvxpy as cvx

from activitysim.core import tracing


import cylp
import cvxpy as cvx

STATUS_TEXT = {
    cvx.OPTIMAL: 'OPTIMAL',
    cvx.INFEASIBLE: 'INFEASIBLE',
    cvx.UNBOUNDED: 'UNBOUNDED',
    cvx.OPTIMAL_INACCURATE: 'OPTIMAL_INACCURATE',
    cvx.INFEASIBLE_INACCURATE: 'INFEASIBLE_INACCURATE',
    cvx.UNBOUNDED_INACCURATE: 'UNBOUNDED_INACCURATE',
    None: 'FAILED'
}

STATUS_SUCCESS = ['OPTIMAL', 'OPTIMAL_INACCURATE']

# - solver list: http://www.cvxpy.org/en/latest/tutorial/advanced/
# cvx.installed_solvers(): ['ECOS_BB', 'SCS', 'ECOS', 'LS']
# ['CBC', 'CVXOPT', 'ECOS_BB', 'GLPK_MI', 'SCS', 'ECOS', 'GLPK', 'LS']
CVX_SOLVER = cvx.ECOS
CVX_SOLVER = cvx.GLPK_MI
CVX_MAX_ITERS = 300


logger = logging.getLogger(__name__)

DUMP = True


def dump_df(dump_switch, df, trace_label, fname):
    if dump_switch:
        trace_label = "%s.DUMP.%s" % (trace_label, fname)

        if len(df.index) > 0:
            tracing.write_csv(df, file_name=trace_label, transpose=False)


class SimulIntegerizer(object):

    def __init__(self,
                 incidence_df,
                 parent_weights, parent_controls_df,
                 sub_weights, sub_controls_df,
                 control_spec, total_hh_control_col
                 ):

        sub_control_zones = list(sub_weights.columns)

        sample_count = len(parent_weights.index)
        sub_zone_count = len(sub_weights.columns)

        assert len(sub_weights.index) == sample_count
        assert len(incidence_df.index) == sample_count
        assert len(sub_controls_df.index) == sub_zone_count
        assert len(sub_weights.columns) == sub_zone_count
        assert total_hh_control_col in sub_controls_df.columns
        assert total_hh_control_col in parent_controls_df.columns

        self.incidence_df = incidence_df
        self.parent_weights = parent_weights
        self.parent_controls_df = parent_controls_df
        self.sub_weights = sub_weights
        self.sub_controls_df = sub_controls_df
        self.total_hh_control_col = total_hh_control_col

        parent_countrol_cols = list(parent_controls_df.columns)
        parent_control_spec = control_spec[control_spec.target.isin(parent_countrol_cols)]

        # print "\nparent_countrol_cols\n", parent_countrol_cols
        # print "\nparent_control_spec.target\n", parent_control_spec.target

        # should be in same order
        assert (parent_countrol_cols == parent_control_spec.target).all()

        self.parent_countrol_importance = parent_control_spec.importance

        sub_countrol_cols = list(sub_controls_df.columns)
        sub_control_spec = control_spec[control_spec.target.isin(sub_countrol_cols)]
        assert (sub_countrol_cols == sub_control_spec.target).all()

        self.sub_countrol_importance = sub_control_spec.importance
        self.sub_control_is_hh_based = sub_control_spec['seed_table'] == 'households'


        # print "\nparent_controls_df\n", self.parent_controls_df
        #
        # print "\nparent_countrol_importance\n", self.parent_countrol_importance
        # print "\nparent_weights\n", self.parent_weights
        #
        # print "\nsub_controls_df\n", self.sub_controls_df
        # print "\nsub_countrol_importance\n", self.sub_countrol_importance
        # print "\nsub_weights\n", self.sub_weights



    def integerize(self):

        parent_incidence = self.incidence_df[self.parent_controls_df.columns]
        parent_incidence = parent_incidence.as_matrix().astype(np.float64)

        print "\nparent_incidence_df\n", self.incidence_df[self.parent_controls_df.columns]
        print "\nparent_incidence\n", parent_incidence

        sub_incidence = self.incidence_df[self.sub_controls_df.columns]
        total_hh_sub_control_index = sub_incidence.columns.get_loc(self.total_hh_control_col),

        sub_incidence = sub_incidence.as_matrix().astype(np.float64)

        print "\nsub_incidence_df\n", self.incidence_df[self.sub_controls_df.columns]
        print "\nsub_incidence\n", sub_incidence

        # parent_float_weights = np.asanyarray(self.parent_weights).astype(np.float64)
        # print "self.parent_weights\n", self.parent_weights
        # print "parent_float_weights\n", parent_float_weights

        parent_control_totals = np.asanyarray(self.parent_controls_df).astype(np.int)
        print "\nparent_control_totals\n", parent_control_totals

        # relaxed_parent_control_totals = np.dot(parent_float_weights, parent_incidence)
        # print "relaxed_parent_control_totals\n", relaxed_parent_control_totals

        sub_float_weights = self.sub_weights.as_matrix().transpose().astype(np.float64)
        sub_int_weights = sub_float_weights.astype(int)
        sub_resid_weights = sub_float_weights % 1.0

        print "\nself.sub_weights\n", self.sub_weights
        print "\nsub_float_weights\n", sub_float_weights
        print "\nsub_int_weights\n", sub_int_weights
        print "\nsub_resid_weights\n", sub_resid_weights

        sub_control_totals = np.asanyarray(self.sub_controls_df).astype(np.int)
        print "\nsub_control_totals\n", sub_control_totals

        relaxed_sub_control_totals = np.dot(sub_float_weights, sub_incidence)
        print "\nrelaxed_sub_control_totals\n", relaxed_sub_control_totals

        sub_zone_count = len(self.sub_weights.columns)
        for i in range(sub_zone_count):
            print "sub_control_totals", i, np.dot(sub_float_weights[i], sub_incidence)

        sub_countrol_importance = np.asanyarray(self.sub_countrol_importance).astype(np.float64)
        print "\nself.sub_countrol_importance\n", self.sub_countrol_importance
        print "\nsub_countrol_importance\n", sub_countrol_importance

        sub_control_is_hh_based = np.asanyarray(self.sub_control_is_hh_based).astype(bool)

        #################################################


        sample_count, sub_control_count = sub_incidence.shape
        print "sample_count", sample_count
        print "control_count", sub_control_count

        # - lp_right_hand_side - relaxed_control_shortfall
        lp_right_hand_side = np.round(relaxed_sub_control_totals) - np.dot(sub_int_weights, sub_incidence)
        lp_right_hand_side = np.maximum(lp_right_hand_side, 0.0)

        print "\nlp_right_hand_side\n", lp_right_hand_side

        # - create the inequality constraint upper bounds
        max_sub_incidence_value = np.amax(sub_incidence, axis=0)
        assert (max_sub_incidence_value[sub_control_is_hh_based] <= 1).all()
        num_households = relaxed_sub_control_totals[total_hh_sub_control_index]
        relax_ge_upper_bound = np.maximum(max_sub_incidence_value * num_households - lp_right_hand_side, 0)

        print "\nmax_sub_incidence_value\n", max_sub_incidence_value
        print "\nrelax_ge_upper_bound\n", relax_ge_upper_bound

        # - Decision variables for optimization
        x = cvx.Variable(sub_zone_count, sample_count)

        # 1.0 unless resid_weights is zero
        x_max = (~(sub_float_weights == sub_int_weights)).astype(float)
        print "\nx_max\n", x_max

        # - Create positive continuous constraint relaxation variables
        relax_le = cvx.Variable(sub_zone_count, sub_control_count)
        relax_ge = cvx.Variable(sub_zone_count, sub_control_count)

        # - Set objective

        LOG_OVERFLOW = -725
        log_resid_weights = np.log(np.maximum(sub_resid_weights, np.exp(LOG_OVERFLOW))).flatten('F')
        assert not np.isnan(log_resid_weights).any()

        print "\nlog_resid_weights\n", log_resid_weights

        # cvxpy overloads * so following is matrix multiplication
        objective = cvx.Maximize(
            cvx.sum_entries(cvx.mul_elemwise(log_resid_weights, cvx.vec(x))) -
            cvx.sum_entries(relax_le * sub_countrol_importance) -
            cvx.sum_entries(relax_ge * sub_countrol_importance)
        )

        total_hh_right_hand_side = lp_right_hand_side[:, total_hh_sub_control_index]

        print "\ntotal_hh_right_hand_side\n", total_hh_right_hand_side
        print "\ntotal_hh_right_hand_side.shape\n", total_hh_right_hand_side.shape

        print "\nlp_right_hand_side\n", lp_right_hand_side

        hh_constraint_ge_bound = np.maximum(sub_control_totals * max_sub_incidence_value, lp_right_hand_side)
        print "\nhh_constraint_ge_bound\n", hh_constraint_ge_bound

        constraints = [
            (x * sub_incidence) - relax_le  >= 0,
            (x * sub_incidence) - relax_le <= lp_right_hand_side,
            (x * sub_incidence) + relax_ge >= lp_right_hand_side,
            (x * sub_incidence) + relax_ge <= hh_constraint_ge_bound,
            # cvx.vec(x * sub_incidence) - relax_le >= 0,
            # cvx.vec(x * sub_incidence) - relax_le <= lp_right_hand_side,
            # cvx.vec(x * sub_incidence) + relax_ge >= lp_right_hand_side,
            # cvx.vec(x * sub_incidence) + relax_ge <= hh_constraint_ge_bound,

            x >= 0.0,
            x <= x_max,
            #
            relax_le >= 0.0,
            relax_le <= lp_right_hand_side,
            #
            relax_ge >= 0.0,
            relax_ge <= relax_ge_upper_bound,
            #
            # # equality constraint for the total households control
            cvx.sum_entries(x, axis=1) == total_hh_right_hand_side,
        ]

        prob = cvx.Problem(objective, constraints)

        try:
            prob.solve(solver=CVX_SOLVER, verbose=True, max_iters=CVX_MAX_ITERS)
        except cvx.SolverError as e:
            logging.exception('Solver error: %s' % e)
            logging.exception('Solver error encountered in SimulIntegerizer. Weights will be rounded.')

        print "\nstatus:", STATUS_TEXT[prob.status]
        print "\nx\n", x.value
        print "\nrelax_le\n", relax_le.value
        print "\nrelax_ge\n", relax_ge.value

        if np.any(x.value):
            resid_weights_out = np.asarray(x.value)[0]
        else:
            resid_weights_out = sub_resid_weights

        return sub_int_weights, resid_weights_out, STATUS_TEXT[prob.status]



def do_simul_integerizing(
        incidence_df,
        parent_weights, parent_controls_df,
        sub_weights, sub_controls_df,
        parent_geography, parent_id,
        sub_geography,
        control_spec, total_hh_control_col):

    trace_label = "do_simul_integerizing_%s_%s" % (parent_geography, parent_id)

    zero_weight_rows = (parent_weights == 0)
    if zero_weight_rows.any():
        logger.info("omitting %s zero weight rows out of %s"
                    % (zero_weight_rows.sum(), len(incidence_df.index)))
        incidence_df = incidence_df[~zero_weight_rows]
        parent_weights = parent_weights[~zero_weight_rows]
        sub_weights = sub_weights[~zero_weight_rows]



    sample_count = len(parent_weights.index)
    sub_zone_count = len(sub_weights.columns)

    assert len(sub_weights.index) == sample_count
    assert len(incidence_df.index) == sample_count
    assert len(sub_controls_df.index) == sub_zone_count
    assert len(sub_weights.columns) == sub_zone_count
    assert total_hh_control_col in sub_controls_df.columns
    assert total_hh_control_col in parent_controls_df.columns

    print "parent_geography", parent_geography, parent_id
    print "sub_geography", sub_geography

    # dump_df(DUMP, incidence_df, trace_label, 'incidence_df')
    # dump_df(DUMP, incidence_df[control_spec.target], trace_label, 'incidence_table')
    # dump_df(DUMP, sub_weights, trace_label, 'sub_weights')

    integerizer = SimulIntegerizer(
        incidence_df,
        parent_weights,
        parent_controls_df,
        sub_weights,
        sub_controls_df,
        control_spec,
        total_hh_control_col
    )

    # otherwise, solve for the integer weights using the Mixed Integer Programming solver.
    status = integerizer.integerize()

    assert False

    # restore zero_weight_rows to integerized_weights
    integerized_weights = pd.Series(0.0, index=zero_weight_rows.index)
    nonzero_integerized_weights = integerizer.weights['integerized_weight']
    integerized_weights.update(nonzero_integerized_weights)

    logger.debug("SimulIntegerizer status %s" % (status,))

    return integerized_weights, status
