# PopulationSim
# See full license in LICENSE.txt.

import logging

import numpy as np
import pandas as pd

import cylp
import cvxpy as cvx

from activitysim.core import tracing
from .integerizer import smart_round
from .sequential_integerizer import do_round_infeasible_subzones
from .sequential_integerizer import do_sequential_integerizing


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
#CVX_SOLVER = cvx.ECOS
CVX_SOLVER = cvx.GLPK_MI
CVX_MAX_ITERS = 1000


logger = logging.getLogger(__name__)

class SimulIntegerizer(object):

    def __init__(self,
                 incidence_df,
                 parent_weights, parent_controls_df,
                 sub_weights, sub_controls_df,
                 control_spec, total_hh_control_col
                 ):

        #########################################


        relaxed_control_totals = \
            np.round(np.dot(np.asanyarray(parent_weights), incidence_df.as_matrix()))

        relaxed_control_totals = \
            pd.Series(relaxed_control_totals, index=incidence_df.columns.values)

        alt_parent_controls_df = relaxed_control_totals.to_frame().T
        alt_parent_controls_df.update(parent_controls_df)

        #print "\nparent_controls_df\n", parent_controls_df
        #print "\nalt_parent_controls_df\n", alt_parent_controls_df

        parent_controls_df = alt_parent_controls_df

        #########################################

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
        self.sub_weights = sub_weights
        self.total_hh_control_col = total_hh_control_col

        # control spec rows and control_df columns should be same and in same order
        sub_countrol_cols = list(sub_controls_df.columns)
        sub_control_spec = control_spec[control_spec.target.isin(sub_countrol_cols)]
        self.sub_controls_df = sub_controls_df[sub_control_spec.target.values]
        self.sub_countrol_importance = sub_control_spec.importance
        self.sub_control_is_hh_based = sub_control_spec['seed_table'] == 'households'

        # only care about parent control columns NOT in sub_controls
        # control spec rows and control_df columns should be same and in same order
        parent_countrol_cols = [c for c in parent_controls_df.columns if c not in sub_controls_df.columns]
        parent_control_spec = control_spec[control_spec.target.isin(parent_countrol_cols)]
        self.parent_controls_df = parent_controls_df[parent_control_spec.target.values]
        self.parent_countrol_importance = parent_control_spec.importance
        self.parent_control_is_hh_based = parent_control_spec['seed_table'] == 'households'


    def integerize(self):

        sub_incidence = self.incidence_df[self.sub_controls_df.columns]
        total_hh_sub_control_index = sub_incidence.columns.get_loc(self.total_hh_control_col),

        sub_incidence = sub_incidence.as_matrix().astype(np.float64)

        sub_float_weights = self.sub_weights.as_matrix().transpose().astype(np.float64)
        sub_int_weights = sub_float_weights.astype(int)
        sub_resid_weights = sub_float_weights % 1.0

        sub_control_totals = np.asanyarray(self.sub_controls_df).astype(np.int)
        # print "\nsub_control_totals\n", sub_control_totals

        relaxed_sub_control_totals = np.dot(sub_float_weights, sub_incidence)

        sub_zone_count = len(self.sub_weights.columns)

        sub_countrol_importance = np.asanyarray(self.sub_countrol_importance).astype(np.float64)

        sub_control_is_hh_based = np.asanyarray(self.sub_control_is_hh_based).astype(bool)

        sample_count, sub_control_count = sub_incidence.shape

        # - lp_right_hand_side
        lp_right_hand_side = np.round(relaxed_sub_control_totals) - np.dot(sub_int_weights, sub_incidence)
        lp_right_hand_side = np.maximum(lp_right_hand_side, 0.0)

        # - create the inequality constraint upper bounds
        max_sub_incidence_value = np.amax(sub_incidence, axis=0)
        assert (max_sub_incidence_value[sub_control_is_hh_based] <= 1).all()
        sub_num_households = relaxed_sub_control_totals[:,total_hh_sub_control_index]

        relax_ge_upper_bound = np.maximum(max_sub_incidence_value * sub_num_households - lp_right_hand_side, 0)

        ############### parent stuff

        parent_incidence = self.incidence_df[self.parent_controls_df.columns]
        parent_incidence = parent_incidence.as_matrix().astype(np.float64)

        _, parent_control_count = parent_incidence.shape

        # print "\nparent_incidence_df\n", self.incidence_df[self.parent_controls_df.columns]

        parent_float_weights = np.sum(sub_float_weights, axis=0)
        parent_int_weights = np.sum(sub_int_weights, axis=0)
        parent_resid_weights = np.sum(sub_resid_weights, axis=0)

        # print "\nself.parent_weights\n", self.parent_weights
        # print "\nparent_float_weights\n", parent_float_weights
        # print "\nparent_int_weights\n", parent_int_weights
        # print "\nparent_resid_weights\n", parent_resid_weights

        parent_control_totals = np.asanyarray(self.parent_controls_df).astype(np.int).flatten()
        print "\nparent_control_totals\n", parent_control_totals

        relaxed_parent_control_totals = np.dot(parent_float_weights, parent_incidence)
        print "relaxed_parent_control_totals\n", relaxed_parent_control_totals

        parent_countrol_importance = np.asanyarray(self.parent_countrol_importance).astype(np.float64)
        # print "\nself.parent_countrol_importance\n", self.parent_countrol_importance

        parent_control_is_hh_based = np.asanyarray(self.parent_control_is_hh_based).astype(bool)

        # - lp_right_hand_side
        parent_lp_right_hand_side = np.round(relaxed_parent_control_totals) - np.dot(parent_int_weights, parent_incidence)
        parent_lp_right_hand_side = np.maximum(parent_lp_right_hand_side, 0.0)

        # print "\nnp.dot(parent_int_weights, parent_incidence)\n", np.dot(parent_int_weights, parent_incidence)
        # print "\nparent_lp_right_hand_side\n", parent_lp_right_hand_side


        # - create the inequality constraint upper bounds
        parent_max_incidence_value = np.amax(parent_incidence, axis=0)
        assert (parent_max_incidence_value[parent_control_is_hh_based] <= 1).all()
        parent_num_households = np.sum(sub_num_households)
        parent_relax_ge_upper_bound = np.maximum(parent_max_incidence_value * parent_num_households - parent_lp_right_hand_side, 0)

        # print "\nrelaxed_sub_control_totals[total_hh_sub_control_index]\n", relaxed_sub_control_totals[total_hh_sub_control_index]
        # print "\nparent_num_households\n", parent_num_households
        # print "\nparent_max_incidence_value\n", parent_max_incidence_value
        # print "\nparent_relax_ge_upper_bound\n", parent_relax_ge_upper_bound

        #################################################




        # - Decision variables for optimization
        x = cvx.Variable(sub_zone_count, sample_count)

        # 1.0 unless resid_weights is zero
        x_max = (~(sub_float_weights == sub_int_weights)).astype(float)
        # print "\nx_max\n", x_max

        # - Create positive continuous constraint relaxation variables
        relax_le = cvx.Variable(sub_zone_count, sub_control_count)
        relax_ge = cvx.Variable(sub_zone_count, sub_control_count)

        parent_relax_le = cvx.Variable(parent_control_count)
        parent_relax_ge = cvx.Variable(parent_control_count)

        # - Set objective

        LOG_OVERFLOW = -725
        log_resid_weights = np.log(np.maximum(sub_resid_weights, np.exp(LOG_OVERFLOW))).flatten('F')
        assert not np.isnan(log_resid_weights).any()

        log_parent_resid_weights = np.log(np.maximum(parent_resid_weights, np.exp(LOG_OVERFLOW))).flatten('F')
        assert not np.isnan(log_parent_resid_weights).any()

        # print "\nsub_resid_weights\n", sub_resid_weights
        # print "\nparent_resid_weights\n", parent_resid_weights
        # print "\nlog_parent_resid_weights\n", log_parent_resid_weights



        #print "\nlog_resid_weights\n", log_resid_weights

        # cvxpy overloads * so following is matrix multiplication
        objective = cvx.Maximize(
            cvx.sum_entries(cvx.mul_elemwise(log_resid_weights, cvx.vec(x))) \
            - cvx.sum_entries(relax_le * sub_countrol_importance) \
            - cvx.sum_entries(relax_ge * sub_countrol_importance) \
            + cvx.sum_entries(cvx.mul_elemwise(log_parent_resid_weights, cvx.vec(cvx.sum_entries(x, axis=0))))
            - cvx.sum_entries(cvx.mul_elemwise(parent_countrol_importance, parent_relax_le)) \
            - cvx.sum_entries(cvx.mul_elemwise(parent_countrol_importance, parent_relax_ge)) \
            )

        total_hh_right_hand_side = lp_right_hand_side[:, total_hh_sub_control_index]

        #print "\ntotal_hh_right_hand_side\n", total_hh_right_hand_side
        #print "\ntotal_hh_right_hand_side.shape\n", total_hh_right_hand_side.shape

        #print "\nlp_right_hand_side\n", lp_right_hand_side

        # integerizer.java line 346
        # ct_ge_bound[i] = Math.max( controlTotals[totalHouseholdsControlIndex]*maxIncidenceValue[i], lpRightHandSide[i] );

        #hh_constraint_ge_bound = np.maximum(sub_control_totals * max_sub_incidence_value, lp_right_hand_side)
        hh_constraint_ge_bound = np.maximum(sub_num_households * max_sub_incidence_value, lp_right_hand_side)

        #parent_hh_constraint_ge_bound = np.maximum(parent_control_totals * parent_max_incidence_value, parent_lp_right_hand_side)
        parent_hh_constraint_ge_bound = np.maximum(parent_num_households * parent_max_incidence_value, parent_lp_right_hand_side)


        constraints = [
            (x * sub_incidence) - relax_le  >= 0,
            (x * sub_incidence) - relax_le <= lp_right_hand_side,
            (x * sub_incidence) + relax_ge >= lp_right_hand_side,
            (x * sub_incidence) + relax_ge <= hh_constraint_ge_bound,

            x >= 0.0,
            x <= x_max,

            relax_le >= 0.0,
            relax_le <= lp_right_hand_side,

            relax_ge >= 0.0,
            relax_ge <= relax_ge_upper_bound,

            # # equality constraint for the total households control
            cvx.sum_entries(x, axis=1) == total_hh_right_hand_side,

            cvx.vec(cvx.sum_entries(x, axis=0) * parent_incidence) - parent_relax_le >= 0,
            cvx.vec(cvx.sum_entries(x, axis=0) * parent_incidence) - parent_relax_le <= parent_lp_right_hand_side,
            cvx.vec(cvx.sum_entries(x, axis=0) * parent_incidence) + parent_relax_ge >= parent_lp_right_hand_side,
            cvx.vec(cvx.sum_entries(x, axis=0) * parent_incidence) + parent_relax_ge <= parent_hh_constraint_ge_bound,

            parent_relax_le >= 0.0,
            parent_relax_le <= parent_lp_right_hand_side,

            parent_relax_ge >= 0.0,
            parent_relax_ge <= parent_relax_ge_upper_bound,
        ]

        prob = cvx.Problem(objective, constraints)

        try:
            print "prob.solve..."
            prob.solve(solver=CVX_SOLVER, verbose=True, max_iters=CVX_MAX_ITERS)
        except cvx.SolverError as e:
            logging.exception('Solver error: %s' % e)
            logging.exception('Solver error encountered in SimulIntegerizer. Weights will be rounded.')

        # print "\nstatus:", STATUS_TEXT[prob.status]
        # print "\nx\n", x.value
        # print "\nrelax_le\n", relax_le.value
        # print "\nrelax_ge\n", relax_ge.value
        # print "\nparent_relax_le\n", parent_relax_le.value
        # print "\nparent_relax_ge\n", parent_relax_ge.value

        if np.any(x.value):
            resid_weights_out = np.asarray(x.value)
        else:
            resid_weights_out = sub_resid_weights

        total_household_controls = sub_control_totals[:, total_hh_sub_control_index].flatten()
        # print "\ntotal_household_controls", total_household_controls

        integerized_weights = np.empty_like(sub_int_weights)
        for i in range(sub_zone_count):
            integerized_weights[i] = smart_round(sub_int_weights[i], resid_weights_out[i], total_household_controls[i])

        delta = (integerized_weights != np.round(sub_float_weights)).sum()
        logger.debug("Integerizer: %s out of %s different from round" % (delta, sub_float_weights.size))

        self.integerized_weights = pd.DataFrame(data=integerized_weights.T,
                                                columns=self.sub_weights.columns,
                                                index=self.incidence_df.index)

        # print "\nself.sub_weights\n", self.sub_weights
        # print "\nself.integerized_weights\n", self.integerized_weights

        return STATUS_TEXT[prob.status]



def try_simul_integerizing(
        incidence_df,
        parent_weights, parent_controls_df,
        sub_weights, sub_controls_df,
        parent_geography, parent_id,
        sub_geography,
        control_spec, total_hh_control_col,
        sub_control_zones):

    zero_weight_rows = (parent_weights == 0)
    if zero_weight_rows.any():
        logger.info("omitting %s zero weight rows out of %s"
                    % (zero_weight_rows.sum(), len(incidence_df.index)))

    integerizer = SimulIntegerizer(
        incidence_df[~zero_weight_rows],
        parent_weights[~zero_weight_rows],
        parent_controls_df,
        sub_weights[~zero_weight_rows],
        sub_controls_df,
        control_spec,
        total_hh_control_col
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

    # print "\nintegerized_weights\n", integerized_weights

    integerized_weights_df = reshape_result(sub_weights, integerized_weights, sub_geography, sub_control_zones)
    # print "\nintegerized_weights_df\n", integerized_weights_df

    logger.debug("SimulIntegerizer status %s" % (status,))

    return status, integerized_weights_df


def reshape_result(float_weights, integerized_weights, sub_geography, sub_control_zones):

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
        incidence_df,
        parent_weights, parent_controls_df,
        sub_weights, sub_controls_df,
        parent_geography, parent_id,
        sub_geography,
        control_spec, total_hh_control_col,
        sub_control_zones):

    trace_label = "do_simul_integerizing_%s_%s" % (parent_geography, parent_id)

    status,  integerized_weights_df = try_simul_integerizing(
        incidence_df,
        parent_weights, parent_controls_df,
        sub_weights, sub_controls_df,
        parent_geography, parent_id,
        sub_geography,
        control_spec, total_hh_control_col,
        sub_control_zones)

    if status in STATUS_SUCCESS:
        logger.info("do_simul_integerizing succeeded for %s status %s. " % (trace_label, status))
        return integerized_weights_df

    logger.warn("do_simul_integerizing failed for %s status %s. " % (trace_label, status))

    feasible_subzones, infeasible_subzones, rounded_weights_df = do_round_infeasible_subzones(
        incidence_df,
        sub_weights, sub_controls_df,
        control_spec, total_hh_control_col,
        sub_control_zones,
        sub_geography,
        parent_geography,
        parent_id)

    #bug - priont number

    print "\ninfeasible_subzones\n", infeasible_subzones
    #print "\nfeasible_subzones\n", feasible_subzones

    logger.info("do_simul_integerizing %s infeasable subzones for %s. " % (len(infeasible_subzones), trace_label))

    if len(infeasible_subzones) > 0:

        sub_controls_df = sub_controls_df.loc[feasible_subzones.index]
        sub_weights = sub_weights[feasible_subzones]
        sub_control_zones = feasible_subzones

        status,  integerized_weights_df = try_simul_integerizing(
            incidence_df,
            parent_weights, parent_controls_df,
            sub_weights, sub_controls_df,
            parent_geography, parent_id,
            sub_geography,
            control_spec, total_hh_control_col,
            sub_control_zones)

        print "status from fallback try_simul_integerizing", status

        assert status in STATUS_SUCCESS

        if status in STATUS_SUCCESS:

            logger.warn("do_simul_integerizing retry succeeded for %s status %s. " % (trace_label, status))

            integerized_weights_df = pd.concat([integerized_weights_df, rounded_weights_df])
            return integerized_weights_df

        logger.error("do_simul_integerizing retry failed for %s status %s. " % (trace_label, status))

    logger.warn("do_simul_integerizing %s falling back to sequential integerizing." % (trace_label, ))

    # sequentially integerize the (supposedly) feasible_subzones since they failed simul_integerizing
    integerized_weights_df = do_sequential_integerizing(
        incidence_df,
        sub_weights, sub_controls_df,
        control_spec, total_hh_control_col,
        sub_control_zones,
        sub_geography,
        parent_geography,
        parent_id)

    integerized_weights_df = pd.concat([integerized_weights_df, rounded_weights_df])
    return integerized_weights_df
