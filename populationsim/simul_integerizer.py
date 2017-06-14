# PopulationSim
# See full license in LICENSE.txt.

import logging

import numpy as np
import pandas as pd


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

STATUS_SUCCESS = ['OPTIMAL', 'FEASIBLE', 'OPTIMAL_INACCURATE']


logger = logging.getLogger(__name__)


class SimulIntegerizer(object):

    def __init__(self,
                 incidence_table,
                 initial_weights,
                 controls,
                 sub_control_zones,
                 total_hh_control_col):
        """

        Parameters
        ----------
        incidence_table : pandas DataFrame
        initial_weights :
        controls : pandas DataFrame
            parent zone controls
            one row per control,
            columns : name, importance, total + one column per sub_zone
        sub_control_zones
        total_hh_control_col
        """
        assert isinstance(incidence_table, pd.DataFrame)
        assert len(initial_weights.index) == len(incidence_table.index)
        assert len(incidence_table.columns) == len(controls.index)

        assert 'total' in controls
        assert 'importance' in controls

        # remove zero weight rows
        # remember series so we can add zero weight rows back into result after balancing
        self.positive_weight_rows = initial_weights > 0
        logger.debug("%s positive weight rows out of %s"
                     % (self.positive_weight_rows.sum(), len(incidence_table.index)))

        self.incidence_table = incidence_table[self.positive_weight_rows]
        self.weights = \
            pd.DataFrame({'aggregate_target': initial_weights[self.positive_weight_rows]})

        self.controls = controls
        self.sub_control_zones = sub_control_zones

        self.total_hh_control_col = total_hh_control_col
        self.master_control_index = self.incidence_table.columns.get_loc(total_hh_control_col)

    def integerize(self):

        sample_count = len(self.incidence_table.index)
        control_count = len(self.incidence_table.columns)

        incidence = self.incidence_table.as_matrix().transpose()
        float_weights = np.asanyarray(self.float_weights).astype(np.float64)
        control_totals = np.asanyarray(self.control_totals).astype(np.int)
        relaxed_control_totals = np.asanyarray(self.relaxed_control_totals).astype(np.float64)
        control_is_hh_based = np.asanyarray(self.control_is_hh_based).astype(np.int)
        control_importance_weights = \
            np.asanyarray(self.control_importance_weights).astype(np.float64)

        assert len(float_weights) == sample_count
        assert len(control_totals) == control_count
        assert len(relaxed_control_totals) == control_count
        assert len(control_is_hh_based) == control_count
        assert len(self.incidence_table.columns) == control_count

        integerized_weights, status = np_integerizer_cvx(
            incidence=incidence,
            float_weights=float_weights,
            control_importance_weights=control_importance_weights,
            control_totals=control_totals,
            relaxed_control_totals=relaxed_control_totals,
            total_hh_control_index=self.total_hh_control_index,
        )

        self.weights = pd.DataFrame(index=self.incidence_table.index)
        self.weights['integerized_weight'] = integerized_weights

        delta = (integerized_weights != np.round(float_weights)).sum()
        logger.debug("Integerizer: %s out of %s different from round" % (delta, len(float_weights)))

        logger.debug("total_hh float %s int %s control %s\n" %
                     (float_weights.sum(), integerized_weights.sum(),
                      control_totals[self.total_hh_control_index]))

        return status


def np_integerizer_cvx(incidence,
                       float_weights,
                       control_importance_weights,
                       control_totals,
                       relaxed_control_totals,
                       total_hh_control_index):

    assert not np.isnan(incidence).any()
    assert not np.isnan(float_weights).any()

    if (float_weights == 0).any():
        # not sure this matters...
        logger.warn("np_integerizer_cvx: %s zero weights" % ((float_weights == 0).sum(), ))

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
        # try to match float_weights
        cvx.vec(x*incidence) <= resid_control_totals + relax_ge,
        cvx.vec(x*incidence) >= resid_control_totals - relax_le,
        ###
        # try to match controls
        # cvx.vec(x * incidence) - relax_le >= 0,
        # cvx.vec(x * incidence) - relax_le <= lp_right_hand_side,
        # cvx.vec(x * incidence) + relax_ge >= lp_right_hand_side,
        # cvx.vec(x * incidence) + relax_ge <= hh_constraint_ge_bound,
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
        cvx.sum_entries(x) >= total_hh_right_hand_side
    ]

    prob = cvx.Problem(objective, constraints)

    try:
        # - solver list: http://www.cvxpy.org/en/latest/tutorial/advanced/
        # cvx.installed_solvers(): ['ECOS_BB', 'SCS', 'ECOS', 'LS']
        # ['CVXOPT', 'ECOS_BB', 'GLPK_MI', 'SCS', 'ECOS', 'GLPK', 'LS']
        # prob.solve(solver=cvx.ECOS, verbose=True)

        print cvx.installed_solvers()
        prob.solve(solver=cvx.CBC, max_iters=10,  verbose=True)

    except cvx.SolverError:
        logging.exception(
            'Solver error encountered in weight discretization. Weights will be rounded.')

    if np.any(x.value):
        resid_weights_out = np.asarray(x.value)[0]
        relax_le_out = np.asarray(relax_le.value)
        relax_ge_out = np.asarray(relax_ge.value)
    else:
        resid_weights_out = resid_weights

    # Make results binary
    resid_weights_out = np.array(resid_weights_out > 0.5).astype(int)
    weights_out = int_weights + resid_weights_out

    return weights_out, CVX_STATUS[prob.status]


def do_simul_integerizing(
        incidence_df,
        parent_weights, parent_controls,
        sub_weights, sub_controls,
        control_spec, total_hh_control_col,
        sub_control_zones,
        parent_geography, sub_geography):
    """

    Parameters
    ----------
    incidence_df : pandas dataframe
        one row per sample, one column per control (plus other columns we dont' care about)
    parent_weights : pandas series
    parent_controls : pandas dataframe
        one row per control, columns: [name, importance, total]
    sub_weights :  pandas dataframe
        columns of sub zone balanced (float) weights to integerize
    sub_controls pandas dataframe
        table with one row per subzone, one column per control
    control_spec : pandas dataframe
    total_hh_control_col : str
    sub_control_zones : pandas series
        series mapping zone_id (index) to zone_name (value)
    parent_geography : str
    sub_geography : str

    Returns
    -------
        zone_weights_df: pandas dataframe
            [hh_id, <parent_geography>, <sub_geobraphy>, balanced_weight, integer_weight]

    """

    zero_weight_rows = parent_weights > 0
    if zero_weight_rows.any():
        logger.info("omitting %s zero weight rows out of %s"
                    % (zero_weight_rows.sum(), len(incidence_df.index)))
        incidence_df = incidence_df[~zero_weight_rows]

    print "incidence_df", incidence_df.shape

    sample_count = len(parent_weights.index)
    zone_count = len(sub_control_zones)
    # incidence_df = incidence_df[control_spec.target]

    parent_countrol_cols = list(parent_controls['name'])
    print "parent_countrol_cols\n", parent_countrol_cols

    sub_countrol_cols = list(sub_controls.columns)
    print "sub_countrol_cols", sub_countrol_cols

    assert len(sub_weights.index) == sample_count
    assert len(incidence_df.index) == sample_count
    assert len(sub_controls.index) == zone_count
    assert len(sub_weights.columns) == zone_count
    assert list(sub_weights.columns) == list(sub_control_zones)
    assert total_hh_control_col in sub_countrol_cols
    assert total_hh_control_col in parent_countrol_cols

    incidence_table = incidence_df[control_spec.target].as_matrix()
    print "incidence_table", incidence_table.shape

    sub_weights = sub_weights.as_matrix()
    print "sub_weights", sub_weights.shape

    print "sub_controls_df\n", sub_controls

    # zone_id, zone_name in sub_control_zones.iteritems()
    # control_totals=sub_controls.loc[zone_id],
    # sub_weights[zone_name]

    assert False

    integerized_weights = pd.Series(0.0, index=zero_weight_rows.index)
    integerized_weights.update(integerizer.weights['integerized_weight'])

    return integerized_weights, status
