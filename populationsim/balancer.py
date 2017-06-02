# PopulationSim
# See full license in LICENSE.txt.

import logging
import numpy as np

import pandas as pd

from doppel import balance_cvx
from doppel import balance_multi_cvx


logger = logging.getLogger(__name__)

MAX_ITERATIONS = 10000

MAX_GAP = 1.0e-9

IMPORTANCE_ADJUST = 2
IMPORTANCE_ADJUST_COUNT = 100
MIN_IMPORTANCE = 1.0
MAX_RELAXATION_FACTOR = 1000000
MIN_CONTROL_VALUE = 0.1
MAX_INT = (1 << 31)


class ListBalancer(object):

    def __init__(self,
                 incidence_table,
                 initial_weights,
                 control_totals,
                 control_importance_weights,
                 lb_weights,
                 ub_weights,
                 master_control_index):

        assert isinstance(incidence_table, pd.DataFrame)

        self.incidence_table = incidence_table

        assert len(initial_weights) == len(self.incidence_table.index)

        self.control_totals = control_totals
        self.initial_weights = initial_weights
        self.control_importance_weights = control_importance_weights
        self.lb_weights = lb_weights
        self.ub_weights = ub_weights
        self.master_control_index = master_control_index

        assert len(self.incidence_table.columns) == len(self.control_totals)

    def balance(self):

        assert len(self.incidence_table.columns) == len(self.control_totals)
        assert \
            np.isscalar(self.control_importance_weights) \
            or len(self.incidence_table.columns) == len(self.control_importance_weights)

        # default values
        if self.control_importance_weights is None:
            self.control_importance_weights = min(1, MIN_IMPORTANCE)
        if self.lb_weights is None:
            self.lb_weights = 0.0
        if self.ub_weights is None:
            self.ub_weights = MAX_INT

        # prepare inputs as numpy (no pandas)
        sample_count = len(self.incidence_table.index)
        control_count = len(self.incidence_table.columns)
        master_control_index = self.master_control_index
        incidence = self.incidence_table.as_matrix().transpose()
        weights_initial = np.asanyarray(self.initial_weights).astype(np.float64)
        weights_lower_bound = np.asanyarray(self.lb_weights).astype(np.float64)
        weights_upper_bound = np.asanyarray(self.ub_weights).astype(np.float64)
        controls_constraint = np.maximum(self.control_totals, MIN_CONTROL_VALUE)
        controls_importance = np.maximum(self.control_importance_weights, MIN_IMPORTANCE)

        # balance
        weights_final, relaxation_factors, status = np_balancer(
            sample_count,
            control_count,
            master_control_index,
            incidence,
            weights_initial,
            weights_lower_bound,
            weights_upper_bound,
            controls_constraint,
            controls_importance)

        # weights dataframe
        weights = pd.DataFrame(index=self.incidence_table.index)
        weights['initial'] = self.initial_weights
        weights['final'] = weights_final

        # controls dataframe
        controls = pd.DataFrame(index=self.incidence_table.columns.tolist())
        controls['control'] = np.maximum(self.control_totals, MIN_CONTROL_VALUE)
        controls['relaxation_factor'] = relaxation_factors
        controls['relaxed_control'] = controls.control * relaxation_factors
        controls['weight_totals'] = \
            [round((self.incidence_table.ix[:, c] * weights['final']).sum(), 2)
             for c in controls.index]

        return status, weights, controls


def np_balancer(
        sample_count,
        control_count,
        master_control_index,
        incidence,
        weights_initial,
        weights_lower_bound,
        weights_upper_bound,
        controls_constraint,
        controls_importance):

    # initial relaxation factors
    relaxation_factors = np.repeat(1.0, control_count)
    importance_adjustment = 1.0

    # make a copy as we change this
    weights_final = weights_initial.copy()

    # array of control indexes for iterating over controls
    control_indexes = range(control_count)
    if master_control_index is not None:
        # reorder indexes so we handle master_control_index last
        control_indexes.append(control_indexes.pop(master_control_index))

    # precompute incidence squared
    incidence2 = incidence * incidence

    for iter in range(MAX_ITERATIONS):

        weights_previous = weights_final.copy()

        # reset gamma every iteration
        gamma = np.array([1.0] * control_count)

        # importance adjustment as number of iterations progress
        if iter > 0 and iter % IMPORTANCE_ADJUST_COUNT == 0:
            importance_adjustment = importance_adjustment / IMPORTANCE_ADJUST

        # for each control
        for c in control_indexes:

            xx = (weights_final * incidence[c]).sum()
            yy = (weights_final * incidence2[c]).sum()

            # adjust importance (unless this is master_control)
            if c == master_control_index:
                importance = controls_importance[c]
            else:
                importance = max(controls_importance[c] * importance_adjustment,
                                 MIN_IMPORTANCE)

            # calculate constraint balancing factors, gamma
            if xx > 0:
                relaxed_constraint = controls_constraint[c] * relaxation_factors[c]
                relaxed_constraint = max(relaxed_constraint, MIN_CONTROL_VALUE)
                gamma[c] = 1.0 - (xx - relaxed_constraint) / (
                    yy + relaxed_constraint / importance)

            # update HH weights
            weights_final[incidence[c] > 0] *= gamma[c]

            # clip weights to upper and lower bounds
            weights_final = np.clip(weights_final, weights_lower_bound, weights_upper_bound)

            relaxation_factors[c] *= pow(1.0 / gamma[c], 1.0 / importance)

            # clip relaxation_factors
            relaxation_factors = np.minimum(relaxation_factors, MAX_RELAXATION_FACTOR)

        max_gamma_dif = np.absolute(gamma - 1).max()

        # delta = (weights_final - weights_previous).abs().sum() / sample_count
        delta = np.absolute(weights_final - weights_previous).sum() / sample_count

        converged = delta < MAX_GAP and max_gamma_dif < MAX_GAP

        #logger.debug("iter %s delta %s max_gamma_dif %s" % (iter, delta, max_gamma_dif))

        if converged:
            break

    status = {
        'converged': converged,
        'iter': iter,
        'delta': delta,
        'max_gamma_dif': max_gamma_dif,
    }

    return weights_final, relaxation_factors, status


def do_seed_balancing(seed_geography, seed_control_spec, seed_id, total_hh_control_col, max_expansion_factor,
                  incidence_df, seed_controls_df):

    # slice incidence rows for this seed geography
    incidence_df = incidence_df[incidence_df[seed_geography] == seed_id]

    # initial hh weights
    initial_weights = incidence_df['sample_weight']

    # incidence table should only have control columns
    incidence_df = incidence_df[seed_control_spec.target]

    control_totals = seed_controls_df.loc[seed_id].values

    control_importance_weights = seed_control_spec.importance

    # master_control_index is total_hh_control_col
    if total_hh_control_col not in incidence_df.columns:
        #print incidence_df.columns
        raise RuntimeError("total_hh_control column '%s' not found in incidence table"
                           % total_hh_control_col)
    total_hh_control_index = incidence_df.columns.get_loc(total_hh_control_col)

    lb_weights = 0
    if max_expansion_factor:

        # number_of_households in this seed geograpy as specified in seed_controlss
        number_of_households = control_totals[total_hh_control_index]

        total_weights = initial_weights.sum()
        ub_ratio = max_expansion_factor * float(number_of_households) / float(total_weights)

        ub_weights = initial_weights * ub_ratio
        ub_weights = ub_weights.round().clip(lower=1).astype(int)

    else:
        ub_weights = None

    balancer = ListBalancer(
        incidence_table=incidence_df,
        initial_weights=initial_weights,
        lb_weights=lb_weights,
        ub_weights=ub_weights,
        control_totals=control_totals,
        control_importance_weights=control_importance_weights,
        master_control_index=total_hh_control_index
    )

    status, weights, controls = balancer.balance()

    return status, weights, controls


def do_seed_balancing_cvx(seed_geography, seed_control_spec, seed_id, total_hh_control_col, max_expansion_factor,
                  incidence_df, seed_controls_df):

    #seed_control_spec = seed_control_spec.head()

    #incidence_df = incidence_df.head(500)

    seed_controls_df = seed_controls_df[seed_control_spec.target]

    total_hh_control_index = incidence_df.columns.get_loc(total_hh_control_col)

    # slice incidence rows for this seed geography
    incidence_df = incidence_df[incidence_df[seed_geography] == seed_id]

    positive_weight_rows = incidence_df['sample_weight'] > 0
    incidence_df = incidence_df[positive_weight_rows]

    control_totals = seed_controls_df.loc[seed_id]
    control_importance_weights = seed_control_spec.importance_cvx

    print "control_totals\n", control_totals

    # initial hh weights as pandas series
    initial_weights = incidence_df['sample_weight']

    # FIXME - scale the initial weights
    initial_weights = initial_weights * (initial_weights.sum()/control_totals[total_hh_control_index])


    # incidence table should only have control columns
    incidence = incidence_df[seed_control_spec.target]

    hh_table = np.mat(incidence.as_matrix())
    A = np.mat(control_totals)
    w = np.mat(initial_weights)
    mu = np.mat(control_importance_weights)

    # Control importance weights
    # < 1 means not important (thus relaxing the constraint in the solver)
    #mu = np.mat([1] * n_controls)

    # Our trade-off coefficient gamma
    # Low values (~1) mean we trust our initial weights, high values
    # (~10000) mean want to fit the marginals.
    gamma = 1.


    print "hh_table %s %s" % (type(hh_table), hh_table.shape)
    print "A %s %s" % (type(A), A.shape)
    print "w %s %s" % (type(w), w.shape)
    print "mu %s %s" % (type(mu), mu.shape)

    # print "hh_table\n", hh_table
    # print "A\n", A
    # print "w\n", w
    print "mu\n", mu

    # weights_out, zs_out, qs_out = balance_multi_cvx(
    #     hh_table, A, B, w_extend, gamma * mu_extend.T, meta_gamma
    # )

    weights_out, zs_out = balance_cvx(
        hh_table, A, w, gamma * mu, total_hh_control_index, verbose_solver=True
    )

    print "weights_out", weights_out.shape
    print "zs_out", zs_out.shape, zs_out

    balanced_weights = pd.Series(0.0, index=positive_weight_rows.index)
    balanced_weights[incidence_df.index] = weights_out.A1

    relaxation_factors = pd.Series(zs_out.A1, index=incidence.columns.tolist())

    return balanced_weights, relaxation_factors


