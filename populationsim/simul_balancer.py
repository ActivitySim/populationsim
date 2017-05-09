# PopulationSim
# See full license in LICENSE.txt.

import logging
import numpy as np

import pandas as pd

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 10000

MAX_DELTA = 1.0e-9
MAX_GAMMA = 1.0e-7

IMPORTANCE_ADJUST = 2
IMPORTANCE_ADJUST_COUNT = 100
MIN_IMPORTANCE = 1.0
MAX_RELAXATION_FACTOR = 1000000
MIN_CONTROL_VALUE = 0.1
MAX_INT = (1 << 31)


class SimultaneousListBalancer(object):

    def __init__(self,
                 incidence_table,
                 weights,
                 controls,
                 sub_control_zones,
                 master_control_index,
                 trace):

        assert isinstance(incidence_table, pd.DataFrame)
        assert len(weights.index) == len(incidence_table.index)
        assert len(incidence_table.columns) == len(controls.index)

        self.incidence_table = incidence_table
        self.weights = weights
        self.controls = controls
        self.sub_control_zones = sub_control_zones
        self.master_control_index = master_control_index
        self.trace = trace

        assert 'total' in self.controls
        assert 'initial_sample' in self.weights
        assert 'final_seed' in self.weights

    def balance(self):

        assert len(self.incidence_table.columns) == len(self.controls.index)
        assert len(self.weights.index) == len(self.incidence_table.index)

        # default values
        if 'importance' not in self.controls:
            self.controls['importance]'] =  min(1, MIN_IMPORTANCE)
        if 'lower_bound' not in self.weights:
            self.weights['lower_bound'] = 0.0
        if 'upper_bound' not in self.weights:
            self.weights['upper_bound'] = MAX_INT

        assert 'total' in self.controls


        self.controls['total'] = np.maximum(self.controls['total'], MIN_CONTROL_VALUE)

        # control relaxation importance weights (higher weights result in lower relaxation factor)
        self.controls['importance'] = np.maximum(self.controls['importance'], MIN_IMPORTANCE)

        # prepare inputs as numpy  (no pandas)
        sample_count = len(self.incidence_table.index)
        control_count = len(self.incidence_table.columns)
        zone_count = len(self.sub_control_zones)

        master_control_index = self.master_control_index
        incidence = self.incidence_table.as_matrix().transpose()
        weights_initial_sample = np.asanyarray(self.weights['initial_sample']).astype(np.float64)
        weights_final_seed = np.asanyarray(self.weights['final_seed']).astype(np.float64)

        weights_lower_bound = np.asanyarray(self.weights['lower_bound']).astype(np.float64)
        weights_upper_bound = np.asanyarray(self.weights['upper_bound']).astype(np.float64)

        controls_total = np.asanyarray(self.controls['total']).astype(np.float64)
        controls_importance = np.asanyarray(self.controls['importance']).astype(np.float64)

        sub_controls = self.controls[self.sub_control_zones].as_matrix().astype('float').transpose()
        sub_weights = self.weights[self.sub_control_zones].as_matrix().astype('float').transpose()

        # balance
        weights_final, relaxation_factors, status = np_simul_balancer(
            sample_count,
            control_count,
            zone_count,
            master_control_index,
            incidence,
            weights_initial_sample,
            weights_final_seed,
            weights_lower_bound,
            weights_upper_bound,
            sub_weights,
            controls_total,
            controls_importance,
            sub_controls,
            self.trace)

        # save results
        # self.weights_initial = sub_weights
        # self.weights_final = weights_final

        self.weights_initial = pd.DataFrame(
            data=sub_weights.transpose(),
            columns=self.sub_control_zones,
            index=self.incidence_table.index
        )
        self.weights_final = pd.DataFrame(
            data=weights_final.transpose(),
            columns=self.sub_control_zones,
            index=self.incidence_table.index
        )

        self.results = self.controls[['name']].copy()
        for zone, zone_name in self.sub_control_zones.iteritems():
            #self.results["%s_control"% zone_name] = self.controls[zone_name]
            x = [round((self.incidence_table.ix[:, c] * self.weights_final[zone_name]).sum(), 2)
                 for c in self.controls.index]
            self.results[zone_name] = x
        self.results['total'] = self.results[self.sub_control_zones.values].sum(axis=1)
        self.results = self.results[['total'] + self.sub_control_zones.tolist()]

        self.relaxation_factors = relaxation_factors
        self.status = status

        return self.status

    def print_status(self):

        print "status", self.status
        print "weights_initial\n", self.weights_initial
        print "weights_final\n", self.weights_final

        print "weights['final_seed']\n", self.weights['final_seed']
        print "row sum sub_weights\n", self.weights_final.sum(axis=1)

        print "weights['final_seed']", self.weights['final_seed'].sum()
        print "sum sub_weights", self.weights_final.values.sum()

        print "controls\n", self.controls
        print "results\n", self.results



def np_simul_balancer(
        sample_count,
        control_count,
        zone_count,
        master_control_index,
        incidence,
        weights_initial_sample,
        weights_final_seed,
        weights_lower_bound,
        weights_upper_bound,
        sub_weights,
        controls_total,
        controls_importance,
        sub_controls,
        trace):
    """
    Parameters
    ----------
    sample_count : int
    control_count : int
    zone_count : int
    master_control_index
    incidence
    weights_initial_sample
    weights_final_seed
    weights_lower_bound
    weights_upper_bound
    sub_weights
    controls_total
    controls_importance
    sub_controls
    trace: bool
    """

    # print "###################### np_simul_balancer - params"
    # print "sample_count", sample_count
    # print "control_count", control_count
    # print "zone_count", zone_count
    # print "master_control_index", master_control_index
    # print "weights_initial_sample", weights_initial_sample
    # print "weights_final_seed", weights_final_seed
    # print "######################"

    # initial relaxation factors
    relaxation_factors = np.ones((zone_count, control_count))

    importance_adjustment = 1.0

    # make a copy as we change this
    weights_final = sub_weights.copy()

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
        gamma = np.ones((zone_count, control_count))

        # importance adjustment as number of iterations progress
        if iter > 0 and iter % IMPORTANCE_ADJUST_COUNT == 0:
            importance_adjustment = importance_adjustment / IMPORTANCE_ADJUST

        # for each control
        for c in control_indexes:

            # adjust importance (unless this is master_control)
            if c == master_control_index:
                importance = controls_importance[c]
            else:
                importance = max(controls_importance[c] * importance_adjustment,
                                 MIN_IMPORTANCE)

            for z in range(zone_count):

                xx = (weights_final[z] * incidence[c]).sum()
                yy = (weights_final[z] * incidence2[c]).sum()

                # calculate constraint balancing factors, gamma
                if xx > 0:
                    relaxed_constraint = sub_controls[z, c] * relaxation_factors[z, c]
                    relaxed_constraint = max(relaxed_constraint, MIN_CONTROL_VALUE)
                    gamma[z, c] = 1.0 - (xx - relaxed_constraint) / (yy + relaxed_constraint / importance)

                # update HH weights
                weights_final[z][incidence[c] > 0] *= gamma[z, c]

                # clip weights to upper and lower bounds
                weights_final[z] = np.clip(weights_final[z], weights_lower_bound, weights_upper_bound)

                relaxation_factors[z, c] *= pow(1.0 / gamma[z, c], 1.0 / importance)

                # clip relaxation_factors
                relaxation_factors[z] = np.minimum(relaxation_factors[z], MAX_RELAXATION_FACTOR)

        # FIXME - avoid division by zero!
        # rescale weights to sum to weights_final_seed
        print "weights_final_seed", weights_final_seed
        print "weights_final", weights_final
        zzz = np.sum(weights_final, axis=0)
        print "np.sum(weights_final, axis=0)", type(zzz)
        print "np.sum(weights_final, axis=0).min()", zzz.min()
        scale = weights_final_seed/ np.sum(weights_final, axis=0)
        print "scale", scale
        weights_final *=  scale
        print "scaled weights_final", weights_final

        max_gamma_dif = np.absolute(gamma - 1).max()

        delta = np.absolute(weights_final - weights_previous).sum() / sample_count

        converged = delta < MAX_DELTA and max_gamma_dif < MAX_GAMMA

        logger.debug("iter %s delta %s max_gamma_dif %s" % (iter, delta, max_gamma_dif))

        assert False

        if converged:
            break

    status = {
        'converged': converged,
        'iter': iter,
        'delta': delta,
        'max_gamma_dif': max_gamma_dif,
    }


    return weights_final, relaxation_factors, status
