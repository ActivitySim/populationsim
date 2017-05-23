# PopulationSim
# See full license in LICENSE.txt.

import logging
import numpy as np

import pandas as pd


logger = logging.getLogger(__name__)

MAX_ITERATIONS = 10000

MAX_DELTA = 1.0e-9
MAX_GAMMA = 1.0e-7

# delta to check for non-convergence without progress
ALT_MAX_DELTA = 1.0e-10

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
                 seed_id,
                 sub_zone,
                 master_control_index):

        assert isinstance(incidence_table, pd.DataFrame)
        assert len(weights.index) == len(incidence_table.index)
        assert len(incidence_table.columns) == len(controls.index)

        self.incidence_table = incidence_table
        self.weights = weights
        self.controls = controls
        self.sub_control_zones = sub_control_zones

        # these are all just to label result
        self.seed_id = seed_id
        self.sub_zone = sub_zone
        self.master_control_index = master_control_index

        assert 'total' in self.controls
        assert 'aggregate_target' in self.weights
        # FIXME - do we also need sample_weights? (as the spec suggests?)

    def balance(self):

        assert len(self.incidence_table.columns) == len(self.controls.index)
        assert len(self.weights.index) == len(self.incidence_table.index)

        # default values
        if 'importance' not in self.controls:
            self.controls['importance]'] = min(1, MIN_IMPORTANCE)
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

        # FIXME - do we also need sample_weights? (as the spec suggests?)
        weights_agg_target = np.asanyarray(self.weights['aggregate_target']).astype(np.float64)

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
            weights_agg_target,
            weights_lower_bound,
            weights_upper_bound,
            sub_weights,
            controls_total,
            controls_importance,
            sub_controls)

        # save results in convenient form for sub_zone balancing and debugging
        self.sub_zone_weights = pd.DataFrame(
            data=weights_final.transpose(),
            columns=self.sub_control_zones,
            index=self.incidence_table.index
        )

        # save results in convenient form for final result processing
        sub_zone_col_name = self.sub_zone
        self.results = pd.DataFrame(
            {'seed_id': self.seed_id,
             sub_zone_col_name: np.repeat(self.sub_control_zones.index.tolist(), sample_count),
             'hh_id': np.tile(np.asanyarray(self.incidence_table.index), zone_count),
             'initial_weight': np.asanyarray(sub_weights).flatten(),
             'final_weight': np.asanyarray(weights_final).flatten()
             }
        )
        cols = ['seed_id', sub_zone_col_name, 'hh_id', 'initial_weight', 'final_weight']
        self.results = self.results[cols]

        self.relaxation_factors = relaxation_factors
        self.status = status

        return self.status


def np_simul_balancer(
        sample_count,
        control_count,
        zone_count,
        master_control_index,
        incidence,
        weights_aggregate_target,
        weights_lower_bound,
        weights_upper_bound,
        sub_weights,
        controls_total,
        controls_importance,
        sub_controls):

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
                importance = max(controls_importance[c] * importance_adjustment, MIN_IMPORTANCE)

            for z in range(zone_count):

                xx = (weights_final[z] * incidence[c]).sum()
                yy = (weights_final[z] * incidence2[c]).sum()

                # calculate constraint balancing factors, gamma
                if xx > 0:
                    relaxed_constraint = sub_controls[z, c] * relaxation_factors[z, c]
                    relaxed_constraint = max(relaxed_constraint, MIN_CONTROL_VALUE)
                    gamma[z, c] \
                        = 1.0 - (xx - relaxed_constraint) / (yy + relaxed_constraint / importance)

                # update HH weights
                weights_final[z][incidence[c] > 0] *= gamma[z, c]

                # clip weights to upper and lower bounds
                weights_final[z] = np.clip(weights_final[z],
                                           weights_lower_bound, weights_upper_bound)

                relaxation_factors[z, c] *= pow(1.0 / gamma[z, c], 1.0 / importance)

                # clip relaxation_factors
                relaxation_factors[z] = np.minimum(relaxation_factors[z], MAX_RELAXATION_FACTOR)

        bug
        # FIXME - can't rescale weights and expect to converge
        # FIXME - also zero weight hh should have been sliced out?

        # rescale weights to sum to weights_seed
        scale = weights_aggregate_target / np.sum(weights_final, axis=0)
        # FIXME - what to do for rows where sum(weights_final) are zero?
        scale = np.nan_to_num(scale)
        weights_final *= scale

        max_gamma_dif = np.absolute(gamma - 1).max()
        assert not np.isnan(max_gamma_dif)

        delta = np.absolute(weights_final - weights_previous).sum() / sample_count
        assert not np.isnan(delta)

        logger.debug("iter %s delta %s max_gamma_dif %s" % (iter, delta, max_gamma_dif))

        # standard convergence criteria
        converged = delta < MAX_DELTA and max_gamma_dif < MAX_GAMMA

        # even if not converged, no point in further iteration if weights aren't changing
        no_progress = delta < ALT_MAX_DELTA

        if converged or no_progress:
            break

    status = {
        'converged': converged,
        'iter': iter,
        'delta': delta,
        'max_gamma_dif': max_gamma_dif
    }

    return weights_final, relaxation_factors, status
