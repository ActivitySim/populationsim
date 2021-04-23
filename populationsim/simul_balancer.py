
# PopulationSim
# See full license in LICENSE.txt.

from builtins import zip
from builtins import range
from builtins import object
import logging
import numpy as np

import pandas as pd

from activitysim.core.config import setting

logger = logging.getLogger(__name__)

DEFAULT_MAX_ITERATIONS = 1000

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
    """
    Dual-zone simultaneous list balancer using Newton-Raphson method with control relaxation.

    Simultaneously balances the household weights across multiple subzones of a parent zone,
    ensuring that the total weight of each household across sub-zones sums to the parent hh weight.

    The resulting weights are float weights, so need to be integerized to integer household weights
    """

    def __init__(self,
                 incidence_table,
                 parent_weights,
                 controls,
                 sub_control_zones,
                 total_hh_control_col):
        """

        Parameters
        ----------
        incidence_table : pandas DataFrame
            incidence table with only columns for sub contols
        parent_weights : pandas: Series
            parent zone balanced (possibly integerized) aggregate target weights
        controls : pandas DataFrame
            parent zone controls
            one row per control,
            columns : name, importance, total + one column per sub_zone
        sub_control_zones : pandas.Series
            index is zone id and value is zone label (e.g. TAZ_101)
            for use in sub_controls_df column names
        total_hh_control_col : str
            name of the total_hh control column
        """
        assert isinstance(incidence_table, pd.DataFrame)
        assert len(parent_weights.index) == len(incidence_table.index)
        assert len(incidence_table.columns) == len(controls.index)

        assert 'total' in controls
        assert 'importance' in controls

        # remove zero weight rows
        # remember series so we can add zero weight rows back into result after balancing
        self.positive_weight_rows = parent_weights > 0
        logger.debug("%s positive weight rows out of %s"
                     % (self.positive_weight_rows.sum(), len(incidence_table.index)))

        self.incidence_table = incidence_table[self.positive_weight_rows]
        self.weights = pd.DataFrame({'parent': parent_weights[self.positive_weight_rows]})

        self.controls = controls
        self.sub_control_zones = sub_control_zones

        self.total_hh_control_col = total_hh_control_col
        self.master_control_index = self.incidence_table.columns.get_loc(total_hh_control_col)

    def balance(self):

        assert len(self.incidence_table.columns) == len(self.controls.index)
        assert len(self.weights.index) == len(self.incidence_table.index)

        self.weights['upper_bound'] = self.weights['parent']
        if 'lower_bound' not in self.weights:
            self.weights['lower_bound'] = 0.0

        # set initial sub zone weights proportionate to number of households
        total_hh_controls = self.controls.iloc[self.master_control_index]
        total_hh = int(total_hh_controls['total'])
        sub_zone_hh_fractions = total_hh_controls[self.sub_control_zones] / total_hh
        for zone, zone_name in list(self.sub_control_zones.items()):
            self.weights[zone_name] = self.weights['parent'] * sub_zone_hh_fractions[zone_name]

        self.controls['total'] = np.maximum(self.controls['total'], MIN_CONTROL_VALUE)

        # control relaxation importance weights (higher weights result in lower relaxation factor)
        self.controls['importance'] = np.maximum(self.controls['importance'], MIN_IMPORTANCE)

        # prepare inputs as numpy  (no pandas)
        sample_count = len(self.incidence_table.index)
        control_count = len(self.incidence_table.columns)
        zone_count = len(self.sub_control_zones)

        master_control_index = self.master_control_index
        incidence = self.incidence_table.values.transpose().astype(np.float64)

        # FIXME - do we also need sample_weights? (as the spec suggests?)
        parent_weights = np.asanyarray(self.weights['parent']).astype(np.float64)

        weights_lower_bound = np.asanyarray(self.weights['lower_bound']).astype(np.float64)
        weights_upper_bound = np.asanyarray(self.weights['upper_bound']).astype(np.float64)

        parent_controls = np.asanyarray(self.controls['total']).astype(np.float64)
        controls_importance = np.asanyarray(self.controls['importance']).astype(np.float64)

        sub_controls = self.controls[self.sub_control_zones].values.astype('float').transpose()
        sub_weights = self.weights[self.sub_control_zones].values.astype('float').transpose()

        # balance
        weights_final, relaxation_factors, status = np_simul_balancer(
            sample_count,
            control_count,
            zone_count,
            master_control_index,
            incidence,
            parent_weights,
            weights_lower_bound,
            weights_upper_bound,
            sub_weights,
            parent_controls,
            controls_importance,
            sub_controls)

        # dataframe with sub_zone_weights in columns, and zero weight rows restored
        self.sub_zone_weights = pd.DataFrame(index=self.positive_weight_rows.index)
        for i, c in zip(list(range(len(self.sub_control_zones))), self.sub_control_zones):
            self.sub_zone_weights[c] = pd.Series(weights_final[i], self.weights.index)
        self.sub_zone_weights.fillna(value=0.0, inplace=True)

        # series mapping zone_id to column names
        self.sub_zone_ids = self.sub_control_zones.index.values

        # dataframe with relaxation factors for each control in columns and one row per subzone
        self.relaxation_factors = pd.DataFrame(
            data=relaxation_factors,
            columns=self.controls.name,
            index=self.sub_control_zones.index)

        self.status = status

        return self.status


def np_simul_balancer(
        sample_count,
        control_count,
        zone_count,
        master_control_index,
        incidence,
        parent_weights,
        weights_lower_bound,
        weights_upper_bound,
        sub_weights,
        parent_controls,
        controls_importance,
        sub_controls):
    """
        Simultaneous balancer using only numpy (no pandas) data types.
        Separate function to ensure that no pandas data types leak in from object instance variables
        since they are often silently accepted as numpy arguments but slow things down
    """

    logger.debug("np_simul_balancer sample_count %s control_count %s zone_count %s" %
                 (sample_count, control_count, zone_count))

    # initial relaxation factors
    relaxation_factors = np.ones((zone_count, control_count))

    # Note: importance_adjustment must always be a float to ensure
    # correct "true division" in both Python 2 and 3
    importance_adjustment = 1.0

    # FIXME - make a copy as we change this (not really necessary as caller doesn't use it...)
    sub_weights = sub_weights.copy()

    # array of control indexes for iterating over controls
    control_indexes = list(range(control_count))
    if master_control_index is not None:
        # reorder indexes so we handle master_control_index last
        control_indexes.append(control_indexes.pop(master_control_index))

    # precompute incidence squared
    incidence2 = incidence * incidence

    max_iterations = setting('MAX_BALANCE_ITERATIONS_SIMULTANEOUS', DEFAULT_MAX_ITERATIONS)
    for iter in range(max_iterations):

        weights_previous = sub_weights.copy()

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

                xx = (sub_weights[z] * incidence[c]).sum()

                # calculate constraint balancing factors, gamma
                if xx > 0:
                    yy = (sub_weights[z] * incidence2[c]).sum()
                    relaxed_constraint = sub_controls[z, c] * relaxation_factors[z, c]
                    relaxed_constraint = max(relaxed_constraint, MIN_CONTROL_VALUE)
                    gamma[z, c] = 1.0 - float(xx - relaxed_constraint) / (
                        yy + (relaxed_constraint / float(importance)))

                # update HH weights
                sub_weights[z] *= pow(gamma[z, c], incidence[c])

                # clip weights to upper and lower bounds
                sub_weights[z] = np.clip(sub_weights[z], weights_lower_bound, weights_upper_bound)

                relaxation_factors[z, c] *= pow(1.0 / gamma[z, c], 1.0 / importance)

                # clip relaxation_factors
                relaxation_factors[z] = np.minimum(relaxation_factors[z], MAX_RELAXATION_FACTOR)

        # FIXME - can't rescale weights and expect to converge
        # FIXME - also zero weight hh should have been sliced out?
        # rescale sub_weights so weight of each hh across sub zones sums to parent_weight
        scale = parent_weights / np.sum(sub_weights, axis=0)
        # FIXME - what to do for rows where sum(sub_weights) are zero?
        scale = np.nan_to_num(scale)
        sub_weights *= scale

        max_gamma_dif = np.absolute(gamma - 1).max()
        assert not np.isnan(max_gamma_dif)

        # ensure float division
        delta = np.absolute(sub_weights - weights_previous).sum() / float(sample_count)
        assert not np.isnan(delta)

        # standard convergence criteria
        converged = delta < MAX_DELTA and max_gamma_dif < MAX_GAMMA

        # even if not converged, no point in further iteration if weights aren't changing
        no_progress = delta < ALT_MAX_DELTA

        if (iter % 100) == 0:
            logger.debug("np_simul_balancer iteration %s delta %s max_gamma_dif %s" %
                         (iter, delta, max_gamma_dif))

        if converged or no_progress:
            break

    status = {
        'converged': converged,
        'iter': iter,
        'delta': delta,
        'max_gamma_dif': max_gamma_dif
    }

    return sub_weights, relaxation_factors, status
