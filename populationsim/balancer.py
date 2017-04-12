# PopulationSim
# See full license in LICENSE.txt.

import numpy as np

import pandas as pd

MAX_ITERATIONS = 10000

MAX_GAP = 1.0e-9
IMPORTANCE_ADJUST = 2
IMPORTANCE_ADJUST_COUNT = 100
MINIMUM_IMPORTANCE = 1.0
MAXIMUM_RELAXATION_FACTOR = 1000000
MIN_CONTROL_VALUE = 0.1
MAX_INT = (1 << 31)


def list_balancer(incidence_table,
                  control_totals,
                  initial_weights,
                  control_importance_weights=None,
                  lb_weights=None,
                  ub_weights=None,
                  master_control_index=None,
                  max_iterations=MAX_ITERATIONS
                  ):

    sample_count = len(incidence_table.index)
    control_count = len(incidence_table.columns)

    weights = pd.DataFrame(index=incidence_table.index)
    weights['initial'] = initial_weights
    weights['lower_bound'] = lb_weights if lb_weights is not None else 0.0
    weights['upper_bound'] = ub_weights if ub_weights is not None else MAX_INT

    # one row for every column in incidenceTable
    controls = pd.DataFrame(index=range(control_count))

    # assign incidence_table column names to corresponding control rows (informational)
    controls['name'] = incidence_table.columns.tolist()

    controls['constraint'] = control_totals
    controls.constraint = np.maximum(controls.constraint, MIN_CONTROL_VALUE)

    # initial relaxation factors
    controls['relaxation_factor'] = 1.0

    # control relaxation importance weights (higher weights result in lower relaxation factor)
    if control_importance_weights is None:
        controls['importance'] = min(1, MINIMUM_IMPORTANCE)
    else:
        controls['importance'] = np.maximum(control_importance_weights, MINIMUM_IMPORTANCE)

    # indices of active controls
    control_cols = controls.index.tolist()
    if master_control_index is not None:
        control_cols.append(control_cols.pop(master_control_index))

    weights_final = np.asanyarray(weights['initial']).astype(np.float64)
    incidence = incidence_table.as_matrix().transpose()
    incidence2 = incidence * incidence
    relaxation_factors = controls.relaxation_factor.values

    importance_adjustment = 1.0

    for iter in range(max_iterations):

        weights_previous = weights_final.copy()

        # reset gamma every iteration
        gamma = np.array([1.0] * control_count)

        # importance adjustment as number of iterations progress
        if iter > 0 and iter % IMPORTANCE_ADJUST_COUNT == 0:
            importance_adjustment = importance_adjustment / IMPORTANCE_ADJUST

        # for each control
        for c in control_cols:

            xx = (weights_final * incidence[c]).sum()
            yy = (weights_final * incidence2[c]).sum()

            # adjust importance (unless this is master_control)
            if c == master_control_index:
                importance = controls.importance[c]
            else:
                importance = max(controls.importance[c] * importance_adjustment, MINIMUM_IMPORTANCE)

            # calculate constraint balancing factors, gamma
            if xx > 0:
                relaxed_constraint = controls.constraint[c] * relaxation_factors[c]
                relaxed_constraint = max(relaxed_constraint, MIN_CONTROL_VALUE)
                gamma[c] = 1.0 - (xx - relaxed_constraint) / (yy + relaxed_constraint / importance)

            # update HH weights
            weights_final[incidence[c] > 0] *= gamma[c]

            # clip weights to upper and lower bounds
            weights_final = np.clip(weights_final, weights.lower_bound, weights.upper_bound)

            relaxation_factors[c] *= pow(1.0 / gamma[c], 1.0 / importance)

        # clip relaxation_factors
            relaxation_factors = np.minimum(relaxation_factors, MAXIMUM_RELAXATION_FACTOR)

        max_gamma_dif = np.absolute(gamma - 1).max()

        delta = (weights_final - weights_previous).abs().sum() / sample_count

        converged = delta < MAX_GAP and max_gamma_dif < MAX_GAP

        if converged:
            break

    weights['final'] = weights_final
    controls.relaxation_factor = relaxation_factors

    # convenient values
    controls['relaxed_constraint'] = controls.constraint * controls.relaxation_factor
    controls['weighted_sum'] = \
        [round((incidence_table.ix[:, c] * weights.final).sum(), 2) for c in controls.index]

    weights = weights[['initial', 'final']]
    controls = \
        controls[['name', 'constraint', 'relaxed_constraint', 'relaxation_factor', 'weighted_sum']]

    status = {
        'converged': converged,
        'iter': iter,
        'delta': delta,
        'max_gamma_dif': max_gamma_dif,
    }

    return weights, controls, status
