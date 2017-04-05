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


# FIXME - not supporting tazTotalHouseholdsControlIndex

def list_balancer(incidence_table,
                  constraints,
                  initial_weights,
                  control_importance_weights=None,
                  lb_weights = None,
                  ub_weights = None,
                  master_control_index = None
                  ):

    # initial_weights, importance_weights, incidence_table, constraint_totals

    """

    Params
    incidence_table : pandas.DataFrame

    initial_weights :

    """
    sample_count = len(incidence_table.index)
    control_count = len(incidence_table.columns)

    weights = pd.DataFrame(index=incidence_table.index)
    weights['initial'] = initial_weights
    weights['lower_bound'] = lb_weights if lb_weights is not None else 0.0
    weights['upper_bound'] = ub_weights if ub_weights is not None else MAX_INT

    # one row for every column in incidenceTable
    controls = pd.DataFrame(index=range(control_count))

    controls['constraint'] = constraints
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

    weights['final'] = weights['initial']
    weights['previous'] = weights['initial']

    importance_adjustment  = 1.0

    for iter in range(MAX_ITERATIONS):

        #print "### iter", iter

        current = str(iter)
        weights.final = weights.previous

        # reset gamma every iteration
        gamma = np.array([1.0] * control_count)
        relaxation_factor = controls.relaxation_factor.values

        # importance adjustment as number of iterations progress
        if iter > 0 and iter % IMPORTANCE_ADJUST_COUNT == 0:
            importance_adjustment = importance_adjustment / IMPORTANCE_ADJUST

        # for each control
        for c in control_cols:

            # column from incidence table for this constraint
            incidence = incidence_table.ix[:, c]

            xx = (weights.final * incidence).sum()
            yy = (weights.final * incidence * incidence).sum()

            if c == master_control_index:
                # don't adjust importance of master_control
                importance = controls.importance[c]
            else:
                # importance of this constraint, clipped
                importance = max(controls.importance[c] * importance_adjustment, MINIMUM_IMPORTANCE)

            # calculate constraint balancing factors, gamma
            if xx > 0:
                relaxed_constraint = controls.constraint[c] * relaxation_factor[c]
                gamma[c] = 1.0 - (xx - relaxed_constraint) / (yy + relaxed_constraint / importance)

                # constraint = controls.constraint[c]
                # gamma[c] = 1.0 - (xx - constraint) / (yy + constraint)

            # adjust weights of incident rows by gamma
            weights.ix[incidence > 0, 'final'] *= gamma[c]

            # clip weights to upper and lower bounds
            weights.final = np.clip(weights.final, weights.lower_bound, weights.upper_bound)

            weights[str(c)] = weights.final

            assert not (weights.final <= 0.0).any()

            relaxation_factor[c] *= pow(1.0 / gamma[c], 1.0 / importance)
            #print "gamma %s importance %s relation_factor %s" % (gamma[c], importance, relaxation_factor[c])

        # clip relaxation_factors
        controls.relaxation_factor = np.minimum(relaxation_factor, MAXIMUM_RELAXATION_FACTOR)

        max_gamma_dif = np.absolute(gamma - 1).max()

        delta = (weights.final - weights.previous).abs().sum() / sample_count

        weights.previous = weights.final

        # for debugging
        #weights[str(iter)] = weights.final

        if delta < MAX_GAP and max_gamma_dif < MAX_GAP:
            break

    print "final iter", iter
    print "final delta", delta


    print "\nweights\n", weights[['initial', 'final']]

    s = pd.DataFrame(index=controls.index)
    s['name'] = incidence_table.columns.tolist()
    s['constraint'] = controls.constraint
    #s['relaxation_factor'] = controls.relaxation_factor
    s['relaxed_constraint'] = controls.constraint * controls.relaxation_factor
    s['weighted_sum'] = [round((incidence_table.ix[:, c] * weights.final).sum(), 2) for c in s.index]

    published_final_weights = [1.36,25.66,7.98,27.79, 18.45,8.64,1.47,8.64]
    s['pub_weighted_sum'] = [round((incidence_table.ix[:, c] * published_final_weights).sum(), 2) for c in
                         s.index]

    print s


