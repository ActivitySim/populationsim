
# PopulationSim
# See full license in LICENSE.txt.

import logging

from activitysim.core.config import setting
from . import lp_cvx
from . import lp_ortools


STATUS_OPTIMAL = 'OPTIMAL'
STATUS_FEASIBLE = 'FEASIBLE'
STATUS_SUCCESS = [STATUS_OPTIMAL, STATUS_FEASIBLE]


def use_cvxpy():

    return setting('USE_CVXPY', False)


def get_single_integerizer():
    """
    Return single integerizer function using installed/configured Linear Programming library.

    Different LP packages can be used for integerization (e.g. ortools of cvx) and this function
    hides the specifics of the individual packages so they can be swapped with minimal impact.

    Returns
    -------
    integerizer_func : function pointer to single integerizer function with call signature:

        def np_integerizer(
            incidence,
            resid_weights,
            log_resid_weights,
            control_importance_weights,
            total_hh_control_index,
            lp_right_hand_side,
            relax_ge_upper_bound,
            hh_constraint_ge_bound)

    """

    if use_cvxpy():
        integerizer_func = lp_cvx.np_integerizer_cvx
    else:
        integerizer_func = lp_ortools.np_integerizer_ortools

    return integerizer_func


def get_simul_integerizer():
    """
    Return simul-integerizer function using installed/configured Linear Programming library.

    Different LP packages can be used for integerization (e.g. ortools of cvx) and this function
    hides the specifics of the individual packages so they can be swapped with minimal impact.

    Returns
    -------
    integerizer_func : function pointer to simul-integerizer function with call signature:

    def np_simul_integerizer(
            sub_int_weights,
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

    """

    if use_cvxpy():
        integerizer_func = lp_cvx.np_simul_integerizer_cvx
    else:
        integerizer_func = lp_ortools.np_simul_integerizer_ortools

    return integerizer_func
