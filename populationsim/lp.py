# PopulationSim
# See full license in LICENSE.txt.

import logging

from util import setting
import lp_cvx
import lp_ortools


STATUS_OPTIMAL = 'OPTIMAL'
STATUS_FEASIBLE = 'FEASIBLE'
STATUS_SUCCESS = [STATUS_OPTIMAL, STATUS_FEASIBLE]


def use_cvxpy():

    return setting('USE_CVXPY', False)


def get_single_integerizer():

    if use_cvxpy():
        integerizer_func = lp_cvx.np_integerizer_cvx
    else:
        integerizer_func = lp_ortools.np_integerizer_ortools

    return integerizer_func


def get_simul_integerizer():
    if use_cvxpy():
        integerizer_func = lp_cvx.np_simul_integerizer_cvx
    else:
        integerizer_func = lp_ortools.np_simul_integerizer_ortools

    return integerizer_func
