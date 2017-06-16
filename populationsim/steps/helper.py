# PopulationSim
# See full license in LICENSE.txt.

import logging
import orca

from activitysim.core import pipeline


def control_table_name(geography):
    return '%s_controls' % geography


def get_control_table(geography):
    return pipeline.get_table(control_table_name(geography))


def weight_table_name(geography, sparse=False):
    if sparse:
        return '%s_weights_sparse' % geography
    else:
        return '%s_weights' % geography


def get_weight_table(geography, sparse=False):
    name = weight_table_name(geography, sparse)
    if orca.is_table(name):
        return pipeline.get_table(name)
    else:
        return None
