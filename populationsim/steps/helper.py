# PopulationSim
# See full license in LICENSE.txt.

import logging

from activitysim.core import pipeline
from activitysim.core import inject


def control_table_name(geography):
    return '%s_controls' % geography


def get_control_table(geography):
    return pipeline.get_table(control_table_name(geography))


def get_control_data_table(geography):
    control_data_table_name = '%s_control_data' % geography
    return pipeline.get_table(control_data_table_name)


def weight_table_name(geography, sparse=False):
    if sparse:
        return '%s_weights_sparse' % geography
    else:
        return '%s_weights' % geography


def get_weight_table(geography, sparse=False):
    name = weight_table_name(geography, sparse)
    weight_table = inject.get_table(name, default=None)
    if weight_table is not None:
        weight_table = weight_table.to_frame()
    return weight_table
