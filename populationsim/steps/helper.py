# PopulationSim
# See full license in LICENSE.txt.

import logging
import orca

def control_table_name(geography):
    return '%s_controls' % geography

def get_control_table(geography):

    return orca.get_table(control_table_name(geography)).to_frame()
