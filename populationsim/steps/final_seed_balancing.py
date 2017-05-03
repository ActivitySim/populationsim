# PopulationSim
# See full license in LICENSE.txt.

import logging
import os

import orca
import pandas as pd
import numpy as np

from activitysim.core import assign
from ..balancer import ListBalancer

logger = logging.getLogger(__name__)

def dump_table(table_name, table):

    print "\ntable_name\n", table


@orca.step()
def final_seed_balancing(settings, geo_cross_walk, control_spec, incidence_table, final_seed_controls):

    pass
