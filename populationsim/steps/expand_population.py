# PopulationSim
# See full license in LICENSE.txt.

import logging

import orca
import pandas as pd
import numpy as np

from populationsim.util import setting
from helper import get_control_table
from helper import get_weight_table

from helper import weight_table_name

logger = logging.getLogger(__name__)

GROUP_BY_INCIDENCE_SIGNATURE = setting('GROUP_BY_INCIDENCE_SIGNATURE')


@orca.step()
def expand_population():

    geographies = setting('geographies')
    low_geography = geographies[-1]

    weights = get_weight_table(low_geography)


    print weights.head()

    assert False
