import orca
from activitysim import tracing
import pandas as pd
import numpy as np
import os

# read input tables, processes with pandas expressions,
# and creates tables in the datastore
orca.run(['input_pre_processor'])
