import orca
from activitysim import defaults
from activitysim import tracing
import pandas as pd
import numpy as np
import os

#read input tables, processes with pandas expressions, 
#and creates tables in the datastore
orca.run(['input_pre_processor'])

#setup geographic correspondence, seeds, control sets, 
#weights, expansion factors, and incidence tables
orca.run(['setup_data_structures']) 

#seed (puma) balancing, meta level balancing, meta 
#control factoring, and meta final balancing
orca.run(['initial_seed_balancing'])

#final balancing for each seed (puma) zone with aggregated 
#low and mid-level controls and distributed meta-level controls
orca.run(['final_seed_balancing'])

#iteratively loop through zones and list balance on each 
#mid-level zone within a meta zone and then each low-level
#zone within a mid-level zone. This is the current procedure, 
#which will likely be revised.
orca.run(['mid_and_low_level_allocation'])

#expand household and person records with final weights 
#to one household and one person record per weight with unique IDs
orca.run(['expand_population'])

#write the household and person files to CSV files
orca.run(['write_results'])

