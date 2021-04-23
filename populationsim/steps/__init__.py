# PopulationSim
# See full license in LICENSE.txt.


from activitysim.core import inject

from . import input_pre_processor
from . import setup_data_structures
from . import initial_seed_balancing
from . import meta_control_factoring
from . import final_seed_balancing
from . import integerize_final_seed_weights
from . import sub_balancing
from . import expand_households
from . import summarize
from . import write_synthetic_population
from . import repop_balancing

from activitysim.core.steps.output import write_data_dictionary
from activitysim.core.steps.output import write_tables


@inject.injectable(cache=True)
def preload_injectables():
    inject.add_step('write_data_dictionary', write_data_dictionary)
    inject.add_step('write_tables', write_tables)
    return True
