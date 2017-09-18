import os

import pandas as pd
import orca

from activitysim.core import inject_defaults
from activitysim.core import tracing
from activitysim.core import pipeline

from populationsim import steps

configs_dir = os.path.join(os.path.dirname(__file__), 'configs')
orca.add_injectable("configs_dir", configs_dir)

data_dir = os.path.join(os.path.dirname(__file__), 'data')
orca.add_injectable("data_dir", data_dir)

output_dir = os.path.join(os.path.dirname(__file__), 'output')
orca.add_injectable("output_dir", output_dir)


def test_full_run1():

    orca.clear_cache()

    tracing.config_logger()

    _MODELS = [
        'input_pre_processor',
        'setup_data_structures',
        'initial_seed_balancing',
        'meta_control_factoring',
        'final_seed_balancing',
        'integerize_final_seed_weights',
        'sub_balancing.geography = mid',
        'sub_balancing.geography=low',
        'expand_population',
        'summarize',
        'write_results'
    ]

    pipeline.run(models=_MODELS, resume_after=None)

    assert isinstance(pipeline.get_table('expanded_household_ids'), pd.DataFrame)
    assert isinstance(pipeline.get_table('meta_1_summary'), pd.DataFrame)

    # tables will no longer be available after pipeline is closed
    pipeline.close()

    orca.clear_cache()
