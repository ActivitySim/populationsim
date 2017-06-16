import os

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
        'sub_balancing',
        'low_balancing',
        'expand_population',
        'summarize'
    ]

    pipeline.run(models=_MODELS, resume_after=None)

    # tables will no longer be available after pipeline is closed
    pipeline.close()

    orca.clear_cache()
