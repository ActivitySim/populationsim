import os

import orca

from activitysim.core import inject_defaults
from activitysim.core import tracing
from activitysim.core import pipeline

from activitysim.core.tracing import print_elapsed_time

from populationsim import steps


def test_full_run1():

    orca.clear_cache()

    configs_dir = os.path.join(os.path.dirname(__file__), 'configs')
    orca.add_injectable("configs_dir", configs_dir)

    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    orca.add_injectable("data_dir", data_dir)

    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    orca.add_injectable("output_dir", output_dir)

    tracing.config_logger()

    _MODELS = [
        'input_pre_processor',
        'setup_data_structures',
        'initial_seed_balancing',
        'meta_control_factoring',
        'final_seed_balancing',
        'integerize_final_seed_weights',
        'simultaneous_sub_balancing',
        'integerize_sub_weights'
    ]

    pipeline.run(models=_MODELS, resume_after=None)

    assert 'sub_results' in pipeline.checkpointed_tables()

    df = pipeline.get_table('sub_results')

    # FIXME - need to check stuff

    # tables will no longer be available after pipeline is closed
    pipeline.close()

    orca.clear_cache()
