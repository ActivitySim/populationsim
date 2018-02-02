import os

import pandas as pd
import orca

from activitysim.core import inject_defaults
from activitysim.core import tracing
from activitysim.core import pipeline
from activitysim.core import inject

from populationsim import steps


def teardown_function(func):
    orca.clear_cache()
    inject.reinject_decorated_tables()


def test_full_run2():

    configs_dir = os.path.join(os.path.dirname(__file__), 'configs2')
    orca.add_injectable("configs_dir", configs_dir)

    data_dir = os.path.join(os.path.dirname(__file__), 'data2')
    orca.add_injectable("data_dir", data_dir)

    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    orca.add_injectable("output_dir", output_dir)

    orca.clear_cache()

    tracing.config_logger()

    _MODELS = [
        'input_pre_processor',
        'setup_data_structures',
        'initial_seed_balancing',
        'meta_control_factoring',
        'final_seed_balancing',
        'integerize_final_seed_weights',
        'sub_balancing.geography = DISTRICT',
        'sub_balancing.geography = TRACT',
        'sub_balancing.geography=TAZ',
        'expand_households',
        'summarize',
        'write_tables'
    ]

    pipeline.run(models=_MODELS, resume_after=None)

    assert isinstance(pipeline.get_table('expanded_household_ids'), pd.DataFrame)

    # output tables list action: include
    assert os.path.exists(os.path.join(output_dir, 'expanded_household_ids.csv'))
    assert os.path.exists(os.path.join(output_dir, 'summary_DISTRICT.csv'))
    assert not os.path.exists(os.path.join(output_dir, 'summary_TAZ.csv'))

    # tables will no longer be available after pipeline is closed
    pipeline.close_pipeline()

    orca.clear_cache()
