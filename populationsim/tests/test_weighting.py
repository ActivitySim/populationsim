import os

import pandas as pd

from activitysim.core import config
from activitysim.core import tracing
from activitysim.core import pipeline
from activitysim.core import inject

from populationsim import steps


def teardown_function(func):
    inject.clear_cache()
    inject.reinject_decorated_tables()


def test_weighting():

    configs_dir = os.path.join(os.path.dirname(__file__), '..', '..',
                               'example_survey_weighting', 'configs')
    inject.add_injectable("configs_dir", configs_dir)

    data_dir = os.path.join(os.path.dirname(__file__), '..', '..',
                            'example_survey_weighting', 'data')
    inject.add_injectable("data_dir", data_dir)

    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    inject.add_injectable("output_dir", output_dir)

    inject.clear_cache()

    tracing.config_logger()

    _MODELS = [
        'input_pre_processor',
        'setup_data_structures',
        'initial_seed_balancing',
        'meta_control_factoring',
        'final_seed_balancing',
        'summarize',
        'write_tables'
    ]

    pipeline.run(models=_MODELS, resume_after=None)

    summary_hh_weights = pipeline.get_table('summary_hh_weights')
    total_summary_hh_weights = summary_hh_weights['SUBREGCluster_balanced_weight'].sum()

    seed_households = pd.read_csv(os.path.join(data_dir, 'seed_households.csv'))
    total_seed_households_weights = seed_households['HHweight'].sum()

    assert abs(total_summary_hh_weights - total_seed_households_weights) < 1

    # tables will no longer be available after pipeline is closed
    pipeline.close_pipeline()

    inject.clear_cache()
