import pandas as pd
from tests.data_hash import hash_dataframe
from pathlib import Path

from activitysim.core import config
from activitysim.core import tracing
from activitysim.core import pipeline
from activitysim.core import inject

from populationsim import steps


def teardown_function(func):
    inject.clear_cache()
    inject.reinject_decorated_tables()


def test_weighting():

    example_dir = Path(__file__).parent.parent / 'examples'
    
    configs_dir = (example_dir / 'example_survey_weighting' / 'configs')
    data_dir = (example_dir / 'example_survey_weighting' / 'data')
    output_dir = Path(__file__).parent / 'output'
    
    inject.add_injectable("data_dir", data_dir.__str__())
    inject.add_injectable("configs_dir", configs_dir.__str__())
    inject.add_injectable("output_dir", output_dir.__str__())

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

    seed_households = pd.read_csv(data_dir / 'seed_households.csv')
    total_seed_households_weights = seed_households['HHweight'].sum()

    assert abs(total_summary_hh_weights - total_seed_households_weights) < 1
    
    
    # This hash is the md5 of the dataframe string file previously generated
    # by the pipeline. It is used to check that the pipeline is generating the same output.
    assert hash_dataframe(summary_hh_weights) == 'e3eced420bdf3ff294e6493a7f8afa25'

    # tables will no longer be available after pipeline is closed
    pipeline.close_pipeline()

    inject.clear_cache()
