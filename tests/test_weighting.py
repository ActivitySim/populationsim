import pandas as pd
from tests.data_hash import hash_dataframe
from pathlib import Path

from populationsim.core import tracing, inject, pipeline



def teardown_function(func):
    inject.clear_cache()
    inject.reinject_decorated_tables()


def test_weighting():

    example_dir = Path(__file__).parent.parent / 'examples'
    
    configs_dir = (example_dir / 'example_survey_weighting' / 'configs')
    data_dir = (example_dir / 'example_survey_weighting' / 'data')
    output_dir = Path(__file__).parent / 'output'
    
    inject.add_injectable("data_dir", data_dir)
    inject.add_injectable("configs_dir", configs_dir)
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

    seed_households = pd.read_csv(data_dir / 'seed_households.csv')
    total_seed_households_weights = seed_households['HHweight'].sum()

    assert abs(total_summary_hh_weights - total_seed_households_weights) < 1
    
    # This hash is the md5 of the dataframe string file previously generated
    # by the pipeline. It is used to check that the pipeline is generating the same output.
    result_hash = hash_dataframe(summary_hh_weights.round(6), sort_by=['hh_id'])
    assert result_hash == 'ee59e3c79732c12745240aadfe20f317'

    # tables will no longer be available after pipeline is closed
    pipeline.close_pipeline()

    inject.clear_cache()
