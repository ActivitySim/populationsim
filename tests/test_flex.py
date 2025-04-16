from pathlib import Path
import pandas as pd
from tests.data_hash import hash_dataframe

from populationsim.core import config
from populationsim.core import tracing
from populationsim.core import pipeline
from populationsim.core import inject


def teardown_function(func):
    inject.clear_cache()


def test_full_run_flex():

    example_dir = Path(__file__).parent.parent / 'examples'

    configs_dir = (example_dir / 'example_test' / 'configs_flex')
    data_dir = (example_dir / 'example_test' / 'data_flex')
    output_dir = Path(__file__).parent / 'output'
    
    inject.add_injectable("data_dir", data_dir)    
    inject.add_injectable("configs_dir", configs_dir)
    inject.add_injectable("output_dir", output_dir)

    inject.clear_cache()

    tracing.config_logger()

    tracing.delete_output_files('csv')
    tracing.delete_output_files('txt')
    tracing.delete_output_files('yaml')

    _MODELS = [
        'input_pre_processor',
        'setup_data_structures',
        'initial_seed_balancing',
        'meta_control_factoring',
        'final_seed_balancing',
        'integerize_final_seed_weights',
        'sub_balancing.geography=DISTRICT',
        'sub_balancing.geography=TRACT',
        'sub_balancing.geography=TAZ',
        'expand_households',
        'summarize',
        'write_tables'
    ]

    pipeline.run(models=_MODELS, resume_after=None)

    assert isinstance(pipeline.get_table('expanded_household_ids'), pd.DataFrame)

    # output tables list action: include
    assert Path(config.output_file_path('expanded_household_ids.csv')).exists()
    assert Path(config.output_file_path('summary_DISTRICT.csv')).exists()
    assert not Path(config.output_file_path('summary_TAZ.csv')).exists()

    # This hash is the md5 of the dataframe string file previously generated
    # by the pipeline. It is used to check that the pipeline is generating the same output.
    expanded_household_ids = pipeline.get_table('expanded_household_ids')
    assert hash_dataframe(expanded_household_ids) == '55ee10a9fb0a64cd1b230f3c8690576c'

    # tables will no longer be available after pipeline is closed
    pipeline.close_pipeline()

    inject.clear_cache()
