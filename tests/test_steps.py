from pathlib import Path
import pandas as pd

from tests.data_hash import hash_dataframe
from activitysim.core import config
from activitysim.core import tracing
from activitysim.core import pipeline
from activitysim.core import inject

from populationsim import steps


example_dir = Path(__file__).parent.parent / 'examples'

example_configs_dir = (example_dir / 'example_test' / 'configs').__str__()
configs_dir = (Path(__file__).parent / 'configs').__str__()
output_dir = Path(__file__).parent / 'output'
data_dir = (example_dir / 'example_test' / 'data').__str__()

def setup_function():

    inject.reinject_decorated_tables()
    
    inject.add_injectable("configs_dir", [configs_dir, example_configs_dir])
    inject.add_injectable("output_dir", output_dir)
    inject.add_injectable("data_dir", data_dir)

    inject.clear_cache()

    tracing.config_logger()

    tracing.delete_output_files('csv')
    tracing.delete_output_files('txt')
    tracing.delete_output_files('yaml')


def teardown_function(func):
    inject.clear_cache()
    inject.reinject_decorated_tables()


TAZ_COUNT = 36
TAZ_100_HH_COUNT = 33
TAZ_100_HH_REPOP_COUNT = 26


def test_full_run1():

    _MODELS = [
        'input_pre_processor',
        'setup_data_structures',
        'initial_seed_balancing',
        'meta_control_factoring',
        'final_seed_balancing',
        'integerize_final_seed_weights',
        'sub_balancing.geography=TRACT',
        'sub_balancing.geography=TAZ',
        'expand_households',
        'summarize',
        'write_tables',
        'write_synthetic_population',
    ]

    pipeline.run(models=_MODELS, resume_after=None)

    expanded_household_ids = pipeline.get_table('expanded_household_ids')
    assert isinstance(expanded_household_ids, pd.DataFrame)
    taz_hh_counts = expanded_household_ids.groupby('TAZ').size()
    assert len(taz_hh_counts) == TAZ_COUNT
    assert taz_hh_counts.loc[100] == TAZ_100_HH_COUNT

    # output_tables action: skip
    output_dir = inject.get_injectable('output_dir')
    assert not (output_dir / 'households.csv').exists()
    assert (output_dir / 'summary_DISTRICT_1.csv').exists()
    
    # This hash is the md5 of the json string of the expanded_household_ids file previously generated
    # by the pipeline. It is used to check that the pipeline is generating the same output.
    result_hash = hash_dataframe(expanded_household_ids, sort_by = ['hh_id', 'TRACT', 'TAZ', 'PUMA'])
    assert result_hash == '05d7f8d0bf5d8e5c7ee29b67c13d858f'

    # tables will no longer be available after pipeline is closed
    pipeline.close_pipeline()

    inject.clear_cache()


def test_full_run2_repop_replace():
    # Note: tests are run in alphabetical order.
    # This tests expects to find the pipeline h5 file from
    # test_full_run1 in the output folder

    _MODELS = [
        'input_pre_processor.table_list=repop_input_table_list;repop',
        'repop_setup_data_structures',
        'initial_seed_balancing.final=true;repop',
        'integerize_final_seed_weights.repop',
        'repop_balancing',
        'expand_households.repop;replace',
        'write_synthetic_population.repop',
        'write_tables.repop',
    ]

    pipeline.run(models=_MODELS, resume_after='summarize')

    expanded_household_ids = pipeline.get_table('expanded_household_ids')
    assert isinstance(expanded_household_ids, pd.DataFrame)
    taz_hh_counts = expanded_household_ids.groupby('TAZ').size()
    assert len(taz_hh_counts) == TAZ_COUNT
    assert taz_hh_counts.loc[100] == TAZ_100_HH_REPOP_COUNT
    
    # This hash is the md5 of the json string of the expanded_household_ids file previously generated
    # by the pipeline. It is used to check that the pipeline is generating the same output.
    result_hash = hash_dataframe(expanded_household_ids, sort_by = ['hh_id', 'TRACT', 'TAZ', 'PUMA'])
    assert result_hash == 'aec5dc9f7d575e367874c33ba6a61ee3'

    # tables will no longer be available after pipeline is closed
    pipeline.close_pipeline()

    inject.clear_cache()


def test_full_run2_repop_append():

    _MODELS = [
        'input_pre_processor.table_list=repop_input_table_list;repop',
        'repop_setup_data_structures',
        'initial_seed_balancing.final=true;repop',
        'integerize_final_seed_weights.repop',
        'repop_balancing',
        'expand_households.repop;append',
        'write_synthetic_population.repop',
    ]

    pipeline.run(models=_MODELS, resume_after='summarize')

    expanded_household_ids = pipeline.get_table('expanded_household_ids')
    assert isinstance(expanded_household_ids, pd.DataFrame)
    taz_hh_counts = expanded_household_ids.groupby('TAZ').size()
    assert len(taz_hh_counts) == TAZ_COUNT
    assert taz_hh_counts.loc[100] == TAZ_100_HH_COUNT + TAZ_100_HH_REPOP_COUNT

    # This hash is the md5 of the dataframe string file previously generated
    # by the pipeline. It is used to check that the pipeline is generating the same output.
    result_hash = hash_dataframe(expanded_household_ids, sort_by = ['hh_id', 'TRACT', 'TAZ', 'PUMA'])
    assert result_hash == 'b791dffd6435774e5a5ddc52f40697f1'


    # tables will no longer be available after pipeline is closed
    pipeline.close_pipeline()

    inject.clear_cache()
