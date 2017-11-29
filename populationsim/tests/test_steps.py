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


TAZ_COUNT = 36
TAZ_100_HH_COUNT = 25
TAZ_100_HH_REPOP_COUNT = 26


def test_full_run1():

    configs_dir = os.path.join(os.path.dirname(__file__), 'configs')
    orca.add_injectable("configs_dir", configs_dir)

    data_dir = os.path.join(os.path.dirname(__file__), 'data')
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
        'sub_balancing.geography = TRACT',
        'sub_balancing.geography=TAZ',
        'expand_households',
        'write_results',
        'summarize'
    ]

    pipeline.run(models=_MODELS, resume_after=None)

    expanded_household_ids = pipeline.get_table('expanded_household_ids')
    assert isinstance(expanded_household_ids, pd.DataFrame)
    taz_hh_counts = expanded_household_ids.groupby('TAZ').size()
    assert len(taz_hh_counts) == TAZ_COUNT
    assert taz_hh_counts.loc[100] == TAZ_100_HH_COUNT

    assert os.path.exists(os.path.join(output_dir, 'summary_DISTRICT_1.csv'))

    # tables will no longer be available after pipeline is closed
    pipeline.close_pipeline()

    orca.clear_cache()


def test_full_run2_repop_replace():

    configs_dir = os.path.join(os.path.dirname(__file__), 'configs')
    orca.add_injectable("configs_dir", configs_dir)

    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    orca.add_injectable("data_dir", data_dir)

    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    orca.add_injectable("output_dir", output_dir)

    orca.clear_cache()

    tracing.config_logger()

    _MODELS = [
        'input_pre_processor.table_list=repop_input_table_list',
        'repop_setup_data_structures',
        'initial_seed_balancing.final=true',
        'integerize_final_seed_weights.repop',
        'repop_balancing',
        'expand_households.repop;replace',
        'write_results.repop',
        'summarize.repop'
    ]

    pipeline.run(models=_MODELS, resume_after='summarize')

    assert os.path.exists(os.path.join(output_dir, 'summary_DISTRICT_1.csv'))

    expanded_household_ids = pipeline.get_table('expanded_household_ids')
    assert isinstance(expanded_household_ids, pd.DataFrame)
    taz_hh_counts = expanded_household_ids.groupby('TAZ').size()
    assert len(taz_hh_counts) == TAZ_COUNT
    assert taz_hh_counts.loc[100] == TAZ_100_HH_REPOP_COUNT

    # tables will no longer be available after pipeline is closed
    pipeline.close_pipeline()

    orca.clear_cache()


def test_full_run2_repop_append():

    configs_dir = os.path.join(os.path.dirname(__file__), 'configs')
    orca.add_injectable("configs_dir", configs_dir)

    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    orca.add_injectable("data_dir", data_dir)

    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    orca.add_injectable("output_dir", output_dir)

    orca.clear_cache()

    tracing.config_logger()

    _MODELS = [
        'input_pre_processor.table_list=repop_input_table_list',
        'repop_setup_data_structures',
        'initial_seed_balancing.final=true',
        'integerize_final_seed_weights.repop',
        'repop_balancing',
        'expand_households.repop;append',
        'write_results.repop',
        'summarize.repop'
    ]

    pipeline.run(models=_MODELS, resume_after='summarize')

    assert os.path.exists(os.path.join(output_dir, 'summary_DISTRICT_1.csv'))

    expanded_household_ids = pipeline.get_table('expanded_household_ids')
    assert isinstance(expanded_household_ids, pd.DataFrame)
    taz_hh_counts = expanded_household_ids.groupby('TAZ').size()
    assert len(taz_hh_counts) == TAZ_COUNT
    assert taz_hh_counts.loc[100] == TAZ_100_HH_COUNT + TAZ_100_HH_REPOP_COUNT

    # tables will no longer be available after pipeline is closed
    pipeline.close_pipeline()

    orca.clear_cache()
