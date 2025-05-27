from pathlib import Path
import pandas as pd

from populationsim.core import tracing, inject, pipeline

TAZ_COUNT = 36
TAZ_100_HH_COUNT = 33
TAZ_100_HH_REPOP_COUNT = 26


def setup_function():

    example_dir = Path(__file__).parent.parent / "examples"

    example_configs_dir = example_dir / "example_test" / "configs"
    configs_dir = Path(__file__).parent / "configs"
    output_dir = Path(__file__).parent / "output"
    data_dir = example_dir / "example_test" / "data"

    inject.reinject_decorated_tables()

    inject.add_injectable("configs_dir", [configs_dir, example_configs_dir])
    inject.add_injectable("output_dir", output_dir)
    inject.add_injectable("data_dir", data_dir)

    inject.clear_cache()

    tracing.config_logger()

    tracing.delete_output_files("csv")
    tracing.delete_output_files("txt")
    tracing.delete_output_files("yaml")


def teardown_function(func):
    inject.clear_cache()
    inject.reinject_decorated_tables()


def test_full_run1():

    _MODELS = [
        "input_pre_processor",
        "setup_data_structures",
        "initial_seed_balancing",
        "meta_control_factoring",
        "final_seed_balancing",
        "integerize_final_seed_weights",
        "sub_balancing.geography=TRACT",
        "sub_balancing.geography=TAZ",
        "expand_households",
        "summarize",
        "write_tables",
        "write_synthetic_population",
    ]

    pipeline.run(models=_MODELS, resume_after=None)

    expanded_household_ids = pipeline.get_table("expanded_household_ids")
    assert isinstance(expanded_household_ids, pd.DataFrame)
    taz_hh_counts = expanded_household_ids.groupby("TAZ").size()
    assert len(taz_hh_counts) == TAZ_COUNT
    assert taz_hh_counts.loc[100] == TAZ_100_HH_COUNT

    # output_tables action: skip
    output_dir = Path(__file__).parent / "output"

    # Check that the output files are created as expected
    assert not (output_dir / "households.csv").exists()
    assert (output_dir / "summary_DISTRICT_1.csv").exists()

    expected_hh_ids = pd.read_parquet(
        Path(__file__).parent / "expected" / "expanded.parquet"
    )

    # Compare the two dataframes
    assert expanded_household_ids.equals(expected_hh_ids)

    # tables will no longer be available after pipeline is closed
    pipeline.close_pipeline()

    inject.clear_cache()


def test_full_run2_repop_replace():
    # Note: tests are run in alphabetical order.
    # This tests expects to find the pipeline h5 file from
    # test_full_run1 in the output folder

    _MODELS = [
        "input_pre_processor.table_list=repop_input_table_list;repop",
        "repop_setup_data_structures",
        "initial_seed_balancing.final=true;repop",
        "integerize_final_seed_weights.repop",
        "repop_balancing",
        "expand_households.repop;replace",
        "write_synthetic_population.repop",
        "write_tables.repop",
    ]

    pipeline.run(models=_MODELS, resume_after="summarize")

    expanded_household_ids = pipeline.get_table("expanded_household_ids")
    assert isinstance(expanded_household_ids, pd.DataFrame)
    taz_hh_counts = expanded_household_ids.groupby("TAZ").size()
    assert len(taz_hh_counts) == TAZ_COUNT
    assert taz_hh_counts.loc[100] == TAZ_100_HH_REPOP_COUNT

    expected_hh_ids = pd.read_parquet(
        Path(__file__).parent / "expected" / "expanded_repop_replace.parquet"
    )

    # Compare the two dataframes
    assert expanded_household_ids.equals(expected_hh_ids)

    # tables will no longer be available after pipeline is closed
    pipeline.close_pipeline()

    inject.clear_cache()


def test_full_run2_repop_append():

    _MODELS = [
        "input_pre_processor.table_list=repop_input_table_list;repop",
        "repop_setup_data_structures",
        "initial_seed_balancing.final=true;repop",
        "integerize_final_seed_weights.repop",
        "repop_balancing",
        "expand_households.repop;append",
        "write_synthetic_population.repop",
    ]

    pipeline.run(models=_MODELS, resume_after="summarize")

    expanded_household_ids = pipeline.get_table("expanded_household_ids")
    assert isinstance(expanded_household_ids, pd.DataFrame)
    taz_hh_counts = expanded_household_ids.groupby("TAZ").size()
    assert len(taz_hh_counts) == TAZ_COUNT
    assert taz_hh_counts.loc[100] == TAZ_100_HH_COUNT + TAZ_100_HH_REPOP_COUNT

    expected_hh_ids = pd.read_parquet(
        Path(__file__).parent / "expected" / "expanded_repop_append.parquet"
    )

    # Compare the two dataframes
    assert expanded_household_ids.equals(expected_hh_ids)

    # tables will no longer be available after pipeline is closed
    pipeline.close_pipeline()

    inject.clear_cache()
