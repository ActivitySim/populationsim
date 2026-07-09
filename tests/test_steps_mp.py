from pathlib import Path
import pandas as pd

from populationsim.core import tracing, inject, pipeline, mp_tasks
from tests import assert_expanded_close, expected_path

TAZ_COUNT = 36
TAZ_100_HH_COUNT = 33
TAZ_100_HH_REPOP_COUNT = 26


def setup_function(func):
    example_dir = Path(__file__).parent.parent / "examples"
    example_configs_dir = example_dir / "example_test" / "configs"
    configs_dir = Path(__file__).parent / "configs"
    mp_configs_dir = example_dir / "example_test" / "configs_mp"
    output_dir = Path(__file__).parent / "output"
    data_dir = example_dir / "example_test" / "data"

    inject.reinject_decorated_tables()

    inject.add_injectable(
        "configs_dir", [mp_configs_dir, configs_dir, example_configs_dir]
    )
    inject.add_injectable("output_dir", output_dir)
    inject.add_injectable("data_dir", data_dir)

    inject.clear_cache()

    tracing.config_logger()

    tracing.delete_output_files("csv")
    tracing.delete_output_files("txt")
    tracing.delete_output_files("yaml")


def regress():

    expanded_household_ids = pipeline.get_table("expanded_household_ids")
    assert isinstance(expanded_household_ids, pd.DataFrame)
    taz_hh_counts = expanded_household_ids.groupby("TAZ").size()
    assert len(taz_hh_counts) == TAZ_COUNT
    assert taz_hh_counts.loc[100] == TAZ_100_HH_COUNT

    # output_tables action: skip
    output_dir = Path(__file__).parent / "output"

    assert not (output_dir / "households.csv").exists()
    assert (output_dir / "summary_DISTRICT_1.csv").exists()

    expected_hh_ids = pd.read_parquet(expected_path("expanded_mp"))

    # Compare zone-level distribution with a tolerance.
    assert_expanded_close(expanded_household_ids, expected_hh_ids)


def teardown_function(func):
    if pipeline.is_open():
        pipeline.close_pipeline()
    inject.clear_cache()
    inject.reinject_decorated_tables()


def test_mp_run():

    # Debugging ----------------------
    run_list = mp_tasks.get_run_list()
    mp_tasks.print_run_list(run_list)
    # --------------------------------

    # do this after config.handle_standard_args, as command line args
    # may override injectables
    injectables = ["data_dir", "configs_dir", "output_dir"]
    injectables = {k: inject.get_injectable(k) for k in injectables}

    mp_tasks.run_multiprocess(injectables)

    pipeline.open_pipeline("_")
    regress()
    pipeline.close_pipeline()
