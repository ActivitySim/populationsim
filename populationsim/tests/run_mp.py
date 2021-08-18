import os

import pandas as pd

from activitysim.core import config
from activitysim.core import tracing
from activitysim.core import pipeline
from activitysim.core import inject
from activitysim.core import mp_tasks

from populationsim import steps

TAZ_COUNT = 36
TAZ_100_HH_COUNT = 33
TAZ_100_HH_REPOP_COUNT = 26


def setup_dirs():

    configs_dir = os.path.join(os.path.dirname(__file__), "configs")
    mp_configs_dir = os.path.join(os.path.dirname(__file__), "configs_mp")
    inject.add_injectable("configs_dir", [mp_configs_dir, configs_dir])

    output_dir = os.path.join(os.path.dirname(__file__), "output")
    inject.add_injectable("output_dir", output_dir)

    data_dir = os.path.join(os.path.dirname(__file__), "data")
    inject.add_injectable("data_dir", data_dir)

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
    output_dir = inject.get_injectable("output_dir")
    assert not os.path.exists(os.path.join(output_dir, "households.csv"))
    assert os.path.exists(os.path.join(output_dir, "summary_DISTRICT_1.csv"))


def test_mp_run():

    setup_dirs()

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


if __name__ == "__main__":

    test_mp_run()
