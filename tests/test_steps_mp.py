from pathlib import Path
import pandas as pd
from tests.data_hash import hash_dataframe

from populationsim.core import tracing
from populationsim.core import pipeline
from populationsim.core import inject
from populationsim.core import mp_tasks

TAZ_COUNT = 36
TAZ_100_HH_COUNT = 33
TAZ_100_HH_REPOP_COUNT = 26


example_dir = Path(__file__).parent.parent / 'examples'
example_configs_dir = (example_dir / 'example_test' / 'configs')
configs_dir = (Path(__file__).parent / 'configs')    
mp_configs_dir = (example_dir / 'example_test' / 'configs_mp')
output_dir = Path(__file__).parent / 'output'
data_dir = (example_dir / 'example_test' / 'data')


def regress():

    expanded_household_ids = pipeline.get_table("expanded_household_ids")
    assert isinstance(expanded_household_ids, pd.DataFrame)
    taz_hh_counts = expanded_household_ids.groupby("TAZ").size()
    assert len(taz_hh_counts) == TAZ_COUNT
    assert taz_hh_counts.loc[100] == TAZ_100_HH_COUNT

    # output_tables action: skip
    assert not (output_dir / 'households.csv').exists()
    assert (output_dir / 'summary_DISTRICT_1.csv').exists()
    
    result_hash = hash_dataframe(expanded_household_ids, sort_by = ['hh_id', 'TRACT', 'TAZ', 'PUMA'])
    assert result_hash == '05d7f8d0bf5d8e5c7ee29b67c13d858f'

def teardown_function(func):
    inject.clear_cache()
    inject.reinject_decorated_tables()


def test_mp_run():

    inject.add_injectable("configs_dir", [mp_configs_dir, configs_dir, example_configs_dir])
    inject.add_injectable("output_dir", output_dir)
    inject.add_injectable("data_dir", data_dir)

    tracing.config_logger()

    tracing.delete_output_files("csv")
    tracing.delete_output_files("txt")
    tracing.delete_output_files("yaml")

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
