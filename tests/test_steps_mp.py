# ActivitySim
# See full license in LICENSE.txt.
import subprocess
from tests.data_hash import hash_dataframe
from pathlib import Path
import pandas as pd

from activitysim.core import inject


def teardown_function(func):
    inject.clear_cache()
    inject.reinject_decorated_tables()


def test_mp_run():

    file_path = Path(__file__).parent / 'run_mp.py'
    output_dir = Path(__file__).parent / 'output'

    subprocess.check_call(['coverage', 'run', file_path])
    
    # This hash is the md5 of the dataframe string file previously generated
    # by the pipeline. It is used to check that the pipeline is generating the same output.
    expanded_household_ids = pd.read_csv(output_dir / 'expanded_household_ids.csv')
    assert hash_dataframe(expanded_household_ids) == '37b263bfa2d25c48fac9c591a87d91df'

if __name__ == '__main__':

    test_mp_run()
