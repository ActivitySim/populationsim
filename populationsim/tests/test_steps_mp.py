# ActivitySim
# See full license in LICENSE.txt.
import os
import subprocess
import hashlib
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
    
    # This hash is the md5 of the json string of the synthetic_*.csv file previously generated
    # by the pipeline. It is used to check that the pipeline is generating the same output.
    expected_hash = {
        'households': 'cb51c372272c3984ad4e8c43177b8737',
        'persons': 'de854cabd4e5db51a3b45ae0f6c50f3f'
    }
    for (table, expected) in expected_hash.items():
        result_df = pd.read_csv(output_dir / f"synthetic_{table}.csv")
        result_bytes = result_df.to_json().encode('utf-8')
        result_hash = hashlib.md5(result_bytes).hexdigest()

        assert result_hash == expected

if __name__ == '__main__':

    test_mp_run()
