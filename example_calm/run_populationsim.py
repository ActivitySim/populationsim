# ActivitySim
# See full license in LICENSE.txt.

import sys
import argparse

from activitysim.core.config import setting
from activitysim.core import inject

from activitysim.cli.run import add_run_args, run
from populationsim import steps


@inject.injectable()
def log_settings():

    return [
        'multiprocess',
        'num_processes',
        'resume_after',
        'GROUP_BY_INCIDENCE_SIGNATURE',
        'INTEGERIZE_WITH_BACKSTOPPED_CONTROLS',
        'SUB_BALANCE_WITH_FLOAT_SEED_WEIGHTS',
        'meta_control_data',
        'control_file_name',
        'USE_CVXPY',
        'USE_SIMUL_INTEGERIZER'
    ]


if __name__ == '__main__':

    assert inject.get_injectable('preload_injectables', None)

    parser = argparse.ArgumentParser()
    add_run_args(parser)
    args = parser.parse_args()

    sys.exit(run(args))
