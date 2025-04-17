import os
import sys
import argparse

import populationsim
from populationsim.core import inject


@inject.injectable()
def log_settings():

    return [
        "multiprocess",
        "num_processes",
        "resume_after",
        "GROUP_BY_INCIDENCE_SIGNATURE",
        "INTEGERIZE_WITH_BACKSTOPPED_CONTROLS",
        "SUB_BALANCE_WITH_FLOAT_SEED_WEIGHTS",
        "meta_control_data",
        "control_file_name",
        "USE_CVXPY",
        "USE_SIMUL_INTEGERIZER",
    ]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    populationsim.add_run_args(parser)
    args = parser.parse_args()
    args.working_dir = os.path.dirname(__file__)

    sys.exit(populationsim.run(args))
