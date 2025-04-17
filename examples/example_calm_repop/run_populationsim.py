# ActivitySim
# See full license in LICENSE.txt.

import os
import sys
import argparse
from pathlib import Path
import shutil

from populationsim.core import inject
import populationsim


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

    base_output = Path(__file__).parent.parent / "example_calm" / "output"
    repop_output = Path(__file__).parent / "output"

    # Copy the pipeline output from the example_survey_weighting/output
    if not (repop_output / "pipeline.h5").exists():

        if not (
            (base_output / "pipeline.h5").exists()
            and (base_output / "final_expanded_household_ids.csv").exists()
        ):
            msg = f"Pipeline output not found at {base_output}. Ensure the example_calm pipeline has been run."
            raise FileNotFoundError(msg)

        shutil.copy(
            (base_output / "pipeline.h5").__str__(),
            (repop_output / "pipeline.h5").__str__(),
        )

    parser = argparse.ArgumentParser()
    populationsim.add_run_args(parser)
    args = parser.parse_args()
    args.working_dir = os.path.dirname(__file__)

    sys.exit(populationsim.run(args))
