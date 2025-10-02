"""Script to convert existing population and household files to a pipeline hdf5 file
to be used for the repop feature of PopulationSim"""

import pandas as pd
import warnings
from pathlib import Path
from tables import NaturalNameWarning

warnings.simplefilter("ignore", NaturalNameWarning)


class ConstructPipe:
    """Class to convert existing population and household files to a pipeline hdf5 for repop"""

    working_dir: Path
    template_dir: Path
    persons: pd.DataFrame
    households: pd.DataFrame
    checkpoints: pd.DataFrame
    xwalk: pd.DataFrame

    required_tables = [
        "/trace_TAZ_weights/sub_balancing.geography=TAZ",
        "/trace_MAZ_weights/sub_balancing.geography=MAZ",
        "/summary_hh_weights/summarize",
        "/summary_TAZ_aggregate/summarize",
        "/summary_TAZ_PUMA/summarize",
        "/summary_TAZ/summarize",
        "/summary_REGION_4/summarize",
        "/summary_MAZ_aggregate/summarize",
        "/summary_MAZ_PUMA/summarize",
        "/summary_MAZ/summarize",
        "/incidence_table/setup_data_structures",
        "/household_groups/setup_data_structures",
        "/crosswalk/setup_data_structures",
        "/control_spec/setup_data_structures",
        "/TAZ_weights_sparse/sub_balancing.geography=TAZ",
        "/TAZ_controls/setup_data_structures",
        "/TAZ_control_data/input_pre_processor",
        "/REGION_controls/setup_data_structures",
        "/REGION_control_data/input_pre_processor",
        "/PUMA_weights/integerize_final_seed_weights",
        "/PUMA_controls/setup_data_structures",
        "/MAZ_weights_sparse/sub_balancing.geography=MAZ",
        "/MAZ_weights/sub_balancing.geography=MAZ",
        "/MAZ_controls/setup_data_structures",
        "/MAZ_control_data/input_pre_processor",
    ]

    def __init__(self, working_dir, template_dir=None):
        self.output_dir = Path(working_dir) / "output"
        self.data_dir = Path(working_dir) / "data"
        self.conf_dir = Path(working_dir) / "configs"

        if template_dir:
            self.template_dir = Path(template_dir)
        else:
            self.template_dir = Path(__file__).parent.resolve() / "pipeline_templates"

        # Check that all paths are valid
        if not self.data_dir.exists():
            raise ValueError(f"Data directory {self.data_dir} does not exist.")

        if not self.output_dir.exists():
            raise ValueError(f"Pipeline directory {self.output_dir} does not exist.")

        if not self.template_dir.exists():
            raise ValueError(f"Template directory {self.template_dir} does not exist.")

        self.read_inputs()
        self.construct_pipe()

    def read_inputs(self):

        pd.read_csv(self.conf_dir / "repop_controls.csv")

        self.xwalk = pd.read_csv(self.data_dir / "geo_crosswalks.csv")
        self.persons = pd.read_csv(self.template_dir / "persons.csv")
        self.households = pd.read_csv(self.template_dir / "households.csv")
        self.checkpoints = pd.read_csv(self.template_dir / "checkpoints.csv")

    def construct_pipe(self):
        """Construct the pipeline hdf5 file"""

        # Delete existing pipeline if it exists
        if (self.output_dir / "pipeline.h5").exists():
            (self.output_dir / "pipeline.h5").unlink()

        pipe = pd.HDFStore(self.output_dir / "pipeline.h5")

        # Join households with xwalk to get PUMA and TAZ
        pipehouseholds = self.households.merge(
            self.xwalk[["MAZ", "PUMA"]], how="left", on="MAZ"
        )
        pipehouseholds["weight"] = 1

        # Filter to only include those with unittype = 0 (single family)
        hh = pipehouseholds[pipehouseholds.unittype == 0]
        persons = self.persons[self.persons.hh_id.isin(hh.hh_id.unique())]
        self.xwalk["STATE"] = 6

        # Populate the pipeline with the data
        pipe[r"/checkpoints"] = self.checkpoints
        pipe[r"/geo_cross_walk/input_pre_processor"] = self.xwalk
        pipe[r"/persons/setup_data_structures"] = persons
        pipe[r"/households/setup_data_structures"] = hh.set_index("hh_id")
        pipe[r"/expanded_household_ids/expand_households"] = hh[
            ["PUMA", "TAZ", "MAZ", "hh_id"]
        ]
        pipe[r"/TAZ_weights/sub_balancing.geography=TAZ"] = pd.DataFrame(
            {
                x: [0]
                for x in list(self.xwalk.columns)
                + ["hh_id", "balanced_weight", "integer_weight"]
                if x not in ["MAZ"]
            }
        )

        # Spoof the remaining required tables
        for table in set(self.required_tables) - set(pipe.keys()):
            pipe[table] = pd.DataFrame()

        pipe.close()


if __name__ == "__main__":

    # Example usage
    outdir = Path(__file__).parent
    ConstructPipe(outdir)
