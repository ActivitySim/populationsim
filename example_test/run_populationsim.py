import os

import orca


from activitysim.core import inject_defaults

from activitysim.core import tracing
from activitysim.core import pipeline

from activitysim.core.tracing import print_elapsed_time

from populationsim import steps

tracing.config_logger()

t0 = print_elapsed_time()

_MODELS = [
    'input_pre_processor',
    'setup_data_structures',
    'initial_seed_balancing',
    'meta_control_factoring',
    'final_seed_balancing',

    # iteratively loop through zones and list balance each
    # lower-level zone within a meta zone and then each next-lower-level
    # zone within a lower-level zone, etc.  This is the current procedure,
    # which is being revised.
    # 'lower_geography_allocation',

    # expand household and person records with final weights
    # to one household and one person record per weight with unique IDs
    # 'expand_population',

    # write the household and person files to CSV files
    # 'write_results'
]


# If you provide a resume_after argument to pipeline.run
# the pipeline manager will attempt to load checkpointed tables from the checkpoint store
# and resume pipeline processing on the next submodel step after the specified checkpoint
resume_after = None
# resume_after = 'meta_control_factoring'

pipeline.run(models=_MODELS, resume_after=resume_after)


# write final versions of all checkpointed dataframes to CSV files to review results
if True:
    t0 = print_elapsed_time()
    for table_name in pipeline.checkpointed_tables():
        file_name = "%s.csv" % table_name
        file_path = os.path.join(orca.get_injectable("output_dir"), file_name)
        pipeline.get_table(table_name).to_csv(file_path, index=True)
    t0 = print_elapsed_time("write final versions of all checkpointed dataframes to CSV", t0)

# tables will no longer be available after pipeline is closed
pipeline.close()

# write checkpoints (this can be called whether or not pipeline is open)
file_path = os.path.join(orca.get_injectable("output_dir"), "checkpoints.csv")
pipeline.get_checkpoints().to_csv(file_path)

t0 = print_elapsed_time("all models", t0)
