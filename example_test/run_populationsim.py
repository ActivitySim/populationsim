import os

import orca


from activitysim.core import inject_defaults

from activitysim.core import tracing
from activitysim.core import pipeline

from activitysim.core.tracing import print_elapsed_time

import extensions


tracing.config_logger()

t0 = print_elapsed_time()

_MODELS = [
    'input_pre_processor'
]

# If you provide a resume_after argument to pipeline.run
# the pipeline manager will attempt to load checkpointed tables from the checkpoint store
# and resume pipeline processing on the next submodel step after the specified checkpoint
resume_after = None
# resume_after = 'input_pre_processor'

pipeline.run(models=_MODELS, resume_after=resume_after)


# write final versions of all checkpointed dataframes to CSV files to review results
for table_name in pipeline.checkpointed_tables():
    file_name = "final_%s_table.csv" % table_name
    file_path = os.path.join(orca.get_injectable("output_dir"), file_name)
    pipeline.get_table(table_name).to_csv(file_path)


# tables will no longer be available after pipeline is closed
pipeline.close()

# write checkpoints (this can be called whether or not pipeline is open)
file_path = os.path.join(orca.get_injectable("output_dir"), "checkpoints.csv")
pipeline.get_checkpoints().to_csv(file_path)

t0 = print_elapsed_time("all models", t0)
