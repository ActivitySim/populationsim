import os

import orca


from activitysim.core import inject_defaults

from activitysim.core import tracing
from activitysim.core import pipeline

from activitysim.core.tracing import print_elapsed_time
from activitysim.core.config import handle_standard_args

from populationsim import steps
from populationsim.util import setting

handle_standard_args()

tracing.config_logger()

t0 = print_elapsed_time()

MODELS = setting('models')

# If you provide a resume_after argument to pipeline.run
# the pipeline manager will attempt to load checkpointed tables from the checkpoint store
# and resume pipeline processing on the next submodel step after the specified checkpoint
resume_after = None
# resume_after = 'integerize_final_seed_weights'

pipeline.run(models=MODELS, resume_after=resume_after)


# incidence_table = pipeline.get_table('incidence_table')
# control_spec = pipeline.get_table('control_spec')
#
# g = list(control_spec.target.values)
# #g = ['num_hh',  'hh_size_1',  'hh_size_2']
#
# print g
# print incidence_table.columns.values
# print "households:", len(incidence_table.index)
#
# households_by_controls = incidence_table.groupby(g)[['PUMA']].count()
#
# print "unique:", len(households_by_controls.index)
# print households_by_controls

# write final versions of all checkpointed dataframes to CSV files to review results
if True:
    for table_name in pipeline.checkpointed_tables():
        if table_name in ['households', 'persons']:
            continue
        file_name = "%s.csv" % table_name
        print "writing", file_name
        file_path = os.path.join(orca.get_injectable("output_dir"), file_name)
        pipeline.get_table(table_name).to_csv(file_path, index=True)

# tables will no longer be available after pipeline is closed
pipeline.close()

# write checkpoints (this can be called whether or not pipeline is open)
file_path = os.path.join(orca.get_injectable("output_dir"), "checkpoints.csv")
pipeline.get_checkpoints().to_csv(file_path)

t0 = print_elapsed_time("all models", t0)
