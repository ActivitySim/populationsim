import os


from activitysim.core import inject_defaults

from activitysim.core import tracing
from activitysim.core import pipeline
from activitysim.core import inject

from activitysim.core.tracing import print_elapsed_time
from activitysim.core.config import handle_standard_args

from populationsim import steps
from populationsim.util import setting

handle_standard_args()

tracing.config_logger()

t0 = print_elapsed_time()

# get the run list (name was possibly specified on the command line)
run_list_name = inject.get_injectable('run_list_name', 'run_list')

# run list from settings file is dict with list of 'steps' and optional 'resume_after'
run_list = setting(run_list_name)
assert 'steps' in run_list, "Did not find steps in run_list"

# list of steps and possible resume_after in run_list
steps = run_list.get('steps')
resume_after = run_list.get('resume_after', None)

# they may have overridden resume_after on command line
resume_after = inject.get_injectable('resume_after', resume_after)

pipeline.run(models=steps, resume_after=resume_after)

expanded_household_ids = pipeline.get_table('expanded_household_ids')

taz_hh_counts = expanded_household_ids.groupby('TAZ').size()
print taz_hh_counts
print len(taz_hh_counts)
print taz_hh_counts.loc[100]

# tables will no longer be available after pipeline is closed
pipeline.close_pipeline()

# write checkpoints (this can be called whether or not pipeline is open)
# file_path = os.path.join(inject.get_injectable("output_dir"), "checkpoints.csv")
# pipeline.get_checkpoints().to_csv(file_path)

t0 = print_elapsed_time("all models", t0)
