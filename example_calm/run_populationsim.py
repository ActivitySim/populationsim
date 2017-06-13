import os
import logging

import orca


from activitysim.core import inject_defaults
from populationsim import steps

from activitysim.core import tracing
from activitysim.core import pipeline
from activitysim.core.tracing import print_elapsed_time
from populationsim.util import setting

tracing.config_logger()

t0 = print_elapsed_time()


logger = logging.getLogger('populationsim')

logger.info("USE_CVX: %s" % setting('USE_CVX'))

logger.info("GROUP_BY_INCIDENCE_SIGNATURE: %s" % setting('GROUP_BY_INCIDENCE_SIGNATURE'))
logger.info("INTEGERIZE_WITH_BACKSTOPPED_CONTROLS: %s" % setting('INTEGERIZE_WITH_BACKSTOPPED_CONTROLS'))
logger.info("SUB_BALANCE_WITH_FLOAT_SEED_WEIGHTS: %s" % setting('SUB_BALANCE_WITH_FLOAT_SEED_WEIGHTS'))
logger.info("meta_control_data: %s" % setting('meta_control_data'))
logger.info("control_file_name: %s" % setting('control_file_name'))


_MODELS = [
    'input_pre_processor',
    'setup_data_structures',
    'initial_seed_balancing',
    'meta_control_factoring',
    'final_seed_balancing',
    'integerize_final_seed_weights',
    'sub_balancing',
    'low_balancing',
    #
    # # expand household and person records with final weights
    # # to one household and one person record per weight with unique IDs
    'expand_population',

    'summarize'

    # write the household and person files to CSV files
    # 'write_results'
]

# If you provide a resume_after argument to pipeline.run
# the pipeline manager will attempt to load checkpointed tables from the checkpoint store
# and resume pipeline processing on the next submodel step after the specified checkpoint
resume_after = None
#resume_after = 'low_balancing'

pipeline.run(models=_MODELS, resume_after=resume_after)


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

t0 = ("all models", t0)
