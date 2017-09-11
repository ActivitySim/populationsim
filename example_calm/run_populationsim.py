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

logger.info("USE_CVXPY: %s"
            % setting('USE_CVXPY'))
logger.info("USE_SIMUL_INTEGERIZER: %s"
            % setting('USE_SIMUL_INTEGERIZER'))
logger.info("GROUP_BY_INCIDENCE_SIGNATURE: %s"
            % setting('GROUP_BY_INCIDENCE_SIGNATURE'))
logger.info("INTEGERIZE_WITH_BACKSTOPPED_CONTROLS: %s"
            % setting('INTEGERIZE_WITH_BACKSTOPPED_CONTROLS'))
logger.info("SUB_BALANCE_WITH_FLOAT_SEED_WEIGHTS: %s"
            % setting('SUB_BALANCE_WITH_FLOAT_SEED_WEIGHTS'))
logger.info("LOW_BALANCE_WITH_FLOAT_SEED_WEIGHTS: %s"
            % setting('LOW_BALANCE_WITH_FLOAT_SEED_WEIGHTS'))
logger.info("meta_control_data: %s"
            % setting('meta_control_data'))
logger.info("control_file_name: %s"
            % setting('control_file_name'))


_MODELS = [
    'input_pre_processor',
    'setup_data_structures',
    'initial_seed_balancing',
    'meta_control_factoring',
    'final_seed_balancing',
    'integerize_final_seed_weights',
    'sub_balancing',
    'low_balancing',
    # 'expand_population',
    'summarize',
    'write_results'
]

# If you provide a resume_after argument to pipeline.run
# the pipeline manager will attempt to load checkpointed tables from the checkpoint store
# and resume pipeline processing on the next submodel step after the specified checkpoint
resume_after = None
resume_after = 'sub_balancing'

pipeline.run(models=_MODELS, resume_after=resume_after)


# tables will no longer be available after pipeline is closed
pipeline.close()

t0 = ("all models", t0)
