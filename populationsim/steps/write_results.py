# PopulationSim
# See full license in LICENSE.txt.

import logging

import os
import orca
import pandas as pd

from activitysim.core import pipeline

from populationsim.util import setting

logger = logging.getLogger(__name__)


@orca.step()
def write_results(output_dir):

    output_tables = setting('output_tables')
    if output_tables is None:
        output_tables = pipeline.checkpointed_tables()

    print "\npipeline.checkpointed_tables()\n", pipeline.checkpointed_tables()

    skip_output_tables = setting('skip_output_tables')
    if skip_output_tables is None:
        skip_output_tables = []

    output_tables = [t for t in output_tables if t not in skip_output_tables]

    for table_name in output_tables:
        df = pipeline.get_table(table_name)
        file_name = "%s.csv" % table_name
        logger.info("writing output file %s" % file_name)
        file_path = os.path.join(output_dir, file_name)
        write_index = df.index.name is not None
        df.to_csv(file_path, index=write_index)

    # # write checkpoints (this can be called whether or not pipeline is open)
    # file_path = os.path.join(orca.get_injectable("output_dir"), "checkpoints.csv")
    # pipeline.get_checkpoints().to_csv(file_path)
