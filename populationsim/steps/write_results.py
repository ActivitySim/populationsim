# PopulationSim
# See full license in LICENSE.txt.

import logging
import os

from activitysim.core import pipeline
from activitysim.core import inject

from populationsim.util import setting

logger = logging.getLogger(__name__)


@inject.step()
def write_results(output_dir):

    output_tables = setting('output_tables')
    if output_tables is None:
        output_tables = pipeline.checkpointed_tables()

    # explicit list of tables to skip (or [] to skip none)
    skip_output_tables = setting('skip_output_tables')

    # if not specified, skip only the input tables
    if skip_output_tables is None:
        skip_output_tables = setting('input_table_list')

    output_tables = [t for t in output_tables if t not in skip_output_tables]

    for table_name in output_tables:
        df = pipeline.get_table(table_name)
        file_name = "%s.csv" % table_name
        logger.info("writing output file %s" % file_name)
        file_path = os.path.join(output_dir, file_name)
        write_index = df.index.name is not None
        df.to_csv(file_path, index=write_index)

    # # write checkpoints (this can be called whether or not pipeline is open)
    # file_path = os.path.join(inject.get_injectable("output_dir"), "checkpoints.csv")
    # pipeline.get_checkpoints().to_csv(file_path)
