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

    output_tables_settings_name = 'output_tables'

    output_tables_settings = setting(output_tables_settings_name)

    output_tables = pipeline.checkpointed_tables()

    if output_tables_settings is not None:

        action = output_tables_settings.get('action')
        tables = output_tables_settings.get('tables')

        if action not in ['include', 'skip']:
            raise "expected %s action '%s' to be either 'include' or 'skip'" % \
                  (output_tables_settings_name, action)

        if action == 'include':
            output_tables = tables
        elif action == 'skip':
            output_tables = [t for t in output_tables if t not in tables]

    # should provide option to also write checkpoints?
    # output_tables.append("checkpoints.csv")

    for table_name in output_tables:
        df = pipeline.get_table(table_name)
        file_name = "%s.csv" % table_name
        logger.info("writing output file %s" % file_name)
        file_path = os.path.join(output_dir, file_name)
        write_index = df.index.name is not None
        df.to_csv(file_path, index=write_index)
