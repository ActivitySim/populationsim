# PopulationSim
# See full license in LICENSE.txt.

import logging
import os

from activitysim.core import pipeline
from activitysim.core import inject

from populationsim.util import setting

logger = logging.getLogger(__name__)


@inject.step()
def write_tables(output_dir):
    """
    Write pipeline tables as csv files (in output directory) as specified by output_tables list
    in settings file.

    Pipeline tables are intermediate computational tables, not to be confused with the
    synthetic population tables written by the write_synthetic_population step.

    'output_tables' can specify either a list of output tables to include or to skip
    if no output_tables list is specified, then no checkpointed tables will be written

    Intermediate tables likely to be of particular interest or utility are the controls and weights
    tables for the various geographies. For example, if one of your geographies is TRACT, then:
    TRACT_controls has control totals for every TRACT (and aggregated subzone) controls.
    TRACT_weights has balanced_weight and integer_weight for every TRACT.

    To write all output tables EXCEPT the households and persons tables:

    ::

      output_tables:
        action: skip
        tables:
          - households
          - persons

    To write ONLY the expanded_household_ids table:

    ::

      output_tables:
        action: include
        tables:
           - expanded_household_ids

    Parameters
    ----------
    output_dir: str

    """

    output_tables_settings_name = 'output_tables'

    output_tables_settings = setting(output_tables_settings_name)

    output_tables_list = pipeline.checkpointed_tables()

    if output_tables_settings is None:
        logger.info("No output_tables specified in settings file. Nothing to write.")
        return

    action = output_tables_settings.get('action')
    tables = output_tables_settings.get('tables')

    if action not in ['include', 'skip']:
        raise RuntimeError("expected %s action '%s' to be either 'include' or 'skip'" %
                           (output_tables_settings_name, action))

    if action == 'include':
        output_tables_list = tables
    elif action == 'skip':
        output_tables_list = [t for t in output_tables_list if t not in tables]

    # should provide option to also write checkpoints?
    # output_tables_list.append("checkpoints.csv")

    for table_name in output_tables_list:
        table = inject.get_table(table_name, None)

        if table is None:
            logger.warn("Skipping '%s': Table not found." % table_name)
            continue

        df = table.to_frame()
        file_name = "%s.csv" % table_name
        logger.info("writing output file %s" % file_name)
        file_path = os.path.join(output_dir, file_name)
        write_index = df.index.name is not None
        df.to_csv(file_path, index=write_index)
