# PopulationSim
# See full license in LICENSE.txt.

import logging
import os

import orca
import pandas as pd
import numpy as np

from activitysim.core import assign


logger = logging.getLogger(__name__)


@orca.step()
def input_pre_processor(settings, configs_dir, data_dir):

    table_list = settings.get('input_pre_processor', None)

    if not table_list:
        logger.warn('No input_pre_processor table list in settings.')
        return

    for table in table_list:
        print "input_pre_processor processing %s" % table

        table_info = settings.get(table, None)


        # read the csv file
        data_filename = table_info.get('filename', None)
        data_file_path = os.path.join(data_dir, data_filename)
        if not os.path.exists(data_file_path):
            raise RuntimeError("input_pre_processor %s - input file not found: %" % (table, data_file_path, ))
        df = pd.read_csv(data_file_path, comment='#')

        # rename columns
        column_map = table_info.get('column_map', None)
        if column_map:
            df.rename(columns=column_map, inplace=True)

        # set index
        index_col = table_info.get('index_col', None)
        if index_col is not None:
            if index_col in df.columns:
                df.set_index(index_col, inplace=True)
            else:
                df.index.names = [index_col]

        # read expression file
        expression_filename = table_info.get('expression_filename', None)
        if expression_filename:
            expression_file_path = os.path.join(configs_dir, expression_filename)
            if not os.path.exists(expression_file_path):
                raise RuntimeError("input_pre_processor %s - expression file not found: %" % (table, expression_file_path, ))
            spec = assign.read_assignment_spec(expression_file_path)

            df_alias = table_info.get('df_alias', table)

            locals_d = {}

            results, trace_results, trace_assigned_locals \
                = assign.assign_variables(spec, df, locals_d, df_alias=df_alias)
            # for column in results.columns:
            #     orca.add_column(table, column, results[column])

            df = pd.concat([df, results], axis=1)

        logger.info("adding table %s" % table)
        orca.add_table(table, df)
