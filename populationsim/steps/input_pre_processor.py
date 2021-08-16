# PopulationSim
# See full license in LICENSE.txt.

import logging
import os

import pandas as pd
import numpy as np

from activitysim.core import inject
from activitysim.core import config
from activitysim.core import input

logger = logging.getLogger(__name__)


@inject.step()
def input_pre_processor():
    """
    Read input text files and save them as pipeline tables for use in subsequent steps.

    The files to read as specified by table_list, and array of dicts that specify the
    input file name, the name of the pipeline table, along with keys allow the specification
    of pre-processing steps.

    By default, reads table_list from 'input_table_list' in settings.yaml,
    unless an alternate table_list name is specified as a model step argument 'table_list'.
    (This allows alternate/additional input files to be read for repop)

    In the case of repop, this step is being run after an initial run has completed,
    in which case the input_table_list may specify replacement tables.
    (e.g. lowest geography controls that will replace the previous low controls dataframe.)

    See input_table_list in settings.yaml in the example folder for a working example

    +--------------+----------------------------------------------------------+
    | key          | description                                              |
    +==============+=========================================+================+
    | tablename    | name of pipeline table in which to store dataframe       |
    +--------------+----------------------------------------------------------+
    | filename     | name of csv file to read (in data_dir)                   |
    +--------------+----------------------------------------------------------+
    | column_map   | list of input columns to rename from_name: to_name       |
    +--------------+----------------------------------------------------------+
    | index_col    | name of column to set as dataframe index column          |
    +--------------+----------------------------------------------------------+
    | drop_columns | list of column names of columns to drop                  |
    +--------------+----------------------------------------------------------+

    """

    # alternate table list name may have been provided as a model argument
    table_list_name = inject.get_step_arg('table_list', default='input_table_list')
    table_list = config.setting(table_list_name)

    assert table_list is not None, "no table list '%s' found in settings." % table_list_name

    logger.info('Using table list: %s' % table_list)

    for table_info in table_list:

        tablename = table_info.get('tablename')
        df = input.read_from_table_info(table_info)
        logger.info('registering table %s' % tablename)

        # add (or replace) pipeline table
        repop = inject.get_step_arg('repop', default=False)
        inject.add_table(tablename, df, replace=repop)
