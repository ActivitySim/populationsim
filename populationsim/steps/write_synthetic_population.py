# PopulationSim
# See full license in LICENSE.txt.

import logging
import os
import pandas as pd

from activitysim.core import pipeline
from activitysim.core import inject

from activitysim.core.config import setting

logger = logging.getLogger(__name__)


def merge_seed_data(expanded_household_ids, seed_data_df, seed_columns, trace_label):

    seed_geography = setting('seed_geography')
    hh_col = setting('household_id_col')

    df_columns = seed_data_df.columns.values

    # warn of any columns that aren't in seed_data_df
    for c in seed_columns:
        if c not in df_columns and c != hh_col:
            logger.warning("column '%s' not in %s" % (c, trace_label))

    # remove any columns that aren't in seed_data_df
    df_columns = [c for c in seed_columns if c in df_columns]

    # seed_geography column in seed_data_df is redundant (already in expanded_household_ids table)
    if seed_geography in df_columns:
        df_columns.remove(seed_geography)

    # join to seed_data on either index or hh_col (for persons)
    right_index = (seed_data_df.index.name == hh_col)
    right_on = hh_col if hh_col in seed_data_df.columns and not right_index else None
    assert right_index or right_on

    if right_on and hh_col not in df_columns:
        df_columns.append(hh_col)

    merged_df = pd.merge(
        how="left",
        left=expanded_household_ids,
        right=seed_data_df[df_columns],
        left_on=hh_col,
        right_index=right_index,
        right_on=right_on
    )

    if hh_col not in seed_columns:
        del merged_df[hh_col]

    return merged_df


@inject.step()
def write_synthetic_population(expanded_household_ids, households, persons, output_dir):
    """
    Write synthetic households and persons tables to output dir as csv files.
    The settings file allows specification of output file names, household_id column name,
    and seed data attribute columns to include in output files.

    Parameters
    ----------
    expanded_household_ids : pipeline table
    households : pipeline table
    persons : pipeline table
    output_dir : str

    Returns
    -------

    """

    if setting('NO_INTEGERIZATION_EVER', False):
        logger.warning("skipping write_synthetic_population: NO_INTEGERIZATION_EVER")
        return

    expanded_household_ids = expanded_household_ids.to_frame()
    households = households.to_frame()
    persons = persons.to_frame()

    SETTINGS_NAME = 'output_synthetic_population'
    synthetic_tables_settings = setting(SETTINGS_NAME)
    if synthetic_tables_settings is None:
        raise RuntimeError("'%s' not found in settings" % SETTINGS_NAME)

    hh_col = setting('household_id_col')
    synthetic_hh_col = synthetic_tables_settings.get('household_id', 'HH_ID')

    # - assign household_ids to synthetic population
    expanded_household_ids.reset_index(drop=True, inplace=True)
    expanded_household_ids['synthetic_hh_id'] = expanded_household_ids.index + 1

    # - households

    TABLE_NAME = 'households'
    options = synthetic_tables_settings.get(TABLE_NAME, None)
    if options is None:
        raise RuntimeError("Options for '%s' not found in '%s' in settings" %
                           (TABLE_NAME, SETTINGS_NAME))

    seed_columns = options.get('columns')

    if synthetic_hh_col.lower() in [c.lower() for c in seed_columns]:
        raise RuntimeError("synthetic household_id column '%s' also appears in seed column list" %
                           synthetic_hh_col)

    df = merge_seed_data(
        expanded_household_ids,
        households,
        seed_columns=seed_columns,
        trace_label=TABLE_NAME)

    # synthetic_hh_id is index
    df.rename(columns={'synthetic_hh_id': synthetic_hh_col}, inplace=True)
    df.set_index(synthetic_hh_col, inplace=True)

    filename = options.get('filename', '%s.csv' % TABLE_NAME)
    file_path = os.path.join(output_dir, filename)
    df.to_csv(file_path, index=True)

    # - persons

    TABLE_NAME = 'persons'
    options = synthetic_tables_settings.get(TABLE_NAME, None)
    if options is None:
        raise RuntimeError("Options for '%s' not found in '%s' in settings" %
                           (TABLE_NAME, SETTINGS_NAME))

    seed_columns = options.get('columns')
    if synthetic_hh_col.lower() in [c.lower() for c in seed_columns]:
        raise RuntimeError("synthetic household_id column '%s' also appears in seed column list" %
                           synthetic_hh_col)

    df = merge_seed_data(
        expanded_household_ids,
        persons,
        seed_columns=seed_columns,
        trace_label=TABLE_NAME)

    # FIXME drop or rename old seed hh_id column?
    df.rename(columns={'synthetic_hh_id': synthetic_hh_col}, inplace=True)

    filename = options.get('filename', '%s.csv' % TABLE_NAME)
    file_path = os.path.join(output_dir, filename)
    df.to_csv(file_path, index=False)
