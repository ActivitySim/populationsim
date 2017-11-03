# PopulationSim
# See full license in LICENSE.txt.

import logging
import os
import pandas as pd

from activitysim.core import pipeline
from activitysim.core import inject

from populationsim.util import setting

logger = logging.getLogger(__name__)


def merge_seed_data(expanded_household_ids, seed_data_df, options, trace_label):

    seed_geography = setting('seed_geography')
    hh_col = setting('household_id_col')

    df_columns = seed_data_df.columns.values

    if options:
        action = options.get('action')
        columns = options.get('columns')

        # ensure we grok action
        if action not in ['include', 'skip']:
            raise "expected %s action '%s' to be either 'include' or 'skip'" % \
                  (trace_label, action)

        # warn of any columns that aren't in seed_data_df
        for c in columns:
            if c not in df_columns:
                logger.warn("can't %s column '%s': not in %s" % (action, c, trace_label))

        # remove any columns that aren't in seed_data_df
        columns = [c for c in columns if c in df_columns]

        if action == 'include':
            df_columns = columns
        elif action == 'skip':
            df_columns = [c for c in df_columns if c not in columns]

    # seed_geography column in seed_data_df is redundant (already in expanded_household_ids table)
    if seed_geography in df_columns:
        df_columns.remove(seed_geography)

    merged_df = pd.merge(
        left=expanded_household_ids,
        right=seed_data_df[df_columns],
        left_on=hh_col,
        right_index=True)

    return merged_df


@inject.step()
def synthesize_population(expanded_household_ids, households, persons):

    expanded_household_ids = expanded_household_ids.to_frame()
    households = households.to_frame()
    persons = persons.to_frame()

    synthetic_tables_settings_name = 'synthetic_tables'
    synthetic_tables_settings = setting(synthetic_tables_settings_name, {})

    synthetic_households = merge_seed_data(
        expanded_household_ids,
        households,
        options=synthetic_tables_settings.get('households'),
        trace_label='households')

    inject.add_table('synthetic_households', synthetic_households)

    synthetic_persons = merge_seed_data(
        expanded_household_ids,
        persons,
        options=synthetic_tables_settings.get('persons'),
        trace_label='persons')

    inject.add_table('synthetic_persons', synthetic_persons)
