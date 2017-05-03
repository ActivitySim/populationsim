
# PopulationSim
# See full license in LICENSE.txt.

import logging
import os

import orca
import pandas as pd
import numpy as np

from ..assign import assign_variable

logger = logging.getLogger(__name__)


@orca.injectable(cache=True)
def control_spec(settings, configs_dir):

    # read the csv file
    data_filename = settings.get('control_file_name', 'controls.csv')
    data_file_path = os.path.join(configs_dir, data_filename)
    if not os.path.exists(data_file_path):
        raise RuntimeError(
            "initial_seed_balancing - control file not found: %s" % (data_file_path,))

    logger.info("Reading control file %s" % data_file_path)
    df = pd.read_csv(data_file_path)

    validate_geography_settings(settings, df)

    return df


def validate_geography_settings(settings, control_spec):

    if 'lower_level_geographies' not in settings:
        raise RuntimeError("lower_level_geographies not specified in settings")

    if 'geographies' not in settings:
        raise RuntimeError("'geographies' not found in settings")

    sub_geographies = settings['lower_level_geographies']
    geographies = settings['geographies']

    for g in sub_geographies:
        if g not in geographies:
            raise RuntimeError("lower_level_geography '%s' not found in geographies" % g)

    if 'geography' not in control_spec.columns:
        raise RuntimeError("missing geography column in controls file")

    for g in control_spec.geography.unique():
        if g not in geographies:
            raise RuntimeError("unknown geography column '%s' in control file" % g)


def build_incidence_table(control_spec, settings, households_df, persons_df):

    hh_col = settings['household_id_col']

    incidence_table = pd.DataFrame(index=households_df.index)

    seed_tables = {
        'households': households_df,
        'persons': persons_df,
    }

    print "households index len %s" % len(households_df.index)
    print "persons index len %s" % len(persons_df.index)

    for control_row in control_spec.itertuples():

        logger.info("control target %s" % control_row.target)
        logger.debug("control_row.seed_table %s" % control_row.seed_table)
        logger.debug("control_row.expression %s" % control_row.expression)

        incidence, trace_results = assign_variable(
            target=control_row.target,
            expression=control_row.expression,
            df=seed_tables[control_row.seed_table],
            locals_dict={'np': np},
            df_alias=control_row.seed_table,
            trace_rows=None
        )

        # convert boolean True/False values to 1/0
        incidence = incidence * 1

        # aggregate person incidence counts to household
        if control_row.seed_table == 'persons':
            df = pd.DataFrame({
                hh_col: persons_df[hh_col],
                'incidence': incidence
            })
            incidence = df.groupby([hh_col], as_index=True).sum()

        incidence_table[control_row.target] = incidence

    return incidence_table


def build_control_table(control_spec, geographies, settings, geo_cross_walk_df):

    sub_geographies = settings['lower_level_geographies']
    seed_col = geographies['seed'].get('id_column')

    # only want controls for sub_geographies
    control_spec = control_spec[control_spec['geography'].isin(sub_geographies)]
    seed_controls = []


    for g in sub_geographies:

        # control spec rows for this geography
        spec = control_spec[ control_spec['geography'] == g ]

        # control_data for this geography
        geography = geographies[g]
        control_data_table_name = geography['control_data_table']
        control_data_df = orca.get_table(control_data_table_name).to_frame()

        # add seed_col to control_data table
        if seed_col not in control_data_df.columns:
            geog_col = geography['id_column']
            geog_to_seed = geo_cross_walk_df[[geog_col, seed_col]].groupby(geog_col, as_index=True).min()[seed_col]
            control_data_df[seed_col] = control_data_df[geog_col].map(geog_to_seed)

        # sum controls to seed level
        cols = [seed_col] + spec.control_field.tolist()
        controls = control_data_df[cols].groupby(seed_col, as_index=True).sum()

        seed_controls.append(controls)

    # concat geography columns
    seed_controls = pd.concat(seed_controls, axis=1)

    # rename columns from seed_col to target
    columns = { column: target for column, target in zip(control_spec.control_field, control_spec.target)}
    seed_controls.rename(columns=columns, inplace=True)

    # reorder columns to match order of control_spec rows
    seed_controls = seed_controls[ control_spec.target ]

    return seed_controls

@orca.step()
def setup_data_structures(settings, households, persons, control_spec, geo_cross_walk):

    geographies = settings.get('geographies')

    hh_weight_col = settings['household_weight_col']

    households_df = households.to_frame()
    persons_df = persons.to_frame()

    incidence_table = build_incidence_table(control_spec, settings, households_df, persons_df)

    # add seed_col to incidence table
    seed_col = geographies['seed'].get('id_column')
    incidence_table[seed_col] = households_df[seed_col]

    # add meta_col to incidence table
    meta_col = geographies['meta'].get('id_column')
    geo_cross_walk_df = geo_cross_walk.to_frame()
    seed_to_meta = geo_cross_walk_df[[seed_col, meta_col]].groupby(seed_col, as_index=True).min()[meta_col]
    incidence_table[meta_col] = incidence_table[seed_col].map(seed_to_meta)

    incidence_table['initial_weight'] = households_df[hh_weight_col]

    #print "incidence_table\n", incidence_table

    orca.add_table('incidence_table', incidence_table)

    seed_controls = build_control_table(control_spec, geographies, settings, geo_cross_walk_df)
    orca.add_table('seed_controls', seed_controls)

