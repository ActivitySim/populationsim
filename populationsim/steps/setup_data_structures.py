
# PopulationSim
# See full license in LICENSE.txt.

import logging
import os

import orca
import pandas as pd
import numpy as np

from ..assign import assign_variable
from helper import control_table_name

logger = logging.getLogger(__name__)


def validate_geography_settings(settings):
    if 'geographies' not in settings:
        raise RuntimeError("geographies not specified in settings")

    if 'geography_settings' not in settings:
        raise RuntimeError("'geography_settings' not found in settings")

    if 'crosswalk_data_table' not in settings:
        raise RuntimeError("'crosswalk_data_table' not found in settings")

    geographies = settings['geographies']
    geography_settings = settings['geography_settings']

    for g in geographies:
        if g not in geography_settings:
            raise RuntimeError("geography '%s' not found in geography_settings" % g)


def read_control_spec(settings, configs_dir):
    # read the csv file
    data_filename = settings.get('control_file_name', 'controls.csv')
    data_file_path = os.path.join(configs_dir, data_filename)
    if not os.path.exists(data_file_path):
        raise RuntimeError(
            "initial_seed_balancing - control file not found: %s" % (data_file_path,))

    logger.info("Reading control file %s" % data_file_path)
    control_spec = pd.read_csv(data_file_path, comment='#')

    geographies = settings['geographies']

    if 'geography' not in control_spec.columns:
        raise RuntimeError("missing geography column in controls file")

    for g in control_spec.geography.unique():
        if g not in geographies:
            raise RuntimeError("unknown geography column '%s' in control file" % g)

    return control_spec


def build_incidence_table(control_spec, settings, households_df, persons_df):

    hh_col = settings['household_id_col']

    incidence_table = pd.DataFrame(index=households_df.index)

    seed_tables = {
        'households': households_df,
        'persons': persons_df,
    }

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


def build_control_table(geo, control_spec, settings, crosswalk_df):

    geography_settings = settings.get('geography_settings')

    control_geographies = settings['geographies']
    assert geo in control_geographies
    control_geographies = control_geographies[control_geographies.index(geo):]

    # only want controls for control_geographies
    control_spec = control_spec[control_spec['geography'].isin(control_geographies)]
    controls_list = []

    for g in control_geographies:

        # control spec rows for this geography
        spec = control_spec[control_spec['geography'] == g]

        # are there any controls specified for this geography? (e.g. seed has none)
        if len(spec.index) == 0:
            continue

        # control_data for this geography
        control_data_table_name = geography_settings[g].get('control_data_table')
        control_data_df = orca.get_table(control_data_table_name).to_frame()

        column_map = {geography_settings.get(g)['id_column']: g for g in control_geographies}
        control_data_df = control_data_df.rename(columns=column_map)

        control_data_columns = [geo] + spec.control_field.tolist()

        if g == geo:
            # for top level, we expect geo_col, and need to group and sum
            assert geo in control_data_df.columns
            controls = control_data_df[control_data_columns]
            controls.set_index(geo, inplace=True)
        else:
            # sum sub geography controls to target geo level

            # add geo_col to control_data table
            if geo not in control_data_df.columns:
                # create series mapping sub_geo id to geo id
                sub_to_geog = crosswalk_df[[g, geo]].groupby(g, as_index=True).min()[geo]

                print control_data_df

                control_data_df[geo] = control_data_df[g].map(sub_to_geog)

            # sum controls to geo level
            controls = control_data_df[control_data_columns].groupby(geo, as_index=True).sum()

        controls_list.append(controls)

    # concat geography columns
    controls = pd.concat(controls_list, axis=1)

    # rename columns from seed_col to target
    columns = {c: t for c, t in zip(control_spec.control_field, control_spec.target)}
    controls.rename(columns=columns, inplace=True)

    # reorder columns to match order of control_spec rows
    controls = controls[control_spec.target]

    return controls


def build_crosswalk_table(settings):

    geography_settings = settings.get('geography_settings')
    geographies = settings.get('geographies')

    crosswalk_data_table = orca.get_table(settings.get('crosswalk_data_table')).to_frame()
    column_map = {geography_settings.get(g)['id_column'] : g for g in geographies}
    crosswalk = crosswalk_data_table.rename(columns = column_map)

    # dont need any other geographies
    crosswalk = crosswalk[geographies]

    # filter geo_cross_walk_df to only include geo_ids with lowest_geography controls
    # (just in case geo_cross_walk_df table contains rows for unused geographies)
    lowest_geography = geographies[-1]
    low_col = geography_settings[lowest_geography].get('id_column')
    low_control_data_table_name = geography_settings[lowest_geography].get('control_data_table')
    low_control_data_df = orca.get_table(low_control_data_table_name).to_frame()
    rows_in_low_controls = crosswalk[lowest_geography].isin(low_control_data_df[low_col])
    crosswalk = crosswalk[rows_in_low_controls]

    return crosswalk


@orca.step()
def setup_data_structures(settings, configs_dir, households, persons, geo_cross_walk):

    validate_geography_settings(settings)
    geography_settings = settings.get('geography_settings')

    geographies = settings.get('geographies')
    seed_geography = settings.get('seed_geography')
    meta_geography = geographies[0]

    households_df = households.to_frame()
    persons_df = persons.to_frame()

    crosswalk_df = build_crosswalk_table(settings)
    orca.add_table('crosswalk', crosswalk_df)

    # FIXME - adding as a table (instead of a
    control_spec = read_control_spec(settings, configs_dir)
    orca.add_table('control_spec', control_spec)

    incidence_table = build_incidence_table(control_spec, settings, households_df, persons_df)

    # add seed_col to incidence table
    seed_col = geography_settings[seed_geography].get('id_column')
    incidence_table[seed_geography] = households_df[seed_col]

    # create series mapping seed id to meta geography id
    seed_to_meta \
        = crosswalk_df[[seed_geography, meta_geography]].groupby(seed_geography, as_index=True).min()[meta_geography]
    incidence_table[meta_geography] = incidence_table[seed_geography].map(seed_to_meta)

    # add sample_weight col to incidence table
    hh_weight_col = settings['household_weight_col']
    incidence_table['sample_weight'] = households_df[hh_weight_col]

    geographies = settings['geographies']
    for g in geographies:
        controls = build_control_table(g, control_spec, settings, crosswalk_df)
        orca.add_table(control_table_name(g), controls)

    # remove any rows in incidence table not in seed zones
    # FIXME - assumes geo_cross_walk_df has extra rows
    seed_ids = crosswalk_df[seed_geography].unique()
    rows_in_seed_zones = incidence_table[seed_geography].isin(seed_ids)
    if len(rows_in_seed_zones) != rows_in_seed_zones.sum():
        incidence_table = incidence_table[rows_in_seed_zones]
        rows_dropped = len(rows_in_seed_zones) - len(incidence_table)
        logger.info("dropped %s rows from incidence table" % rows_dropped)
        logger.info("kept %s rows in incidence table" % len(incidence_table))

    orca.add_table('incidence_table', incidence_table)
