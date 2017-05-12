
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
    df = pd.read_csv(data_file_path, comment='#')

    validate_geography_settings(settings, df)

    return df


def validate_geography_settings(settings, control_spec):

    if 'geographies' not in settings:
        raise RuntimeError("geographies not specified in settings")

    if 'geography_settings' not in settings:
        raise RuntimeError("'geography_settings' not found in settings")

    geographies = settings['geographies']
    geography_settings = settings['geography_settings']

    for g in geographies:
        if g not in geography_settings:
            raise RuntimeError("geography '%s' not found in geography_settings" % g)

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


def build_control_table(geo, control_spec, settings, geo_cross_walk_df):

    geography_settings = settings.get('geography_settings')

    control_geographies = settings['geographies']
    assert geo in control_geographies
    control_geographies = control_geographies[control_geographies.index(geo):]

    geo_col = geography_settings[geo].get('id_column')

    # only want controls for control_geographies
    control_spec = control_spec[control_spec['geography'].isin(control_geographies)]
    controls_list = []

    for g in control_geographies:

        print "processing g", g

        # control spec rows for this geography
        spec = control_spec[control_spec['geography'] == g]

        # are there any controls specified for this geography? (e.g. seed has none)
        if len(spec.index) == 0:
            continue

        # control_data for this geography
        control_data_table_name = geography_settings[g].get('control_data_table')
        control_data_df = orca.get_table(control_data_table_name).to_frame()

        control_data_columns = [geo_col] + spec.control_field.tolist()

        if g == geo:
            # for top level, we expect geo_col, and need to group and sum
            assert geo_col in control_data_df.columns
            controls = control_data_df[control_data_columns]
            controls.set_index(geo_col, inplace=True)
        else:
            # sum controls to geo level

            # add geo_col to control_data table
            if geo_col not in control_data_df.columns:
                sub_geo_col = geography_settings[g].get('id_column')
                # create series mapping sub_geo id to geo id
                sub_to_geog = geo_cross_walk_df[[sub_geo_col, geo_col]]\
                    .groupby(sub_geo_col, as_index=True).min()[geo_col]
                control_data_df[geo_col] = control_data_df[sub_geo_col].map(sub_to_geog)

            # sum controls to geo level
            controls = control_data_df[control_data_columns].groupby(geo_col, as_index=True).sum()

        controls_list.append(controls)

    # concat geography columns
    controls = pd.concat(controls_list, axis=1)

    # rename columns from seed_col to target
    columns = {c: t for c, t in zip(control_spec.control_field, control_spec.target)}
    controls.rename(columns=columns, inplace=True)

    # reorder columns to match order of control_spec rows
    controls = controls[control_spec.target]

    return controls


@orca.step()
def setup_data_structures(settings, households, persons, control_spec, geo_cross_walk):

    geography_settings = settings.get('geography_settings')

    hh_weight_col = settings['household_weight_col']

    households_df = households.to_frame()
    persons_df = persons.to_frame()
    geo_cross_walk_df = geo_cross_walk.to_frame()

    incidence_table = build_incidence_table(control_spec, settings, households_df, persons_df)
    incidence_table['sample_weight'] = households_df[hh_weight_col]

    # add seed_col to incidence table
    seed_col = geography_settings['seed'].get('id_column')
    incidence_table[seed_col] = households_df[seed_col]

    # add meta_col to incidence table
    meta_col = geography_settings['meta'].get('id_column')
    # create series mapping seed id to meta geography id
    seed_to_meta \
        = geo_cross_walk_df[[seed_col, meta_col]].groupby(seed_col, as_index=True).min()[meta_col]
    incidence_table[meta_col] = incidence_table[seed_col].map(seed_to_meta)

    orca.add_table('incidence_table', incidence_table)

    geographies = settings['geographies']
    for g in geographies:
        controls = build_control_table(g, control_spec, settings, geo_cross_walk_df)
        control_table_name = '%s_controls' % g
        orca.add_table(control_table_name, controls)

        print "\n", control_table_name
        print controls
