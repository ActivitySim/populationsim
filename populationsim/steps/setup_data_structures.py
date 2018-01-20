
# PopulationSim
# See full license in LICENSE.txt.

import logging
import os

import pandas as pd
import numpy as np

from activitysim.core import inject
from activitysim.core import pipeline

from ..assign import assign_variable
from helper import control_table_name
from helper import get_control_table
from helper import get_control_data_table

from populationsim.util import setting

logger = logging.getLogger(__name__)


def read_control_spec(data_filename, configs_dir):

    # read the csv file
    data_file_path = os.path.join(configs_dir, data_filename)
    if not os.path.exists(data_file_path):
        raise RuntimeError(
            "initial_seed_balancing - control file not found: %s" % (data_file_path,))

    logger.info("Reading control file %s" % data_file_path)
    control_spec = pd.read_csv(data_file_path, comment='#')

    geographies = setting('geographies')

    if 'geography' not in control_spec.columns:
        raise RuntimeError("missing geography column in controls file")

    for g in control_spec.geography.unique():
        if g not in geographies:
            raise RuntimeError("unknown geography column '%s' in control file" % g)

    return control_spec


def build_incidence_table(control_spec, households_df, persons_df, crosswalk_df):

    hh_col = setting('household_id_col')

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


def add_geography_columns(incidence_table, households_df, crosswalk_df):
    """
    Add seed and meta geography columns to incidence_table

    Parameters
    ----------
    incidence_table
    households_df
    crosswalk_df

    Returns
    -------

    """

    geographies = setting('geographies')
    meta_geography = geographies[0]
    seed_geography = setting('seed_geography')

    # add seed_geography col to incidence table
    incidence_table[seed_geography] = households_df[seed_geography]

    # add meta column to incidence table
    seed_to_meta = \
        crosswalk_df[[seed_geography, meta_geography]] \
        .groupby(seed_geography, as_index=True).min()[meta_geography]
    incidence_table[meta_geography] = incidence_table[seed_geography].map(seed_to_meta)

    return incidence_table


def build_control_table(geo, control_spec, crosswalk_df):

    # control_geographies is list with target geography and the geographies beneath it
    control_geographies = setting('geographies')
    assert geo in control_geographies
    control_geographies = control_geographies[control_geographies.index(geo):]

    # only want controls for control_geographies
    control_spec = control_spec[control_spec['geography'].isin(control_geographies)]
    controls_list = []

    # for each geography at or beneath target geography
    for g in control_geographies:

        # control spec rows for this geography
        spec = control_spec[control_spec['geography'] == g]

        # are there any controls specified for this geography? (e.g. seed has none)
        if len(spec.index) == 0:
            continue

        # control_data for this geography
        control_data_df = get_control_data_table(g)

        control_data_columns = [geo] + spec.control_field.tolist()

        if g == geo:
            # for top level, we expect geo_col, and need to group and sum
            assert geo in control_data_df.columns
            controls = control_data_df[control_data_columns]
            controls.set_index(geo, inplace=True)
        else:
            # aggregate sub geography control totals to the target geo level

            # add geo_col to control_data table
            if geo not in control_data_df.columns:
                # create series mapping sub_geo id to geo id
                sub_to_geog = crosswalk_df[[g, geo]].groupby(g, as_index=True).min()[geo]

                control_data_df[geo] = control_data_df[g].map(sub_to_geog)

            # aggregate (sum) controls to geo level
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


def build_crosswalk_table():
    """
    build crosswalk table filtered to include only zones in lowest geography
    """

    geographies = setting('geographies')

    crosswalk_data_table = inject.get_table('geo_cross_walk').to_frame()

    # dont need any other geographies
    crosswalk = crosswalk_data_table[geographies]

    # filter geo_cross_walk_df to only include geo_ids with lowest_geography controls
    # (just in case geo_cross_walk_df table contains rows for unused low zones)
    low_geography = geographies[-1]
    low_control_data_df = get_control_data_table(low_geography)
    rows_in_low_controls = crosswalk[low_geography].isin(low_control_data_df[low_geography])
    crosswalk = crosswalk[rows_in_low_controls]

    return crosswalk


def build_grouped_incidence_table(incidence_table, control_spec, seed_geography):

    hh_incidence_table = incidence_table
    household_id_col = setting('household_id_col')

    hh_groupby_cols = list(control_spec.target) + [seed_geography]
    hh_grouper = hh_incidence_table.groupby(hh_groupby_cols)
    group_incidence_table = hh_grouper.max()
    group_incidence_table['sample_weight'] = hh_grouper.sum()['sample_weight']
    group_incidence_table['group_size'] = hh_grouper.count()['sample_weight']
    group_incidence_table = group_incidence_table.reset_index()

    logger.info("grouped incidence table has %s entries, ungrouped has %s"
                % (len(group_incidence_table.index), len(hh_incidence_table.index)))

    # add group_id of each hh to hh_incidence_table
    group_incidence_table['group_id'] = group_incidence_table.index
    hh_incidence_table['group_id'] = hh_incidence_table[hh_groupby_cols].merge(
        group_incidence_table[hh_groupby_cols + ['group_id']],
        on=hh_groupby_cols,
        how='left').group_id.astype(int).values

    # it doesn't really matter what the incidence_table index is until we create population
    # when we need to expand each group to constituent households
    # but incidence_table should have the same name whether grouped or ungrouped
    # so that the rest of the steps can handle them interchangeably
    group_incidence_table.index.name = hh_incidence_table.index.name

    # create table mapping household_groups to households and their sample_weights
    # explicitly provide hh_id as a column to make it easier for use when expanding population
    household_groups = hh_incidence_table[['group_id', 'sample_weight']].copy()
    household_groups[household_id_col] = household_groups.index.astype(int)

    return group_incidence_table, household_groups


def filter_households(households_df, persons_df, crosswalk_df):
    """
    Filter households and persons tables, removing zero weight households
    and any households not in seed zones.

    Returns filtered households_df and persons_df
    """

    # drop any zero weight households (there are some in calm data)
    hh_weight_col = setting('household_weight_col')
    households_df = households_df[households_df[hh_weight_col] > 0]

    # remove any households not in seed zones
    seed_geography = setting('seed_geography')
    seed_ids = crosswalk_df[seed_geography].unique()

    rows_in_seed_zones = households_df[seed_geography].isin(seed_ids)
    if rows_in_seed_zones.any():
        households_df = households_df[rows_in_seed_zones]
        logger.info("dropped %s households not in seed zones" % (~rows_in_seed_zones).sum())
        logger.info("kept %s households in seed zones" % len(households_df))

    return households_df, persons_df


@inject.step()
def setup_data_structures(output_dir, settings, configs_dir, households, persons):
    """
    Setup geographic correspondence (crosswalk), control sets, and incidence tables.

    If GROUP_BY_INCIDENCE_SIGNATURE setting is enabled, then incidence table rows are
    household group ids and and additional household_groups table is created mapping hh group ids
    to actual hh_ids.

    Parameters
    ----------
    output_dir: str
    settings: dict
        contents of settings.yaml as dict
    configs_dir: str
    households: pipeline table
    persons: pipeline table

    creates pipeline tables:
        crosswalk
        controls
        geography-specific controls
        incidence_table
        household_groups (if GROUP_BY_INCIDENCE_SIGNATURE setting is enabled)

    modifies tables:
        households
        persons

    """

    seed_geography = setting('seed_geography')

    households_df = households.to_frame()
    persons_df = persons.to_frame()

    crosswalk_df = build_crosswalk_table()
    inject.add_table('crosswalk', crosswalk_df)

    control_spec = read_control_spec(setting('control_file_name', 'controls.csv'), configs_dir)
    inject.add_table('control_spec', control_spec)

    geographies = settings['geographies']
    for g in geographies:
        controls = build_control_table(g, control_spec, crosswalk_df)
        inject.add_table(control_table_name(g), controls)

    households_df, persons_df = filter_households(households_df, persons_df, crosswalk_df)
    pipeline.replace_table('households', households_df)
    pipeline.replace_table('persons', persons_df)

    incidence_table = \
        build_incidence_table(control_spec, households_df, persons_df, crosswalk_df)

    incidence_table = add_geography_columns(incidence_table, households_df, crosswalk_df)

    # add sample_weight col to incidence table
    hh_weight_col = setting('household_weight_col')
    incidence_table['sample_weight'] = households_df[hh_weight_col]

    if setting('GROUP_BY_INCIDENCE_SIGNATURE'):
        group_incidence_table, household_groups \
            = build_grouped_incidence_table(incidence_table, control_spec, seed_geography)

        inject.add_table('household_groups', household_groups)
        inject.add_table('incidence_table', group_incidence_table)
    else:
        inject.add_table('incidence_table', incidence_table)


@inject.step()
def repop_setup_data_structures(settings, configs_dir, households, persons):

    seed_geography = setting('seed_geography')
    geographies = setting('geographies')
    low_geography = geographies[-1]

    # replace crosswalk table
    crosswalk_df = build_crosswalk_table()
    pipeline.replace_table('crosswalk', crosswalk_df)

    # replace control_spec
    control_file_name = setting('repop_control_file_name', 'repop_controls.csv')
    control_spec = read_control_spec(control_file_name, configs_dir)
    pipeline.replace_table('control_spec', control_spec)

    # build incidence_table with repop controls and households in repop zones
    # filter households (dropping any not in crosswalk) in order to build incidence_table
    # We DO NOT REPLACE households and persons as we need full tables to synthesize population
    households_df = households.to_frame()
    persons_df = persons.to_frame()
    households_df, repop_persons_df = filter_households(households_df, persons_df, crosswalk_df)
    incidence_table = build_incidence_table(control_spec, households_df, persons_df, crosswalk_df)
    incidence_table = add_geography_columns(incidence_table, households_df, crosswalk_df)
    # add sample_weight col to incidence table
    hh_weight_col = setting('household_weight_col')
    incidence_table['sample_weight'] = households_df[hh_weight_col]

    # rebuild control tables with only the low level controls (aggregated at higher levels)
    for g in geographies:
        controls = build_control_table(g, control_spec, crosswalk_df)
        pipeline.replace_table(control_table_name(g), controls)

    if setting('GROUP_BY_INCIDENCE_SIGNATURE'):
        group_incidence_table, household_groups \
            = build_grouped_incidence_table(incidence_table, control_spec, seed_geography)

        pipeline.replace_table('household_groups', household_groups)
        pipeline.replace_table('incidence_table', group_incidence_table)
    else:
        pipeline.replace_table('incidence_table', incidence_table)
