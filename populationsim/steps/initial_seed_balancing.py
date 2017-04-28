# PopulationSim
# See full license in LICENSE.txt.

import logging
import os

import orca
import pandas as pd
import numpy as np

from activitysim.core import assign
from ..balancer import ListBalancer


logger = logging.getLogger(__name__)


@orca.injectable()
def control_spec(settings, configs_dir):

    # read the csv file
    data_filename = settings.get('control_file_name', 'controls.csv')
    data_file_path = os.path.join(configs_dir, data_filename)
    if not os.path.exists(data_file_path):
        raise RuntimeError(
            "initial_seed_balancing - control file not found: %s" % (data_file_path,))

    logger.info("Reading control file %s" % data_file_path)
    df = pd.read_csv(data_file_path)

    return df


def assign_variable(target, expression, df, locals_dict, df_alias=None, trace_rows=None):
    """
    Evaluate an expression of a given data table.

    Expressions are evaluated using Python's eval function.
    Python expressions have access to variables in locals_d (and df being
    accessible as variable df.) They also have access to previously assigned
    targets as the assigned target name.

    Users should take care that expressions should result in
    a Pandas Series (scalars will be automatically promoted to series.)

    Parameters
    ----------
    assignment_expressions : pandas.DataFrame of target assignment expressions
        target: target column names
        expression: pandas or python expression to evaluate
    df : pandas.DataFrame
    locals_d : Dict
        This is a dictionary of local variables that will be the environment
        for an evaluation of "python" expression.
    trace_rows: series or array of bools to use as mask to select target rows to trace

    Returns
    -------
    result : pandas.Series
        Will have the index of `df` and columns named by target and containing
        the result of evaluating expression
    trace_df : pandas.Series or None
        a series containing the eval result values for the assignment expression
    """

    np_logger = assign.NumpyLogger(logger)

    def to_series(x, target=None):
        if x is None or np.isscalar(x):
            if target:
                logger.warn("WARNING: assign_variables promoting scalar %s to series" % target)
            return pd.Series([x] * len(df.index), index=df.index)
        return x

    trace_results = None

    # avoid touching caller's passed-in locals_d parameter (they may be looping)
    locals_dict = locals_dict.copy() if locals_dict is not None else {}
    if df_alias:
        locals_dict[df_alias] = df
    else:
        locals_dict['df'] = df

    try:

        # FIXME - log any numpy warnings/errors but don't raise
        np_logger.target = str(target)
        np_logger.expression = str(expression)
        saved_handler = np.seterrcall(np_logger)
        save_err = np.seterr(all='log')

        values = to_series(eval(expression, globals(), locals_dict), target=target)

        # if expression.startswith('@'):
        #     values = to_series(eval(expression, globals(), locals_dict), target=target)
        # else:
        #     values = df.eval(expression)

        np.seterr(**save_err)
        np.seterrcall(saved_handler)

    except Exception as err:
        logger.error("assign_variables error: %s: %s" % (type(err).__name__, str(err)))
        logger.error("assign_variables expression: %s = %s"
                     % (str(target), str(expression)))

        # values = to_series(None, target=target)
        raise err

    if trace_rows is not None:
        trace_results = values[trace_rows]

    return values, trace_results


def seed_balancer(control_spec, seed_id, geographies, settings,
                  households_df, persons_df, geo_cross_walk_df):

    hh_weight_col = settings['household_weight_col']
    hh_col = settings['household_id_col']

    seed_col = geographies['seed'].get('id_column')
    lower_level_geographies = geographies['lower_level_geographies']

    # subset of crosswalk table for this seed geography
    seed_crosswalk_df = geo_cross_walk_df[geo_cross_walk_df[seed_col] == seed_id]

    seed_households_df = households_df[households_df[seed_col] == seed_id]
    seed_persons_df = persons_df[persons_df[seed_col] == seed_id]

    seed_tables = {
        'households': seed_households_df,
        'persons': seed_persons_df,
    }

    print "households index len %s" % len(seed_households_df.index)
    print "persons index len %s" % len(seed_persons_df.index)

    balancer = ListBalancer(
        incidence_table=seed_households_df.index,
        initial_weights=seed_households_df[hh_weight_col]
    )

    # aggregate sub-geographies
    for sub_geog in lower_level_geographies:

        geography = geographies[sub_geog]
        geog_col = geography['id_column']

        logger.info("aggregating %s controls for seed geography id: %s" % (sub_geog, seed_id))
        logger.info("geog_col for %s sub_geog: %s" % (sub_geog, geog_col))

        control_data_table_name = geography['control_data_table']
        control_data_df = orca.get_table(control_data_table_name).to_frame()

        if seed_col in control_data_df.columns:
            logger.info("seed_col %s in control_data_df" % seed_col)
            sub_geog_control_data_df = control_data_df[control_data_df[seed_col] == seed_id]
        else:
            logger.info("seed_col %s not in control_data_df" % seed_col)
            # unique ids for this geography level in current seed_geography
            geog_ids = seed_crosswalk_df[geog_col].unique()
            sub_geog_control_data_df = control_data_df[control_data_df[geog_col].isin(geog_ids)]

        for control_row in control_spec[control_spec['geography'] == sub_geog].itertuples():

            logger.info("control target %s" % control_row.target)
            logger.debug("control_row.geography %s" % control_row.geography)
            logger.debug("control_row.seed_table %s" % control_row.seed_table)
            logger.debug("control_row.expression %s" % control_row.expression)
            logger.debug("control_row.control_field %s" % control_row.control_field)

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
                    hh_col: seed_persons_df[hh_col],
                    'incidence': incidence
                })
                incidence = df.groupby([hh_col], as_index=True).sum()

            # control total is sum of control_field for sub_geography
            control_total = sub_geog_control_data_df[control_row.control_field].sum()

            balancer.add_control_column(
                target=control_row.target,
                incidence=incidence,
                control_total=control_total,
                control_importance_weight=int(control_row.importance)
            )

    return balancer


@orca.step()
def initial_seed_balancing(settings, geo_cross_walk, households, persons, control_spec):

    # seed (puma) balancing, meta level balancing, meta
    # control factoring, and meta final balancing

    # controls.csv
    #
    # Description        | Num HHs
    # Total HH Control   | TRUE         | True for only one columns
    # Geography          | low
    # Seed Table         | households
    # Importance         | 10000
    # Control Field      | HHBASE       | field in control data table to sum as control/constraint
    # Expression         | (WGTP > 0)   | whether the incidence table has a one or a zero

    geographies = settings.get('geographies')
    seed_col = geographies['seed'].get('id_column')
    lower_level_geographies = geographies['lower_level_geographies']
    hh_col = settings['household_id_col']

    households_df = households.to_frame()
    persons_df = persons.to_frame()
    geo_cross_walk_df = geo_cross_walk.to_frame()

    seed_ids = geo_cross_walk_df[seed_col].unique()

    print "################ for seed_id in seed_ids:"
    for seed_id in seed_ids:

        logger.info("seed id %s" % seed_id)

        balancer = seed_balancer(
            control_spec,
            seed_id,
            geographies,
            settings,
            households_df,
            persons_df,
            geo_cross_walk_df)

        print "aggregated controls for seed geography id: %s" % (seed_id,)

        balancer.dump()

        status = balancer.balance()

        logger.info("seed_balancer status: %s" % status)
