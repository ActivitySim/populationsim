
# PopulationSim
# See full license in LICENSE.txt.

import logging
import os

import orca
import pandas as pd
import numpy as np

from activitysim.core import assign


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


@orca.step()
def setup_data_structures(settings, households, persons, control_spec):

    geographies = settings.get('geographies')
    seed_col = geographies['seed'].get('id_column')
    hh_weight_col = settings['household_weight_col']

    households_df = households.to_frame()
    persons_df = persons.to_frame()

    incidence_table = build_incidence_table(control_spec, settings, households_df, persons_df)

    # remember these before we add in geog selection cols
    incidence_cols = incidence_table.columns.values
    orca.add_injectable('incidence_cols', incidence_cols)

    incidence_table[seed_col] = households_df[seed_col]
    incidence_table['initial_weight'] = households_df[hh_weight_col]

    print "incidence_table\n", incidence_table

    orca.add_table('incidence_table', incidence_table)
