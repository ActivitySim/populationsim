
# PopulationSim
# See full license in LICENSE.txt.

from builtins import str
import logging
import os

import pandas as pd
import numpy as np

from activitysim.core import assign

logger = logging.getLogger(__name__)


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
                logger.warning("WARNING: assign_variables promoting scalar %s to series" % target)
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

        # log any numpy warnings/errors but don't raise
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
