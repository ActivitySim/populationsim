# PopulationSim
# See full license in LICENSE.txt.

import logging

import numpy as np
import pandas as pd

import cylp
import cvxpy as cvx

from activitysim.core import tracing


import cylp
import cvxpy as cvx


CVX_STATUS = {
    cvx.OPTIMAL: 'OPTIMAL',
    cvx.INFEASIBLE: 'INFEASIBLE',
    cvx.UNBOUNDED: 'UNBOUNDED',
    cvx.OPTIMAL_INACCURATE: 'OPTIMAL_INACCURATE',
    cvx.INFEASIBLE_INACCURATE: 'INFEASIBLE_INACCURATE',
    cvx.UNBOUNDED_INACCURATE: 'UNBOUNDED_INACCURATE',
    None: 'FAILED'
}

STATUS_SUCCESS = ['OPTIMAL', 'FEASIBLE', 'OPTIMAL_INACCURATE']

logger = logging.getLogger(__name__)

DUMP = True


def dump_df(dump_switch, df, trace_label, fname):
    if dump_switch:
        trace_label = "%s.DUMP.%s" % (trace_label, fname)

        if len(df.index) > 0:
            tracing.write_csv(df, file_name=trace_label, transpose=False)


class SimulIntegerizer(object):

    def __init__(self,
                 incidence_df,
                 parent_weights, parent_controls_df,
                 sub_weights, sub_controls_df,
                 control_spec, total_hh_control_col
                 ):

        sub_control_zones = list(sub_weights.columns)

        sample_count = len(parent_weights.index)
        sub_zone_count = len(sub_weights.columns)

        assert len(sub_weights.index) == sample_count
        assert len(incidence_df.index) == sample_count
        assert len(sub_controls_df.index) == sub_zone_count
        assert len(sub_weights.columns) == sub_zone_count
        assert total_hh_control_col in sub_controls_df.columns
        assert total_hh_control_col in parent_controls_df.columns

        self.incidence_df = incidence_df
        self.parent_weights = parent_weights
        self.parent_controls_df = parent_controls_df
        self.sub_weights = sub_weights
        self.sub_controls_df = sub_controls_df

        parent_countrol_cols = list(parent_controls_df.columns)

        parent_control_spec = control_spec[control_spec.target.isin(parent_countrol_cols)]
        # should be in same order
        assert (parent_countrol_cols == parent_control_spec.target).all()

        self.parent_countrol_importance = parent_control_spec.importance

        relaxed_parent_control_totals = \
            np.round(np.dot(np.asanyarray(parent_weights), incidence_df[parent_countrol_cols].as_matrix()))
        relaxed_parent_control_totals = \
            pd.Series(relaxed_parent_control_totals, index=parent_countrol_cols)


        sub_countrol_cols = list(sub_controls_df.columns)
        sub_control_spec = control_spec[control_spec.target.isin(sub_countrol_cols)]
        assert (sub_countrol_cols == sub_control_spec.target).all()

        self.sub_countrol_importance = sub_control_spec.importance


        print "\nincidence_df\n", self.incidence_df

        print "\nparent_controls_df\n", self.parent_controls_df
        print "\nrelaxed_parent_control_totals\n", relaxed_parent_control_totals

        print "\nparent_countrol_importance\n", self.parent_countrol_importance
        print "\nparent_weights\n", self.parent_weights

        print "\nsub_controls_df\n", self.sub_controls_df
        print "\nsub_countrol_importance\n", self.sub_countrol_importance
        print "\nsub_weights\n", self.sub_weights



    def integerize(self):

        sub_incidence_df = self.incidence_df[self.sub_controls_df.columns]

        print "sub_incidence_df\n", sub_incidence_df


        assert False

        weights_out = None
        status = None
        return weights_out, status

        return status




def do_simul_integerizing(
        incidence_df,
        parent_weights, parent_controls_df,
        sub_weights, sub_controls_df,
        parent_geography, parent_id,
        sub_geography,
        control_spec, total_hh_control_col):

    trace_label = "do_simul_integerizing_%s_%s" % (parent_geography, parent_id)

    zero_weight_rows = (parent_weights == 0)
    if zero_weight_rows.any():
        logger.info("omitting %s zero weight rows out of %s"
                    % (zero_weight_rows.sum(), len(incidence_df.index)))
        incidence_df = incidence_df[~zero_weight_rows]
        parent_weights = parent_weights[~zero_weight_rows]
        sub_weights = sub_weights[~zero_weight_rows]

    sample_count = len(parent_weights.index)

    sub_zone_count = len(sub_weights.columns)

    assert len(sub_weights.index) == sample_count
    assert len(incidence_df.index) == sample_count
    assert len(sub_controls_df.index) == sub_zone_count
    assert len(sub_weights.columns) == sub_zone_count
    assert total_hh_control_col in sub_controls_df.columns
    assert total_hh_control_col in parent_controls_df.columns

    print "parent_geography", parent_geography, parent_id
    print "sub_geography", sub_geography

    # dump_df(DUMP, incidence_df, trace_label, 'incidence_df')
    # dump_df(DUMP, incidence_df[control_spec.target], trace_label, 'incidence_table')
    # dump_df(DUMP, sub_weights, trace_label, 'sub_weights')

    integerizer = SimulIntegerizer(
        incidence_df,
        parent_weights,
        parent_controls_df,
        sub_weights,
        sub_controls_df,
        control_spec,
        total_hh_control_col
    )

    # otherwise, solve for the integer weights using the Mixed Integer Programming solver.
    status = integerizer.integerize()

    assert False

    # restore zero_weight_rows to integerized_weights
    integerized_weights = pd.Series(0.0, index=zero_weight_rows.index)
    nonzero_integerized_weights = integerizer.weights['integerized_weight']
    integerized_weights.update(nonzero_integerized_weights)

    logger.debug("SimulIntegerizer status %s" % (status,))

    return integerized_weights, status
