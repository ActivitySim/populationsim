# PopulationSim
# See full license in LICENSE.txt.

import logging

import numpy as np
import pandas as pd

from populationsim.core import config
from populationsim.integerizing.smart_round import smart_round
from populationsim.integerizing import lp_ortools, lp_cvx

logger = logging.getLogger(__name__)


class SimulIntegerizer:

    def __init__(
        self,
        incidence_df,
        sub_weights,
        sub_controls_df,
        control_spec,
        total_hh_control_col,
        trace_label="",
    ):

        sample_count = len(sub_weights.index)
        sub_zone_count = len(sub_weights.columns)

        assert len(sub_weights.index) == sample_count
        assert len(incidence_df.index) == sample_count
        assert len(sub_controls_df.index) == sub_zone_count
        assert len(sub_weights.columns) == sub_zone_count
        assert total_hh_control_col in sub_controls_df.columns
        assert total_hh_control_col in incidence_df.columns
        assert (incidence_df.columns == control_spec.target).all()

        self.incidence_df = incidence_df
        self.sub_weights = sub_weights
        self.total_hh_control_col = total_hh_control_col

        # control spec rows and control_df columns should be same and in same order
        sub_countrol_cols = list(sub_controls_df.columns)
        sub_control_spec = control_spec[control_spec.target.isin(sub_countrol_cols)]
        self.sub_controls_df = sub_controls_df[sub_control_spec.target.values]
        self.sub_control_importance = sub_control_spec.importance

        # only care about parent control columns NOT in sub_controls
        # control spec rows and control_df columns should be same and in same order
        parent_control_spec = control_spec[
            ~control_spec.target.isin(self.sub_controls_df.columns)
        ]
        self.parent_countrol_cols = parent_control_spec.target.values
        self.parent_countrol_importance = parent_control_spec.importance

        assert total_hh_control_col not in self.parent_countrol_cols

        self.trace_label = trace_label

        # Choose the integerizer function based on configuration
        if config.setting("USE_CVXPY", False):
            self.integerizer_func = lp_cvx.np_simul_integerizer_cvx
        else:
            self.integerizer_func = lp_ortools.np_simul_integerizer_ortools

        self.timeout_in_seconds = config.setting("INTEGIZER_TIMEOUT", 60)

    def integerize(self):

        # - subzone

        total_hh_sub_control_index = self.sub_controls_df.columns.get_loc(
            self.total_hh_control_col
        )

        # FIXME - shouldn't need this?
        total_hh_parent_control_index = -1

        sub_incidence = self.incidence_df[self.sub_controls_df.columns]
        sub_incidence = sub_incidence.values.astype(np.float64)

        sub_float_weights = self.sub_weights.values.transpose().astype(np.float64)
        sub_int_weights = sub_float_weights.astype(int)
        sub_resid_weights = sub_float_weights % 1.0

        # print "sub_float_weights\n", sub_float_weights
        # print "sub_int_weights\n", sub_int_weights
        # print "sub_resid_weights\n", sub_resid_weights

        sub_control_totals = np.asanyarray(self.sub_controls_df).astype(np.int64)
        sub_control_importance = np.asanyarray(self.sub_control_importance).astype(
            np.float64
        )

        relaxed_sub_control_totals = np.dot(sub_float_weights, sub_incidence)

        # lp_right_hand_side
        lp_right_hand_side = np.round(relaxed_sub_control_totals) - np.dot(
            sub_int_weights, sub_incidence
        )
        lp_right_hand_side = np.maximum(lp_right_hand_side, 0.0)

        # inequality constraint upper bounds
        sub_num_households = relaxed_sub_control_totals[
            :, (total_hh_sub_control_index,)
        ]
        sub_max_control_values = np.amax(sub_incidence, axis=0) * sub_num_households
        relax_ge_upper_bound = np.maximum(
            sub_max_control_values - lp_right_hand_side, 0
        )
        hh_constraint_ge_bound = np.maximum(sub_max_control_values, lp_right_hand_side)

        # equality constraint for the total households control
        total_hh_right_hand_side = lp_right_hand_side[:, total_hh_sub_control_index]

        # - parent
        parent_incidence = self.incidence_df[self.parent_countrol_cols]
        parent_incidence = parent_incidence.values.astype(np.float64)

        # note:
        # sum(sub_int_weights) might be different from parent_float_weights.astype(int)
        # parent_resid_weights might be > 1.0, OK as we are using in objective, not for rounding
        parent_float_weights = np.sum(sub_float_weights, axis=0)
        parent_int_weights = np.sum(sub_int_weights, axis=0)
        parent_resid_weights = np.sum(sub_resid_weights, axis=0)

        # print "parent_float_weights\n", parent_float_weights
        # print "parent_int_weights\n", parent_int_weights
        # print "parent_resid_weights\n", parent_resid_weights

        # - parent control totals based on sub_zone balanced weights
        relaxed_parent_control_totals = np.dot(parent_float_weights, parent_incidence)

        parent_countrol_importance = np.asanyarray(
            self.parent_countrol_importance
        ).astype(np.float64)

        parent_lp_right_hand_side = np.round(relaxed_parent_control_totals) - np.dot(
            parent_int_weights, parent_incidence
        )
        parent_lp_right_hand_side = np.maximum(parent_lp_right_hand_side, 0.0)

        # - create the inequality constraint upper bounds
        parent_num_households = np.sum(sub_num_households)

        parent_max_possible_control_values = (
            np.amax(parent_incidence, axis=0) * parent_num_households
        )
        parent_relax_ge_upper_bound = np.maximum(
            parent_max_possible_control_values - parent_lp_right_hand_side, 0
        )
        parent_hh_constraint_ge_bound = np.maximum(
            parent_max_possible_control_values, parent_lp_right_hand_side
        )

        # how could this not be the case?
        if not (
            parent_hh_constraint_ge_bound == parent_max_possible_control_values
        ).all():
            print("\nSimulIntegerizer integerizing", self.trace_label)
            logger.warning(
                "parent_hh_constraint_ge_bound != parent_max_possible_control_values"
            )
            logger.warning(
                "parent_hh_constraint_ge_bound:      %s" % parent_hh_constraint_ge_bound
            )
            logger.warning(
                "parent_max_possible_control_values: %s"
                % parent_max_possible_control_values
            )
            print("\n")
            # assert (parent_hh_constraint_ge_bound == parent_max_possible_control_values).all()

        resid_weights_out, status_text = self.integerizer_func(
            sub_int_weights,
            parent_countrol_importance,
            parent_relax_ge_upper_bound,
            sub_control_importance,
            sub_float_weights,
            sub_resid_weights,
            lp_right_hand_side,
            parent_hh_constraint_ge_bound,
            sub_incidence,
            parent_incidence,
            total_hh_right_hand_side,
            relax_ge_upper_bound,
            parent_lp_right_hand_side,
            hh_constraint_ge_bound,
            parent_resid_weights,
            total_hh_sub_control_index,
            total_hh_parent_control_index,
            timeout_in_seconds=self.timeout_in_seconds,
        )

        # smart round resid_weights_out for each sub_zone
        total_household_controls = sub_control_totals[
            :, total_hh_sub_control_index
        ].flatten()
        integerized_weights = np.empty_like(sub_int_weights)

        sub_zone_count = len(self.sub_weights.columns)
        for i in range(sub_zone_count):
            integerized_weights[i] = smart_round(
                sub_int_weights[i], resid_weights_out[i], total_household_controls[i]
            )

        # integerized_weights df: one column of integerized weights per sub_zone
        self.integerized_weights = pd.DataFrame(
            data=integerized_weights.T,
            columns=self.sub_weights.columns,
            index=self.incidence_df.index,
        )

        return status_text
