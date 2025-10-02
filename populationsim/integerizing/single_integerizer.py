# PopulationSim
# See full license in LICENSE.txt.

import logging

import numpy as np
import pandas as pd

from populationsim.core import config
from populationsim.integerizing.constants import STATUS_OPTIMAL
from populationsim.integerizing.smart_round import smart_round
from populationsim.integerizing import lp_cvx, lp_ortools

logger = logging.getLogger(__name__)


class Integerizer:
    def __init__(
        self,
        incidence_table,
        control_importance_weights,
        float_weights,
        relaxed_control_totals,
        total_hh_control_value,
        total_hh_control_index,
        control_is_hh_based,
        trace_label="",
    ):
        """

        Parameters
        ----------
        control_totals : pandas.Series
            targeted control totals (either explict or backstopped) we are trying to hit
        incidence_table : pandas.Dataframe
            incidence table with columns only for targeted controls
        control_importance_weights : pandas.Series
            importance weights (from control_spec) of targeted controls
        float_weights
            blanaced float weights to integerize
        relaxed_control_totals
        total_hh_control_index : int
        control_is_hh_based : bool
        """

        self.incidence_table = incidence_table
        self.control_importance_weights = control_importance_weights
        self.float_weights = float_weights
        self.relaxed_control_totals = relaxed_control_totals

        self.total_hh_control_value = total_hh_control_value
        self.total_hh_control_index = total_hh_control_index
        self.control_is_hh_based = control_is_hh_based

        self.trace_label = trace_label

        # Choose the integerizer function based on configuration
        if config.setting("USE_CVXPY", False):
            self.integerizer_func = lp_cvx.np_integerizer_cvx
        else:
            self.integerizer_func = lp_ortools.np_integerizer_ortools

        self.timeout_in_seconds = config.setting("INTEGIZER_TIMEOUT", 60)

    def integerize(self):

        sample_count = len(self.incidence_table.index)
        control_count = len(self.incidence_table.columns)

        incidence = self.incidence_table.values.transpose().astype(np.float64)
        float_weights = np.asanyarray(self.float_weights).astype(np.float64)
        relaxed_control_totals = np.asanyarray(self.relaxed_control_totals).astype(
            np.float64
        )
        control_is_hh_based = np.asanyarray(self.control_is_hh_based).astype(bool)
        control_importance_weights = np.asanyarray(
            self.control_importance_weights
        ).astype(np.float64)

        assert len(float_weights) == sample_count
        assert len(relaxed_control_totals) == control_count
        assert len(control_is_hh_based) == control_count
        assert len(self.incidence_table.columns) == control_count
        assert (relaxed_control_totals == np.round(relaxed_control_totals)).all()
        assert not np.isnan(incidence).any()
        assert not np.isnan(float_weights).any()
        assert (incidence[self.total_hh_control_index] == 1).all()

        int_weights = float_weights.astype(int)
        resid_weights = float_weights % 1.0

        if (resid_weights == 0.0).all():
            # not sure this matters...
            logger.info(
                "Integerizer: all %s resid_weights zero. Returning success."
                % ((resid_weights == 0).sum(),)
            )

            integerized_weights = int_weights
            status = STATUS_OPTIMAL

        else:

            # - lp_right_hand_side - relaxed_control_shortfall
            lp_right_hand_side = relaxed_control_totals - np.dot(
                int_weights, incidence.T
            )
            lp_right_hand_side = np.maximum(lp_right_hand_side, 0.0)

            # - max_incidence_value of each control
            max_incidence_value = np.amax(incidence, axis=1)
            assert (max_incidence_value[control_is_hh_based] <= 1).all()

            # - create the inequality constraint upper bounds
            num_households = relaxed_control_totals[self.total_hh_control_index]
            relax_ge_upper_bound = np.maximum(
                max_incidence_value * num_households - lp_right_hand_side, 0
            )
            hh_constraint_ge_bound = np.maximum(
                self.total_hh_control_value * max_incidence_value, lp_right_hand_side
            )

            # popsim3 does does something rather peculiar, which I am not sure is right
            # it applies a huge penalty to rounding a near-zero residual upwards
            # the documentation justifying this is sparse and possibly confused:
            # // Set objective: min sum{c(n)*x(n)} + 999*y(i) - 999*z(i)}
            # objective_function_coefficients = -1.0 * np.log(resid_weights)
            # objective_function_coefficients[(resid_weights <= np.exp(-999))] = 999
            # We opt for an alternate interpretation of what they meant to do: avoid log overflow
            # There is not much difference in effect...
            LOG_OVERFLOW = -725
            log_resid_weights = np.log(np.maximum(resid_weights, np.exp(LOG_OVERFLOW)))
            assert not np.isnan(log_resid_weights).any()

            if (float_weights == 0).any():
                # not sure this matters...
                logger.warning(
                    "Integerizer: %s zero weights out of %s"
                    % ((float_weights == 0).sum(), sample_count)
                )

                raise AssertionError(
                    "Integerizer: %s zero weights out of %s"
                    % ((float_weights == 0).sum(), sample_count)
                )

            if (resid_weights == 0.0).any():
                # not sure this matters...
                logger.info(
                    "Integerizer: %s zero resid_weights out of %s"
                    % ((resid_weights == 0).sum(), sample_count)
                )
                # assert False

            resid_weights, status = self.integerizer_func(
                incidence=incidence,
                resid_weights=resid_weights,
                log_resid_weights=log_resid_weights,
                control_importance_weights=control_importance_weights,
                total_hh_control_index=self.total_hh_control_index,
                lp_right_hand_side=lp_right_hand_side,
                relax_ge_upper_bound=relax_ge_upper_bound,
                hh_constraint_ge_bound=hh_constraint_ge_bound,
                timeout_in_seconds=self.timeout_in_seconds,
            )

            integerized_weights = smart_round(
                int_weights, resid_weights, self.total_hh_control_value
            )

        self.weights = pd.DataFrame(index=self.incidence_table.index)
        self.weights["integerized_weight"] = integerized_weights

        delta = (integerized_weights != np.round(float_weights)).sum()
        logger.debug(
            "Integerizer: %s out of %s different from round"
            % (delta, len(float_weights))
        )

        return status
