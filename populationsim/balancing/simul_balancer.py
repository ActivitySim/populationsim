# PopulationSim
# See full license in LICENSE.txt.

import logging
import numpy as np
import pandas as pd

from populationsim.core.config import setting
from populationsim.balancing.balancers import np_simul_balancer_py
from populationsim.balancing.balancers_numba import np_simul_balancer_numba
from populationsim.balancing.constants import (
    MIN_CONTROL_VALUE,
    MIN_IMPORTANCE,
    DEFAULT_MAX_ITERATIONS,
)

logger = logging.getLogger(__name__)


class SimultaneousListBalancer:
    """
    Dual-zone simultaneous list balancer using Newton-Raphson method with control relaxation.

    Simultaneously balances the household weights across multiple subzones of a parent zone,
    ensuring that the total weight of each household across sub-zones sums to the parent hh weight.

    The resulting weights are float weights, so need to be integerized to integer household weights
    """

    def __init__(
        self,
        incidence_table,
        parent_weights,
        controls,
        sub_control_zones,
        total_hh_control_col,
        use_numba,
        numba_precision,
    ):
        """

        Parameters
        ----------
        incidence_table : pandas DataFrame
            incidence table with only columns for sub contols
        parent_weights : pandas: Series
            parent zone balanced (possibly integerized) aggregate target weights
        controls : pandas DataFrame
            parent zone controls
            one row per control,
            columns : name, importance, total + one column per sub_zone
        sub_control_zones : pandas.Series
            index is zone id and value is zone label (e.g. TAZ_101)
            for use in sub_controls_df column names
        total_hh_control_col : str
            name of the total_hh control column
        """
        assert isinstance(incidence_table, pd.DataFrame)
        assert len(parent_weights.index) == len(incidence_table.index)
        assert len(incidence_table.columns) == len(controls.index)

        assert "total" in controls
        assert "importance" in controls

        # remove zero weight rows
        # remember series so we can add zero weight rows back into result after balancing
        self.positive_weight_rows = parent_weights > 0
        logger.debug(
            "%s positive weight rows out of %s"
            % (self.positive_weight_rows.sum(), len(incidence_table.index))
        )

        self.incidence_table = incidence_table[self.positive_weight_rows]
        self.weights = pd.DataFrame(
            {"parent": parent_weights[self.positive_weight_rows]}
        )

        self.controls = controls
        self.sub_control_zones = sub_control_zones

        self.total_hh_control_col = total_hh_control_col
        self.master_control_index = self.incidence_table.columns.get_loc(
            total_hh_control_col
        )

        self.balancer = np_simul_balancer_numba if use_numba else np_simul_balancer_py
        self.numba_precision = numba_precision

    def balance(self):

        assert len(self.incidence_table.columns) == len(self.controls.index)
        assert len(self.weights.index) == len(self.incidence_table.index)

        self.weights["upper_bound"] = self.weights["parent"]
        if "lower_bound" not in self.weights:
            self.weights["lower_bound"] = 0.0

        # set initial sub zone weights proportionate to number of households
        total_hh_controls = self.controls.iloc[self.master_control_index]
        total_hh = int(total_hh_controls["total"])
        sub_zone_hh_fractions = total_hh_controls[self.sub_control_zones] / total_hh

        # Using numpy outer product to create a matrix of sub_zone_hh_fractions for each hh rather than pandas looping
        new_weights = np.outer(
            self.weights["parent"], sub_zone_hh_fractions[self.sub_control_zones.values]
        )
        new_weights = pd.DataFrame(
            new_weights,
            index=self.weights.index,
            columns=self.sub_control_zones.tolist(),
        )

        if len(set(new_weights.columns).intersection(self.weights.columns)) > 0:
            self.weights.update(new_weights)
        else:
            self.weights = self.weights.join(new_weights)

        self.controls["total"] = np.maximum(self.controls["total"], MIN_CONTROL_VALUE)

        # control relaxation importance weights (higher weights result in lower relaxation factor)
        self.controls["importance"] = np.maximum(
            self.controls["importance"], MIN_IMPORTANCE
        )

        # prepare inputs as numpy  (no pandas)
        sample_count = len(self.incidence_table.index)
        control_count = len(self.incidence_table.columns)
        zone_count = len(self.sub_control_zones)

        master_control_index = self.master_control_index
        incidence = self.incidence_table.values.transpose().astype(np.float64)

        # FIXME - do we also need sample_weights? (as the spec suggests?)
        parent_weights = np.asanyarray(self.weights["parent"]).astype(np.float64)

        weights_lower_bound = np.asanyarray(self.weights["lower_bound"]).astype(
            np.float64
        )
        weights_upper_bound = np.asanyarray(self.weights["upper_bound"]).astype(
            np.float64
        )

        parent_controls = np.asanyarray(self.controls["total"]).astype(np.float64)
        controls_importance = np.asanyarray(self.controls["importance"]).astype(
            np.float64
        )

        sub_controls = (
            self.controls[self.sub_control_zones].values.astype("float").transpose()
        )
        sub_weights = (
            self.weights[self.sub_control_zones].values.astype("float").transpose()
        )

        max_iterations = (
            setting("MAX_BALANCE_ITERATIONS_SIMULTANEOUS", DEFAULT_MAX_ITERATIONS) // 10
        )

        # balance
        weights_final, relaxation_factors, status = self.balancer(
            sample_count,
            control_count,
            zone_count,
            master_control_index,
            incidence,
            parent_weights,
            weights_lower_bound,
            weights_upper_bound,
            sub_weights,
            parent_controls,
            controls_importance,
            sub_controls,
            max_iterations,
            # max_delta=
        )

        status = dict(zip(("converged", "iter", "delta", "max_gamma_dif"), status))

        # dataframe with sub_zone_weights in columns, and zero weight rows restored
        self.sub_zone_weights = pd.DataFrame(
            weights_final[range(len(self.sub_control_zones))].transpose(),
            index=self.weights.index,
            columns=self.sub_control_zones.tolist(),
        )
        self.sub_zone_weights = self.sub_zone_weights.reindex(
            self.positive_weight_rows.index
        )
        self.sub_zone_weights.fillna(value=0.0, inplace=True)

        # series mapping zone_id to column names
        self.sub_zone_ids = self.sub_control_zones.index.values

        # dataframe with relaxation factors for each control in columns and one row per subzone
        self.relaxation_factors = pd.DataFrame(
            data=relaxation_factors,
            columns=self.controls.name,
            index=self.sub_control_zones.index,
        )

        self.status = status

        return self.status
