
# PopulationSim
# See full license in LICENSE.txt.

import logging
import os

import pandas as pd

from activitysim.core import inject

from ..integerizer import do_integerizing
from .helper import get_control_table
from .helper import weight_table_name
from .helper import get_weight_table
from activitysim.core.config import setting

logger = logging.getLogger(__name__)


@inject.step()
def integerize_final_seed_weights(settings, crosswalk, control_spec, incidence_table):
    """
    Final balancing for each seed (puma) zone with aggregated low and mid-level controls and
    distributed meta-level controls.

    Adds integer_weight column to seed-level weight table

    Parameters
    ----------
    settings : dict (settings.yaml as dict)
    crosswalk : pipeline table
    control_spec : pipeline table
    incidence_table : pipeline table

    Returns
    -------

    """

    if setting('NO_INTEGERIZATION_EVER', False):
        logger.warning("skipping integerize_final_seed_weights: NO_INTEGERIZATION_EVER")
        return

    crosswalk_df = crosswalk.to_frame()
    incidence_df = incidence_table.to_frame()
    control_spec = control_spec.to_frame()

    seed_geography = settings.get('seed_geography')
    seed_controls_df = get_control_table(seed_geography)

    seed_weights_df = get_weight_table(seed_geography)

    # FIXME - I assume we want to integerize using meta controls too?
    control_cols = control_spec.target
    assert (seed_controls_df.columns == control_cols).all()

    # determine master_control_index if specified in settings
    total_hh_control_col = setting('total_hh_control')

    # run balancer for each seed geography
    weight_list = []

    seed_ids = crosswalk_df[seed_geography].unique()
    for seed_id in seed_ids:

        logger.info("integerize_final_seed_weights seed id %s" % seed_id)

        # slice incidence rows for this seed geography
        seed_incidence = incidence_df[incidence_df[seed_geography] == seed_id]

        balanced_seed_weights = \
            seed_weights_df.loc[seed_weights_df[seed_geography] == seed_id, 'balanced_weight']

        trace_label = "%s_%s" % (seed_geography, seed_id)

        integer_weights, status = do_integerizing(
            trace_label=trace_label,
            control_spec=control_spec,
            control_totals=seed_controls_df.loc[seed_id],
            incidence_table=seed_incidence[control_cols],
            float_weights=balanced_seed_weights,
            total_hh_control_col=total_hh_control_col
        )

        weight_list.append(integer_weights)

    # bulk concat all seed level results
    integer_seed_weights = pd.concat(weight_list)

    inject.add_column(weight_table_name(seed_geography), 'integer_weight', integer_seed_weights)
