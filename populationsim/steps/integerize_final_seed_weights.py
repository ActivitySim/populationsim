# PopulationSim
# See full license in LICENSE.txt.

import logging
import os

import orca
import pandas as pd
import numpy as np

from ..integerizer import do_integerizing
from helper import get_control_table
from helper import weight_table_name
from helper import get_weight_table
from populationsim.util import setting

logger = logging.getLogger(__name__)


@orca.step()
def integerize_final_seed_weights(settings, crosswalk, control_spec, incidence_table):

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
    total_hh_control_col = settings.get('total_hh_control')

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

    orca.add_column(weight_table_name(seed_geography), 'integer_weight', integer_seed_weights)
