# PopulationSim
# See full license in LICENSE.txt.

import logging
import os

import orca
import pandas as pd
import numpy as np

from ..integerizer import do_integerizing
from helper import get_control_table

logger = logging.getLogger(__name__)


@orca.step()
def integerize_final_seed_weights(settings, crosswalk, control_spec, incidence_table):

    crosswalk_df = crosswalk.to_frame()
    incidence_df = incidence_table.to_frame()
    control_spec = control_spec.to_frame()

    seed_geography = settings.get('seed_geography')
    seed_controls_df = get_control_table(seed_geography)

    # FIXME - I assume we want to integerize using meta controls too?
    control_cols = control_spec.target

    # FIXME - ensure columns are in right order for orca-extended table
    seed_controls_df = seed_controls_df[control_cols]

    # determine master_control_index if specified in settings
    total_hh_control_col = settings.get('total_hh_control')

    # run balancer for each seed geography
    weight_list = []
    seed_ids = crosswalk_df[seed_geography].unique()
    for seed_id in seed_ids:

        logger.info("integerize_final_seed_weights seed id %s" % seed_id)

        # slice incidence rows for this seed geography
        seed_incidence = incidence_df[incidence_df[seed_geography] == seed_id]

        integer_weights, status = do_integerizing(
            label=seed_geography,
            id=seed_id,
            control_spec=control_spec,
            control_totals=seed_controls_df.loc[seed_id],
            incidence_table=seed_incidence[control_cols],
            float_weights=seed_incidence['final_seed_weight'],
            total_hh_control_col=total_hh_control_col
        )

        weight_list.append(integer_weights)

    # bulk concat all seed level results
    integer_seed_weights = pd.concat(weight_list)

    orca.add_column('incidence_table', 'integer_seed_weight', integer_seed_weights)