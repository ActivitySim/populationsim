# PopulationSim
# See full license in LICENSE.txt.

import logging
import os
import orca
import pandas as pd

from ..balancer import do_seed_balancing
from helper import get_control_table


logger = logging.getLogger(__name__)


@orca.step()
def final_seed_balancing(settings, crosswalk, control_spec, incidence_table):

    crosswalk_df = crosswalk.to_frame()
    incidence_df = incidence_table.to_frame()
    control_spec = control_spec.to_frame()

    seed_geography = settings.get('seed_geography')

    seed_controls_df = get_control_table(seed_geography)

    # we use all control_spec rows, so no need to filter on geography as for initial_seed_balancing

    # FIXME - ensure columns are in right order for orca-extended table
    seed_controls_df = seed_controls_df[control_spec.target]

    # determine master_control_index if specified in settings
    total_hh_control_col = settings.get('total_hh_control')

    max_expansion_factor = settings.get('max_expansion_factor', None)

    relaxation_factors = pd.DataFrame(index=seed_controls_df.columns.tolist())

    # run balancer for each seed geography
    weight_list = []
    seed_ids = crosswalk_df[seed_geography].unique()
    for seed_id in seed_ids:

        logger.info("initial_seed_balancing seed id %s" % seed_id)

        status, weights_df, controls_df = do_seed_balancing(
            seed_geography=seed_geography,
            seed_control_spec=control_spec,
            seed_id=seed_id,
            total_hh_control_col=total_hh_control_col,
            max_expansion_factor=max_expansion_factor,
            incidence_df=incidence_df,
            seed_controls_df=seed_controls_df)

        logger.info("seed_balancer status: %s" % status)
        if not status['converged']:
            raise RuntimeError("final_seed_balancing for seed_id %s did not converge" % seed_id)

        weight_list.append(weights_df['final'])

        relaxation_factors[seed_id] = controls_df['relaxation_factor']

    # bulk concat all seed level results
    final_seed_weights = pd.concat(weight_list)

    orca.add_column('incidence_table', 'final_seed_weight', final_seed_weights)
