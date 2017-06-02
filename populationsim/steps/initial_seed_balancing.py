# PopulationSim
# See full license in LICENSE.txt.

import logging
import orca
import pandas as pd

from ..balancer import do_seed_balancing
from ..balancer import do_seed_balancing_cvx

from helper import get_control_table


logger = logging.getLogger(__name__)


@orca.step()
def initial_seed_balancing(settings, crosswalk, control_spec, incidence_table):

    crosswalk_df = crosswalk.to_frame()
    incidence_df = incidence_table.to_frame()
    control_spec = control_spec.to_frame()

    seed_geography = settings.get('seed_geography')
    seed_controls_df = get_control_table(seed_geography)

    # only want control_spec rows for sub_geographies
    geographies = settings['geographies']
    sub_geographies = geographies[geographies.index(seed_geography)+1:]
    seed_control_spec = control_spec[control_spec['geography'].isin(sub_geographies)]

    # determine master_control_index if specified in settings
    total_hh_control_col = settings.get('total_hh_control')

    max_expansion_factor = settings.get('max_expansion_factor', None)

    # run balancer for each seed geography
    weight_list = []
    seed_ids = crosswalk_df[seed_geography].unique()
    for seed_id in seed_ids:

        logger.info("initial_seed_balancing seed id %s" % seed_id)

        USE_CVX = False

        if USE_CVX:

            balanced_weights, relaxation_factors = do_seed_balancing_cvx(
                seed_geography=seed_geography,
                seed_control_spec=seed_control_spec,
                seed_id=seed_id,
                total_hh_control_col=total_hh_control_col,
                max_expansion_factor=max_expansion_factor,
                incidence_df=incidence_df,
                seed_controls_df=seed_controls_df)

        else:

            status, weights_df, controls_df = do_seed_balancing(
                seed_geography=seed_geography,
                seed_control_spec=seed_control_spec,
                seed_id=seed_id,
                total_hh_control_col=total_hh_control_col,
                max_expansion_factor=max_expansion_factor,
                incidence_df=incidence_df,
                seed_controls_df=seed_controls_df)

            logger.info("seed_balancer status: %s" % status)
            if not status['converged']:
                raise RuntimeError("initial_seed_balancing for seed_id %s did not converge" % seed_id)

            balanced_weights = weights_df['final']

        logger.info("Total balanced weights for seed %s = %s" % (seed_id, balanced_weights.sum()))

        weight_list.append(balanced_weights)

    # bulk concat all seed level results
    weights = pd.concat(weight_list)

    orca.add_column('incidence_table', 'initial_seed_weight', weights)
