# PopulationSim
# See full license in LICENSE.txt.

import logging
import os
import orca
import pandas as pd

from ..balancer import seed_balancer


logger = logging.getLogger(__name__)


@orca.step()
def final_seed_balancing(settings, geo_cross_walk, control_spec,
                         incidence_table, seed_controls):

    geo_cross_walk_df = geo_cross_walk.to_frame()
    incidence_df = incidence_table.to_frame()
    seed_controls_df = seed_controls.to_frame()
    control_spec = control_spec.to_frame()

    seed_col = settings.get('geography_settings')['seed'].get('id_column')

    # we use all control_spec rows, so no need to filter on geography as for initial_seed_balancing

    # determine master_control_index if specified in settings
    total_hh_control_col = settings.get('total_hh_control')

    max_expansion_factor = settings.get('max_expansion_factor', None)

    relaxation_factors = pd.DataFrame(index = seed_controls_df.columns.tolist())

    # run balancer for each seed geography
    weight_list = []
    seed_ids = geo_cross_walk_df[seed_col].unique()
    for seed_id in seed_ids:

        logger.info("initial_seed_balancing seed id %s" % seed_id)

        balancer = seed_balancer(
            seed_control_spec=control_spec,
            seed_id=seed_id,
            seed_col=seed_col,
            total_hh_control_col=total_hh_control_col,
            max_expansion_factor=max_expansion_factor,
            incidence_df=incidence_df,
            seed_controls_df=seed_controls_df)

        # balancer.dump()
        status = balancer.balance()

        logger.info("seed_balancer status: %s" % status)
        if not status['converged']:
            raise RuntimeError("final_seed_balancing for seed_id %s did not converge" % seed_id)

        weight_list.append(balancer.weights['final'])

        relaxation_factors[seed_id] = balancer.controls['relaxation_factor']

        # print "balancer.initial_weights\n", balancer.initial_weights
        # print "balancer.ub_weights\n", balancer.ub_weights
        # print "balancer.weights\n", balancer.weights

    # bulk concat all seed level results
    final_seed_weights = pd.concat(weight_list)

    relaxation_factors = relaxation_factors.transpose()

    orca.add_table('seed_control_relaxation_factors', relaxation_factors)

    orca.add_column('incidence_table', 'final_seed_weight', final_seed_weights)
