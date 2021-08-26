
# PopulationSim
# See full license in LICENSE.txt.

import logging
import os
import pandas as pd

from activitysim.core import inject

from activitysim.core.config import setting

from ..balancer import do_balancing
from .helper import get_control_table
from .helper import weight_table_name
from .helper import get_weight_table


logger = logging.getLogger(__name__)


@inject.step()
def final_seed_balancing(settings, crosswalk, control_spec, incidence_table):
    """
    Balance the household weights for each of the seed geographies (independently)
    using the seed level controls and the aggregated sub-zone controls totals.

    Create the seed_weights table with one row per household and columns contaiing
    household_id, seed geography (e.g. PUMA), and float preliminary_balanced_weights

    Adds column balanced_weight  to the seed_weights table

    Parameters
    ----------
    settings : dict (settings.yaml as dict)
    crosswalk : pipeline table
    control_spec : pipeline table
    incidence_table : pipeline table

    Returns
    -------

    """

    crosswalk_df = crosswalk.to_frame()
    incidence_df = incidence_table.to_frame()
    control_spec = control_spec.to_frame()

    seed_geography = settings.get('seed_geography')
    seed_weight_table_name = weight_table_name(seed_geography)

    # if there are no meta controls, then balanced_weight is simply preliminary_balanced_weight
    geographies = settings['geographies']
    if not (control_spec.geography == geographies[0]).any():
        logger.warning("no need for final_seed_balancing because no meta controls")
        seed_weights_df = get_weight_table(seed_geography)
        if 'balanced_weight' not in seed_weights_df:
            final_seed_weights = seed_weights_df['preliminary_balanced_weight']
            inject.add_column(seed_weight_table_name, 'balanced_weight', final_seed_weights)
        return

    # we use all control_spec rows, so no need to filter on geography as for initial_seed_balancing
    seed_controls_df = get_control_table(seed_geography)
    assert (seed_controls_df.columns == control_spec.target).all()

    # determine master_control_index if specified in settings
    total_hh_control_col = setting('total_hh_control')

    max_expansion_factor = settings.get('max_expansion_factor', None)
    min_expansion_factor = settings.get('min_expansion_factor', None)
    absolute_upper_bound = settings.get('absolute_upper_bound', None)
    absolute_lower_bound = settings.get('absolute_lower_bound', None)

    relaxation_factors = pd.DataFrame(index=seed_controls_df.columns.tolist())

    # run balancer for each seed geography
    weight_list = []

    seed_ids = crosswalk_df[seed_geography].unique()
    for seed_id in seed_ids:

        logger.info("final_seed_balancing seed id %s" % seed_id)

        seed_incidence_df = incidence_df[incidence_df[seed_geography] == seed_id]

        status, weights_df, controls_df = do_balancing(
            control_spec=control_spec,
            total_hh_control_col=total_hh_control_col,
            max_expansion_factor=max_expansion_factor,
            min_expansion_factor=min_expansion_factor,
            absolute_lower_bound=absolute_lower_bound,
            absolute_upper_bound=absolute_upper_bound,
            incidence_df=seed_incidence_df,
            control_totals=seed_controls_df.loc[seed_id],
            initial_weights=seed_incidence_df['sample_weight'])

        logger.info("seed_balancer status: %s" % status)
        if not status['converged']:
            raise RuntimeError("final_seed_balancing for seed_id %s did not converge" % seed_id)

        weight_list.append(weights_df['final'])

        relaxation_factors[seed_id] = controls_df['relaxation_factor']

    # bulk concat all seed level results
    final_seed_weights = pd.concat(weight_list)

    inject.add_column(seed_weight_table_name, 'balanced_weight', final_seed_weights)
