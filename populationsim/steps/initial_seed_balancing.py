
# PopulationSim
# See full license in LICENSE.txt.

import logging
import pandas as pd

from activitysim.core import inject
from activitysim.core import pipeline

from activitysim.core.config import setting

from ..balancer import do_balancing

from .helper import get_control_table
from .helper import weight_table_name


logger = logging.getLogger(__name__)


@inject.step()
def initial_seed_balancing(settings, crosswalk, control_spec, incidence_table):
    """
    Balance the household weights for each of the seed geographies (independently)
    using the seed level controls and the aggregated sub-zone controls totals.

    Create the seed_weights table with one row per household and columns contaiing
    household_id, seed geography (e.g. PUMA), and float preliminary_balanced_weights

    Adds seed_weights table to pipeline named <seed_geography>_weights (e.g. PUMA_weights):

    +--------+------+-----------------------------+-------+
    | index  | PUMA | preliminary_balanced_weight | hh_id |
    | hh_id  |      |                             |       |
    +========+======+=============================+=======+
    | 0      | 600  |                   0.313555  |    0  |
    | 1      | 601  |                   0.627110  |    1  |
    | 2      | 602  |                   0.313555  |    2  |
    | ...    |      |                             |       |
    +--------+------+-----------------------------+-------+

    Parameters
    ----------
    settings : dict (settings.yaml as dict)
    crosswalk : pipeline table
    control_spec : pipeline table
    incidence_table : pipeline table

    """
    crosswalk_df = crosswalk.to_frame()
    incidence_df = incidence_table.to_frame()
    control_spec = control_spec.to_frame()

    seed_geography = settings.get('seed_geography')
    seed_controls_df = get_control_table(seed_geography)

    # only want control_spec rows for seed geography and below
    geographies = settings['geographies']
    seed_geographies = geographies[geographies.index(seed_geography):]
    seed_control_spec = control_spec[control_spec['geography'].isin(seed_geographies)]

    # determine master_control_index if specified in settings
    total_hh_control_col = setting('total_hh_control')

    max_expansion_factor = settings.get('max_expansion_factor', None)
    min_expansion_factor = settings.get('min_expansion_factor', None)
    absolute_upper_bound = settings.get('absolute_upper_bound', None)
    absolute_lower_bound = settings.get('absolute_lower_bound', None)

    # run balancer for each seed geography
    weight_list = []
    sample_weight_list = []

    seed_ids = crosswalk_df[seed_geography].unique()
    for seed_id in seed_ids:

        logger.info("initial_seed_balancing seed id %s" % seed_id)

        seed_incidence_df = incidence_df[incidence_df[seed_geography] == seed_id]

        status, weights_df, controls_df = do_balancing(
            control_spec=seed_control_spec,
            total_hh_control_col=total_hh_control_col,
            max_expansion_factor=max_expansion_factor,
            min_expansion_factor=min_expansion_factor,
            absolute_upper_bound=absolute_upper_bound,
            absolute_lower_bound=absolute_lower_bound,
            incidence_df=seed_incidence_df,
            control_totals=seed_controls_df.loc[seed_id],
            initial_weights=seed_incidence_df['sample_weight'])

        logger.info("seed_balancer status: %s" % status)
        if not status['converged']:
            raise RuntimeError("initial_seed_balancing for seed_id %s did not converge" % seed_id)

        balanced_weights = weights_df['final']

        logger.info("Total balanced weights for seed %s = %s" % (seed_id, balanced_weights.sum()))

        weight_list.append(balanced_weights)
        sample_weight_list.append(seed_incidence_df['sample_weight'])

    # bulk concat all seed level results
    weights = pd.concat(weight_list)
    sample_weights = pd.concat(sample_weight_list)

    # build canonical weights table
    seed_weights_df = incidence_df[[seed_geography]].copy()
    seed_weights_df['preliminary_balanced_weight'] = weights

    seed_weights_df['sample_weight'] = sample_weights

    # copy household_id_col index to named column
    seed_weights_df[setting('household_id_col')] = seed_weights_df.index

    # this is just a convenience if there are no meta controls
    if inject.get_step_arg('final', default=False):
        seed_weights_df['balanced_weight'] = seed_weights_df['preliminary_balanced_weight']

    repop = inject.get_step_arg('repop', default=False)
    inject.add_table(weight_table_name(seed_geography), seed_weights_df, replace=repop)
