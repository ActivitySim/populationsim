
# PopulationSim
# See full license in LICENSE.txt.

import logging
import pandas as pd

from activitysim.core import inject

from activitysim.core.config import setting

from .helper import get_control_table
from .helper import weight_table_name
from .helper import get_weight_table

from ..balancer import do_balancing
from ..integerizer import do_integerizing


logger = logging.getLogger(__name__)


@inject.step()
def repop_balancing(settings, crosswalk, control_spec, incidence_table):
    """

    Balance and integerize all zones at a lowest geographic level.


    Creates a weight table for the repop zones target geography
    with float 'balanced_weight' and 'integer_weight' columns.

    Parameters
    ----------
    settings : dict (settings.yaml as dict)
    crosswalk : pipeline table
    control_spec: pipeline table
    incidence_table : pipeline table
    """

    crosswalk_df = crosswalk.to_frame()
    incidence_df = incidence_table.to_frame()
    control_spec = control_spec.to_frame()

    geographies = settings['geographies']
    low_geography = geographies[-1]

    seed_geography = settings.get('seed_geography')
    seed_controls_df = get_control_table(seed_geography)

    all_seed_weights_df = get_weight_table(seed_geography)
    assert all_seed_weights_df is not None

    # only want control_spec rows for low_geography
    low_control_spec = control_spec[control_spec['geography'] == low_geography]
    low_controls_df = get_control_table(low_geography)

    household_id_col = setting('household_id_col')
    total_hh_control_col = setting('total_hh_control')

    max_expansion_factor = settings.get('max_expansion_factor', None)
    min_expansion_factor = settings.get('min_expansion_factor', None)
    absolute_upper_bound = settings.get('absolute_upper_bound', None)
    absolute_lower_bound = settings.get('absolute_lower_bound', None)
    hard_constraints = settings.get('hard_constraints', None)

    # run balancer for each low geography
    low_weight_list = []

    seed_ids = crosswalk_df[seed_geography].unique()
    for seed_id in seed_ids:

        logger.info("initial_seed_balancing seed id %s" % seed_id)

        seed_incidence_df = incidence_df[incidence_df[seed_geography] == seed_id]
        seed_crosswalk_df = crosswalk_df[crosswalk_df[seed_geography] == seed_id]

        # initial seed weights in series indexed by hh id
        seed_weights_df = all_seed_weights_df[all_seed_weights_df[seed_geography] == seed_id]
        seed_weights_df = seed_weights_df.set_index(household_id_col)

        # number of hh in seed zone (for scaling low zone weights)
        seed_zone_hh_count = seed_controls_df[total_hh_control_col].loc[seed_id]

        low_ids = seed_crosswalk_df[low_geography].unique()
        for low_id in low_ids:

            trace_label = "%s_%s_%s_%s" % (seed_geography, seed_id, low_geography, low_id)
            logger.info("balance and integerize %s" % trace_label)

            # weights table for this zone with household_id index and low_geography column
            zone_weights_df = pd.DataFrame(index=seed_weights_df.index)
            zone_weights_df[low_geography] = low_id

            # scale seed weights by relative hh counts
            # it doesn't makes sense to repop balance with integer weights
            low_zone_hh_count = low_controls_df[total_hh_control_col].loc[low_id]
            scaling_factor = float(low_zone_hh_count)/seed_zone_hh_count
            initial_weights = seed_weights_df['balanced_weight'] * scaling_factor

            # - balance
            status, weights_df, controls_df = do_balancing(
                control_spec=low_control_spec,
                total_hh_control_col=total_hh_control_col,
                max_expansion_factor=max_expansion_factor,
                min_expansion_factor=min_expansion_factor,
                absolute_upper_bound=absolute_upper_bound,
                absolute_lower_bound=absolute_lower_bound,
                incidence_df=seed_incidence_df,
                control_totals=low_controls_df.loc[low_id],
                initial_weights=initial_weights,
                use_hard_constraints=hard_constraints)

            logger.info("repop_balancing balancing %s status: %s" % (trace_label, status))
            if not status['converged']:
                raise RuntimeError("repop_balancing for %s did not converge" % trace_label)

            zone_weights_df['balanced_weight'] = weights_df['final']

            # - integerize
            integer_weights, status = do_integerizing(
                trace_label=trace_label,
                control_spec=control_spec,
                control_totals=low_controls_df.loc[low_id],
                incidence_table=seed_incidence_df,
                float_weights=weights_df['final'],
                total_hh_control_col=total_hh_control_col)

            logger.info("repop_balancing integerizing status: %s" % status)

            zone_weights_df['integer_weight'] = integer_weights

            logger.info("Total balanced weights for %s = %s" %
                        (trace_label, zone_weights_df['balanced_weight'].sum()))
            logger.info("Total integerized weights for %s = %s" %
                        (trace_label, zone_weights_df['integer_weight'].sum()))

            low_weight_list.append(zone_weights_df)

    # concat all low geography zone level results
    low_weights_df = pd.concat(low_weight_list).reset_index()

    # add higher level geography id columns to facilitate summaries
    crosswalk_df = crosswalk_df.set_index(low_geography)\
        .loc[low_weights_df[low_geography]]\
        .reset_index(drop=True)
    low_weights_df = pd.concat([low_weights_df, crosswalk_df], axis=1)

    inject.add_table(weight_table_name(low_geography),
                     low_weights_df, replace=True)
    inject.add_table(weight_table_name(low_geography, sparse=True),
                     low_weights_df[low_weights_df['integer_weight'] > 0],
                     replace=True)
