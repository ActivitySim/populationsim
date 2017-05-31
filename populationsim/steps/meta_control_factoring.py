# PopulationSim
# See full license in LICENSE.txt.

import logging
import os

import orca
import pandas as pd


from helper import get_control_table
from helper import control_table_name

logger = logging.getLogger(__name__)


def dump_table(table_name, table):

    return

    print "\n%s\n" % table_name, table


@orca.step()
def meta_control_factoring(settings, control_spec, incidence_table):
    """
    Apply simple factoring to summed household fractional weights based on original
    meta control values relative to summed household fractional weights by meta zone.

    The resulting factored meta control weights will be new meta controls, to be
    appended to the original controls, for final balancing.

    Parameters
    ----------
    settings
    control_spec
    incidence_table

    Returns
    -------

    """

    # FIXME - if there is only one seed zone in the meta zone, just copy meta control values?

    incidence_df = incidence_table.to_frame()
    control_spec = control_spec.to_frame()

    geographies = settings.get('geographies')
    seed_geography = settings.get('seed_geography')
    meta_geography = geographies[0]

    meta_controls_df = get_control_table(meta_geography)

    meta_controls_spec = control_spec[control_spec.geography == meta_geography]
    meta_control_targets = meta_controls_spec['target']

    # weights of meta targets at hh (incidence table) level
    hh_level_weights = incidence_df[[seed_geography, meta_geography]].copy()
    for target in meta_control_targets:
        hh_level_weights[target] = incidence_df[target] * incidence_df['initial_seed_weight']

    # weights of meta targets at seed level
    factored_seed_weights = hh_level_weights.groupby([seed_geography, meta_geography], as_index=False).sum()
    factored_seed_weights.set_index(seed_geography, inplace=True)
    dump_table("factored_seed_weights", factored_seed_weights)

    # weights of meta targets summed from seed level to  meta level
    factored_meta_weights = factored_seed_weights.groupby(meta_geography, as_index=True).sum()
    dump_table("factored_meta_weights", factored_meta_weights)

    # only the meta level controls from meta_controls table
    meta_controls_df = meta_controls_df[meta_control_targets]
    dump_table("meta_controls_df", meta_controls_df)

    # compute the scaling factors to be applied to the seed-level totals:
    meta_factors = pd.DataFrame(index=meta_controls_df.index)
    for target in meta_control_targets:
        meta_factors[target] = meta_controls_df[target] / factored_meta_weights[target]
    dump_table("meta_factors", meta_factors)

    # compute seed-level controls from meta-level controls
    seed_level_meta_controls = pd.DataFrame(index=factored_seed_weights.index)
    for target in meta_control_targets:
        #  meta level scaling_factor for this meta_control
        scaling_factor = factored_seed_weights[meta_geography].map(meta_factors[target])
        # scale the seed_level_meta_controls by meta_level scaling_factor
        seed_level_meta_controls[target] = factored_seed_weights[target] * scaling_factor
        # FIXME - why round scaled factored seed_weights to int prior to final seed balancing?
        seed_level_meta_controls[target] = seed_level_meta_controls[target].round().astype(int)
    dump_table("seed_level_meta_controls", seed_level_meta_controls)

    # create final balancing controls
    # add newly created seed_level_meta_controls to the existing set of seed level controls
    for column_name in seed_level_meta_controls.columns:
        orca.add_column(control_table_name(seed_geography), column_name,
                        seed_level_meta_controls[column_name])
        logger.info("adding column %s" % column_name)
