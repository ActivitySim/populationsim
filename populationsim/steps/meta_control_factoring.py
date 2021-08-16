
# PopulationSim
# See full license in LICENSE.txt.

import logging
import os

import pandas as pd

from activitysim.core import pipeline
from activitysim.core import inject

from .helper import get_control_table
from .helper import control_table_name
from .helper import get_weight_table

logger = logging.getLogger(__name__)


def dump_table(table_name, table):

    return

    print("\n%s\n" % table_name, table)


@inject.step()
def meta_control_factoring(settings, control_spec, incidence_table):
    """
    Apply simple factoring to summed household fractional weights based on original
    meta control values relative to summed household fractional weights by meta zone.

    The resulting factored meta control weights will be new meta controls appended as
    additional columns to the seed control table, for final balancing.

    Parameters
    ----------
    settings : dict (settings.yaml as dict)
    control_spec : pipeline table
    incidence_table : pipeline table

    Returns
    -------

    """

    # FIXME - if there is only one seed zone in the meta zone, just copy meta control values?

    incidence_df = incidence_table.to_frame()
    control_spec = control_spec.to_frame()

    geographies = settings.get('geographies')
    seed_geography = settings.get('seed_geography')
    meta_geography = geographies[0]

    # - if there are no meta controls, then we don't have to do anything
    if not (control_spec.geography == meta_geography).any():
        logger.warning("meta_control_factoring: no meta targets so nothing to do")
        return

    meta_controls_df = get_control_table(meta_geography)
    dump_table("meta_controls_df", meta_controls_df)

    # slice control_spec to select only the rows for meta level controls
    meta_controls_spec = control_spec[control_spec.geography == meta_geography]
    meta_control_targets = meta_controls_spec['target']

    logger.info("meta_control_factoring %s targets" % len(meta_control_targets))

    dump_table("meta_controls_spec", meta_controls_spec)
    dump_table("meta_control_targets", meta_control_targets)

    # seed level weights of all households (rows aligned with incidence_df rows)
    seed_weights_df = get_weight_table(seed_geography)
    assert len(incidence_df.index) == len(seed_weights_df.index)

    # expand person weights by incidence (incidnece will simply be 1 for household targets)
    hh_level_weights = incidence_df[[seed_geography, meta_geography]].copy()
    for target in meta_control_targets:
        hh_level_weights[target] = \
            incidence_df[target] * seed_weights_df['preliminary_balanced_weight']

    dump_table("hh_level_weights", hh_level_weights)

    # weights of meta targets at seed level
    factored_seed_weights = \
        hh_level_weights.groupby([seed_geography, meta_geography], as_index=False).sum()
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

    seed_controls_df = get_control_table(seed_geography)

    assert len(seed_controls_df.index) == len(seed_level_meta_controls.index)
    seed_controls_df = pd.concat([seed_controls_df, seed_level_meta_controls], axis=1)

    # ensure columns are in right order for orca-extended table
    seed_controls_df = seed_controls_df[control_spec.target]
    assert (seed_controls_df.columns == control_spec.target).all()

    dump_table("seed_controls_df", seed_controls_df)

    pipeline.replace_table(control_table_name(seed_geography), seed_controls_df)
