# PopulationSim
# See full license in LICENSE.txt.

import logging
import os

import orca
import pandas as pd

from ..simul_balancer import SimultaneousListBalancer

from populationsim.util import setting

from helper import get_control_table
from helper import weight_table_name
from helper import get_weight_table

from ..simul_integerizer import do_simul_integerizing
from ..sequential_integerizer import do_sequential_integerizing


logger = logging.getLogger(__name__)


def balance(
        parent_geography,
        parent_id,
        sub_geographies,
        control_spec,
        sub_controls_df,
        initial_weights,
        incidence_df,
        total_hh_control_col
        ):

    sub_geography = sub_geographies[0]
    sub_control_spec = control_spec[control_spec['geography'].isin(sub_geographies)]

    # standard names for sub_control zone columns in controls and weights
    sub_control_zone_names = ['%s_%s' % (sub_geography, z) for z in sub_controls_df.index]
    sub_control_zones = pd.Series(sub_control_zone_names, index=sub_controls_df.index)

    # controls - organized in legible form
    controls = pd.DataFrame({'name': sub_control_spec.target})
    controls['importance'] = sub_control_spec.importance
    controls['total'] = sub_controls_df.sum(axis=0).values
    for zone, zone_name in sub_control_zones.iteritems():
        controls[zone_name] = sub_controls_df.loc[zone].values

    # incidence table should only have control columns
    sub_incidence_df = incidence_df[sub_control_spec.target]

    balancer = SimultaneousListBalancer(
        incidence_table=sub_incidence_df,
        initial_weights=initial_weights,
        controls=controls,
        sub_control_zones=sub_control_zones,
        total_hh_control_col=total_hh_control_col
    )

    status = balancer.balance()

    logger.debug("%s %s converged %s iter %s"
                 % (parent_geography, parent_id, status['converged'], status['iter']))

    return balancer.sub_zone_weights

def integerize(
        sub_zone_weights,
        parent_geography,
        parent_id,
        sub_geographies,
        control_spec,
        sub_controls_df,
        parent_controls_df,
        parent_weights,
        incidence_df,
        total_hh_control_col,
        USE_SIMUL_INTEGERIZER):

    sub_geography = sub_geographies[0]

    # standard names for sub_control zone columns in controls and weights
    sub_control_zone_names = ['%s_%s' % (sub_geography, z) for z in sub_controls_df.index]
    sub_control_zones = pd.Series(sub_control_zone_names, index=sub_controls_df.index)

    if USE_SIMUL_INTEGERIZER:

        integer_weights_df = do_simul_integerizing(
            incidence_df=incidence_df,
            parent_weights=parent_weights,
            parent_controls_df=parent_controls_df,
            sub_weights=sub_zone_weights,
            sub_controls_df=sub_controls_df,
            parent_geography=parent_geography,
            parent_id=parent_id,
            sub_geography=sub_geography,
            control_spec=control_spec,
            total_hh_control_col=total_hh_control_col,
            sub_control_zones=sub_control_zones
        )
    else:

        integer_weights_df = do_sequential_integerizing(
            incidence_df=incidence_df,
            sub_weights=sub_zone_weights,
            sub_controls=sub_controls_df,
            control_spec=control_spec,
            total_hh_control_col=total_hh_control_col,
            sub_control_zones=sub_control_zones,
            sub_geography=sub_geography,
            parent_geography=parent_geography,
            parent_id=parent_id,
        )

    integer_weights_df[parent_geography] = parent_id

    return integer_weights_df


def balance_and_integerize(
        parent_geography,
        parent_id,
        sub_geographies,
        control_spec,
        sub_controls_df,
        parent_controls_df,
        initial_weights,
        incidence_df,
        crosswalk_df,
        total_hh_control_col,
        USE_SIMUL_INTEGERIZER):

    sub_geography = sub_geographies[0]

    # only want subcontrol rows for current geography geo_id
    sub_ids = crosswalk_df.loc[crosswalk_df[parent_geography] == parent_id, sub_geography].unique()
    sub_controls_df = sub_controls_df.loc[sub_ids]

    parent_controls_df = parent_controls_df.loc[[parent_id]]

    # only care about the control columns
    incidence_df = incidence_df[control_spec.target]

    # FIXME - any reason not to just drop out any empty zones?
    empty_sub_zones = (sub_controls_df[total_hh_control_col] == 0)
    if empty_sub_zones.any():
        logger.info("dropping %s empty %s  in %s %s"
                    % (empty_sub_zones.sum(), sub_geography, parent_geography, parent_id))
        sub_controls_df = sub_controls_df[~empty_sub_zones]

    sub_zone_weights = balance(
        parent_geography=parent_geography,
        parent_id=parent_id,
        sub_geographies=sub_geographies,
        control_spec=control_spec,
        sub_controls_df=sub_controls_df,
        initial_weights=initial_weights,
        incidence_df=incidence_df,
        total_hh_control_col=total_hh_control_col)

    zone_weights_df = integerize(
        sub_zone_weights,
        parent_geography=parent_geography,
        parent_id=parent_id,
        sub_geographies=sub_geographies,
        control_spec=control_spec,
        sub_controls_df=sub_controls_df,
        parent_controls_df=parent_controls_df,
        parent_weights=initial_weights,
        incidence_df=incidence_df,
        total_hh_control_col=total_hh_control_col,
        USE_SIMUL_INTEGERIZER=USE_SIMUL_INTEGERIZER)

    return zone_weights_df

@orca.step()
def sub_balancing(settings, crosswalk, control_spec, incidence_table):

    crosswalk_df = crosswalk.to_frame()
    incidence_df = incidence_table.to_frame()
    control_spec = control_spec.to_frame()

    geographies = settings.get('geographies')
    seed_geography = settings.get('seed_geography')
    meta_geography = geographies[0]

    parent_geography = seed_geography
    sub_geographies = geographies[geographies.index(parent_geography) + 1:]
    sub_geography = sub_geographies[0]

    total_hh_control_col = settings.get('total_hh_control')

    sub_controls_df = get_control_table(sub_geography)
    parent_controls_df = get_control_table(parent_geography)

    #################################
    # control spec rows and control_df columns should be in same order
    # seed control columns may not be in right order, se we re-order them here
    countrol_cols = list(parent_controls_df.columns)
    parent_control_spec = control_spec[control_spec.target.isin(countrol_cols)]
    parent_controls_df = parent_controls_df[parent_control_spec.target.values]
    assert (list(parent_controls_df.columns) == parent_control_spec.target).all()

    # check that these are already in the right order
    countrol_cols = list(sub_controls_df.columns)
    sub_control_spec = control_spec[control_spec.target.isin(countrol_cols)]
    assert (list(sub_controls_df.columns) == sub_control_spec.target).all()
    #################################

    integer_weights_list = []

    seed_ids = crosswalk_df[seed_geography].unique()
    for seed_id in seed_ids:

        logger.info("sub_balancing seed id %s" % seed_id)

        seed_incidence_df = incidence_df[incidence_df[seed_geography] == seed_id]

        seed_crosswalk_df = crosswalk_df[crosswalk_df[seed_geography] == seed_id]

        assert len(seed_crosswalk_df[meta_geography].unique()) == 1
        meta_id = seed_crosswalk_df[meta_geography].max()

        if setting('SUB_BALANCE_WITH_FLOAT_SEED_WEIGHTS'):
            initial_weights = seed_incidence_df['final_seed_weight']
        else:
            initial_weights = seed_incidence_df['integer_seed_weight']

        zone_weights_df = balance_and_integerize(
            parent_geography=parent_geography,
            parent_id=seed_id,
            sub_geographies=sub_geographies,
            control_spec=control_spec,
            sub_controls_df=sub_controls_df,
            parent_controls_df=parent_controls_df,
            initial_weights=initial_weights,
            incidence_df=seed_incidence_df,
            crosswalk_df=crosswalk_df,
            total_hh_control_col=total_hh_control_col,
            USE_SIMUL_INTEGERIZER=setting('USE_SIMUL_INTEGERIZER'))

        # add meta level geography column to facilitate summaries
        zone_weights_df[meta_geography] = meta_id

        integer_weights_list.append(zone_weights_df)

    integer_weights_df = pd.concat(integer_weights_list)

    orca.add_table(weight_table_name(sub_geography),
                   integer_weights_df)
    orca.add_table(weight_table_name(sub_geography, sparse=True),
                   integer_weights_df[integer_weights_df['integer_weight'] > 0])

    if 'trace_geography' in settings and sub_geography in settings['trace_geography']:
        sub_geography_id = settings.get('trace_geography')[sub_geography]
        df = integer_weights_df[integer_weights_df[sub_geography] == sub_geography_id]
        orca.add_table('trace_%s' % weight_table_name(sub_geography), df)


@orca.step()
def low_balancing(settings, crosswalk, control_spec, incidence_table):

    crosswalk_df = crosswalk.to_frame()
    incidence_df = incidence_table.to_frame()
    control_spec = control_spec.to_frame()

    geographies = settings.get('geographies')
    seed_geography = settings.get('seed_geography')
    meta_geography = geographies[0]

    DEPTH = 1
    parent_geography = geographies[geographies.index(seed_geography) + DEPTH]
    sub_geographies = geographies[geographies.index(parent_geography) + 1:]
    sub_geography = sub_geographies[0]

    total_hh_control_col = settings.get('total_hh_control')

    sub_controls_df = get_control_table(sub_geography)
    parent_controls_df = get_control_table(parent_geography)

    weights_df = get_weight_table(parent_geography)

    integer_weights_list = []

    seed_ids = crosswalk_df[seed_geography].unique()
    for seed_id in seed_ids:

        seed_incidence_df = incidence_df[incidence_df[seed_geography] == seed_id]
        seed_crosswalk_df = crosswalk_df[crosswalk_df[seed_geography] == seed_id]

        assert len(seed_crosswalk_df[meta_geography].unique()) == 1
        meta_id = seed_crosswalk_df[meta_geography].max()

        parent_ids = seed_crosswalk_df[parent_geography].unique()

        for parent_id in parent_ids:

            logger.info("balancing seed %s, %s %s" % (seed_id, parent_geography, parent_id))

            initial_weights = weights_df[weights_df[parent_geography] == parent_id]
            initial_weights = initial_weights.set_index(settings.get('household_id_col'))

            if setting('SUB_BALANCE_WITH_FLOAT_SEED_WEIGHTS'):
                initial_weights = initial_weights['balanced_weight']
            else:
                initial_weights = initial_weights['integer_weight']

            assert len(initial_weights.index) == len(seed_incidence_df.index)

            zone_weights_df = balance_and_integerize(
                parent_geography=parent_geography,
                parent_id=parent_id,
                sub_geographies=sub_geographies,
                control_spec=control_spec,
                sub_controls_df=sub_controls_df,
                parent_controls_df=parent_controls_df,
                initial_weights=initial_weights,
                incidence_df=seed_incidence_df,
                crosswalk_df=crosswalk_df,
                total_hh_control_col=total_hh_control_col,
                USE_SIMUL_INTEGERIZER=setting('USE_SIMUL_INTEGERIZER'))


            # add higher level geography columns to facilitate summaries
            zone_weights_df[seed_geography] = seed_id
            zone_weights_df[meta_geography] = meta_id

            integer_weights_list.append(zone_weights_df)

    integer_weights_df = pd.concat(integer_weights_list)

    orca.add_table(weight_table_name(sub_geography),
                   integer_weights_df)
    orca.add_table(weight_table_name(sub_geography, sparse=True),
                   integer_weights_df[integer_weights_df['integer_weight'] > 0])

    if 'trace_geography' in settings and sub_geography in settings['trace_geography']:
        sub_geography_id = settings.get('trace_geography')[sub_geography]
        df = integer_weights_df[integer_weights_df[sub_geography] == sub_geography_id]
        orca.add_table('trace_%s' % weight_table_name(sub_geography), df)
