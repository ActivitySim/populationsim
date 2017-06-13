# PopulationSim
# See full license in LICENSE.txt.

import logging
import os

import orca
import pandas as pd
import numpy as np

from ..simul_balancer import SimultaneousListBalancer

from ..integerizer import do_integerizing

from populationsim.util import setting

from helper import get_control_table
from helper import weight_table_name
from helper import get_weight_table

logger = logging.getLogger(__name__)

SIMUL_INTEGERIZE = False
if SIMUL_INTEGERIZE:
    from ..simul_integerizer import do_simul_integerizing


SUB_BALANCE_WITH_FLOAT_SEED_WEIGHTS = setting('SUB_BALANCE_WITH_FLOAT_SEED_WEIGHTS')

def sequential_multi_integerize(incidence_df,
                     parent_weights, parent_controls,
                     sub_weights, sub_controls,
                     control_spec, total_hh_control_col,
                     sub_control_zones,
                     parent_geography, sub_geography):

    # integerize the sub_zone weights
    integer_weights_list = []
    for zone_id, zone_name in sub_control_zones.iteritems():
        #control_totals = sub_controls.loc[zone_id].values
        weights = sub_weights[zone_name]

        integer_weights, status = do_integerizing(
            label=sub_geography,
            id=zone_id,
            control_spec=control_spec,
            control_totals=sub_controls.loc[zone_id],
            incidence_table=incidence_df[control_spec.target],
            float_weights=weights,
            total_hh_control_col=total_hh_control_col
        )

        # print "weights\n", weights
        # print "integer_weights\n", integer_weights

        zone_weights_df = pd.DataFrame(index=range(0, len(integer_weights.index)))
        zone_weights_df[weights.index.name] = weights.index
        zone_weights_df[sub_geography] = zone_id
        zone_weights_df['balanced_weight'] = weights.values
        zone_weights_df['integer_weight'] = integer_weights.astype(int).values

        integer_weights_list.append(zone_weights_df)

    integer_weights_df = pd.concat(integer_weights_list)
    return integer_weights_df


def balance(
        parent_geography,
        parent_id,
        sub_geographies,
        control_spec,
        parent_controls_df,
        sub_controls_df,
        initial_weights,
        incidence_df,
        crosswalk_df,
        total_hh_control_col):

    sub_geography = sub_geographies[0]

    sub_control_spec = control_spec[control_spec['geography'].isin(sub_geographies)]

    # only want subcontrol rows for current geography geo_id
    sub_ids = crosswalk_df.loc[crosswalk_df[parent_geography] == parent_id, sub_geography].unique()
    sub_controls = sub_controls_df.loc[sub_ids]

    # FIXME - an rason not to just drop out any empty zones?
    empty_sub_zones = (sub_controls[total_hh_control_col] == 0)
    if empty_sub_zones.any():
        logger.info("dropping %s empty %s  in %s %s"
                    % (empty_sub_zones.sum(), sub_geography, parent_geography, parent_id))
        sub_controls = sub_controls[ ~empty_sub_zones ]

    # standard names for sub_control zone columns in controls and weights
    sub_control_zone_names = ['%s_%s' % (sub_geography, z) for z in sub_controls.index]
    sub_control_zones = pd.Series(sub_control_zone_names, index=sub_controls.index)

    # controls - organized in legible form
    controls = pd.DataFrame({'name': sub_control_spec.target})
    controls['importance'] = sub_control_spec.importance
    controls['total'] = sub_controls.sum(axis=0).values
    for zone, zone_name in sub_control_zones.iteritems():
        controls[zone_name] = sub_controls.loc[zone].values

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
    sub_weights = balancer.sub_zone_weights

    logger.debug("%s %s converged %s iter %s"
                % (parent_geography, parent_id, status['converged'], status['iter']))

    if SIMUL_INTEGERIZE:
        multi_integerize = do_simul_integerizing
    else:
        multi_integerize = sequential_multi_integerize

    integer_weights_df = multi_integerize(
        incidence_df,
        initial_weights, controls,
        sub_weights, sub_controls,
        control_spec, total_hh_control_col,
        sub_control_zones,
        parent_geography, sub_geography
    )
    integer_weights_df[parent_geography] = parent_id

    return integer_weights_df


@orca.step()
def sub_balancing(settings, crosswalk, control_spec, incidence_table):

    crosswalk_df = crosswalk.to_frame()
    incidence_df = incidence_table.to_frame()
    control_spec = control_spec.to_frame()

    geographies = settings.get('geographies')
    seed_geography = settings.get('seed_geography')

    parent_geography = seed_geography
    sub_geographies = geographies[geographies.index(parent_geography) + 1:]
    sub_geography = sub_geographies[0]

    total_hh_control_col = settings.get('total_hh_control')

    # control table for the sub geography below seed
    parent_controls_df = get_control_table(parent_geography)
    sub_controls_df = get_control_table(sub_geography)

    integer_weights_list = []

    seed_ids = crosswalk_df[seed_geography].unique()
    for seed_id in seed_ids:

        logger.info("sub_balancing seed id %s" % seed_id)

        seed_incidence_df = incidence_df[incidence_df[seed_geography] == seed_id]

        if SUB_BALANCE_WITH_FLOAT_SEED_WEIGHTS:
            initial_weights = seed_incidence_df['final_seed_weight']
        else:
            initial_weights = seed_incidence_df['integer_seed_weight']

        zone_weights_df = balance(
            parent_geography=parent_geography,
            parent_id=seed_id,
            sub_geographies=sub_geographies,
            control_spec=control_spec,
            parent_controls_df=parent_controls_df,
            sub_controls_df=sub_controls_df,
            initial_weights=initial_weights,
            incidence_df=seed_incidence_df,
            crosswalk_df=crosswalk_df,
            total_hh_control_col=total_hh_control_col)

        integer_weights_list.append(zone_weights_df)


    integer_weights_df = pd.concat(integer_weights_list)
    integer_weights_df['integer_weight'] = integer_weights_df['integer_weight'].astype(int)

    orca.add_table(weight_table_name(sub_geography), integer_weights_df)
    orca.add_table(weight_table_name(sub_geography, sparse=True), integer_weights_df[integer_weights_df['integer_weight'] > 0])

    if 'trace_geography' in settings and sub_geography in settings['trace_geography']:
        sub_geography_id = settings.get('trace_geography')[sub_geography]
        df = integer_weights_df[ integer_weights_df[sub_geography] ==  sub_geography_id]
        orca.add_table('trace_%s' % weight_table_name(sub_geography), df)




@orca.step()
def low_balancing(settings, crosswalk, control_spec, incidence_table):

    crosswalk_df = crosswalk.to_frame()
    incidence_df = incidence_table.to_frame()
    control_spec = control_spec.to_frame()

    geographies = settings.get('geographies')
    seed_geography = settings.get('seed_geography')

    DEPTH = 1
    parent_geography = geographies[geographies.index(seed_geography) + DEPTH]
    sub_geographies = geographies[geographies.index(parent_geography) + 1:]
    sub_geography = sub_geographies[0]

    total_hh_control_col = settings.get('total_hh_control')

    parent_controls_df = get_control_table(parent_geography)
    sub_controls_df = get_control_table(sub_geography)

    weights_df = get_weight_table(parent_geography)

    integer_weights_list = []

    seed_ids = crosswalk_df[seed_geography].unique()
    for seed_id in seed_ids:

        seed_incidence_df = incidence_df[incidence_df[seed_geography] == seed_id]
        seed_crosswalk_df = crosswalk_df[crosswalk_df[seed_geography] == seed_id]

        parent_ids = seed_crosswalk_df[parent_geography].unique()

        #parent_ids = [10900]

        for parent_id in parent_ids:

            logger.info("balancing seed %s, %s %s" % (seed_id, parent_geography, parent_id))

            initial_weights = weights_df[weights_df[parent_geography] == parent_id]
            initial_weights = initial_weights.set_index(settings.get('household_id_col'))
            initial_weights = initial_weights['integer_weight']

            # print "\n\n\n######################################## initial_weights"
            # print initial_weights

            assert len(initial_weights.index) == len(seed_incidence_df.index)

            zone_weights_df = balance(
                parent_geography=parent_geography,
                parent_id=parent_id,
                sub_geographies=sub_geographies,
                control_spec=control_spec,
                parent_controls_df=parent_controls_df,
                sub_controls_df=sub_controls_df,
                initial_weights=initial_weights,
                incidence_df=seed_incidence_df,
                crosswalk_df=crosswalk_df,
                total_hh_control_col=total_hh_control_col)

            # facilitate summaries
            zone_weights_df[seed_geography] = seed_id

            # print "######################################## zone_weights_df"
            # print zone_weights_df

            integer_weights_list.append(zone_weights_df)


    integer_weights_df = pd.concat(integer_weights_list)
    integer_weights_df['integer_weight'] = integer_weights_df['integer_weight'].astype(int)

    orca.add_table(weight_table_name(sub_geography), integer_weights_df)
    orca.add_table(weight_table_name(sub_geography, sparse=True), integer_weights_df[integer_weights_df['integer_weight'] > 0])

    if 'trace_geography' in settings and sub_geography in settings['trace_geography']:
        sub_geography_id = settings.get('trace_geography')[sub_geography]
        df = integer_weights_df[ integer_weights_df[sub_geography] ==  sub_geography_id]
        orca.add_table('trace_%s' % weight_table_name(sub_geography), df)
