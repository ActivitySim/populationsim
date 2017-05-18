# PopulationSim
# See full license in LICENSE.txt.

import logging
import os

import orca
import pandas as pd
import numpy as np

from ..simul_balancer import SimultaneousListBalancer


logger = logging.getLogger(__name__)


def dump_table(table_name, table):

    print "\n%s\n" % table_name, table.head(100)


def simul_balancer(control_spec,
                   geography, geo_id, settings, max_expansion_factor,
                   incidence_df, geo_cross_walk_df,
                   aggregate_target_weights):

    trace = False

    geographies = settings.get('geographies')
    geography_settings = settings.get('geography_settings')

    total_hh_control_col = settings.get('total_hh_control')

    assert geography in geographies
    assert geographies.index(geography) < len(geographies)

    geographies = geographies[geographies.index(geography):]
    sub_geographies = geographies[geographies.index(geography) + 1:]
    sub_geography = sub_geographies[0]
    geo_col = geography_settings[geography].get('id_column')

    control_spec = control_spec[control_spec['geography'].isin(sub_geographies)]
    trace and dump_table('control_spec', control_spec)

    # only want subcontrol rows for current geography geo_id
    sub_control_table_name = geography_settings[sub_geography].get('controls_table')
    sub_controls_df = orca.get_table(sub_control_table_name).to_frame()
    sub_geo_col = geography_settings[sub_geography].get('id_column')
    sub_ids = geo_cross_walk_df.loc[geo_cross_walk_df[geo_col] == geo_id, sub_geo_col].unique()
    sub_controls_df = sub_controls_df.loc[sub_ids]
    # trace and dump_table('sub_controls_df', sub_controls_df)

    # standard names for sub_control zone columns in controls and weights
    sub_control_zone_names = ['%s_%s' % (sub_geo_col, z) for z in sub_controls_df.index]
    sub_control_zones = pd.Series(sub_control_zone_names, index=sub_controls_df.index)

    # controls - organized in legible form
    controls = pd.DataFrame({'name': control_spec.target})
    controls['importance'] = control_spec.importance
    controls['total'] = sub_controls_df.sum(axis=0).values
    for zone, zone_name in sub_control_zones.iteritems():
        controls[zone_name] = sub_controls_df.loc[zone].values
    trace and dump_table('controls', controls)

    # # incidence_df - only want rows for this seed geography
    seed_col = geography_settings['seed'].get('id_column')
    seed_id = geo_cross_walk_df.loc[geo_cross_walk_df[geo_col] == geo_id, seed_col].max()
    incidence_df = incidence_df[incidence_df[seed_col] == seed_id]

    # weights
    weights = pd.DataFrame(index=incidence_df.index)
    weights['aggregate_target'] = aggregate_target_weights

    if max_expansion_factor:

        # FIXME - how to apply max_expansion_factor to compute sub_zone ub_weights
        # FIXME - scale by the number of households in each sub_zone compared to the target zone?
        # FIXME - or should upper bounds remain the same for each subzone?

        # number_of_households in this seed geograpy as specified in seed_controlss
        number_of_households = controls.loc[controls['name'] == total_hh_control_col, 'total']
        # convert single-element int array to scalar float
        number_of_households = float(number_of_households)
        total_weights = weights['aggregate_target'].sum()
        ub_ratio = max_expansion_factor * number_of_households / total_weights
        ub_weights = weights['aggregate_target'] * ub_ratio
        weights['upper_bound'] = ub_weights.round().clip(lower=1).astype(int)

        # print "number_of_households", number_of_households
        # print "total_weights", total_weights
        # print "number_of_households/total_weights", number_of_households/total_weights
        # print "ub_ratio", ub_ratio
        # print "ub_weights\n", ub_weights
        # print "aggregate_target_weights\n", aggregate_target_weights

    total_hh = sub_controls_df[total_hh_control_col].sum()
    sub_zone_hh_fraction = sub_controls_df[total_hh_control_col] / total_hh
    for zone, zone_name in sub_control_zones.iteritems():
        weights[zone_name] = weights['aggregate_target'] * sub_zone_hh_fraction[zone]
    trace and dump_table('weights', weights)

    # incidence table should only have control columns
    incidence_df = incidence_df[control_spec.target]
    trace and dump_table('incidence_df', incidence_df)

    total_hh_control_index = incidence_df.columns.get_loc(total_hh_control_col)

    balancer = SimultaneousListBalancer(
        incidence_table=incidence_df,
        weights=weights,
        controls=controls,
        sub_control_zones=sub_control_zones,
        seed_id=seed_id,
        sub_zone=sub_geo_col,
        master_control_index=total_hh_control_index
    )

    return balancer


@orca.step()
def simultaneous_sub_balancing(settings, geo_cross_walk, control_spec, incidence_table):

    geography_settings = settings.get('geography_settings')
    geo_cross_walk_df = geo_cross_walk.to_frame()
    incidence_df = incidence_table.to_frame()

    max_expansion_factor = settings.get('max_expansion_factor', None)
    geographies = settings.get('geographies')

    USE_INTEGER_SEED_WEIGHT = settings.get('USE_INTEGER_SEED_WEIGHT', True)

    def log_status(geography, geo_col, geo_id, status):

        logger.info("%s %s converged %s iter %s"
                    % (geography, geo_id, status['converged'], status['iter']))

    # FIXME - should do this up front and save sanitized geo_cross_walk table
    # filter geo_cross_walk_df to only include geo_ids with lowest_geography controls
    # (just in case geo_cross_walk_df table contains rows for unused geographies)
    lowest_geography = geographies[-1]
    low_col = geography_settings[lowest_geography].get('id_column')
    low_control_table_name = geography_settings[lowest_geography].get('controls_table')
    low_controls_df = orca.get_table(low_control_table_name).to_frame()
    if len(geo_cross_walk_df.index) != len(low_controls_df.index) \
            or len(geo_cross_walk_df.index) != len(low_controls_df.index):
        logger.warn("geo_cross_walk '%s' doesn't match '%s' control table")
        rows_in_low_controls = geo_cross_walk_df[low_col].isin(low_controls_df.index)
        geo_cross_walk_df = geo_cross_walk_df[rows_in_low_controls]

    def sub_balance(target_geographies, geo_cross_walk_df, sub_zone_weights, sub_control_zones):

        geography = target_geographies[0]
        sub_geographies = target_geographies[1:]

        geo_col = geography_settings[geography].get('id_column')
        result_list = []

        geo_ids = geo_cross_walk_df[geo_col].unique()
        for geo_id in geo_ids:

            assert geo_id in sub_control_zones.index
            aggregate_target_weights = sub_zone_weights[sub_control_zones[geo_id]]

            balancer = simul_balancer(
                control_spec=control_spec,
                geography=geography,
                geo_id=geo_id,
                settings=settings,
                max_expansion_factor=max_expansion_factor,
                incidence_df=incidence_df,
                geo_cross_walk_df=geo_cross_walk_df,
                aggregate_target_weights=aggregate_target_weights
            )

            logger.info("balancing %s %s" % (geography, geo_id))

            status = balancer.balance()

            log_status(geography, geo_col, geo_id, status)

            INCLUDE_INTERMEDIATE_ZONE_RESULTS = True
            if INCLUDE_INTERMEDIATE_ZONE_RESULTS or len(sub_geographies) == 1:
                result_list.append(balancer.results)

            # FIXME - untested as we don't have that many geographies
            if len(sub_geographies) > 1:
                assert False
                sub_results = sub_balance(
                    target_geographies=sub_geographies,
                    geo_cross_walk_df=geo_cross_walk_df[geo_cross_walk_df[geo_col] == geo_id],
                    sub_zone_weights=balancer.weights_final,
                    sub_control_zones=balancer.sub_control_zones
                )

                result_table_list.extend(sub_results)

        return result_list

    result_table_list = []

    sub_geographies = geographies[geographies.index('seed') + 1:]
    seed_col = geography_settings['seed'].get('id_column')
    seed_ids = geo_cross_walk_df[seed_col].unique()
    for seed_id in seed_ids:

        logger.info("simultaneous_sub_balancing seed id %s" % seed_id)

        seed_incidence_df = incidence_df[incidence_df[seed_col] == seed_id]

        # FIXME - do we need to use integerized weights?
        # FIXME - or can we use final_seed_weight and wait until the end to integerize?
        if USE_INTEGER_SEED_WEIGHT:
            initial_weights = seed_incidence_df['integer_seed_weight']
        else:
            initial_weights = seed_incidence_df['final_seed_weight']

        balancer = simul_balancer(
            control_spec=control_spec,
            geography='seed',
            geo_id=seed_id,
            settings=settings,
            max_expansion_factor=max_expansion_factor,
            incidence_df=seed_incidence_df,
            geo_cross_walk_df=geo_cross_walk_df,
            aggregate_target_weights=initial_weights
        )

        status = balancer.balance()

        log_status('seed', seed_col, seed_id, status)

        seed_results = balancer.results

        sub_results = sub_balance(
            target_geographies=sub_geographies,
            geo_cross_walk_df=geo_cross_walk_df[geo_cross_walk_df[seed_col] == seed_id],
            sub_zone_weights=balancer.sub_zone_weights,
            sub_control_zones=balancer.sub_control_zones,
            )

        result_table_list.extend(sub_results)
        # add seed level results
        result_table_list.append(seed_results)

    results_df = pd.concat(result_table_list, axis=0)
    orca.add_table('sub_results', results_df)
