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


# balancer = simul_balancer(
#     control_spec=control_spec,
#     geo='seed',
#     geo_col=seed_col,
#     geo_id=seed_id,
#     sub_geo='mid',
#     incidence_df=incidence_df,
#     seed_controls_df=seed_controls_df)


def simul_balancer(control_spec,
                   geography, geo_id, settings, max_expansion_factor,
                   incidence_df, geo_cross_walk_df,
                   initial_weights,
                   trace):

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
    sub_controls_df = orca.get_table(geography_settings[sub_geography].get('controls_table')).to_frame()
    sub_geo_col = geography_settings[sub_geography].get('id_column')
    sub_ids = geo_cross_walk_df.loc[geo_cross_walk_df[geo_col]==geo_id, sub_geo_col].unique()
    sub_controls_df = sub_controls_df.loc[sub_ids]
    #trace and dump_table('sub_controls_df', sub_controls_df)

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
    # FIXME - not sure initial_sample should be different
    weights['initial_sample'] = initial_weights
    weights['final_seed'] = initial_weights

    if max_expansion_factor:

        # FIXME - scale the max_expansion factor by the number of households in each sub_zone compared to the target zone?
        # FIXME - or should upper bounds remain the same for each subzone?

        # number_of_households in this seed geograpy as specified in seed_controlss
        number_of_households = float(controls.loc[ controls['name'] == total_hh_control_col, 'total' ])
        total_weights = weights['initial_sample'].sum()
        ub_ratio = max_expansion_factor * number_of_households / total_weights
        ub_weights = weights['initial_sample'] * ub_ratio
        weights['upper_bound'] = ub_weights.round().clip(lower=1).astype(int)

        # print "number_of_households", number_of_households
        # print "total_weights", total_weights
        # print "number_of_households/total_weights", number_of_households/total_weights
        # print "ub_ratio", ub_ratio
        # print "ub_weights\n", ub_weights
        # print "initial_weights\n", initial_weights
        # assert False


    total_hh = sub_controls_df[total_hh_control_col].sum()
    sub_zone_hh_fraction = sub_controls_df[total_hh_control_col] / total_hh
    for zone, zone_name in sub_control_zones.iteritems():
        weights[zone_name] = weights['final_seed'] * sub_zone_hh_fraction[zone]
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
        master_control_index=total_hh_control_index,
        trace=trace
    )

    return balancer


@orca.step()
def simultaneous_sub_balancing(settings, geo_cross_walk, control_spec, incidence_table):

    geography_settings = settings.get('geography_settings')
    geo_cross_walk_df = geo_cross_walk.to_frame()
    incidence_df = incidence_table.to_frame()

    max_expansion_factor = settings.get('max_expansion_factor', None)

    geographies = settings.get('geographies')
    target_geographies = geographies[geographies.index('seed'):]
    lowest_geography = target_geographies.pop()

    #print "target_geographies", target_geographies
    #print "lowest_geography", lowest_geography

    # filter geo_cross_walk_df to only include geo_ids with lowest_geography controls
    # (just in case geo_cross_walk_df table contains rows for unused geographies)
    low_col = geography_settings[lowest_geography].get('id_column')
    low_controls_df = orca.get_table(geography_settings[lowest_geography].get('controls_table')).to_frame()
    if len(geo_cross_walk_df.index) != len(low_controls_df.index) \
            or len(geo_cross_walk_df.index) != len(low_controls_df.index):
        logger.warn("geo_cross_walk '%s' doesn't match '%s' control table")
        geo_cross_walk_df = geo_cross_walk_df[ geo_cross_walk_df[low_col].isin(low_controls_df.index) ]

    def sub_balance(target_geographies, geo_cross_walk_df, sub_zone_weights, sub_control_zones, trace):

        geography = target_geographies[0]
        sub_geographies = target_geographies[1:]

        geo_col = geography_settings[geography].get('id_column')
        weight_list = []

        geo_ids = geo_cross_walk_df[geo_col].unique()
        for geo_id in geo_ids:

            trace = trace and (geo_id == geo_ids[0])

            if geography == 'seed':
                seed_incidence_df = incidence_df[incidence_df[geo_col] == geo_id]
                initial_weights = seed_incidence_df['final_seed_weight']
            else:
                assert geo_id in sub_control_zones.index
                initial_weights = sub_zone_weights[sub_control_zones[geo_id]]

            balancer = simul_balancer(
                control_spec=control_spec,
                geography=geography,
                geo_id=geo_id,
                settings=settings,
                max_expansion_factor=max_expansion_factor,
                incidence_df=incidence_df,
                geo_cross_walk_df=geo_cross_walk_df,
                initial_weights=initial_weights,
                trace=trace
            )


            status = balancer.balance()
            final_weights = balancer.weights_final

            assert False

            logger.info("%s %s converged %s iter %s delta %s max_gamma_dif %s"
                        % (geography, geo_id,
                           status['converged'], status['iter'], status['delta'], status['max_gamma_dif']))

            if sub_geographies:
                sub_weights = sub_balance(sub_geographies, geo_cross_walk_df[ geo_cross_walk_df[geo_col] == geo_id],
                                          sub_zone_weights=balancer.weights_final, sub_control_zones=balancer.sub_control_zones,
                                        trace=trace)

                #final_weights = pd.concat([final_weights] + sub_weights, axis=1)
                final_weights = sub_weights

            if geography=='seed':
                final_weights = pd.concat(final_weights, axis=1)

                # print "\n----- seed %s" % geo_id
                # print "final_weights\n", final_weights
                # print "-----\n"

                seed_table_name = "final_seed_weights_%s" % geo_id
                orca.add_table(seed_table_name, final_weights)
            else:
                weight_list.append(final_weights)

        return weight_list

    sub_balance(target_geographies, geo_cross_walk_df,
                sub_zone_weights=None, sub_control_zones=None,
                trace=True)

@orca.step()
def psimultaneous_sub_balancing(settings, geo_cross_walk, control_spec, incidence_table):

    geography_settings = settings.get('geography_settings')

    seed_col = geography_settings['seed'].get('id_column')

    geo_cross_walk_df = geo_cross_walk.to_frame()
    incidence_df = incidence_table.to_frame()

    max_expansion_factor = settings.get('max_expansion_factor', None)

    geographies = settings.get('geographies')
    target_geographies = geographies[geographies.index('seed'):]
    lowest_geography = target_geographies.pop()

    print "target_geographies", target_geographies
    print "lowest_geography", lowest_geography

    # filter geo_cross_walk_df to only include geo_ids with lowest_geography controls
    # (just in case geo_cross_walk_df table contains rows for unused geographies)
    low_col = geography_settings[lowest_geography].get('id_column')
    low_controls_df = orca.get_table(geography_settings[lowest_geography].get('controls_table')).to_frame()
    if len(geo_cross_walk_df.index) != len(low_controls_df.index) \
            or len(geo_cross_walk_df.index) != len(low_controls_df.index):
        logger.warn("geo_cross_walk '%s' doesn't match '%s' control table")
        geo_cross_walk_df = geo_cross_walk_df[ geo_cross_walk_df[low_col].isin(low_controls_df.index) ]

    seed_ids = geo_cross_walk_df[seed_col].unique()
    for seed_id in seed_ids:

        logger.info("simultaneous_sub_balancing seed id %s" % seed_id)

        balancer = simul_balancer(
            control_spec=control_spec,
            geography='seed',
            geo_id=seed_id,
            settings=settings,
            max_expansion_factor=max_expansion_factor,
            incidence_df=incidence_df,
            geo_cross_walk_df=geo_cross_walk_df
        )

        status = balancer.balance()

        print "status", status

        balancer.print_status()

        sub_weights = balancer.weights_final

        break



