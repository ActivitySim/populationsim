# PopulationSim
# See full license in LICENSE.txt.

import logging
import os

import orca
import pandas as pd
import numpy as np

from ..simul_balancer import SimultaneousListBalancer
from ..integerizer import do_integerizing
from helper import get_control_table

logger = logging.getLogger(__name__)


def dump_table(table_name, table):

    print "\n%s\n" % table_name, table.head(100)


def log_status(geography, geo_id, status):

    logger.info("%s %s converged %s iter %s"
                % (geography, geo_id, status['converged'], status['iter']))

@orca.step()
def sub_balancing(settings, crosswalk, control_spec, incidence_table):

    trace = False

    geographies = settings.get('geographies')
    geography_settings = settings.get('geography_settings')
    total_hh_control_col = settings.get('total_hh_control')

    crosswalk_df = crosswalk.to_frame()
    incidence_df = incidence_table.to_frame()
    control_spec = control_spec.to_frame()

    # control table for the sub geography below seed
    seed_geography = settings.get('seed_geography')
    sub_geographies = geographies[geographies.index(seed_geography) + 1:]
    sub_geography = sub_geographies[0]
    sub_controls_df = get_control_table(sub_geography)

    integer_weights_list = []

    seed_geography = settings.get('seed_geography')
    seed_ids = crosswalk_df[seed_geography].unique()
    for seed_id in seed_ids:

        logger.info("simultaneous_sub_balancing seed id %s" % seed_id)

        seed_incidence = incidence_df[incidence_df[seed_geography] == seed_id]

        initial_weights = seed_incidence['integer_seed_weight']

        control_spec = control_spec[control_spec['geography'].isin(sub_geographies)]
        trace and dump_table('control_spec', control_spec)

        # only want subcontrol rows for current geography geo_id
        sub_ids = crosswalk_df.loc[crosswalk_df[seed_geography] == seed_id, sub_geography].unique()
        sub_controls = sub_controls_df.loc[sub_ids]
        trace and dump_table('sub_controls', sub_controls)

        # standard names for sub_control zone columns in controls and weights
        sub_control_zone_names = ['%s_%s' % (sub_geography, z) for z in sub_controls.index]
        sub_control_zones = pd.Series(sub_control_zone_names, index=sub_controls.index)

        # controls - organized in legible form
        controls = pd.DataFrame({'name': control_spec.target})
        controls['importance'] = control_spec.importance
        controls['total'] = sub_controls.sum(axis=0).values
        for zone, zone_name in sub_control_zones.iteritems():
            controls[zone_name] = sub_controls.loc[zone].values
        trace and dump_table('controls', controls)

        # incidence table should only have control columns
        seed_incidence = seed_incidence[control_spec.target]
        trace and dump_table('seed_incidence', seed_incidence)

        balancer = SimultaneousListBalancer(
            incidence_table=seed_incidence,
            initial_weights=initial_weights,
            controls=controls,
            sub_control_zones=sub_control_zones,
            total_hh_control_col=total_hh_control_col
        )

        status = balancer.balance()

        log_status(seed_geography, seed_id, status)

        # integerize the sub_zone weights
        for zone_id, zone_name in sub_control_zones.iteritems():

            control_totals = sub_controls.loc[zone_id].values
            weights = balancer.sub_zone_weights[zone_name]
            relaxation_factors = balancer.relaxation_factors.loc[zone_id]

            integer_weights = do_integerizing(
                control_spec=control_spec,
                control_totals=control_totals,
                incidence_table=seed_incidence,
                final_weights=weights,
                relaxation_factors=relaxation_factors,
                total_hh_control_col=total_hh_control_col
            )

            print "weights\n", weights
            print "integer_weights\n", integer_weights

            zone_weights_df = pd.DataFrame(index=range(0, len(integer_weights.index)))
            zone_weights_df[weights.index.name] = weights.index
            zone_weights_df[seed_geography] = seed_id
            zone_weights_df[sub_geography] = zone_id
            zone_weights_df['balanced_weight'] = weights.values
            zone_weights_df['integer_weight'] = integer_weights.values
            integer_weights_list.append(zone_weights_df)


    integer_weights_df = pd.concat(integer_weights_list)
    orca.add_table('%s_weights' % (sub_geography,), integer_weights_df)
    orca.add_table('sparse_%s_weights' % (sub_geography,), integer_weights_df[integer_weights_df['integer_weight'] > 0])

    hh_id_col = incidence_df.index.name
    aggegrate_weights = integer_weights_df.groupby([seed_geography, hh_id_col], as_index=False).sum()
    del aggegrate_weights[sub_geography]
    aggegrate_weights.set_index(hh_id_col, inplace=True)
    orca.add_table('aggregated_%s_weights' % (sub_geography,), aggegrate_weights)
