# PopulationSim
# See full license in LICENSE.txt.

import logging
import os

import orca
import pandas as pd
import numpy as np

from activitysim.core import assign
from ..balancer import ListBalancer


logger = logging.getLogger(__name__)


def load_control_data_tables(geographies, seed_id, sub_geographies, geo_cross_walk_df):

    seed_col = geographies['seed'].get('id_column')

    # subset of crosswalk table for this seed geography
    seed_crosswalk_df = geo_cross_walk_df[geo_cross_walk_df[seed_col] == seed_id]

    # preload control_data tables for sub_geographies
    control_data_tables = {}
    for g in sub_geographies:

        geography = geographies[g]
        control_data_table_name = geography['control_data_table']
        control_data_df = orca.get_table(control_data_table_name).to_frame()

        if seed_col in control_data_df.columns:
            logger.info("seed_col %s in control_data_df" % seed_col)
            control_data_df = control_data_df[control_data_df[seed_col] == seed_id]
        else:
            # FIXME - perhaps we should add seed_col if necessary in setup_data_structures?
            logger.info("seed_col %s not in control_data_df" % seed_col)
            # unique ids for this geography level in current seed_geography
            geog_col = geography['id_column']
            geog_ids = seed_crosswalk_df[geog_col].unique()
            control_data_df = control_data_df[control_data_df[geog_col].isin(geog_ids)]

        control_data_tables[g] = control_data_df

    return control_data_tables


def seed_balancer(control_spec, seed_id, geographies, settings,
                  geo_cross_walk_df,
                  incidence_df, incidence_cols):

    sub_geographies = geographies['lower_level_geographies']

    # preload dict of control_data tables for sub_geographies
    control_data_tables \
        = load_control_data_tables(geographies, seed_id, sub_geographies, geo_cross_walk_df)

    # only want controls for sub_geographies
    control_spec = control_spec[control_spec['geography'].isin(sub_geographies)]

    # control total is sum of control_field of control_data_table for specified geography
    control_totals \
        = [control_data_tables.get(geography)[control_field].sum()
           for geography, control_field in zip(control_spec.geography, control_spec.control_field)]

    control_importance_weights = control_spec.importance
    initial_weights = incidence_df['initial_weight']+1

    # we only want the controls for sub_geographies
    incidence_df = incidence_df[control_spec.target]

    balancer = ListBalancer(
        incidence_table=incidence_df,
        initial_weights=initial_weights,
        control_totals=control_totals,
        control_importance_weights=control_importance_weights
    )

    return balancer


@orca.step()
def initial_seed_balancing(settings, geo_cross_walk, control_spec,
                           incidence_table, incidence_cols):

    # seed (puma) balancing, meta level balancing, meta
    # control factoring, and meta final balancing

    geographies = settings.get('geographies')

    seed_col = geographies['seed'].get('id_column')
    geo_cross_walk_df = geo_cross_walk.to_frame()

    incidence_df = incidence_table.to_frame()

    weight_list = []
    seed_ids = geo_cross_walk_df[seed_col].unique()
    for seed_id in seed_ids:

        logger.info("initial_seed_balancing seed id %s" % seed_id)

        balancer = seed_balancer(
            control_spec,
            seed_id,
            geographies,
            settings,
            geo_cross_walk_df,
            incidence_df=incidence_df[incidence_df[seed_col] == seed_id],
            incidence_cols=incidence_cols)

        # balancer.dump()

        status = balancer.balance()

        logger.info("seed_balancer status: %s" % status)

        weights = balancer.weights['final']

        weight_list.append(weights)

    weights = pd.concat(weight_list)
    orca.add_column('incidence_table', 'seed_weight', weights)
