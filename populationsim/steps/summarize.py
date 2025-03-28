
# PopulationSim
# See full license in LICENSE.txt.

import logging
import os
import pandas as pd
import numpy as np

from activitysim.core import inject

from .helper import get_control_table
from .helper import get_weight_table
from activitysim.core.config import setting

logger = logging.getLogger(__name__)

AS_CSV = False


def out_table(table_name, df):

    table_name = "summary_%s" % table_name

    if AS_CSV:
        file_name = "%s.csv" % table_name
        output_dir = inject.get_injectable('output_dir')
        file_path = os.path.join(output_dir, file_name)
        logger.info("writing output file %s" % file_path)
        write_index = df.index.name is not None
        df.to_csv(file_path, index=write_index)
    else:
        logger.info("saving summary table %s" % table_name)
        repop = inject.get_step_arg('repop', default=False)
        inject.add_table(table_name, df, replace=repop)


def summarize_geography(geography, weight_col, hh_id_col,
                        crosswalk_df, results_df, incidence_df):

    # controls_table for current geography level
    controls_table = get_control_table(geography)
    control_names = controls_table.columns.tolist()

    # only want zones from crosswalk for which non-zero control rows exist
    zone_ids = crosswalk_df[geography].unique()
    zone_ids = controls_table.index.intersection(zone_ids).astype(np.int64)
    
    # Using numpy matrix multiplication for efficient aggregation        
    zone_results_df = results_df.loc[results_df[geography].isin(zone_ids), [geography, hh_id_col, weight_col]]    
    
    geo_vec = zone_results_df[geography].to_numpy()
    weights = zone_results_df[weight_col].to_numpy()
    incidence = incidence_df.loc[zone_results_df[hh_id_col], control_names].to_numpy()
    
    results = np.transpose(np.transpose(incidence) * weights)
    results = np.column_stack([results, geo_vec])    

    logger.info("summarizing %s" % geography)
    controls = [controls_table.loc[x].tolist() for x in zone_ids]
    
    summary_df = pd.DataFrame(
        data=results,
        columns=['%s_result' % c for c in control_names] + [geography]
    ).groupby(geography).sum()
    
    controls_df = pd.DataFrame(
        data=np.asanyarray(controls),
        columns=['%s_control' % c for c in control_names],
        index=zone_ids
    )

    dif_df = pd.DataFrame(
        # data=np.asanyarray(results) - np.asanyarray(controls),
        data=np.array(summary_df) - np.array(controls_df),
        columns=['%s_diff' % c for c in control_names],
        index=zone_ids
    )

    summary_df = pd.concat([controls_df, summary_df, dif_df], axis=1, ignore_index=False)

    summary_cols = summary_df.columns.tolist()

    summary_df['geography'] = geography
    summary_df['id'] = summary_df.index
    summary_df.index = summary_df['geography'] + '_' + summary_df['id'].astype(str)
    summary_df = summary_df[['geography', 'id'] + summary_cols]

    return summary_df


def meta_summary(incidence_df, control_spec, top_geography, top_id, sub_geographies, hh_id_col):

    if setting('NO_INTEGERIZATION_EVER', False):
        seed_weight_cols = ['preliminary_balanced_weight', 'balanced_weight']
        sub_weight_cols = ['balanced_weight']
    else:
        seed_weight_cols = ['preliminary_balanced_weight', 'balanced_weight', 'integer_weight']
        sub_weight_cols = ['balanced_weight', 'integer_weight']

    incidence_df = incidence_df[incidence_df[top_geography] == top_id]

    control_cols = control_spec.target.values

    controls_df = get_control_table(top_geography)

    # controls for this geography as series
    controls = controls_df[control_cols].loc[top_id]

    incidence = incidence_df[control_cols]

    summary = pd.DataFrame(index=control_cols)

    summary.index.name = 'control_name'

    summary['control_value'] = controls

    seed_geography = setting('seed_geography')
    seed_weights_df = get_weight_table(seed_geography)

    for c in seed_weight_cols:
        if c in seed_weights_df:
            summary_col_name = '%s_%s' % (top_geography, c)
            summary[summary_col_name] = \
                incidence.multiply(seed_weights_df[c], axis="index").sum(axis=0)

    for g in sub_geographies:

        sub_weights = get_weight_table(g)

        if sub_weights is None:
            continue

        sub_weights = sub_weights[sub_weights[top_geography] == top_id]

        sub_weights = sub_weights[[hh_id_col] + sub_weight_cols].groupby(hh_id_col).sum()

        for c in sub_weight_cols:
            summary['%s_%s' % (g, c)] = \
                incidence.multiply(sub_weights[c], axis="index").sum(axis=0)

    return summary


@inject.step()
def summarize(crosswalk, incidence_table, control_spec):
    """
    Write aggregate summary files of controls and weights for all geographic levels to output dir

    Parameters
    ----------
    crosswalk : pipeline table
    incidence_table : pipeline table
    control_spec : pipeline table

    Returns
    -------

    """

    include_integer_colums = not setting('NO_INTEGERIZATION_EVER', False)

    crosswalk_df = crosswalk.to_frame()
    incidence_df = incidence_table.to_frame()

    geographies = setting('geographies')
    seed_geography = setting('seed_geography')
    meta_geography = geographies[0]
    sub_geographies = geographies[geographies.index(seed_geography) + 1:]
    super_geographies = geographies[:geographies.index(seed_geography)]
    
    hh_id_col = setting('household_id_col')

    meta_ids = crosswalk_df[meta_geography].unique()
    for meta_id in meta_ids:
        meta_summary_df = \
            meta_summary(incidence_df, control_spec, meta_geography,
                         meta_id, sub_geographies, hh_id_col)
        out_table('%s_%s' % (meta_geography, meta_id), meta_summary_df)

    hh_weights_summary = pd.DataFrame(index=incidence_df.index)

    # add seed level summaries
    seed_weights_df = get_weight_table(seed_geography)
    hh_weights_summary['%s_balanced_weight' % seed_geography] = seed_weights_df['balanced_weight']
    if include_integer_colums:
        hh_weights_summary['%s_integer_weight' % seed_geography] = seed_weights_df['integer_weight']

    for geography in sub_geographies:

        weights_df = get_weight_table(geography)

        if weights_df is None:
            continue

        if include_integer_colums:
            hh_weight_cols = [hh_id_col, 'balanced_weight', 'integer_weight']
        else:
            hh_weight_cols = [hh_id_col, 'balanced_weight']

        hh_weights = weights_df[hh_weight_cols].groupby([hh_id_col]).sum()
        hh_weights_summary['%s_balanced_weight' % geography] = hh_weights['balanced_weight']
        if include_integer_colums:
            hh_weights_summary['%s_integer_weight' % geography] = hh_weights['integer_weight']

        # aggregate to seed level
        aggegrate_weights = weights_df.groupby([seed_geography, hh_id_col], as_index=False).sum()
        aggegrate_weights.set_index(hh_id_col, inplace=True)

        if include_integer_colums:
            aggegrate_weight_cols = [seed_geography, 'balanced_weight', 'integer_weight']
        else:
            aggegrate_weight_cols = [seed_geography, 'balanced_weight']

        aggegrate_weights = aggegrate_weights[aggegrate_weight_cols]
        aggegrate_weights['sample_weight'] = incidence_df['sample_weight']
        aggegrate_weights['%s_preliminary_balanced_weight' % seed_geography] = \
            seed_weights_df['preliminary_balanced_weight']
        aggegrate_weights['%s_balanced_weight' % seed_geography] = \
            seed_weights_df['balanced_weight']
        if include_integer_colums:
            aggegrate_weights['%s_integer_weight' % seed_geography] = \
                seed_weights_df['integer_weight']

        out_table('%s_aggregate' % (geography,), aggegrate_weights)

        summary_col = 'integer_weight' if include_integer_colums else 'balanced_weight'
        df_seed = summarize_geography(seed_geography, summary_col, hh_id_col,
                                 crosswalk_df, weights_df, incidence_df)
        out_table('%s_%s' % (geography, seed_geography,), df_seed)

        df_geo = summarize_geography(geography, summary_col, hh_id_col,
                                 crosswalk_df, weights_df, incidence_df)
        out_table('%s' % (geography,), df_geo)
        
        # Aggregate super geographies
        for super_geo in super_geographies:        
            df_super = summarize_geography(super_geo, summary_col, hh_id_col, crosswalk_df, weights_df, incidence_df)
            out_table('%s_%s' % (geography, super_geo,), df_super)

    out_table('hh_weights', hh_weights_summary)
