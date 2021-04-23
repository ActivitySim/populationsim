
# PopulationSim
# See full license in LICENSE.txt.

import os
import numpy as np
import pandas as pd

from activitysim.core import inject

from populationsim.multi_integerizer import do_simul_integerizing
from populationsim.multi_integerizer import do_sequential_integerizing

incidence_df = pd.DataFrame({
    'hh_id': [0, 6, 12, 18, 24, 30],
    'num_hh': [1, 1, 1, 1, 1, 1],
    'hh_size_1': [0, 0, 0, 0, 0, 1],
    'hh_size_2': [0, 0, 0, 0, 1, 0],
    'hh_size_3': [0, 0, 0, 1, 0, 0],
    'hh_size_4_plus': [1, 1, 1, 0, 0, 0],
    'students_by_housing_type': [0, 2, 3, 1, 0, 0],
    'hh_by_type': [0, 1, 1, 1, 1, 1],
    'persons_occ_1': [0, 2, 2, 2, 2, 0],
    'persons_occ_2': [0, 0, 0, 0, 0, 1],
    'persons_occ_3': [10, 2, 3, 1, 0, 0],
}).set_index('hh_id')
incidence_df = incidence_df[
    ['num_hh', 'hh_size_1', 'hh_size_2', 'hh_size_3', 'hh_size_4_plus', 'students_by_housing_type',
     'hh_by_type', 'persons_occ_1', 'persons_occ_2', 'persons_occ_3']]

sub_zone_weights = pd.DataFrame({
    'hh_id': [0, 6, 12, 18, 24, 30],
    'TRACT_1': [0.0, 13.9663617613, 9.97597268661, 48.9423344608, 1.3135301063, 0.838423693168],
    'TRACT_2': [0.0, 0.0336382387492, 0.0240273133923, 0.0576655392034, 45.6864698937,
                29.1615763068],
}).set_index('hh_id')

sub_controls_df = pd.DataFrame({
    'TRACT': [1, 2],
    'num_hh': [75, 75],
    'hh_size_1': [15, 15],
    'hh_size_2': [24, 24],
    'hh_size_3': [24, 24],
    'hh_size_4_plus': [12, 12],
    'students_by_housing_type': [106, 0],
    'hh_by_type': [75, 75],
}).set_index('TRACT')
sub_controls_df = sub_controls_df[
    ['num_hh', 'hh_size_1', 'hh_size_2', 'hh_size_3', 'hh_size_4_plus', 'students_by_housing_type',
     'hh_by_type']]

control_spec = pd.DataFrame({
    'target': ['num_hh', 'hh_size_1', 'hh_size_2', 'hh_size_3', 'hh_size_4_plus',
               'students_by_housing_type', 'hh_by_type', 'persons_occ_1', 'persons_occ_2',
               'persons_occ_3'],
    'geography': ['TAZ', 'TAZ', 'TAZ', 'TAZ', 'TAZ', 'TAZ', 'TRACT', 'DISTRICT', 'DISTRICT',
                  'DISTRICT', ],
    'seed_table': ['households', 'households', 'households', 'households', 'households', 'persons',
                   'households', 'persons', 'persons', 'persons'],
    'importance': [1000000000, 1000, 1000, 1000, 1000, 1000, 100, 1000, 100, 100]
})

sub_control_zones = pd.Series(['TRACT_1', 'TRACT_2'], index=[1, 2])


def test_simul_integerizer():

    configs_dir = os.path.join(os.path.dirname(__file__), 'configs')
    inject.add_injectable("configs_dir", configs_dir)

    # data_dir = os.path.join(os.path.dirname(__file__), 'data')
    # inject.add_injectable("data_dir", data_dir)
    #
    # output_dir = os.path.join(os.path.dirname(__file__), 'output')
    # inject.add_injectable("output_dir", output_dir)

    integer_weights_df = do_simul_integerizing(
        trace_label="label",
        incidence_df=incidence_df,
        sub_weights=sub_zone_weights,
        sub_controls_df=sub_controls_df,
        control_spec=control_spec,
        total_hh_control_col='num_hh',
        sub_geography='TRACT',
        sub_control_zones=sub_control_zones
    )

    assert (integer_weights_df.integer_weight.values == [
        0,
        14,
        10,
        49,
        1,
        1,
        0,
        0,
        0,
        0,
        46,
        29
    ]).all()

    print("\ntest_simul_integerizer integer_weights_df\n", integer_weights_df)


def test_sequential_integerizer():
    configs_dir = os.path.join(os.path.dirname(__file__), 'configs')
    inject.add_injectable("configs_dir", configs_dir)

    # data_dir = os.path.join(os.path.dirname(__file__), 'data')
    # inject.add_injectable("data_dir", data_dir)
    #
    # output_dir = os.path.join(os.path.dirname(__file__), 'output')
    # inject.add_injectable("output_dir", output_dir)

    integer_weights_df = do_sequential_integerizing(
        trace_label="label",
        incidence_df=incidence_df,
        sub_weights=sub_zone_weights,
        sub_controls_df=sub_controls_df,
        control_spec=control_spec,
        total_hh_control_col='num_hh',
        sub_geography='TRACT',
        sub_control_zones=sub_control_zones
    )

    print("\ntest_sequential_integerizer integer_weights_df\n", integer_weights_df)

    assert (integer_weights_df.integer_weight.values == [
        0,
        14,
        10,
        49,
        1,
        1,
        0,
        0,
        0,
        0,
        46,
        29
    ]).all()
