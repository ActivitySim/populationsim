# PopulationSim
# See full license in LICENSE.txt.

"""
build_incidence_table and build_grouped_incidence_table must produce exactly the
tables (values, dtypes, index) of their previous groupby / merge implementations,
which are restated here as the reference.
"""

import sys

import numpy as np
import pandas as pd
import pytest

from populationsim.core import inject
from populationsim.core.assign import assign_variable
from populationsim.steps.setup_data_structures import (
    build_grouped_incidence_table,
    build_incidence_table,
)

HH_ID = "hh_id"
SEED_GEOGRAPHY = "PUMA"

CONTROL_SPEC = pd.DataFrame(
    [
        ("num_hh", "households", "households.hh_size > 0"),
        ("hh_size_3", "households", "households.hh_size.clip(upper=3)"),
        ("persons_total", "persons", "persons.age >= 0"),
        ("age_65_plus", "persons", "persons.age >= 65"),
        ("age_decades", "persons", "persons.age // 10"),
        ("workers", "persons", "np.where(persons.worker, 1, 0)"),
        ("half_weight", "persons", "persons.weight * 0.5"),
        ("too_large_for_float64", "persons", "persons.age * 2**50"),
    ],
    columns=["target", "seed_table", "expression"],
)


def setup_function():
    inject.add_injectable("settings", {"household_id_col": HH_ID})


def teardown_function(func):
    inject.clear_cache()
    inject.reinject_decorated_tables()


def old_build_incidence_table(control_spec, households_df, persons_df):
    incidence_table = pd.DataFrame(index=households_df.index)
    seed_tables = {"households": households_df, "persons": persons_df}
    for control_row in control_spec.itertuples():
        incidence, _ = assign_variable(
            target=control_row.target,
            expression=control_row.expression,
            df=seed_tables[control_row.seed_table],
            locals_dict={"np": np},
            df_alias=control_row.seed_table,
            trace_rows=None,
        )
        incidence = incidence * 1
        if control_row.seed_table == "persons":
            df = pd.DataFrame({HH_ID: persons_df[HH_ID], "incidence": incidence})
            incidence = df.groupby([HH_ID], as_index=True).sum()
        incidence_table[control_row.target] = incidence
    return incidence_table


def old_build_grouped_incidence_table(incidence_table, control_spec, seed_geography):
    hh_incidence_table = incidence_table
    hh_groupby_cols = list(control_spec.target) + [seed_geography]
    hh_grouper = hh_incidence_table.groupby(hh_groupby_cols)
    group_incidence_table = hh_grouper.max()
    group_incidence_table["sample_weight"] = hh_grouper.sum()["sample_weight"]
    group_incidence_table["group_size"] = hh_grouper.count()["sample_weight"]
    group_incidence_table = group_incidence_table.reset_index()
    group_incidence_table = group_incidence_table.astype(np.int64)
    group_incidence_table["group_id"] = group_incidence_table.index.astype(np.int64)
    hh_incidence_table["group_id"] = pd.merge(
        hh_incidence_table[hh_groupby_cols],
        group_incidence_table[hh_groupby_cols + ["group_id"]],
        on=hh_groupby_cols,
        how="left",
    ).group_id.values
    group_incidence_table.index.name = hh_incidence_table.index.name
    household_groups = hh_incidence_table[["group_id", "sample_weight"]].copy()
    household_groups[HH_ID] = household_groups.index.astype(np.int64)
    return group_incidence_table, household_groups


def make_seed(num_households, case, seed=0):
    rng = np.random.default_rng(seed)
    hh_ids = np.arange(1, num_households + 1) * 3
    hh_size = rng.integers(1, 6, num_households)
    if case == "households_without_persons":
        hh_size[rng.random(num_households) < 0.1] = 0

    households = pd.DataFrame(
        {
            HH_ID: hh_ids,
            "hh_size": hh_size,
            SEED_GEOGRAPHY: rng.integers(100, 104, num_households),
            "sample_weight": rng.integers(1, 50, num_households),
        }
    ).set_index(HH_ID)

    num_persons = hh_size.sum()
    persons = pd.DataFrame(
        {
            HH_ID: np.repeat(hh_ids, hh_size),
            "age": rng.integers(0, 95, num_persons),
            "worker": rng.random(num_persons) < 0.5,
            "weight": rng.random(num_persons),
        }
    )
    if case == "shuffled_persons":
        persons = persons.sample(frac=1, random_state=seed)
        persons.index = persons.index * 7 + 11
    elif case == "missing_household_id":
        persons[HH_ID] = persons[HH_ID].astype(float)
        persons.loc[persons.index[::50], HH_ID] = np.nan
    elif case == "persons_outside_households":
        outside = persons.iloc[::50].assign(**{HH_ID: -1})
        persons = pd.concat([persons, outside], ignore_index=True)
    return households, persons


CASES = [
    "matched",
    "shuffled_persons",
    "households_without_persons",
    "missing_household_id",
    "persons_outside_households",
]


@pytest.mark.parametrize("case", CASES)
def test_build_incidence_table_matches_groupby(case, monkeypatch):
    households, persons = make_seed(2000, case)

    bincount_calls = []
    bincount = np.bincount

    def counting_bincount(*args, **kwargs):
        if sys._getframe(1).f_code.co_name == "build_incidence_table":
            bincount_calls.append(1)
        return bincount(*args, **kwargs)

    monkeypatch.setattr(np, "bincount", counting_bincount)

    expected = old_build_incidence_table(CONTROL_SPEC, households, persons)
    result = build_incidence_table(CONTROL_SPEC, households, persons, None)

    pd.testing.assert_frame_equal(result, expected, check_exact=True)
    if case in ("matched", "shuffled_persons", "persons_outside_households"):
        # the four int and bool person controls, not the float or too-large ones
        assert len(bincount_calls) == 4
        assert result.age_decades.dtype == np.int64
        assert result.half_weight.dtype == np.float64
    else:
        assert len(bincount_calls) == 0


@pytest.mark.parametrize("case", CASES)
def test_build_grouped_incidence_table_matches_merge(case):
    households, persons = make_seed(5000, case, seed=1)
    spec = CONTROL_SPEC[
        ~CONTROL_SPEC.target.isin(["half_weight", "too_large_for_float64"])
    ]

    incidence = old_build_incidence_table(spec, households, persons)
    incidence["sample_weight"] = households.sample_weight
    incidence[SEED_GEOGRAPHY] = households[SEED_GEOGRAPHY]

    expected_input = incidence.copy()
    expected = old_build_grouped_incidence_table(expected_input, spec, SEED_GEOGRAPHY)
    result_input = incidence.copy()
    result = build_grouped_incidence_table(result_input, spec, SEED_GEOGRAPHY)

    assert len(result[0]) > 100
    pd.testing.assert_frame_equal(result[0], expected[0], check_exact=True)
    pd.testing.assert_frame_equal(result[1], expected[1], check_exact=True)
    pd.testing.assert_frame_equal(result_input, expected_input, check_exact=True)
