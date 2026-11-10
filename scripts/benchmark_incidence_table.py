"""
Benchmark build_incidence_table and build_grouped_incidence_table against their
previous groupby / merge implementations on synthetic seed data, and check that
both produce identical tables.

usage: python scripts/benchmark_incidence_table.py [num_households ...]
"""

import sys
import time
import tracemalloc

import numpy as np
import pandas as pd

from populationsim.core import inject
from populationsim.core.assign import assign_variable
from populationsim.steps.setup_data_structures import (
    build_grouped_incidence_table,
    build_incidence_table,
)

HH_ID = "hh_id"
SEED_GEOGRAPHY = "PUMA"


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


def make_seed(num_households, seed=0):
    rng = np.random.default_rng(seed)
    hh_size = rng.choice([1, 2, 3, 4, 5], num_households, p=[0.3, 0.3, 0.2, 0.1, 0.1])
    households = pd.DataFrame(
        {
            HH_ID: np.arange(1, num_households + 1),
            "hh_size": hh_size,
            "income": rng.integers(0, 250_000, num_households),
            SEED_GEOGRAPHY: rng.integers(100, 200, num_households),
            "sample_weight": rng.integers(1, 100, num_households),
        }
    ).set_index(HH_ID)
    num_persons = hh_size.sum()
    persons = pd.DataFrame(
        {
            HH_ID: np.repeat(households.index.to_numpy(), hh_size),
            "age": rng.integers(0, 95, num_persons),
            "sex": rng.integers(1, 3, num_persons),
            "worker": rng.random(num_persons) < 0.5,
            "occupation": rng.integers(0, 10, num_persons),
        }
    )
    return households, persons


def make_control_spec():
    rows = [
        ("num_hh", "households", "households.hh_size > 0"),
        ("hh_size_4_plus", "households", "households.hh_size >= 4"),
        ("hh_inc_100k_plus", "households", "households.income >= 100_000"),
        ("persons_total", "persons", "persons.age >= 0"),
        ("workers", "persons", "persons.worker"),
        ("worker_age_sum", "persons", "np.where(persons.worker, persons.age, 0)"),
    ]
    for lo in range(0, 90, 5):
        for sex in (1, 2):
            rows.append(
                (
                    f"age_{lo}_{lo + 4}_sex_{sex}",
                    "persons",
                    f"(persons.age >= {lo}) & (persons.age <= {lo + 4}) & (persons.sex == {sex})",
                )
            )
    for occupation in range(10):
        rows.append(
            (
                f"occupation_{occupation}",
                "persons",
                f"persons.occupation == {occupation}",
            )
        )
    return pd.DataFrame(rows, columns=["target", "seed_table", "expression"])


def measure(fn, *args):
    start = time.perf_counter()
    fn(*args)
    seconds = time.perf_counter() - start
    tracemalloc.start()
    result = fn(*args)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return result, seconds, peak / 2**20


def main(sizes):
    inject.add_injectable("settings", {"household_id_col": HH_ID})
    control_spec = make_control_spec()
    num_person_controls = (control_spec.seed_table == "persons").sum()
    print(
        f"{len(control_spec)} controls ({num_person_controls} person controls); "
        "time is wall clock, peak is the tracemalloc peak of a separate run"
    )
    print(
        f"{'households':>10} {'persons':>9} {'function':<30}"
        f"{'old s':>8} {'new s':>8} {'speedup':>8} {'old MiB':>9} {'new MiB':>9}"
    )

    for num_households in sizes:
        households, persons = make_seed(num_households)

        old, old_s, old_mb = measure(
            old_build_incidence_table, control_spec, households, persons
        )
        new, new_s, new_mb = measure(
            build_incidence_table, control_spec, households, persons, None
        )
        pd.testing.assert_frame_equal(new, old, check_exact=True)
        print(
            f"{num_households:>10,} {len(persons):>9,} {'build_incidence_table':<30}"
            f"{old_s:>8.2f} {new_s:>8.2f} {old_s / new_s:>7.1f}x {old_mb:>9.0f} {new_mb:>9.0f}"
        )

        incidence = old
        incidence["sample_weight"] = households.sample_weight
        incidence[SEED_GEOGRAPHY] = households[SEED_GEOGRAPHY]
        old, old_s, old_mb = measure(
            lambda: old_build_grouped_incidence_table(
                incidence.copy(), control_spec, SEED_GEOGRAPHY
            )
        )
        new, new_s, new_mb = measure(
            lambda: build_grouped_incidence_table(
                incidence.copy(), control_spec, SEED_GEOGRAPHY
            )
        )
        for new_table, old_table in zip(new, old, strict=True):
            pd.testing.assert_frame_equal(new_table, old_table, check_exact=True)
        print(
            f"{num_households:>10,} {len(persons):>9,} {'build_grouped_incidence_table':<30}"
            f"{old_s:>8.2f} {new_s:>8.2f} {old_s / new_s:>7.1f}x {old_mb:>9.0f} {new_mb:>9.0f}"
        )
    print("old and new tables are identical")


if __name__ == "__main__":
    main([int(s) for s in sys.argv[1:]] or [100_000, 1_000_000])
