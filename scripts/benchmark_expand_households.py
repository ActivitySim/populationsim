"""
Benchmark the seed household choice in the expand_households step.

Compares the previous algorithm, which built hh_id/probability lists for every
group in household_groups, with choose_group_households, which builds them only
for the groups present in the expanded weights. Data is synthetic. Both runs use
the same seeded RandomState and the script asserts they choose identical hh ids.

Wall time is measured without tracing; peak memory is the tracemalloc peak of a
separate traced run, counting only allocations made during the call.

Usage: python scripts/benchmark_expand_households.py [--groups 100000 500000] [--used 1000]
"""

import argparse
import time
import tracemalloc

import numpy as np
import pandas as pd

from populationsim.steps.expand_households import choose_group_households

HH_ID_COL = "hh_id"


def all_groups_choices(group_ids, household_groups, household_id_col, prng):
    grouper = household_groups.groupby("group_id")
    group_hh_probs = [0] * len(grouper)
    for group_id, df in grouper:
        hh_ids = list(df[household_id_col])
        probs = list(df.sample_weight / df.sample_weight.sum())
        group_hh_probs[group_id] = [hh_ids, probs]

    def chooser(group_id):
        return prng.choice(group_hh_probs[group_id][0], p=group_hh_probs[group_id][1])

    return group_ids.apply(chooser)


def synthetic_inputs(n_groups, n_used, seed=0):
    rng = np.random.RandomState(seed)
    group_size = rng.randint(1, 6, size=n_groups)
    group_id = np.repeat(np.arange(n_groups), group_size)
    household_groups = pd.DataFrame(
        {
            HH_ID_COL: np.arange(len(group_id)),
            "group_id": group_id,
            "sample_weight": rng.uniform(0.5, 50.0, size=len(group_id)),
        }
    )
    used = rng.choice(n_groups, size=n_used, replace=False)
    draws = rng.randint(1, 20, size=n_used)
    group_ids = pd.Series(np.sort(np.repeat(used, draws)), name="group_id")
    return household_groups, group_ids


def measure(fn, group_ids, household_groups):
    t = time.perf_counter()
    chosen = fn(group_ids, household_groups, HH_ID_COL, np.random.RandomState(42))
    seconds = time.perf_counter() - t

    tracemalloc.start()
    fn(group_ids, household_groups, HH_ID_COL, np.random.RandomState(42))
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return chosen, seconds, peak / 2**20


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--groups", type=int, nargs="+", default=[100_000, 500_000])
    parser.add_argument("--used", type=int, default=1000)
    args = parser.parse_args()

    header = (
        f"{'groups':>8} {'households':>10} {'used':>6} {'draws':>7} "
        f"{'old s':>8} {'new s':>8} {'old peak MiB':>13} {'new peak MiB':>13} identical"
    )
    print(header)
    for n_groups in args.groups:
        household_groups, group_ids = synthetic_inputs(n_groups, args.used)
        old, old_s, old_mib = measure(all_groups_choices, group_ids, household_groups)
        new, new_s, new_mib = measure(
            choose_group_households, group_ids, household_groups
        )
        identical = old.tolist() == new.tolist()
        print(
            f"{n_groups:>8} {len(household_groups):>10} {args.used:>6} {len(group_ids):>7} "
            f"{old_s:>8.2f} {new_s:>8.3f} {old_mib:>13.1f} {new_mib:>13.1f} {identical}"
        )
        assert identical, "choices differ between old and new algorithm"


if __name__ == "__main__":
    main()
