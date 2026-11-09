"""
Benchmark the numba balancers against an earlier version of themselves.

The incidence tables are synthetic and shaped like a household seed: one
row per household, one-hot household categories (income band, household
size, and so on) and person counts by category (age band, and so on). The
baseline balancers are read from a git revision, so the script compares the
working tree against any earlier commit without keeping old code around.

Usage, from the repository root:

    uv run python scripts/benchmark_balancers.py
    uv run python scripts/benchmark_balancers.py --baseline 64449c8 --iterations 50

For each problem size the script reports the share of nonzero incidence
cells, the time per balancing iteration for the baseline and for the working
tree on one thread and on all threads, and the largest relative weight
difference between baseline and working tree after the same number of
iterations. Working-tree results are also checked to be identical on one
thread and on all threads.
"""

import argparse
import importlib.util
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numba
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from populationsim.balancing import balancers_numba as current  # noqa: E402

HOUSEHOLD_CATEGORIES = (10, 7, 5, 4, 4, 3, 2, 2)
PERSON_CATEGORIES = (18, 7, 3, 2)


def load_baseline(revision):
    source = subprocess.run(
        ["git", "show", f"{revision}:populationsim/balancing/balancers_numba.py"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    path = Path(tempfile.mkdtemp()) / "baseline_balancers_numba.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location("baseline_balancers_numba", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_incidence(sample_count, rng):
    rows = [np.ones(sample_count)]
    for levels in HOUSEHOLD_CATEGORIES:
        category = rng.integers(0, levels, sample_count)
        rows.extend((category == level).astype(np.float64) for level in range(levels))

    persons = rng.integers(1, 7, sample_count)
    person_household = np.repeat(np.arange(sample_count), persons)
    for levels in PERSON_CATEGORIES:
        category = rng.integers(0, levels, person_household.size)
        counts = np.zeros((levels, sample_count))
        np.add.at(counts, (category, person_household), 1.0)
        rows.extend(counts)
    return np.vstack(rows)


def single_problem(sample_count, seed):
    rng = np.random.default_rng(seed)
    incidence = synthetic_incidence(sample_count, rng)
    control_count = incidence.shape[0]
    weights = rng.uniform(5.0, 50.0, sample_count)
    controls = (incidence * weights).sum(axis=1) * rng.uniform(
        0.85, 1.15, control_count
    )
    importance = np.full(control_count, 1000.0)
    importance[0] = 1.0e9
    return (
        sample_count,
        control_count,
        0,
        incidence,
        weights,
        np.zeros(sample_count),
        weights * 30.0,
        controls,
        importance,
    )


def simul_problem(sample_count, zone_count, seed):
    rng = np.random.default_rng(seed)
    incidence = synthetic_incidence(sample_count, rng)
    control_count = incidence.shape[0]
    parent_weights = rng.uniform(5.0, 50.0, sample_count)
    sub_weights = parent_weights * rng.dirichlet(np.ones(zone_count), sample_count).T
    sub_controls = (sub_weights @ incidence.T) * rng.uniform(
        0.85, 1.15, (zone_count, control_count)
    )
    importance = np.full(control_count, 1000.0)
    importance[0] = 1.0e9
    return (
        sample_count,
        control_count,
        zone_count,
        0,
        incidence,
        parent_weights,
        np.zeros(sample_count),
        parent_weights.copy(),
        sub_weights,
        sub_controls.sum(axis=0),
        importance,
        sub_controls,
    )


def copy_args(args):
    return tuple(a.copy() if isinstance(a, np.ndarray) else a for a in args)


def timed(balancer, args, iterations):
    balancer(*copy_args(args), 2)
    start = time.perf_counter()
    weights, _, status = balancer(*copy_args(args), iterations)
    return time.perf_counter() - start, weights, status


def compare(label, baseline, current_balancer, args, iterations, incidence):
    threads = numba.get_num_threads()
    numba.set_num_threads(1)
    base_time, base_weights, _ = timed(baseline, args, iterations)
    one_time, one_weights, _ = timed(current_balancer, args, iterations)
    numba.set_num_threads(threads)
    all_time, all_weights, _ = timed(current_balancer, args, iterations)
    assert np.array_equal(one_weights, all_weights), "thread count changed results"
    difference = np.max(
        np.abs(one_weights - base_weights) / np.maximum(np.abs(base_weights), 1e-300)
    )
    per_iteration = 1e3 / iterations
    print(
        f"{label:34s} {np.count_nonzero(incidence) / incidence.size:8.1%}"
        f" {base_time * per_iteration:9.2f}"
        f" {one_time * per_iteration:9.2f} ({base_time / one_time:4.1f}x)"
        f" {all_time * per_iteration:9.2f} ({base_time / all_time:4.1f}x)"
        f" {difference:12.1e}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--baseline", default="64449c8")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=[10_000, 100_000, 500_000]
    )
    parser.add_argument("--zones", type=int, default=10)
    options = parser.parse_args()

    baseline = load_baseline(options.baseline)
    print(
        f"baseline {options.baseline} vs working tree, {options.iterations} iterations,"
        f" ms per iteration, {numba.get_num_threads()} threads available"
    )
    print(
        f"{'problem':34s} {'nonzero':>8s} {'baseline':>9s}"
        f" {'1 thread':>16s} {'all threads':>16s} {'max rel diff':>12s}"
    )
    for size in options.sizes:
        args = single_problem(size, options.seed)
        compare(
            f"single, {size:,} rows x {args[1]} controls",
            baseline.np_balancer_numba,
            current.np_balancer_numba,
            args,
            options.iterations,
            args[3],
        )
    for size in options.sizes:
        args = simul_problem(size // options.zones, options.zones, options.seed)
        compare(
            f"simul, {size // options.zones:,} rows x {options.zones} zones",
            baseline.np_simul_balancer_numba,
            current.np_simul_balancer_numba,
            args,
            options.iterations,
            args[4],
        )


if __name__ == "__main__":
    main()
