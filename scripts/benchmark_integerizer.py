"""
Benchmark the OR-Tools integerizers on synthetic problems: the legacy
per-element pywraplp model construction against the MPModelProto one in
populationsim/integerizing/lp_ortools.py.

Each (problem, size, implementation) runs in its own process. It reports the
model build time (until Solve is called), the solve time (Solve plus reading
the solution), and the peak RSS increase: the process's maximum resident set
size minus its RSS just before the call (the problem generation that precedes
the call peaks far lower than the integerizer at the benchmark sizes).
The script fails unless both implementations export byte-identical models
(ExportModelToProto, deterministic serialization) and return identical
solutions and statuses.

usage: python scripts/benchmark_integerizer.py [--sizes 10000 100000 1000000]
"""

import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SIMUL_ZONES = 20


def problem(kind, size):
    """Synthetic arguments with about `size` household variables."""
    from tests.integerizer_lp_reference import simul_problem, single_problem

    if kind == "single":
        return (), single_problem(size, control_count=24, seed=0)
    args = simul_problem(size // SIMUL_ZONES, SIMUL_ZONES, 16, 16, seed=0)
    return args, {}


def run_one(kind, size, impl):
    """Build and solve one problem; return timings, peak RSS and digests."""
    from ortools.linear_solver import linear_solver_pb2, pywraplp

    from populationsim.integerizing import lp_ortools
    from tests import integerizer_lp_reference as ref

    func = {
        ("single", "legacy"): ref.legacy_integerizer_ortools,
        ("single", "new"): lp_ortools.np_integerizer_ortools,
        ("simul", "legacy"): ref.legacy_simul_integerizer_ortools,
        ("simul", "new"): lp_ortools.np_simul_integerizer_ortools,
    }[kind, impl]
    args, kwargs = problem(kind, size)

    solve_calls = []
    original_solve = pywraplp.Solver.Solve

    def solve(self, *a):
        solve_calls.append((self, time.perf_counter()))
        return original_solve(self, *a)

    pywraplp.Solver.Solve = solve

    baseline = psutil.Process().memory_info().rss
    start = time.perf_counter()
    weights, status = func(*args, **kwargs)
    end = time.perf_counter()
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak *= 1 if sys.platform == "darwin" else 1024  # bytes on macOS, KiB on Linux

    (solver, solve_start), *rest = solve_calls
    assert not rest
    model = linear_solver_pb2.MPModelProto()
    solver.ExportModelToProto(model)
    return {
        "build_s": solve_start - start,
        "solve_s": end - solve_start,
        "total_s": end - start,
        "peak_rss_mb": (peak - baseline) / 2**20,
        "variables": len(model.variable),
        "status": status,
        "model_sha": hashlib.sha256(
            model.SerializeToString(deterministic=True)
        ).hexdigest(),
        "solution_sha": hashlib.sha256(np.ascontiguousarray(weights)).hexdigest()
        + str(weights.dtype)
        + str(weights.shape),
    }


def child(kind, size, impl, out):
    # CBC logs to stdout; keep it out of the report
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    Path(out).write_text(json.dumps(run_one(kind, size, impl)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=[10_000, 100_000, 1_000_000]
    )
    parser.add_argument(
        "--kinds", nargs="+", default=["single", "simul"], choices=["single", "simul"]
    )
    parser.add_argument("--child", nargs=4, help=argparse.SUPPRESS)
    options = parser.parse_args()
    if options.child:
        kind, size, impl, out = options.child
        return child(kind, int(size), impl, out)

    print(
        "| problem | household vars | total vars | impl | build s | solve s "
        "| total s | peak RSS increase MB | status |"
    )
    print("|---|--:|--:|---|--:|--:|--:|--:|---|")
    failures = 0
    for kind in options.kinds:
        for size in options.sizes:
            results = {}
            for impl in ("legacy", "new"):
                with tempfile.NamedTemporaryFile(suffix=".json") as out:
                    subprocess.run(
                        [sys.executable, __file__, "--child", kind, str(size), impl]
                        + [out.name],
                        check=True,
                    )
                    r = results[impl] = json.loads(Path(out.name).read_text())
                print(
                    f"| {kind} | {size:,} | {r['variables']:,} | {impl} "
                    f"| {r['build_s']:.2f} | {r['solve_s']:.2f} | {r['total_s']:.2f} "
                    f"| {r['peak_rss_mb']:.0f} | {r['status']} |",
                    flush=True,
                )
            legacy, new = results["legacy"], results["new"]
            for key in ("model_sha", "solution_sha", "status"):
                if legacy[key] != new[key]:
                    failures += 1
                    print(f"MISMATCH {kind} {size} {key}", flush=True)
    if failures:
        sys.exit("models or solutions differ")
    print("\nAll models byte-identical and all solutions identical.")


if __name__ == "__main__":
    main()
