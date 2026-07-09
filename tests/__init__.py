# PopulationSim
# See full license in LICENSE.txt.

from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_DIR = Path(__file__).parent / "expected"


def expected_path(name: str) -> Path:
    """Return the reference expected parquet file.

    Returns ``expected/<name>.parquet`` (e.g. ``expected/expanded.parquet``).

    A single reference set is committed (generated on the CI platform). Tests
    compare against it with a tolerance rather than bit-for-bit, so one set
    suffices across platforms -- see :func:`assert_expanded_close`.
    """
    return EXPECTED_DIR / f"{name}.parquet"


def assert_expanded_close(actual, expected, geog_col="TAZ", rtol=0.0, atol=2):
    """Assert two ``expanded_household_ids`` tables match at the zone level.

    The MILP solvers (CBC/GLPK) and numba ``fastmath`` produce different but
    equally valid integer solutions across platforms and OS versions, so
    comparing the exact household-id composition bit-for-bit is fragile. What is
    stable -- and what actually matters for a synthetic population -- is the
    per-zone household distribution. This comparator checks that distribution
    within a tolerance instead of ``DataFrame.equals()``.

    Parameters
    ----------
    actual, expected : pandas.DataFrame
        Expanded household id tables (one row per synthesized household).
    geog_col : str
        Geography column to aggregate household counts by (default ``"TAZ"``).
    rtol, atol : float
        Relative / absolute tolerance on per-zone and total counts. ``atol`` is
        in households; a couple of households per zone absorbs solver
        tie-breaking while still catching real regressions.
    """
    assert isinstance(actual, pd.DataFrame)
    assert isinstance(expected, pd.DataFrame)
    assert list(actual.columns) == list(expected.columns), (
        f"column mismatch: {list(actual.columns)} != {list(expected.columns)}"
    )
    assert geog_col in actual.columns, f"missing geography column '{geog_col}'"

    # same set of zones present
    assert set(actual[geog_col]) == set(expected[geog_col]), (
        "zone sets differ between actual and expected"
    )

    # per-zone household counts, aligned on the common (sorted) zone index
    a = actual.groupby(geog_col).size()
    e = expected.groupby(geog_col).size()
    a, e = a.align(e, fill_value=0)
    np.testing.assert_allclose(
        a.values, e.values, rtol=rtol, atol=atol, err_msg="per-zone counts differ"
    )

    # total row count
    np.testing.assert_allclose(
        len(actual.index), len(expected.index), rtol=rtol, atol=atol,
        err_msg="total household count differs",
    )
