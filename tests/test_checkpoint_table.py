# PopulationSim
# See full license in LICENSE.txt.

import warnings

import pandas as pd

from populationsim.core import pipeline


def checkpoint_history(table_count=33, checkpoint_count=12):
    rows = []
    for i in range(checkpoint_count):
        row = {
            pipeline.CHECKPOINT_NAME: f"step_{i}",
            pipeline.TIMESTAMP: f"2026-01-01 00:00:{i:02d}",
        }
        for t in range(min(table_count, 3 * (i + 1))):
            row[f"table_{t}"] = f"step_{t // 3}"
        rows.append(row)
    return pd.DataFrame(rows)


def test_checkpoint_table_is_small_and_round_trips(tmp_path, monkeypatch):
    store_path = tmp_path / "pipeline.h5"
    store = pd.HDFStore(store_path, mode="w")
    monkeypatch.setattr(pipeline, "get_pipeline_store", lambda: store)

    checkpoints = checkpoint_history()
    assert checkpoints.isna().any().any()

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", pd.errors.PerformanceWarning)
            pipeline.write_checkpoints_df(checkpoints)
            pipeline.write_checkpoints_df(checkpoints.iloc[:5])
        stored = store[pipeline.CHECKPOINT_TABLE_NAME]
    finally:
        store.close()

    pd.testing.assert_frame_equal(stored, checkpoints.iloc[:5].fillna(""))
    assert store_path.stat().st_size < 1_000_000
