import numpy as np
import pandas as pd
from pathlib import Path

from populationsim.core import tracing, inject, pipeline


def teardown_function(func):
    inject.clear_cache()
    inject.reinject_decorated_tables()


def test_weighting():

    example_dir = Path(__file__).parent.parent / "examples" / "example_survey_weighting"
    configs_dir = example_dir / "configs"
    data_dir = example_dir / "data"
    output_dir = Path(__file__).parent / "output"
    expect_dir = Path(__file__).parent / "expected"

    inject.add_injectable("data_dir", data_dir)
    inject.add_injectable("configs_dir", configs_dir)
    inject.add_injectable("output_dir", output_dir)

    inject.clear_cache()

    tracing.config_logger()

    _MODELS = [
        "input_pre_processor",
        "setup_data_structures",
        "initial_seed_balancing",
        "meta_control_factoring",
        "final_seed_balancing",
        "summarize",
        # "write_tables",
    ]

    pipeline.run(models=_MODELS, resume_after=None)

    summary_hh_weights = pipeline.get_table("summary_hh_weights")
    total_summary_hh_weights = summary_hh_weights["SUBREGCluster_balanced_weight"].sum()

    seed_households = pipeline.get_table("households")
    total_seed_households_weights = seed_households["HHweight"].sum()

    # Should be pretty close but not exact.
    assert abs(total_summary_hh_weights - total_seed_households_weights) < 1

    expected_wts = pd.read_parquet(expect_dir / "weights.parquet")

    np.allclose(
        summary_hh_weights["SUBREGCluster_balanced_weight"].values,
        expected_wts["SUBREGCluster_balanced_weight"].values,
    )

    # tables will no longer be available after pipeline is closed
    pipeline.close_pipeline()

    inject.clear_cache()
