"""
Generate the reference expected parquet files for regression tests.

MILP solver results vary across platforms (Windows/Linux/macOS) and even OS
versions due to differences in floating-point behavior and compiled CBC/GLPK
binaries. Rather than committing a bit-exact golden per platform, the tests
compare the zone-level household distribution against a single reference set
with a tolerance (see ``tests.assert_expanded_close``). Generate that reference
set on the CI platform (Linux) and commit it.

Usage:
    python tests/generate_expected.py

Output files are written to tests/expected/ with the naming convention:
    <name>.parquet
"""

from pathlib import Path

from populationsim.core import config, tracing, inject, pipeline

TESTS_DIR = Path(__file__).parent
EXPECTED_DIR = TESTS_DIR / "expected"
EXAMPLE_DIR = TESTS_DIR.parent / "examples"


def setup_injectables(configs_dirs, data_dir):
    inject.reinject_decorated_tables()
    inject.add_injectable("configs_dir", configs_dirs)
    inject.add_injectable("output_dir", TESTS_DIR / "output")
    inject.add_injectable("data_dir", data_dir)
    inject.clear_cache()
    tracing.config_logger()


def save_expected(df, name):
    EXPECTED_DIR.mkdir(parents=True, exist_ok=True)
    path = EXPECTED_DIR / f"{name}.parquet"
    df.to_parquet(path)
    print(f"  {name}.parquet: {df.shape}")


def generate_test_steps():
    """Generate expected files for test_steps.py (full run + repop replace + repop append)."""
    print("=== test_steps ===")

    example_configs_dir = EXAMPLE_DIR / "example_test" / "configs"
    configs_dirs = [TESTS_DIR / "configs", example_configs_dir]
    data_dir = EXAMPLE_DIR / "example_test" / "data"

    models_run1 = [
        "input_pre_processor",
        "setup_data_structures",
        "initial_seed_balancing",
        "meta_control_factoring",
        "final_seed_balancing",
        "integerize_final_seed_weights",
        "sub_balancing.geography=TRACT",
        "sub_balancing.geography=TAZ",
        "expand_households",
        "summarize",
        "write_tables",
        "write_synthetic_population",
    ]

    models_repop_replace = [
        "input_pre_processor.table_list=repop_input_table_list;repop",
        "repop_setup_data_structures",
        "initial_seed_balancing.final=true;repop",
        "integerize_final_seed_weights.repop",
        "repop_balancing",
        "expand_households.repop;replace",
        "write_synthetic_population.repop",
        "write_tables.repop",
    ]

    models_repop_append = [
        "input_pre_processor.table_list=repop_input_table_list;repop",
        "repop_setup_data_structures",
        "initial_seed_balancing.final=true;repop",
        "integerize_final_seed_weights.repop",
        "repop_balancing",
        "expand_households.repop;append",
        "write_synthetic_population.repop",
    ]

    # --- Run 1: full run ---
    setup_injectables(configs_dirs, data_dir)
    pipeline.run(models=models_run1, resume_after=None)
    save_expected(pipeline.get_table("expanded_household_ids"), "expanded")
    pipeline.close_pipeline()
    inject.clear_cache()

    # --- Run 2: repop replace (resumes from run1's summarize checkpoint) ---
    setup_injectables(configs_dirs, data_dir)
    pipeline.run(models=models_repop_replace, resume_after="summarize")
    save_expected(pipeline.get_table("expanded_household_ids"), "expanded_repop_replace")
    pipeline.close_pipeline()
    inject.clear_cache()

    # --- Run 3: repop append (needs fresh run1, then append) ---
    setup_injectables(configs_dirs, data_dir)
    pipeline.run(models=models_run1, resume_after=None)
    pipeline.close_pipeline()
    inject.clear_cache()

    setup_injectables(configs_dirs, data_dir)
    pipeline.run(models=models_repop_append, resume_after="summarize")
    save_expected(pipeline.get_table("expanded_household_ids"), "expanded_repop_append")
    pipeline.close_pipeline()
    inject.clear_cache()


def generate_test_steps_mp():
    """Generate expected files for test_steps_mp.py."""
    from populationsim.core import mp_tasks

    print("=== test_steps_mp ===")

    example_configs_dir = EXAMPLE_DIR / "example_test" / "configs"
    mp_configs_dir = EXAMPLE_DIR / "example_test" / "configs_mp"
    configs_dirs = [mp_configs_dir, TESTS_DIR / "configs", example_configs_dir]
    data_dir = EXAMPLE_DIR / "example_test" / "data"

    setup_injectables(configs_dirs, data_dir)

    injectables = ["data_dir", "configs_dir", "output_dir"]
    injectables = {k: inject.get_injectable(k) for k in injectables}
    mp_tasks.run_multiprocess(injectables)

    pipeline.open_pipeline("_")
    save_expected(pipeline.get_table("expanded_household_ids"), "expanded_mp")
    pipeline.close_pipeline()
    inject.clear_cache()
    inject.reinject_decorated_tables()


def generate_test_flex():
    """Generate expected files for test_flex.py (ortools and cvxpy)."""
    print("=== test_flex ===")

    configs_dir = EXAMPLE_DIR / "example_test" / "configs_flex"
    data_dir = EXAMPLE_DIR / "example_test" / "data_flex"

    models = [
        "input_pre_processor",
        "setup_data_structures",
        "initial_seed_balancing",
        "meta_control_factoring",
        "final_seed_balancing",
        "integerize_final_seed_weights",
        "sub_balancing.geography=DISTRICT",
        "sub_balancing.geography=TRACT",
        "sub_balancing.geography=TAZ",
        "expand_households",
        "summarize",
        "write_tables",
    ]

    for solver_name, use_cvxpy in [("ortools", False), ("cvxpy", True)]:
        setup_injectables(configs_dir, data_dir)
        config.override_setting("cleanup_pipeline_after_run", True)
        config.override_setting("NO_INTEGERIZATION_EVER", False)
        config.override_setting("USE_CVXPY", use_cvxpy)

        pipeline.run(models=models, resume_after=None)
        save_expected(pipeline.get_table("expanded_household_ids"), f"expanded_{solver_name}")
        pipeline.close_pipeline()
        inject.clear_cache()
        inject.reinject_decorated_tables()


def generate_test_weighting():
    """Generate expected files for test_weighting.py."""
    print("=== test_weighting ===")

    example_dir = EXAMPLE_DIR / "example_survey_weighting"
    configs_dir = example_dir / "configs"
    data_dir = example_dir / "data"

    setup_injectables(configs_dir, data_dir)

    models = [
        "input_pre_processor",
        "setup_data_structures",
        "initial_seed_balancing",
        "meta_control_factoring",
        "final_seed_balancing",
        "summarize",
    ]

    pipeline.run(models=models, resume_after=None)
    summary = pipeline.get_table("summary_hh_weights")
    save_expected(summary, "weights")
    pipeline.close_pipeline()
    inject.clear_cache()


if __name__ == "__main__":
    print(f"Output directory: {EXPECTED_DIR}")
    EXPECTED_DIR.mkdir(parents=True, exist_ok=True)
    print()

    generate_test_steps()
    generate_test_flex()
    generate_test_steps_mp()
    generate_test_weighting()

    print("\nDone! All expected files generated.")
