from pathlib import Path

from populationsim.core import tracing, inject, pipeline


def setup_function():
    inject.reinject_decorated_tables()


def teardown_function(func):
    if pipeline.is_open():
        pipeline.close_pipeline()
    inject.clear_cache()
    inject.reinject_decorated_tables()


def test_intermediate_geography():

    example_dir = Path(__file__).parent.parent / "examples" / "example_test"
    configs_dir = example_dir / "configs_intermediate"
    data_dir = example_dir / "data_intermediate"
    output_dir = Path(__file__).parent / "output"

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
        "integerize_final_seed_weights",
        "sub_balancing.geography=TRACT",
        "sub_balancing.geography=TAZ",
        "expand_households",
        "summarize",
        "write_tables",
        "write_synthetic_population",
    ]

    pipeline.run(models=_MODELS, resume_after=None)