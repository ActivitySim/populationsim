from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from populationsim.core import config, inject, pipeline, tracing

# examples/example_test/configs_intermediate declares
#     geographies: [REGION, DISTRICT, PUMA, TRACT, TAZ]
#     seed_geography: PUMA
# so DISTRICT sits between the meta geography (REGION) and the seed geography
# (PUMA). Its controls.csv puts persons_occ_1 at DISTRICT and persons_occ_2 and
# persons_occ_3 at REGION, which is what lets this example tell the two apart.
SEED_GEOGRAPHY = "PUMA"
INTERMEDIATE_GEOGRAPHY = "DISTRICT"
INTERMEDIATE_CONTROL = "persons_occ_1"
META_CONTROLS = ("persons_occ_2", "persons_occ_3")

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


def setup_function():
    # Other test modules pin `settings` as a plain injectable via
    # config.override_setting; restore the decorated one so this example's own
    # settings -- in particular its five-level `geographies` list -- are read.
    inject.reinject_decorated_tables()


def teardown_function(func):
    if pipeline.is_open():
        pipeline.close_pipeline()
    inject.clear_cache()
    inject.reinject_decorated_tables()


@pytest.mark.parametrize("integerize", [False, True], ids=["fractional", "integerized"])
@pytest.mark.parametrize("intermediate_levels", [1, 2], ids=["one-level", "two-levels"])
def test_intermediate_geography(tmp_path, integerize, intermediate_levels):

    example_dir = Path(__file__).parent.parent / "examples" / "example_test"

    # configs_dir and data_dir both cascade: the first directory holding a
    # given file wins. The *_intermediate directories therefore carry only what
    # this example actually changes -- its settings, controls, and the crosswalk
    # with the extra geography levels -- and everything shared (seed households
    # and persons, TAZ and TRACT controls, logging) resolves from the base
    # example. Copying those would leave this example silently testing stale
    # inputs whenever the originals changed.
    inject.add_injectable(
        "data_dir", [example_dir / "data_intermediate", example_dir / "data"]
    )
    inject.add_injectable(
        "configs_dir", [example_dir / "configs_intermediate", example_dir / "configs"]
    )
    inject.add_injectable("output_dir", tmp_path)

    inject.clear_cache()

    config.override_setting("NO_INTEGERIZATION_EVER", not integerize)
    intermediate_targets = {INTERMEDIATE_GEOGRAPHY: INTERMEDIATE_CONTROL}
    meta_controls = META_CONTROLS
    geographies = ["REGION", INTERMEDIATE_GEOGRAPHY, SEED_GEOGRAPHY, "TRACT", "TAZ"]

    if intermediate_levels == 2:
        # Insert AREA above DISTRICT, with its own target. Both intermediate
        # targets must be excluded; filtering only the level next to seed fails.
        inputs_dir = tmp_path / "inputs"
        inputs_dir.mkdir()
        crosswalk = pd.read_csv(
            example_dir / "data_intermediate" / "geo_cross_walk.csv", comment="#"
        )
        crosswalk["AREA"] = crosswalk["DISTRICT"].map({1: 1, 2: 1, 3: 2})
        crosswalk.to_csv(inputs_dir / "geo_cross_walk.csv", index=False)
        pd.DataFrame({"AREA": [1, 2], "OCCP2": [180, 120]}).to_csv(
            inputs_dir / "area_controls.csv", index=False
        )
        controls = pd.read_csv(example_dir / "configs_intermediate" / "controls.csv")
        controls.loc[controls.target == "persons_occ_2", "geography"] = "AREA"
        controls.to_csv(inputs_dir / "controls.csv", index=False)
        geographies.insert(1, "AREA")
        config.override_setting("geographies", geographies)
        settings = inject.get_injectable("settings")
        config.override_setting(
            "input_table_list",
            settings["input_table_list"]
            + [{"tablename": "AREA_control_data", "filename": "area_controls.csv"}],
        )
        for directory in ("data_dir", "configs_dir"):
            inject.add_injectable(
                directory, [inputs_dir] + inject.get_injectable(directory)
            )
        intermediate_targets["AREA"] = "persons_occ_2"
        meta_controls = ("persons_occ_3",)

    tracing.config_logger()

    # Completing at all is the primary regression. Without the control_spec
    # filtering in meta_control_factoring / final_seed_balancing /
    # integerize_final_seed_weights, the seed-level controls are indexed with
    # the intermediate geography's target and this raises
    # KeyError: "['persons_occ_1'] not in index".
    pipeline.run(models=_MODELS, resume_after=None)

    crosswalk = pipeline.get_table("crosswalk")
    assert list(crosswalk.columns) == geographies

    # The behaviour under test: controls belonging to a geography between the
    # meta and seed geographies are kept out of the seed-level control table,
    # while the meta geography's controls are still factored into it.
    seed_controls = pipeline.get_table(f"{SEED_GEOGRAPHY}_controls")
    for target in intermediate_targets.values():
        assert target not in seed_controls.columns
    for target in meta_controls:
        assert target in seed_controls.columns

    # Intermediate targets remain available for reporting. This feature does
    # not distribute them to seed zones or enforce them during balancing.
    for geography, target in intermediate_targets.items():
        intermediate_controls = pipeline.get_table(f"{geography}_controls")
        assert target in intermediate_controls.columns
        for sub_geography in ("TRACT", "TAZ"):
            summary = pipeline.get_table(f"summary_{sub_geography}_{geography}")
            assert set(summary["id"]) == set(crosswalk[geography])
            np.testing.assert_allclose(
                summary[f"{target}_control"],
                intermediate_controls.loc[summary["id"], target],
            )
            weights = pipeline.get_table(f"{sub_geography}_weights")
            incidence = pipeline.get_table("incidence_table")
            weight_col = "integer_weight" if integerize else "balanced_weight"
            expected_results = (
                (weights[weight_col] * weights["hh_id"].map(incidence[target]))
                .groupby(weights[geography])
                .sum()
            )
            np.testing.assert_allclose(
                summary[f"{target}_result"], expected_results.loc[summary["id"]]
            )

    # Sub-geography balancing ran and preserved the seed-level household total.
    seed_total = pipeline.get_table(f"{SEED_GEOGRAPHY}_weights")[
        "balanced_weight"
    ].sum()
    assert seed_total > 0
    for geography in ("TRACT", "TAZ"):
        weights = pipeline.get_table(f"{geography}_weights")["balanced_weight"]
        assert (weights >= 0).all()
        assert weights.sum() == pytest.approx(seed_total, rel=1e-6)

    expanded = pipeline.get_table("expanded_household_ids")
    if integerize:
        seed_weights = pipeline.get_table(f"{SEED_GEOGRAPHY}_weights")
        assert (seed_weights.integer_weight >= 0).all()
        assert (seed_weights.integer_weight % 1 == 0).all()
        seed_totals = seed_weights.groupby(SEED_GEOGRAPHY).integer_weight.sum()
        pd.testing.assert_series_equal(
            seed_totals, seed_controls.num_hh, check_dtype=False, check_names=False
        )
        taz_weights = pipeline.get_table("TAZ_weights")
        expected_counts = taz_weights.groupby("TAZ").integer_weight.sum()
        pd.testing.assert_series_equal(
            expanded.groupby("TAZ").size(),
            expected_counts,
            check_dtype=False,
            check_names=False,
        )
        assert len(expanded) == expected_counts.sum()
        assert (tmp_path / "synthetic_households.csv").exists()
        assert (tmp_path / "synthetic_persons.csv").exists()
    else:
        assert expanded.empty
