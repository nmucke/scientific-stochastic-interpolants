"""Options the one-rung, long-rollout NS tree relies on."""

import csv

from common.latex_tables import _variant_rank
from compute_ns_reference_metrics import REF_METHOD, collect
import figure_common


def test_shared_cadences_sort_numerically() -> None:
    variants = ["jacfree", "shared_jac10", "shared_jac5", "shared_jac1"]
    assert sorted(variants, key=_variant_rank) == [
        "shared_jac1", "shared_jac5", "shared_jac10", "jacfree",
    ]


def test_step_curves_keep_classical_rows_only_when_asked(tmp_path, monkeypatch) -> None:
    agg = tmp_path / "navier_stokes" / "aggregated"
    agg.mkdir(parents=True)
    with open(agg / "per_step.csv", "w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["method", "variant", "scenario", "metric", "step", "value", "M"]
        )
        writer.writeheader()
        for method, M in (("EnKF", 50), ("SDA", 100)):
            writer.writerow(dict(method=method, variant="", scenario="s", metric="rmse",
                                 step=0, value=1.0, M=M))
    monkeypatch.setattr(figure_common, "RESULTS", tmp_path)

    default = figure_common.load_metric_vs_step("navier_stokes", "rmse", "s", M=100)
    kept = figure_common.load_metric_vs_step(
        "navier_stokes", "rmse", "s", M=100, include_m_independent=True
    )
    assert set(default) == {("SDA", None)}
    assert set(kept) == {("SDA", None), ("EnKF", None)}


def test_reference_rows_are_renamed_and_carry_their_trajectory(tmp_path) -> None:
    (tmp_path / "ref_sparse_5__traj11.csv").write_text(
        "case,method,scenario,metric,value,std,E,M,seed,NFE,seconds,variant\n"
        "navier_stokes,EnKF,sparse 5%,rmse,0.1,0.0,1000,50,-1,nan,100,\n"
        "navier_stokes,EnKF,sparse 5%,kl_points,nan,0.0,1000,50,-1,nan,100,\n"
    )
    rows = collect(tmp_path)
    assert [r["metric"] for r in rows] == ["rmse"]
    assert rows[0]["method"] == REF_METHOD and rows[0]["test_index"] == "11"
