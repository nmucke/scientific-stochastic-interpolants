#!/usr/bin/env python
"""Audit complete pixel video results against the immutable pre-test protocol."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from scisi.metrics.video import frechet_video_distance

from freeze_video_protocol import (
    FAMILIES,
    RESULTS,
    ROOT,
    SETTINGS,
    read_json,
    require,
    resolve,
    sha256,
    verify_frozen,
)
from report_video_results import QUALITY, validate_comparable


def close(actual, expected, label, rtol=1e-8, atol=1e-8):
    require(
        np.allclose(actual, expected, rtol=rtol, atol=atol), f"Inconsistent {label}"
    )


def verify_evaluation(root, protocol, kind, item, directory, verify_features=True):
    persistence = kind == "persistence"
    expected_model = "persistence" if persistence else FAMILIES[kind]
    require(item["model_class"] == expected_model, f"Wrong model in {kind}")
    for field in (
        "dataset",
        "split",
        "resolution",
        "frame_stride",
        "history_length",
        "forecast_horizon",
        "num_clips",
        "ensemble_size",
        "seed",
        "selected_indices",
        "feedback_clipping",
    ):
        require(item[field] == protocol[field], f"Protocol mismatch {kind}: {field}")
    require(
        item["manifest_sha256"] == protocol["manifest"]["sha256"],
        f"Manifest hash mismatch: {kind}",
    )
    require(
        resolve(root, item["data_manifest"]).resolve()
        == resolve(root, protocol["manifest"]["path"]).resolve(),
        f"Manifest path mismatch: {kind}",
    )
    expected_budget = (
        protocol["persistence_steps"]
        if persistence
        else protocol["models"][kind]["num_steps"]
    )
    require(
        [row["num_steps"] for row in item["fvd"]] == expected_budget,
        f"Wrong step budgets: {kind}",
    )
    require(
        item["i3d_sha256"]
        == protocol["metric_artifacts"]["i3d_google_conversion"]["sha256"],
        f"I3D mismatch: {kind}",
    )
    require(
        item["lpips"] == {"version": "0.1", "network": "alex", "package": "0.1.4"},
        f"LPIPS configuration mismatch: {kind}",
    )
    if persistence:
        require(
            item["checkpoint"] is None and item["checkpoint_sha256"] is None,
            "Persistence must not claim learned weights",
        )
        allowed_config = {
            row["config"]["sha256"] for row in protocol["models"].values()
        }
        require(
            item["config_sha256"] in allowed_config, "Persistence config was not frozen"
        )
    else:
        model = protocol["models"][kind]
        require(
            item["checkpoint_sha256"] == model["checkpoint"]["sha256"]
            and item["config_sha256"] == model["config"]["sha256"],
            f"Model/config changed: {kind}",
        )
        require(
            resolve(root, item["checkpoint"]).resolve()
            == resolve(root, model["checkpoint"]["path"]).resolve(),
            f"Checkpoint path mismatch: {kind}",
        )
    identities = {
        index: case
        for index, case in zip(protocol["selected_indices"], protocol["cases"])
    }
    for row in item["rows"]:
        case = identities[row["case_index"]]
        require(
            row["clip_id"] == case["clip_id"]
            and row["source_video"] == case.get("source_member", case["clip_id"]),
            f"Case identity mismatch: {kind}",
        )
        nfe = 0 if persistence else row["num_steps"]
        require(
            row["nfe"] == nfe and row["cumulative_nfe"] == nfe * row["lead"],
            f"Incorrect network evaluation count: {kind}",
        )
        for key in QUALITY:
            require(key in row, f"Missing metric {key}: {kind}")
            require(
                row[key] is not None or key == "spread_skill_ratio",
                f"Uncomputed metric {key}: {kind}",
            )
        require(
            row["seconds"] >= 0 and row["cumulative_seconds"] >= row["seconds"],
            f"Invalid timing: {kind}",
        )
        if persistence:
            close(row["rms_spread"], 0, "persistence spread")
            close(row["out_of_range_fraction"], 0, "persistence pixel range")
    for steps in expected_budget:
        for case_index in protocol["selected_indices"]:
            rows = sorted(
                (
                    r
                    for r in item["rows"]
                    if r["num_steps"] == steps and r["case_index"] == case_index
                ),
                key=lambda r: r["lead"],
            )
            close(
                [r["cumulative_seconds"] for r in rows],
                np.cumsum([r["seconds"] for r in rows]),
                f"cumulative timing {kind}/{case_index}/{steps}",
            )
    sources = {case.get("source_member", case["clip_id"]) for case in protocol["cases"]}
    for summary in item["summary"]:
        require(
            summary["num_clips"] == protocol["num_clips"]
            and summary["num_source_videos"] == len(sources),
            f"Summary case/source count mismatch: {kind}",
        )
        rows = [
            r
            for r in item["rows"]
            if r["num_steps"] == summary["num_steps"] and r["lead"] == summary["lead"]
        ]
        for key in (*QUALITY, "seconds", "cumulative_seconds", "nfe", "cumulative_nfe"):
            values = [r[key] for r in rows]
            if any(v is None for v in values):
                require(
                    summary[key] is None, f"Undefined summary mismatch {kind}/{key}"
                )
                continue
            expected = np.mean(
                [
                    np.mean([r[key] for r in rows if r["source_video"] == source])
                    for source in sources
                ]
            )
            close(summary[key], expected, f"source-weighted summary {kind}/{key}")
            interval = np.asarray(summary[key + "_ci95"])
            require(
                interval.shape == (2,)
                and np.isfinite(interval).all()
                and interval[0] <= interval[1],
                f"Invalid bootstrap interval {kind}/{key}",
            )
    require(
        read_json(directory / "cases.json") == protocol["cases"],
        f"Saved selected case list differs: {kind}",
    )
    require(
        read_json(directory / "forecasts.json") == item["rows"],
        f"Standalone forecast rows differ: {kind}",
    )
    require(
        len(item["feature_artifacts"]) == len(expected_budget),
        f"Missing FVD feature artifacts: {kind}",
    )
    feature_records = {record["path"]: record for record in item["feature_artifacts"]}
    for fvd in item["fvd"]:
        close(
            fvd["fvd_member_std"],
            np.std(fvd["fvd_member_values"], ddof=1),
            f"FVD member SD {kind}",
        )
        filename = f"features_steps_{fvd['num_steps']}.npz"
        require(
            filename in feature_records, f"Missing feature record {kind}/{filename}"
        )
        path = directory / filename
        require(
            path.exists() and sha256(path) == feature_records[filename]["sha256"],
            f"Feature artifact changed: {path}",
        )
        with np.load(path, allow_pickle=False) as features:
            real, generated = features["real"], features["generated"]
        require(
            real.shape == (protocol["num_clips"], 400)
            and generated.shape
            == (protocol["num_clips"], protocol["ensemble_size"], 400),
            f"Unexpected saved FVD feature shape: {kind}",
        )
        require(
            np.isfinite(real).all() and np.isfinite(generated).all(),
            f"Nonfinite saved FVD feature: {kind}",
        )
        if verify_features:
            close(
                fvd["fvd_pooled"],
                frechet_video_distance(real, generated.reshape(-1, 400)),
                f"Recomputed pooled FVD {kind}",
                rtol=1e-6,
            )
            close(
                fvd["fvd_member_values"],
                [
                    frechet_video_distance(real, generated[:, i])
                    for i in range(protocol["ensemble_size"])
                ],
                f"Recomputed member FVD {kind}",
                rtol=1e-6,
            )
    return {
        "model": kind,
        "forecast_rows": len(item["rows"]),
        "budgets": expected_budget,
        "features_verified": True,
        "fvd_recomputed": verify_features,
    }


def verify_results(dataset, root=ROOT, recompute_fvd=True, write_audit=True):
    root = Path(root).resolve()
    protocol_path = root / RESULTS / f"{dataset}_protocol.json"
    protocol = verify_frozen(root, read_json(protocol_path))
    kinds = [*FAMILIES, "persistence"]
    directories = {kind: root / RESULTS / "test" / dataset / kind for kind in kinds}
    evaluations = {
        kind: read_json(directory / "evaluation.json")
        for kind, directory in directories.items()
    }
    validate_comparable(list(evaluations.values()))
    checks = [
        verify_evaluation(
            root, protocol, kind, evaluations[kind], directories[kind], recompute_fvd
        )
        for kind in kinds
    ]
    # The identical truth cases and frozen detector must give identical real
    # features, independently of model family, budget, or generated members.
    reference = None
    for kind, item in evaluations.items():
        for entry in item["feature_artifacts"]:
            with np.load(
                directories[kind] / entry["path"], allow_pickle=False
            ) as features:
                real = features["real"]
            if reference is None:
                reference = real
            else:
                close(
                    real,
                    reference,
                    f"Common real FVD features {kind}",
                    rtol=1e-6,
                    atol=1e-6,
                )
    audit = {
        "status": "passed",
        "dataset": dataset,
        "audited_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha256(protocol_path),
        "training_stages": len(protocol["training_stages"]),
        "forecast_rows": sum(row["forecast_rows"] for row in checks),
        "checks": checks,
        "frozen_checkpoint_config_manifest_metric_hashes_unchanged": True,
        "selected_video_hashes_unchanged": True,
        "evaluation_artifacts": {
            kind: sha256(path / "evaluation.json") for kind, path in directories.items()
        },
    }
    if write_audit:
        (root / RESULTS / f"{dataset}_audit.json").write_text(
            json.dumps(audit, indent=2, allow_nan=False) + "\n"
        )
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=SETTINGS)
    parser.add_argument(
        "--skip-fvd-recomputation",
        action="store_true",
        help="Still check feature hashes/shapes, but skip recomputing FVD",
    )
    args = parser.parse_args()
    result = verify_results(args.dataset, recompute_fvd=not args.skip_fvd_recomputation)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
