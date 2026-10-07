#!/usr/bin/env python3
"""Audit completed finite NS rollouts independently from saved normalized states.

CPU only. Recompute physical metrics with NumPy, verify artifact hashes, input
and truth alignment, all scalar-row identities, and restart-state consistency.
The audit verifies RNG payload structure; it does not replay GPU RNG streams.
Existing intermediate audit artifacts are never overwritten by default.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
import torch


@lru_cache(None)
def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def close(actual, expected, label):
    if not np.allclose(actual, expected, rtol=1e-11, atol=1e-11):
        raise AssertionError(f"{label}: {actual} != {expected}")


def check_rng_payload(restart):
    # These local generators do not change global RNG or touch a CUDA device.
    state = restart["torch_rng_state"]
    generator = torch.Generator(device="cpu")
    generator.set_state(state)
    assert torch.equal(generator.get_state(), state)
    numpy_rng = np.random.RandomState()
    numpy_rng.set_state(restart["numpy_rng_state"])
    python_rng = random.Random()
    python_rng.setstate(restart["python_rng_state"])
    cuda_states = restart["cuda_rng_states"]
    if cuda_states is not None:
        assert len(cuda_states) >= 1
        for value in cuda_states:
            assert value.device.type == "cpu" and value.dtype == torch.uint8
            assert value.ndim == 1 and value.numel() > 0
    return {
        "cpu_torch_numpy_python_states_restorable": True,
        "cuda_rng_payloads_present": cuda_states is not None,
        "cuda_rng_payloads_structurally_valid": True,
        "gpu_stream_replay_performed": False,
    }


def audit_run(directory, archive, archive_path):
    directory = Path(directory)
    protocol_path = directory / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    summary_path = directory / "diagnostics.json"
    summary = json.loads(summary_path.read_text())
    assert summary["status"] == "complete"
    horizon = summary["completed_leads"]
    assert horizon == summary["requested_horizon"] == protocol["horizon"]
    trajectories = protocol["trajectories"]
    ensemble_size = protocol["ensemble_size"]
    origin, history_length = protocol["origin"], protocol["history_length"]
    ncases = len(trajectories)
    truth_leads = archive.shape[1] - origin
    assert protocol["truth_leads"] == protocol["available_truth_leads"] == truth_leads
    assert summary["available_truth_leads"] == truth_leads
    histories = np.stack(
        [
            archive[t, origin - history_length : origin].transpose(1, 2, 0)[None]
            for t in trajectories
        ]
    )
    truths = np.stack(
        [archive[t, origin:].transpose(1, 2, 0)[None] for t in trajectories]
    )
    assert array_hash(histories) == protocol["initial_history_sha256"]
    assert array_hash(truths) == protocol["truth_sha256"]
    provenance = protocol["provenance"]
    assert sha256(archive_path) == provenance["data_files"][0]["sha256"]
    assert sha256(provenance["checkpoint"]) == provenance["checkpoint_sha256"]
    config_path = Path(provenance["checkpoint"]).with_name("config.yaml")
    assert sha256(config_path) == provenance["config_sha256"]
    time_audit_path = Path(provenance["timestamp_audit"]["path"])
    assert sha256(time_audit_path) == provenance["timestamp_audit"]["sha256"]
    time_audit = json.loads(time_audit_path.read_text())
    assert time_audit["archive_sha256"] == sha256(archive_path)
    assert protocol["snapshot_dt"] == time_audit["time_spacing_mean"]
    assert (
        protocol["time_origin_stored_timestamp"]
        == time_audit["stored_times"][origin - 1]
    )
    assert protocol["feedback_clipping"] is False
    for index in range(ncases):
        close(
            protocol["initial_enstrophy"][index],
            0.5 * np.mean(histories[index, ..., -1].astype(np.float64) ** 2),
            "initial enstrophy",
        )
    std = np.asarray(protocol["inverse_affine"]["std"], dtype=np.float64)[None]
    mean = np.asarray(protocol["inverse_affine"]["mean"], dtype=np.float64)[None]
    assert np.all(std > 0) and np.isfinite(std).all() and np.isfinite(mean).all()
    rows = {(r["trajectory"], r["lead"]): r for r in summary["rows"]}
    assert len(rows) == len(summary["rows"]) == horizon * ncases
    assert set(rows) == {
        (t, lead) for t in trajectories for lead in range(1, horizon + 1)
    }
    journal = [
        json.loads(line)
        for line in (directory / "diagnostics.jsonl").read_text().splitlines()
    ]
    assert journal == summary["rows"]
    for trajectory in trajectories:
        cumulative_seconds = 0.0
        cumulative_nfe = 0
        for lead in range(1, horizon + 1):
            row = rows[trajectory, lead]
            assert row["origin"] == origin and row["ensemble_size"] == ensemble_size
            assert (
                row["num_steps"] == protocol["num_steps"]
                and row["model_class"] == protocol["model_class"]
            )
            assert row["finite_member_fraction"] == 1.0
            z = np.asarray(row["enstrophy_members"], dtype=np.float64)
            assert z.shape == (ensemble_size,) and np.isfinite(z).all()
            close(
                row["enstrophy_member_mean"], z.mean(), "stored member-enstrophy mean"
            )
            close(
                row["enstrophy_member_q05"],
                np.quantile(z, 0.05),
                "stored member-enstrophy q05",
            )
            close(
                row["enstrophy_member_q95"],
                np.quantile(z, 0.95),
                "stored member-enstrophy q95",
            )
            close(
                row["enstrophy_member_mean"],
                row["enstrophy_of_mean"]
                + 0.5 * (ensemble_size - 1) / ensemble_size * row["rms_spread"] ** 2,
                "all-lead enstrophy/variance identity",
            )
            close(
                row["physical_time"],
                lead * protocol["snapshot_dt"],
                "elapsed nominal stored time",
            )
            if lead > truth_leads:
                assert all(
                    row[k] is None
                    for k in ["truth_enstrophy", "rmse", "spread_skill_ratio"]
                )
            else:
                target = archive[trajectory, origin + lead - 1].astype(np.float64)
                close(
                    row["truth_enstrophy"],
                    0.5 * np.mean(target**2),
                    "truth frame index",
                )
                close(
                    row["spread_skill_ratio"],
                    math.sqrt((ensemble_size + 1) / ensemble_size)
                    * row["rms_spread"]
                    / row["rmse"],
                    "spread skill correction",
                )
            assert row["seconds"] >= 0 and row["nfe"] == protocol["num_steps"]
            cumulative_seconds += row["seconds"]
            cumulative_nfe += row["nfe"]
            close(row["cumulative_seconds"], cumulative_seconds, "cumulative timing")
            assert row["cumulative_nfe"] == cumulative_nfe
        close(summary["sampling_seconds"], cumulative_seconds, "summary timing")
        assert summary["total_nfe"] == cumulative_nfe
    expected_leads = {
        lead
        for lead in range(1, horizon + 1)
        if lead <= protocol["raw_dense"]
        or lead % protocol["raw_every"] == 0
        or lead == horizon
    }
    saved = summary["raw_snapshots"]
    assert (
        len(saved) == len(expected_leads)
        and {s["lead"] for s in saved} == expected_leads
    )
    maximum = {}
    for item in saved:
        lead = item["lead"]
        path = directory / item["path"]
        assert sha256(path) == item["sha256"]
        normalized = np.load(path, allow_pickle=False)
        assert list(normalized.shape) == item["shape"]
        assert str(normalized.dtype) == item["dtype"] == "float32"
        assert normalized.shape == (ncases, ensemble_size, 1, *archive.shape[-2:])
        assert np.isfinite(normalized).all()
        for case, trajectory in enumerate(trajectories):
            values = normalized[case].astype(np.float64) * std + mean
            average = values.mean(0)
            z = 0.5 * np.mean(values**2, axis=(1, 2, 3))
            spread = np.sqrt(np.var(values, axis=0, ddof=1).mean())
            row = rows[trajectory, lead]
            close(z, row["enstrophy_members"], "raw member enstrophy")
            expected = {
                "enstrophy_member_mean": z.mean(),
                "enstrophy_member_q05": np.quantile(z, 0.05),
                "enstrophy_member_q95": np.quantile(z, 0.95),
                "enstrophy_of_mean": 0.5 * np.mean(average**2),
                "rms_spread": spread,
                "max_abs_vorticity": np.abs(values).max(),
                "mean_vorticity": values.mean(),
            }
            if lead <= truth_leads:
                target = archive[trajectory, origin + lead - 1][None].astype(np.float64)
                expected["truth_enstrophy"] = 0.5 * np.mean(target**2)
                expected["rmse"] = np.sqrt(np.mean((average - target) ** 2))
                expected["spread_skill_ratio"] = (
                    math.sqrt((ensemble_size + 1) / ensemble_size)
                    * spread
                    / expected["rmse"]
                )
            for key, value in expected.items():
                maximum[key] = max(maximum.get(key, 0.0), abs(float(value) - row[key]))
                close(value, row[key], f"raw metric lead{lead} {key}")
    restart_audits = []
    for path in sorted(directory.glob("restart*.pt")):
        restart = torch.load(path, map_location="cpu", weights_only=False)
        assert restart["protocol_sha256"] == sha256(protocol_path)
        lead = restart["completed_leads"]
        assert 0 < lead <= horizon
        history = restart["history"]
        assert history.device.type == "cpu" and history.dtype == torch.float32
        assert tuple(history.shape) == (
            ncases * ensemble_size,
            1,
            *archive.shape[-2:],
            history_length,
        )
        assert torch.isfinite(history).all()
        state_path = directory / "raw_states" / f"lead_{lead:06d}.npy"
        assert np.array_equal(
            history[..., -1].numpy(),
            np.load(state_path).reshape(ncases * ensemble_size, 1, *archive.shape[-2:]),
        )
        assert restart["summary"]["rows"] == [
            r for r in summary["rows"] if r["lead"] <= lead
        ]
        assert restart["summary"]["total_nfe"] == lead * protocol["num_steps"]
        if path.name == "restart.pt":
            assert restart["completed_leads"] == horizon
            assert restart["summary"] == summary
        restart_audits.append(
            {
                "path": str(path),
                "sha256": sha256(path),
                "completed_leads": lead,
                "history_endpoint_matches_saved_raw_state": True,
                **check_rng_payload(restart),
            }
        )
    assert any(Path(r["path"]).name == "restart.pt" for r in restart_audits)
    return {
        "status": "passed",
        "run_directory": str(directory),
        "model_class": protocol["model_class"],
        "completed_leads": horizon,
        "ensemble_size": ensemble_size,
        "trajectories": trajectories,
        "protocol_sha256": sha256(protocol_path),
        "diagnostics_sha256": sha256(summary_path),
        "all_scalar_rows_checked": len(rows),
        "saved_raw_files_checked": len(saved),
        "saved_raw_leads_checked": sorted(expected_leads),
        "max_absolute_metric_differences": maximum,
        "truth_last_lead": truth_leads,
        "all_raw_snapshot_hashes_match": True,
        "truth_metrics_null_past_archive": True,
        "timestamp_audit_hash_unchanged": True,
        "archive_and_checkpoint_hashes_match": True,
        "diagnostic_journal_matches_summary": True,
        "restart_audits": restart_audits,
        "scope_limit": "No GPU RNG replay or model inference; all saved raw fields independently rescored, remaining leads checked through scalar identities and artifact consistency.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--archive", type=Path, default=Path("data/stochastic_navier_stokes/data.npz")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(
            "Audit output exists; choose a distinct artifact to preserve history"
        )
    torch.set_num_threads(2)
    archive = np.load(args.archive)["state"]
    results = [
        audit_run(directory, archive, args.archive) for directory in args.run_dirs
    ]
    result = {
        "schema_version": 2,
        "status": "passed",
        "cpu_only": True,
        "runs": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            [
                {
                    k: r[k]
                    for k in [
                        "model_class",
                        "completed_leads",
                        "all_scalar_rows_checked",
                        "saved_raw_files_checked",
                        "max_absolute_metric_differences",
                    ]
                }
                for r in results
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
