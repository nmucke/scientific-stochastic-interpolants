#!/usr/bin/env python3
"""Recover timestamp provenance and held-out climatological context without fitting."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8388608), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--archive", type=Path, default=Path("data/stochastic_navier_stokes/data.npz")
    )
    p.add_argument(
        "--full-resolution",
        type=Path,
        default=Path("data/stochastic_navier_stokes/data_file.pt"),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path(
            "research/ito_maps/results/long_rollout/data_reference_audit.json"
        ),
    )
    a = p.parse_args()
    full, times = torch.load(
        a.full_resolution, weights_only=True, mmap=True, map_location="cpu"
    )
    coarse = np.load(a.archive)["state"]
    assert full.shape[:2] == coarse.shape[:2] and full.shape[-1] == 2 * coarse.shape[-1]
    max_error = 0.0
    for i in range(len(coarse)):
        error = np.max(np.abs(full[i, :, ::2, ::2].numpy() - coarse[i]))
        max_error = max(max_error, float(error))
    assert max_error == 0.0, f"Full/coarse arrays differ: {max_error}"
    t = times.numpy().astype(np.float64)
    dt = np.diff(t)
    enstrophy = 0.5 * np.mean(coarse[180:200].astype(np.float64) ** 2, axis=(-2, -1))
    result = {
        "schema_version": 1,
        "archive": str(a.archive),
        "archive_sha256": sha256(a.archive),
        "full_resolution_archive": str(a.full_resolution),
        "full_resolution_sha256": sha256(a.full_resolution),
        "shape": list(coarse.shape),
        "full_shape": list(full.shape),
        "all_values_exactly_equal_to_stride2_downsampling": True,
        "max_abs_downsampling_difference": max_error,
        "stored_times": t.tolist(),
        "time_spacing_mean": float(dt.mean()),
        "time_spacing_min": float(dt.min()),
        "time_spacing_max": float(dt.max()),
        "time_spacing_interpretation": "Approximately0.4999 in stored time coordinates; viscosity/drag/forcing and absolute origin remain unverified.",
        "available_forecast_leads_for_history5": 95,
        "last_verified_lead": 95,
        "heldout_trajectory_ids": list(range(180, 200)),
        "enstrophy_definition": "0.5 * mean_grid(raw_vorticity**2)",
        "heldout_enstrophy_by_trajectory_and_snapshot": enstrophy.tolist(),
        "heldout_enstrophy_pooled_summary": {
            "mean": float(enstrophy.mean()),
            "q05": float(np.quantile(enstrophy, 0.05)),
            "median": float(np.median(enstrophy)),
            "q95": float(np.quantile(enstrophy, 0.95)),
        },
        "reference_band_interpretation": "Descriptive pooled held-out snapshot distribution, not a future trajectory, predictive interval, or stationary-law proof.",
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    reference_path = Path(
        "research/ito_maps/results/long_horizon/reference_enstrophy.json"
    )
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    summary = result["heldout_enstrophy_pooled_summary"]
    reference_path.write_text(
        json.dumps(
            {
                "definition": result["enstrophy_definition"],
                "trajectory_ids": result["heldout_trajectory_ids"],
                "frame_indices": list(range(100)),
                "pooling": "20 held-out trajectories × 100 stored snapshots; descriptive reference only",
                "n_fields": int(enstrophy.size),
                "enstrophy_mean": summary["mean"],
                "enstrophy_q05": summary["q05"],
                "enstrophy_median": summary["median"],
                "enstrophy_q95": summary["q95"],
                "per_trajectory": enstrophy.tolist(),
                "stored_times": t.tolist(),
                "time_spacing_mean": float(dt.mean()),
                "audit_path": str(a.output),
                "interpretation": result["reference_band_interpretation"],
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                k: result[k]
                for k in [
                    "all_values_exactly_equal_to_stride2_downsampling",
                    "time_spacing_mean",
                    "heldout_enstrophy_pooled_summary",
                ]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
