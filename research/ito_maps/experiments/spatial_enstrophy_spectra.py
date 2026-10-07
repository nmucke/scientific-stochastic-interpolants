#!/usr/bin/env python3
"""Parseval-normalized spatial enstrophy spectra for saved NS forecast fields.

This deliberately differs from older shell-average / unnormalized metrics:
Z_k sums 0.5*|FFT_forward(omega)|^2 in nearest-integer radial shells. Including
DC and square-grid corner modes, sum_k Z_k = 0.5*mean_grid(omega^2). The frequency
coordinate is cycles per grid-domain; no physical domain length is inferred.
No FFT is needed when rebuilding figures from the compact derived JSON output.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np


@lru_cache(None)
def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def shell_geometry(size):
    """Nearest radial integer bins [k-0.5,k+0.5), retaining every FFT cell."""
    if not isinstance(size, int) or size < 2:
        raise ValueError("Expected integer grid size >= 2")
    frequency = np.fft.fftfreq(size) * size
    radius = np.hypot(frequency[:, None], frequency[None, :])
    shell = np.floor(radius + 0.5).astype(np.int64)
    k = np.arange(int(shell.max()) + 1)
    counts = np.bincount(shell.ravel(), minlength=len(k))
    assert counts.sum() == size * size and counts[0] == 1
    return {
        "k": k,
        "shell_index": shell,
        "mode_counts": counts,
        "shell_edges": np.arange(len(k) + 1) - 0.5,
        "axis_nyquist": size / 2.0,
        "max_radial_frequency": float(radius.max()),
        "grid_size": size,
    }


def radial_enstrophy_shells(vorticity):
    """Return (integer k, shell sums, Parseval audit) for real [...,N,N] fields.

    Every leading dimension is retained. There is no division by shell mode
    counts and no removal of the spatial mean or zero Fourier coefficient.
    """
    raw = np.asarray(vorticity)
    if np.iscomplexobj(raw) or raw.ndim < 2 or raw.shape[-1] != raw.shape[-2]:
        raise ValueError("Expected real square vorticity fields [...,N,N]")
    if not np.isfinite(raw).all():
        raise ValueError("Nonfinite vorticity cannot define a finite spectrum")
    field = raw.astype(np.float64)
    size = field.shape[-1]
    geometry = shell_geometry(size)
    fourier = np.fft.fft2(field, axes=(-2, -1), norm="forward")
    modal = 0.5 * (fourier.real**2 + fourier.imag**2)
    flat = modal.reshape(-1, size * size)
    spectra = np.stack(
        [
            np.bincount(
                geometry["shell_index"].ravel(),
                weights=row,
                minlength=len(geometry["k"]),
            )
            for row in flat
        ]
    )
    spectra = spectra.reshape(*field.shape[:-2], len(geometry["k"]))
    direct = 0.5 * np.mean(field**2, axis=(-2, -1))
    modal_total = modal.sum(axis=(-2, -1))
    shell_total = spectra.sum(axis=-1)
    if not np.allclose(shell_total, direct, rtol=2e-12, atol=2e-12):
        raise AssertionError("Shell spectrum violates Parseval normalization")
    audit = {
        "fft_normalization": "forward",
        "all_modes_included": True,
        "max_abs_modal_minus_direct_enstrophy": float(
            np.max(np.abs(modal_total - direct))
        ),
        "max_abs_shell_sum_minus_direct_enstrophy": float(
            np.max(np.abs(shell_total - direct))
        ),
        "n_fields": int(flat.shape[0]),
        "direct_enstrophy": np.asarray(direct).tolist(),
        "shell_sum": np.asarray(shell_total).tolist(),
        "dc_enstrophy": np.asarray(spectra[..., 0]).tolist(),
    }
    return geometry["k"], spectra, audit


def spectrum_summary(fields):
    """Summarize member/field spectra; bands are pointwise empirical quantiles."""
    k, spectra, audit = radial_enstrophy_shells(fields)
    if spectra.ndim != 2:
        raise ValueError("Expected [member,N,N] fields for an ensemble summary")
    _, spectrum_of_mean, mean_audit = radial_enstrophy_shells(
        np.asarray(fields).mean(0)
    )
    result = {
        "spectrum_members": spectra.tolist(),
        "spectrum_mean": spectra.mean(0).tolist(),
        "spectrum_q05": np.quantile(spectra, 0.05, axis=0).tolist(),
        "spectrum_q95": np.quantile(spectra, 0.95, axis=0).tolist(),
        "spectrum_of_ensemble_mean": spectrum_of_mean.tolist(),
        "enstrophy_members": audit["direct_enstrophy"],
        "dc_enstrophy_members": audit["dc_enstrophy"],
        "parseval_audit": audit,
        "ensemble_mean_parseval_audit": mean_audit,
        "band_definition": "Pointwise 5th/95th quantiles over generated ensemble members; not confidence intervals",
    }
    return k, result


def derive_model_records(directory, leads, trajectory, archive_path):
    directory = Path(directory)
    protocol_path = directory / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    diagnostics_path = directory / "diagnostics.json"
    diagnostics = json.loads(diagnostics_path.read_text())
    case = protocol["trajectories"].index(trajectory)
    records = []
    raw_by_lead = {item["lead"]: item for item in diagnostics["raw_snapshots"]}
    row_by_lead = {
        row["lead"]: row
        for row in diagnostics["rows"]
        if row["trajectory"] == trajectory
    }
    assert sha256(archive_path) == protocol["provenance"]["data_files"][0]["sha256"]
    checkpoint = Path(protocol["provenance"]["checkpoint"])
    assert sha256(checkpoint) == protocol["provenance"]["checkpoint_sha256"]
    assert (
        sha256(checkpoint.with_name("config.yaml"))
        == protocol["provenance"]["config_sha256"]
    )
    std = np.asarray(protocol["inverse_affine"]["std"], dtype=np.float64)[None]
    mean = np.asarray(protocol["inverse_affine"]["mean"], dtype=np.float64)[None]
    for lead in leads:
        if lead not in raw_by_lead:
            raise ValueError(f"{directory}: saved raw lead {lead} is not available yet")
        raw_record = raw_by_lead[lead]
        raw_path = directory / raw_record["path"]
        assert sha256(raw_path) == raw_record["sha256"]
        raw = np.load(raw_path, allow_pickle=False)
        if raw.shape[2] != 1:
            raise ValueError("This NS enstrophy audit expects one vorticity channel")
        physical = (raw[case].astype(np.float64) * std + mean)[:, 0]
        k, derived = spectrum_summary(physical)
        if not np.allclose(
            derived["enstrophy_members"],
            row_by_lead[lead]["enstrophy_members"],
            rtol=1e-11,
            atol=1e-11,
        ):
            raise AssertionError(
                "Raw physical enstrophy differs from existing diagnostic values"
            )
        records.append(
            {
                "name": directory.name,
                "model_class": protocol["model_class"],
                "num_steps": protocol["num_steps"],
                "ensemble_size": protocol["ensemble_size"],
                "trajectory": trajectory,
                "origin": protocol["origin"],
                "lead": lead,
                "nominal_elapsed_stored_time": row_by_lead[lead]["physical_time"],
                "matched_truth_available": lead <= protocol["available_truth_leads"],
                **derived,
                "provenance": {
                    "run_directory": str(directory),
                    "protocol_sha256": sha256(protocol_path),
                    "diagnostics_sha256": sha256(diagnostics_path),
                    "raw_path": str(raw_path),
                    "raw_sha256": raw_record["sha256"],
                    "checkpoint_sha256": protocol["provenance"]["checkpoint_sha256"],
                    "config_sha256": protocol["provenance"]["config_sha256"],
                    "inverse_affine": protocol["inverse_affine"],
                },
            }
        )
    return k, records, protocol


def archive_context(archive, archive_path, batch_size=64):
    ids = list(range(180, 200))
    fields = archive[ids].reshape(-1, *archive.shape[-2:])
    chunks = []
    audits = []
    for start in range(0, len(fields), batch_size):
        _, spectra, audit = radial_enstrophy_shells(fields[start : start + batch_size])
        chunks.append(spectra)
        audits.append(audit)
    spectra = np.concatenate(chunks)
    totals = spectra.sum(-1)
    return {
        "trajectory_ids": ids,
        "frame_indices": list(range(archive.shape[1])),
        "n_fields": len(fields),
        "spectrum_mean": spectra.mean(0).tolist(),
        "spectrum_q05": np.quantile(spectra, 0.05, axis=0).tolist(),
        "spectrum_q95": np.quantile(spectra, 0.95, axis=0).tolist(),
        "enstrophy_mean": float(totals.mean()),
        "dc_enstrophy_mean": float(spectra[:, 0].mean()),
        "interpretation": "Historical held-out archive context, not matched final-step truth, a predictive interval, or an established stationary target",
        "band_definition": "Pointwise 5th/95th quantiles across 2000 temporally correlated held-out fields; not confidence intervals",
        "provenance": {
            "archive_path": str(archive_path),
            "archive_sha256": sha256(archive_path),
        },
        "parseval_audit": {
            "n_fields": len(fields),
            "all_modes_included": True,
            "max_abs_modal_minus_direct_enstrophy": max(
                a["max_abs_modal_minus_direct_enstrophy"] for a in audits
            ),
            "max_abs_shell_sum_minus_direct_enstrophy": max(
                a["max_abs_shell_sum_minus_direct_enstrophy"] for a in audits
            ),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--leads", nargs="+", type=int, default=[1, 4096])
    parser.add_argument("--trajectory", type=int, default=180)
    parser.add_argument(
        "--archive", type=Path, default=Path("data/stochastic_navier_stokes/data.npz")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(
            "Derived spectrum output exists; choose a new filename to preserve provenance"
        )
    archive = np.load(args.archive)["state"]
    models = []
    origins = []
    for directory in args.run_dirs:
        k, records, protocol = derive_model_records(
            directory, args.leads, args.trajectory, args.archive
        )
        models.extend(records)
        origins.append(protocol["origin"])
    if len(set(origins)) != 1:
        raise ValueError("All model spectra must use the same forecast origin")
    geometry = shell_geometry(archive.shape[-1])
    if not np.array_equal(k, geometry["k"]):
        raise ValueError("Model and archive spectrum grids differ")
    truth = None
    if 1 in args.leads:
        frame = origins[0]
        _, spectrum, audit = radial_enstrophy_shells(archive[args.trajectory, frame])
        truth = {
            "lead": 1,
            "trajectory": args.trajectory,
            "archive_frame": frame,
            "spectrum": spectrum.tolist(),
            "enstrophy": audit["direct_enstrophy"],
            "parseval_audit": audit,
            "provenance": {
                "archive_path": str(args.archive),
                "archive_sha256": sha256(args.archive),
            },
        }
    expected = {
        "FlowMapModel",
        "FollmerItoMapModel",
        "FlowMatchingModel",
        "FollmerStochasticInterpolant",
    }
    classes = {m["model_class"] for m in models}
    complete = classes == expected and set(args.leads) == {1, 4096}
    result = {
        "schema_version": 1,
        "definition": "Z_k = sum_{k-0.5 <= sqrt(kx^2+ky^2) < k+0.5} 0.5*abs(fft2(omega,norm=forward))^2",
        "frequency_units": "integer cycles per grid-domain; no physical inverse-length unit assumed",
        "spectrum_units": "enstrophy per integer radial shell in squared raw-vorticity units",
        "normalization": "Full FFT with norm=forward; sum including DC equals0.5*mean_grid(omega^2); no shell-count division, no total-power renormalization",
        "k": geometry["k"].tolist(),
        "mode_counts": geometry["mode_counts"].tolist(),
        "shell_edges": geometry["shell_edges"].tolist(),
        "axis_nyquist": geometry["axis_nyquist"],
        "max_radial_frequency": geometry["max_radial_frequency"],
        "grid_size": geometry["grid_size"],
        "corner_modes_policy": "All square-grid modes retained, including incomplete radial shells beyond axis Nyquist; mark this region in figures",
        "dc_policy": "DC retained in data and Parseval audit, omitted from logarithmic plots; nonpositive plotted powers masked without an epsilon floor",
        "comparison_complete": complete,
        "comparison_scope": (
            "All four model families"
            if complete
            else "Partial comparison: " + ", ".join(sorted(classes))
        ),
        "requested_leads": args.leads,
        "models": models,
        "truth": truth,
        "last_step_matched_truth": None,
        "archive_context": archive_context(archive, args.archive),
        "computation_script_sha256": sha256(Path(__file__)),
        "numpy_version": np.__version__,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "comparison_complete": complete,
                "model_lead_records": len(models),
                "frequency_shells": len(k),
                "context_fields": result["archive_context"]["n_fields"],
                "largest_parseval_residual": max(
                    [
                        m["parseval_audit"]["max_abs_shell_sum_minus_direct_enstrophy"]
                        for m in models
                    ]
                    + [
                        result["archive_context"]["parseval_audit"][
                            "max_abs_shell_sum_minus_direct_enstrophy"
                        ]
                    ]
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
