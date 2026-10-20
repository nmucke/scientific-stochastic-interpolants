#!/usr/bin/env python
"""Resumable autonomous Navier–Stokes forecasts with sparse raw-state snapshots.

Diagnostic values are streamed at every physical forecast step. Truth-dependent
scores stop at the archive boundary; autonomous statistics continue. Nothing is
clipped or replaced. Finite excursions are recorded, never used as stop rules.
"""

import argparse
import hashlib
import json
import math
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from scisi.bin.main_map_model_train import load_teacher, seed_everything
from scisi.models.follmer_stochastic_interpolant import FollmerStochasticInterpolant


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_hash(tensor):
    value = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(value.tobytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def atomic_torch_save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def inverse_state(preprocesser, states):
    return (
        preprocesser.inverse_transform(target=states.double(), is_batch=True)["target"]
        if preprocesser is not None
        else states.double()
    )


def state_diagnostics(ensemble, truth=None):
    """Physical/archive-unit statistics for [E,C,H,W], without clipping.

    Total enstrophy is .5 times the spatial mean of squared vorticity. Its
    ensemble mean differs from the enstrophy of the ensemble-mean field.
    Undefined/failed-member quantities are represented as JSON null, not NaN.
    """
    ensemble = ensemble.double()
    finite_members = torch.isfinite(ensemble).flatten(1).all(1)
    member_z = [
        float(0.5 * member.square().mean()) if finite else None
        for member, finite in zip(ensemble, finite_members)
    ]
    row = {
        "finite_member_fraction": float(finite_members.double().mean()),
        "enstrophy_members": member_z,
        "enstrophy_member_mean": None,
        "enstrophy_member_q05": None,
        "enstrophy_member_q95": None,
        "enstrophy_of_mean": None,
        "rms_spread": None,
        "rmse": None,
        "spread_skill_ratio": None,
        "truth_enstrophy": None,
        "max_abs_vorticity": None,
        "mean_vorticity": None,
    }
    if truth is not None:
        truth = truth.double()
        if not torch.isfinite(truth).all():
            raise ValueError("Reference truth contains nonfinite values")
        row["truth_enstrophy"] = float(0.5 * truth.square().mean())
    if not finite_members.all():
        return row
    z = torch.tensor(member_z, dtype=torch.float64)
    mean = ensemble.mean(0)
    spread = ensemble.var(0, correction=1).mean().sqrt()
    row.update(
        enstrophy_member_mean=float(z.mean()),
        enstrophy_member_q05=float(torch.quantile(z, 0.05)),
        enstrophy_member_q95=float(torch.quantile(z, 0.95)),
        enstrophy_of_mean=float(0.5 * mean.square().mean()),
        rms_spread=float(spread),
        max_abs_vorticity=float(ensemble.abs().max()),
        mean_vorticity=float(ensemble.mean()),
    )
    if truth is not None:
        error = (mean - truth).square().mean().sqrt()
        row["rmse"] = float(error)
        if error > 0:
            row["spread_skill_ratio"] = float(
                math.sqrt((len(ensemble) + 1) / len(ensemble)) * spread / error
            )
    return row


def save_raw(directory, lead, generated):
    """Save exact normalized model outputs; an orphan from a crash must agree."""
    path = directory / "raw_states" / f"lead_{lead:06d}.npy"
    path.parent.mkdir(exist_ok=True)
    values = generated.detach().cpu().numpy()
    if path.exists():
        previous = np.load(path, allow_pickle=False)
        if previous.shape != values.shape or not np.array_equal(
            previous, values, equal_nan=True
        ):
            raise RuntimeError(
                f"Existing uncommitted raw state disagrees with resumed sampling: {path}"
            )
    else:
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("wb") as stream:
            np.save(stream, values, allow_pickle=False)
        os.replace(temporary, path)
    return {
        "lead": lead,
        "path": str(path.relative_to(directory)),
        "sha256": sha256(path),
        "shape": list(values.shape),
        "dtype": str(values.dtype),
    }


def run_rollout(
    model,
    histories,
    truths,
    *,
    preprocesser=None,
    output_dir,
    trajectories=(180,),
    origin=5,
    num_steps=8,
    ensemble_size=8,
    horizon=4096,
    seed=20261007,
    device="cuda",
    checkpoint_every=16,
    milestones=(1024,),
    raw_every=16,
    raw_dense=100,
    resume=False,
    stop_after_lead=None,
    snapshot_dt=None,
    provenance=None,
    concurrent_workload="unspecified; timings are diagnostic only",
):
    """Histories [B,C,H,W,N] and future truths [B,C,H,W,T] in archive units.

    Raw output files contain [B,E,C,H,W] normalized float32 predictions. This
    intentionally preserves only dense initial states, periodic states, and
    final/failure states, while every lead has complete scalar diagnostics.
    Resume preserves model RNG and conditioning exactly at committed checkpoints.
    """
    if ensemble_size < 2 or horizon < 1 or num_steps < 1:
        raise ValueError("Require ensemble>=2, horizon>=1 and num_steps>=1")
    if checkpoint_every < 1 or raw_every < 1 or raw_dense < 0:
        raise ValueError("Invalid checkpoint/raw-state saving cadence")
    if snapshot_dt is not None and (not math.isfinite(snapshot_dt) or snapshot_dt <= 0):
        raise ValueError("snapshot_dt must be finite and positive")
    if (
        histories.ndim != 5
        or truths.ndim != 5
        or histories.shape[:-1] != truths.shape[:-1]
    ):
        raise ValueError("Expected compatible [B,C,H,W,N/T] histories and truth")
    if len(trajectories) != len(histories) or len(set(trajectories)) != len(
        trajectories
    ):
        raise ValueError("Unique trajectory IDs must match the history batch")
    if not torch.isfinite(histories).all() or not torch.isfinite(truths).all():
        raise ValueError("Input archive data must be finite")
    if preprocesser is not None:
        if not torch.equal(
            preprocesser.base_mean, preprocesser.target_mean
        ) or not torch.equal(preprocesser.base_std, preprocesser.target_std):
            raise ValueError(
                "Autoregressive feedback requires matching base/target affine preprocessing"
            )
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    normalized = (
        preprocesser.transform(field_history=histories, is_batch=True)["field_history"]
        if preprocesser is not None
        else histories.clone()
    )
    initial_z = [
        float(0.5 * row[..., -1].double().square().mean()) for row in histories
    ]
    protocol = {
        "schema_version": 1,
        "model_class": type(model).__name__,
        "trajectories": list(trajectories),
        "origin": origin,
        "history_length": histories.shape[-1],
        "num_steps": num_steps,
        "ensemble_size": ensemble_size,
        "horizon": horizon,
        "seed": seed,
        "device": str(device),
        "initial_history_sha256": tensor_hash(histories),
        "truth_sha256": tensor_hash(truths),
        "available_truth_leads": truths.shape[-1],
        "truth_leads": truths.shape[-1],
        "checkpoint": (provenance or {}).get("checkpoint"),
        "data_sha256": next(iter((provenance or {}).get("data_files", [])), {}).get(
            "sha256"
        ),
        "inverse_affine": (
            None
            if preprocesser is None
            else {
                "mean": preprocesser.target_mean.cpu().tolist(),
                "std": preprocesser.target_std.cpu().tolist(),
            }
        ),
        "snapshot_dt": snapshot_dt,
        "time_units": (
            "stored snapshot intervals"
            if snapshot_dt is None
            else "physical time from supplied verified snapshot spacing"
        ),
        "physical_dt_caveat": "No physical clock is inferred when snapshot_dt is omitted; otherwise lead * mean stored spacing is nominal elapsed time, not an absolute timestamp.",
        "time_origin_archive_frame": origin - 1,
        "time_origin_stored_timestamp": (provenance or {})
        .get("timestamp_audit", {})
        .get("time_origin_stored_timestamp"),
        "physical_time_definition": "lead * mean stored snapshot spacing, elapsed from the last observed frame; extrapolated beyond available archive timestamps",
        "feedback_clipping": False,
        "raw_coordinate_system": "normalized model coordinates, without clipping",
        "diagnostic_coordinate_system": "original archive vorticity units",
        "normalization": (
            None
            if preprocesser is None
            else {
                "target_mean": preprocesser.target_mean.cpu().tolist(),
                "target_std": preprocesser.target_std.cpu().tolist(),
            }
        ),
        "enstrophy_definition": "0.5 * spatial_mean(vorticity**2); no mean subtraction",
        "spread_definition": "rms_spread = sqrt(spatial_mean(unbiased member variance)); spread_skill_ratio = sqrt((E+1)/E) * rms_spread / RMSE; skill only where paired truth exists",
        "stop_rule": "Stop on nonfinite outputs only; finite amplitude excursions do not stop or alter a trajectory",
        "excursion_factors": [10, 100, 1000],
        "initial_enstrophy": initial_z,
        "checkpoint_every": checkpoint_every,
        "milestones": list(milestones),
        "raw_every": raw_every,
        "raw_dense": raw_dense,
        "concurrent_workload": concurrent_workload,
        "timing_interpretation": "Diagnostic wall time, not a controlled inference-speed comparison",
        "provenance": provenance or {},
    }
    path = directory / "protocol.json"
    if path.exists():
        if not resume:
            raise FileExistsError(
                f"Existing rollout: {directory}; use --resume or a new output directory"
            )
        previous_protocol = json.loads(path.read_text())
        protocol["horizon"] = previous_protocol[
            "horizon"
        ]  # Extending the requested horizon does not rewrite the original protocol.
        if previous_protocol != protocol:
            raise ValueError(
                "Resume protocol/input/model provenance differs from the frozen rollout"
            )
    else:
        if resume:
            raise FileNotFoundError(
                "Cannot resume without an existing protocol and restart state"
            )
        with path.open("x") as stream:
            json.dump(protocol, stream, indent=2, allow_nan=False)
            stream.write("\n")
    model.to(device).eval()
    ncases = len(histories)
    state = normalized.repeat_interleave(ensemble_size, dim=0)
    rows, snapshots, crossings = [], [], []
    completed = 0
    sampling_seconds = 0.0
    total_nfe = 0
    first_failure = None
    count = [0]

    def count_call(_module, _args, _output):
        count[0] += 1

    hook = model.model.register_forward_hook(count_call)
    start_wall = time.perf_counter()
    elapsed_prior = 0.0
    requested_stop = (
        min(horizon, stop_after_lead) if stop_after_lead is not None else horizon
    )
    checkpoint_path = directory / "restart.pt"

    def save_checkpoint(status):
        summary = {
            "status": status,
            "completed_leads": completed,
            "requested_horizon": horizon,
            "available_truth_leads": truths.shape[-1],
            "rows": rows,
            "raw_snapshots": snapshots,
            "excursions": crossings,
            "first_failure": first_failure,
            "sampling_seconds": sampling_seconds,
            "total_nfe": total_nfe,
            "elapsed_seconds": elapsed_prior + time.perf_counter() - start_wall,
            "timing_interpretation": protocol["timing_interpretation"],
        }
        restart = {
            "completed_leads": completed,
            "history": state.detach().cpu(),
            "torch_rng_state": torch.random.get_rng_state(),
            "cuda_rng_states": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
            ),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
            "summary": summary,
            "protocol_sha256": sha256(path),
        }
        atomic_torch_save(checkpoint_path, restart)
        if completed in milestones:
            milestone = directory / f"restart_{completed:06d}.pt"
            if not milestone.exists():
                atomic_torch_save(milestone, restart)
        atomic_json(directory / "diagnostics.json", summary)
        print(
            f"lead={completed}/{horizon} status={status} sampling_seconds={sampling_seconds:.3f}",
            flush=True,
        )
        return summary

    try:
        with torch.no_grad():
            # Warmup is outside the reproducible sampling stream and all timings.
            model.sample(
                field_history=state,
                batch_size=len(state),
                num_steps=1,
                base=(
                    state[..., -1]
                    if isinstance(model, FollmerStochasticInterpolant)
                    else None
                ),
            )
            sync(device)
            if resume:
                restart = torch.load(
                    checkpoint_path, map_location="cpu", weights_only=False
                )
                if restart["protocol_sha256"] != sha256(path):
                    raise ValueError(
                        "Restart state does not belong to this frozen protocol"
                    )
                completed, state = restart["completed_leads"], restart["history"]
                summary = restart["summary"]
                rows, snapshots, crossings = (
                    summary["rows"],
                    summary["raw_snapshots"],
                    summary["excursions"],
                )
                sampling_seconds, total_nfe = (
                    summary["sampling_seconds"],
                    summary["total_nfe"],
                )
                elapsed_prior = summary["elapsed_seconds"]
                first_failure = summary["first_failure"]
                if first_failure is not None:
                    raise ValueError(
                        "Cannot resume through recorded nonfinite dynamics"
                    )
                for record in snapshots:
                    if sha256(directory / record["path"]) != record["sha256"]:
                        raise ValueError("Saved raw rollout state changed")
                torch.random.set_rng_state(restart["torch_rng_state"])
                if restart["cuda_rng_states"] is not None:
                    torch.cuda.set_rng_state_all(restart["cuda_rng_states"])
                np.random.set_state(restart["numpy_rng_state"])
                random.setstate(restart["python_rng_state"])
            else:
                seed_everything(seed)
                save_checkpoint("initialized")
            if horizon < completed or requested_stop < completed:
                raise ValueError("Requested end precedes already committed rollout")
            # Discard only uncommitted diagnostic lines after a crash. Raw state
            # orphans are retained and checked against deterministic regeneration.
            journal = directory / "diagnostics.jsonl"
            journal.write_text(
                "".join(json.dumps(row, allow_nan=False) + "\n" for row in rows)
            )
            with journal.open("a") as stream:
                for lead in range(completed + 1, requested_stop + 1):
                    count[0] = 0
                    sync(device)
                    begin = time.perf_counter()
                    generated, state = model.sample(
                        field_history=state,
                        batch_size=len(state),
                        num_steps=num_steps,
                        base=(
                            state[..., -1]
                            if isinstance(model, FollmerStochasticInterpolant)
                            else None
                        ),
                        return_field_history=True,
                    )
                    sync(device)
                    seconds = time.perf_counter() - begin
                    sampling_seconds += seconds
                    total_nfe += count[0]
                    shape = (ncases, ensemble_size, *generated.shape[1:])
                    generated = generated.detach().cpu().reshape(shape)
                    physical = inverse_state(
                        preprocesser, generated.flatten(0, 1)
                    ).reshape(shape)
                    finite = bool(torch.isfinite(generated).all())
                    for case, trajectory in enumerate(trajectories):
                        target = (
                            truths[case, ..., lead - 1]
                            if lead <= truths.shape[-1]
                            else None
                        )
                        metrics = state_diagnostics(physical[case], target)
                        row = {
                            "model_class": type(model).__name__,
                            "trajectory": trajectory,
                            "origin": origin,
                            "lead": lead,
                            "num_steps": num_steps,
                            "ensemble_size": ensemble_size,
                            "physical_time": (
                                None if snapshot_dt is None else lead * snapshot_dt
                            ),
                            "time_units": protocol["time_units"],
                            "seconds": seconds,
                            "cumulative_seconds": sampling_seconds,
                            "nfe": count[0],
                            "cumulative_nfe": total_nfe,
                            **metrics,
                        }
                        rows.append(row)
                        stream.write(json.dumps(row, allow_nan=False) + "\n")
                        z = metrics["enstrophy_member_mean"]
                        if z is not None:
                            for factor in protocol["excursion_factors"]:
                                identity = (trajectory, factor)
                                if z > factor * max(initial_z[case], 1e-30) and not any(
                                    (e["trajectory"], e["factor"]) == identity
                                    for e in crossings
                                ):
                                    crossings.append(
                                        {
                                            "trajectory": trajectory,
                                            "factor": factor,
                                            "first_lead": lead,
                                            "definition": "mean member enstrophy exceeds factor times its initial observed enstrophy",
                                        }
                                    )
                    stream.flush()
                    completed = lead
                    if (
                        lead <= raw_dense
                        or lead % raw_every == 0
                        or lead == requested_stop
                        or not finite
                    ):
                        snapshots.append(save_raw(directory, lead, generated))
                    if not finite:
                        bad = (
                            (~torch.isfinite(generated))
                            .flatten(2)
                            .any(2)
                            .nonzero()
                            .tolist()
                        )
                        first_failure = {
                            "lead": lead,
                            "affected_case_member_indices": bad,
                            "reason": "nonfinite model endpoint; no clipping or replacement",
                        }
                        return save_checkpoint("nonfinite")
                    if lead % checkpoint_every == 0 or lead in milestones:
                        save_checkpoint("running")
            return save_checkpoint(
                "complete" if completed >= horizon else "paused_at_requested_lead"
            )
    except Exception as error:
        atomic_json(
            directory / "error.json",
            {
                "exception": repr(error),
                "last_completed_lead": completed,
                "instruction": "Restart from the last committed restart.pt; inspect unexpected errors before resuming",
            },
        )
        raise
    finally:
        hook.remove()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--horizon", type=int, default=4096)
    parser.add_argument("--ensemble", type=int, default=8)
    parser.add_argument("--trajectories", nargs="+", type=int, default=[180])
    parser.add_argument("--origin", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--checkpoint-every", type=int, default=16)
    parser.add_argument("--milestones", nargs="*", type=int, default=[1024])
    parser.add_argument("--raw-every", type=int, default=16)
    parser.add_argument("--raw-dense", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-lead", type=int)
    parser.add_argument("--snapshot-dt", type=float)
    parser.add_argument(
        "--time-audit",
        help="Verified timestamp/data correspondence audit JSON; uses its mean stored spacing",
    )
    parser.add_argument(
        "--concurrent-workload", default="unspecified; timings are diagnostic only"
    )
    args = parser.parse_args()
    torch.set_num_threads(args.num_threads)
    model, config = load_teacher(args.checkpoint_dir, args.device)
    length = int(config.len_field_history)
    if args.origin < length:
        raise ValueError("Origin must include the complete trained history")
    prep = instantiate(config.preprocesser) if config.preprocesser is not None else None
    data_config = config.test_data.copy()
    lower, upper = min(args.trajectories), max(args.trajectories) + 1
    data_config.tajectories_ids = f"({lower}, {upper})"
    dataset = instantiate(data_config, preprocesser=prep)
    observed = torch.stack([dataset[index - lower]["x"] for index in args.trajectories])
    if args.origin > observed.shape[-1]:
        raise ValueError("Origin lies beyond available observations")
    histories, truths = (
        observed[..., args.origin - length : args.origin],
        observed[..., args.origin :],
    )
    provenance = {
        "checkpoint": str(Path(args.checkpoint_dir) / "model.pth"),
        "checkpoint_sha256": sha256(Path(args.checkpoint_dir) / "model.pth"),
        "config_sha256": sha256(Path(args.checkpoint_dir) / "config.yaml"),
        "data_files": [
            {
                "path": str(Path(folder) / filename),
                "sha256": sha256(Path(folder) / filename),
            }
            for folder, filename in zip(data_config.paths, data_config.files)
        ],
        "parameters": sum(p.numel() for p in model.parameters()),
        "torch_version": str(torch.__version__),
        "platform": platform.platform(),
        "num_threads": args.num_threads,
        "gpu": (
            torch.cuda.get_device_name(args.device)
            if str(args.device).startswith("cuda")
            else None
        ),
    }
    if args.time_audit:
        audit = json.loads(Path(args.time_audit).read_text())
        if (
            not audit.get("all_values_exactly_equal_to_stride2_downsampling")
            or audit["archive_sha256"] != provenance["data_files"][0]["sha256"]
        ):
            raise ValueError("Timestamp audit does not verify this data archive")
        spacing = float(audit["time_spacing_mean"])
        if args.snapshot_dt is not None and not math.isclose(
            args.snapshot_dt, spacing, rel_tol=0, abs_tol=1e-12
        ):
            raise ValueError(
                "Requested spacing differs from the verified mean stored spacing"
            )
        args.snapshot_dt = spacing
        if len(audit["stored_times"]) != observed.shape[-1]:
            raise ValueError(
                "Timestamp audit length differs from the loaded trajectory"
            )
        provenance["timestamp_audit"] = {
            "path": str(args.time_audit),
            "sha256": sha256(args.time_audit),
            "time_spacing_mean": spacing,
            "time_origin_archive_frame": args.origin - 1,
            "time_origin_stored_timestamp": float(
                audit["stored_times"][args.origin - 1]
            ),
            "interpretation": "Nominal elapsed stored time; not an absolute timestamp or a claim about unverified PDE parameters",
        }
    result = run_rollout(
        model,
        histories,
        truths,
        preprocesser=prep,
        output_dir=args.output,
        trajectories=args.trajectories,
        origin=args.origin,
        num_steps=args.steps,
        ensemble_size=args.ensemble,
        horizon=args.horizon,
        seed=args.seed,
        device=args.device,
        checkpoint_every=args.checkpoint_every,
        milestones=args.milestones,
        raw_every=args.raw_every,
        raw_dense=args.raw_dense,
        resume=args.resume,
        stop_after_lead=args.stop_after_lead,
        snapshot_dt=args.snapshot_dt,
        concurrent_workload=args.concurrent_workload,
        provenance=provenance,
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "completed_leads",
                    "sampling_seconds",
                    "first_failure",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
