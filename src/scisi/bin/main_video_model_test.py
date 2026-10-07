"""Pixel-space autoregressive video evaluation shared by all four model families.

All sampling happens before metric extraction for a clip: perceptual networks
cannot consume the RNG used for a subsequent forecast. Metrics see future frames
only; FVD sees the common observed prefix followed by the generated future.
"""

import argparse
import hashlib
import json
import platform
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from PIL import Image

from scisi.bin.main_map_model_train import seed_everything
from scisi.data.video_datasets import VideoFrameDataset
from scisi.metrics.video import (
    I3DFeatures,
    frechet_video_distance,
    video_frame_metrics,
)
from scisi.models.follmer_stochastic_interpolant import FollmerStochasticInterpolant


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def select_cases(dataset, num_clips, seed):
    """Sample eligible annotated clips without sorting/action-prefix bias."""
    if num_clips < 2 or num_clips > len(dataset):
        raise ValueError(f"Need 2 <= num_clips <= eligible clips ({len(dataset)})")
    indices = np.random.default_rng(seed).choice(len(dataset), num_clips, replace=False)
    return [int(i) for i in indices]


def summarize_video_rows(rows, bootstrap_seed=712, n_bootstrap=2000):
    """Average clips within original source videos before source-video bootstrap."""
    groups = defaultdict(list)
    for row in rows:
        groups[(row["num_steps"], row["lead"])].append(row)
    excluded = {"clip_id", "source_video", "case_index", "num_steps", "lead"}
    output = []
    for (steps, lead), records in sorted(groups.items()):
        sources = sorted({r["source_video"] for r in records})
        rng = np.random.default_rng(bootstrap_seed + lead)
        draws = rng.integers(len(sources), size=(n_bootstrap, len(sources)))
        item = {"num_steps": steps, "lead": lead, "num_clips": len(records),
                "num_source_videos": len(sources)}
        for key in records[0]:
            if key in excluded:
                continue
            if any(r[key] is None for r in records):
                item[key] = None
                continue
            values = np.array([np.mean([r[key] for r in records
                                       if r["source_video"] == source])
                               for source in sources], dtype=np.float64)
            item[key] = float(values.mean())
            item[key + "_ci95"] = np.quantile(values[draws].mean(1), [.025, .975]).tolist()
        output.append(item)
    return output


def _contact_sheet(truth, predictions, history_length, destination):
    """Save the same first two ensemble members; never select by quality."""
    horizon = predictions.shape[1]
    leads = sorted(set([0, horizon // 4, horizon // 2, 3 * horizon // 4, horizon - 1]))
    sequences = [truth[history_length:]] + list(predictions[:2])
    rows = []
    for sequence in sequences:
        pictures = []
        for lead in leads:
            pixels = ((sequence[lead].clamp(-1, 1) + 1) * 127.5).byte()
            if pixels.shape[0] == 1:
                pixels = pixels.repeat(3, 1, 1)
            pictures.append(pixels.permute(1, 2, 0).numpy())
        rows.append(np.concatenate(pictures, axis=1))
    Image.fromarray(np.concatenate(rows, axis=0)).save(destination)
    return [lead + 1 for lead in leads]


def evaluate_video_model(
    model, dataset, *, num_steps=(1, 8), ensemble_size=4, num_clips=128,
    forecast_horizon=30, seed=20261007, device="cuda", output_dir=None,
    feature_extractor=None, lpips_model=None, metric_fn=video_frame_metrics,
    selected_indices=None, save_qualitative=3,
):
    """Evaluate a trained sampler, or persistence when model is None.

    Histories and generated outputs remain in [-1,1] normalization, with no
    internal clipping. Pixel/image metrics declare their own clipping policy.
    ``feature_extractor`` accepts [N,T,C,H,W] and returns [N,D] numpy features.
    """
    if ensemble_size < 2 or forecast_horizon < 1:
        raise ValueError("Need at least two members and a positive horizon")
    if not num_steps or any(not isinstance(n, int) or n < 1 for n in num_steps):
        raise ValueError("num_steps must be positive integer budgets")
    if len(set(num_steps)) != len(num_steps):
        raise ValueError("Sampling budgets must be unique")
    if dataset.normalization != "minus_one_one":
        raise ValueError("Video evaluation expects fixed [-1,1] data normalization")
    if feature_extractor is None:
        raise ValueError("An actual pretrained I3D extractor is required for video FVD")
    indices = (select_cases(dataset, num_clips, seed) if selected_indices is None
               else list(selected_indices))
    if len(indices) != num_clips or len(set(indices)) != len(indices):
        raise ValueError("Selected case count/uniqueness mismatch")
    if any(i < 0 or i >= len(dataset) for i in indices):
        raise ValueError("Selected case index outside dataset")
    length = dataset.len_field_history
    output = Path(output_dir) if output_dir is not None else None
    if output:
        output.mkdir(parents=True, exist_ok=True)
    cases = [dataset.clips[i] for i in indices]
    if output:
        (output / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    nfe = [0]
    hook = None
    if model is not None:
        model.to(device).eval()
        def count_calls(_module, _args, _output):
            nfe[0] += 1
        hook = model.model.register_forward_hook(count_calls)
    rows, fvd_rows, feature_provenance = [], [], []
    real_features = None
    try:
        with torch.no_grad():
            if model is not None:
                warm = dataset[indices[0]]["x"][..., :length].unsqueeze(0)
                model.sample(field_history=warm, batch_size=ensemble_size, num_steps=1,
                             base=warm[..., -1] if isinstance(model, FollmerStochasticInterpolant) else None)
                _sync(device)
            for steps in num_steps:
                real_list, generated_list = [], []
                for case_number, index in enumerate(indices):
                    record = cases[case_number]
                    clip = dataset[index]["x"]
                    if clip.shape[-1] < length + forecast_horizon:
                        raise ValueError("A selected clip does not contain the requested horizon")
                    history = clip[..., :length].unsqueeze(0)
                    truth = clip[..., :length + forecast_horizon].permute(3, 0, 1, 2)
                    seed_everything(seed + 100003 * index)
                    predictions, times, calls = [], [], []
                    for lead in range(forecast_horizon):
                        nfe[0] = 0
                        _sync(device)
                        start = time.perf_counter()
                        if model is None:
                            if history.shape[0] == 1:
                                history = history.repeat(ensemble_size, 1, 1, 1, 1)
                            generated = history[..., -1].clone()
                            history = torch.cat([history[..., 1:], generated.unsqueeze(-1)], -1)
                        else:
                            generated, history = model.sample(
                                field_history=history, batch_size=ensemble_size,
                                num_steps=steps, return_field_history=True,
                                base=history[..., -1] if isinstance(model, FollmerStochasticInterpolant) else None,
                            )
                        _sync(device)
                        times.append(time.perf_counter() - start)
                        calls.append(nfe[0])
                        if not torch.isfinite(generated).all():
                            raise FloatingPointError(f"Nonfinite forecast: {record['clip_id']} lead {lead+1}")
                        predictions.append(generated.cpu())
                    # Complete each stochastic rollout before using any metric model.
                    predicted = torch.stack(predictions, 1)  # E,T,C,H,W
                    prefix = truth[:length].unsqueeze(0).repeat(ensemble_size, 1, 1, 1, 1)
                    full_generated = torch.cat([prefix, predicted], 1)
                    generated_list.append(feature_extractor(full_generated))
                    if real_features is None:
                        real_list.append(feature_extractor(truth.unsqueeze(0)))
                    for lead in range(forecast_horizon):
                        metrics = metric_fn(predicted[:, lead], truth[length + lead], lpips_model)
                        for key, value in metrics.items():
                            if value is not None and not np.isfinite(value):
                                raise FloatingPointError(f"Nonfinite metric {key}")
                        rows.append({
                            "clip_id": record["clip_id"],
                            "source_video": record.get("source_member", record["clip_id"]),
                            "case_index": index, "num_steps": steps, "lead": lead + 1,
                            "seconds": times[lead], "cumulative_seconds": sum(times[:lead+1]),
                            "nfe": calls[lead], "cumulative_nfe": sum(calls[:lead+1]),
                            **metrics,
                        })
                    if output and case_number < save_qualitative:
                        prefix_path = output / f"case_{case_number:03d}_steps_{steps}"
                        np.savez_compressed(str(prefix_path) + ".npz", truth=truth.numpy(),
                                            predictions=predicted.numpy(), history_length=length)
                        leads = _contact_sheet(truth, predicted, length, str(prefix_path) + ".png")
                        (Path(str(prefix_path) + ".json")).write_text(json.dumps({
                            "clip_id": record["clip_id"], "future_leads": leads,
                            "rows": ["truth", "member_0", "member_1"],
                            "selection": "first fixed selected cases, first two members",
                        }, indent=2) + "\n")
                    if output and ((case_number + 1) % 8 == 0 or case_number == len(indices)-1):
                        (output / "forecasts.json").write_text(json.dumps(rows, indent=2) + "\n")
                        print(f"steps={steps} clips={case_number+1}/{len(indices)}", flush=True)
                if real_features is None:
                    real_features = np.concatenate(real_list, axis=0)
                generated_features = np.stack(generated_list, axis=0)  # cases,E,D
                member_fvd = [frechet_video_distance(real_features, generated_features[:, e])
                              for e in range(ensemble_size)]
                fvd_rows.append({
                    "num_steps": steps, "real_clips": len(real_features),
                    "generated_clips": len(indices) * ensemble_size,
                    "feature_dimension": int(real_features.shape[1]),
                    "fvd_pooled": frechet_video_distance(real_features, generated_features.reshape(-1, real_features.shape[1])),
                    "fvd_member_values": member_fvd,
                    "fvd_member_mean": float(np.mean(member_fvd)),
                    "fvd_member_std": float(np.std(member_fvd, ddof=1)),
                })
                if output:
                    feature_path = output / f"features_steps_{steps}.npz"
                    np.savez_compressed(feature_path, real=real_features, generated=generated_features)
                    feature_provenance.append({"path": feature_path.name, "sha256": _hash(feature_path)})
    finally:
        if hook is not None:
            hook.remove()
    result = {
        "status": "complete", "study_type": "initial pixel-space pilot",
        "model_class": "persistence" if model is None else type(model).__name__,
        "normalization": "model [-1,1]; proper scores in unbounded unit-pixel coordinates",
        "image_metric_clipping": "clip [0,1] only for PSNR, SSIM, LPIPS and display",
        "fvd_clipping": "clip [-1,1] for I3D; prefix is observed truth",
        "feedback_clipping": False, "history_length": length,
        "forecast_horizon": forecast_horizon, "ensemble_size": ensemble_size,
        "num_clips": len(indices), "seed": seed, "selected_indices": indices,
        "aggregation": "frame summaries: source-video equal weight and pointwise source-video bootstrap; FVD: selected clip distribution",
        "fvd_uncertainty": "member SD describes completion randomness; not a confidence interval",
        "hardware": {"gpu": torch.cuda.get_device_name(device) if str(device).startswith("cuda") else None,
                     "platform": platform.platform(), "torch_version": str(torch.__version__)},
        "rows": rows, "summary": summarize_video_rows(rows), "fvd": fvd_rows,
        "feature_artifacts": feature_provenance,
    }
    if output:
        (output / "evaluation.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main_video_model_test(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True, help="Saved model directory, also supplies dataset config")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest-path")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--steps", nargs="+", type=int, default=[1, 8])
    parser.add_argument("--num-clips", type=int, default=128)
    parser.add_argument("--ensemble-size", type=int, default=4)
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--i3d-path", default="cache/video_metrics/i3d_torchscript.pt")
    parser.add_argument("--persistence", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    torch.set_num_threads(args.num_threads)
    output = Path(args.output_dir)
    if (output / "evaluation.json").exists() and not args.overwrite:
        raise FileExistsError(f"Completed evaluation exists: {output}")
    cfg = OmegaConf.load(Path(args.checkpoint_dir) / "config.yaml")
    horizon = args.horizon or int(cfg.evaluation.forecast_horizon)
    manifest = args.manifest_path or cfg.manifest_path
    dataset = VideoFrameDataset(manifest_path=manifest, split=args.split,
                                len_field_history=int(cfg.len_field_history),
                                train_or_test="test", horizon=horizon)
    model, checkpoint = None, Path(args.checkpoint_dir) / "model.pth"
    if not args.persistence:
        model = instantiate(cfg.model)
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    import lpips
    lpips_model = lpips.LPIPS(net="alex", version="0.1", verbose=False).to(args.device).eval()
    extractor = I3DFeatures(args.i3d_path, args.device)
    result = evaluate_video_model(
        model, dataset, num_steps=args.steps, ensemble_size=args.ensemble_size,
        num_clips=args.num_clips, forecast_horizon=horizon, seed=args.seed,
        device=args.device, output_dir=output, feature_extractor=extractor,
        lpips_model=lpips_model,
    )
    result.update({"checkpoint": None if args.persistence else str(checkpoint),
                   "checkpoint_sha256": None if args.persistence else _hash(checkpoint),
                   "config_sha256": _hash(Path(args.checkpoint_dir) / "config.yaml"),
                   "data_manifest": str(manifest), "manifest_sha256": _hash(manifest),
                   "split": args.split, "dataset": dataset.manifest["dataset"],
                   "frame_stride": dataset.manifest["frame_stride"],
                   "resolution": dataset.manifest["resolution"],
                   "i3d_sha256": _hash(args.i3d_path),
                   "lpips": {"version": "0.1", "network": "alex", "package": "0.1.4"},
                   "command": vars(args)})
    (output / "evaluation.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"fvd": result["fvd"], "last_lead": [r for r in result["summary"] if r["lead"] == horizon]}, indent=2))
    return result


if __name__ == "__main__":
    main_video_model_test()
