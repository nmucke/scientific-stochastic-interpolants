#!/usr/bin/env python
"""Build manuscript setup assets solely from frozen video protocols/previews."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parents[2]
VIDEO = ROOT / "research/video_forecasting"
OUTPUT = PAPER / "figures/video_case"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    datasets = {}
    for name in ("kth", "clevrer"):
        protocol_path = VIDEO / f"results/{name}_protocol.json"
        protocol = json.loads(protocol_path.read_text())
        manifest_path = ROOT / protocol["manifest"]["path"]
        assert digest(manifest_path) == protocol["manifest"]["sha256"]
        manifest = json.loads(manifest_path.read_text())
        split_counts = {
            split: sum(clip["split"] == split for clip in manifest["clips"])
            for split in ("train", "validation", "test")
        }
        stages = {}
        for family in protocol["models"]:
            stage = protocol["training_stages"][family]
            metadata_path = VIDEO / f"results/training/{name}_{family}/metadata.json"
            assert (
                digest(metadata_path) == stage["artifacts"]["metadata.json"]["sha256"]
            )
            metadata = json.loads(metadata_path.read_text())
            stages[family] = {
                "parameters": metadata["parameters"],
                "updates": stage["recorded_updates"],
                "checkpoint_sha256": stage["artifacts"]["model.pth"]["sha256"],
            }
        datasets[name] = {
            "protocol_sha256": digest(protocol_path),
            "split_counts": split_counts,
            "stages": stages,
            "history": protocol["history_length"],
            "horizon": protocol["forecast_horizon"],
            "ensemble": protocol["ensemble_size"],
            "test_cases": protocol["num_clips"],
            "frame_stride": protocol["frame_stride"],
        }
    table = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Model & KTH parameters & CLEVRER parameters & Own updates & Test NFEs \\",
        r"\midrule",
    ]
    families = [
        (
            "follmer_stochastic_interpolant",
            "Stochastic interpolant",
            "20,000",
            "16, 64",
        ),
        ("flow_matching", "Flow matching", "20,000", "16, 64"),
        ("follmer_ito_map", r"It\^o map", "2,000 + 500", "1, 8"),
        ("flow_map", "Flow map", "2,000 + 500", "1, 8"),
    ]
    for family, label, updates, budgets in families:
        kth, clevrer = (
            datasets[name]["stages"][family]["parameters"]
            for name in ("kth", "clevrer")
        )
        table.append(
            f"{label} & {kth:,} & {clevrer:,} & {updates} & {budgets} " + r"\\"
        )
    table += [r"\bottomrule", r"\end{tabular}"]
    (OUTPUT / "training_table.tex").write_text("\n".join(table) + "\n")
    table = [
        r"\begin{tabular}{lccrrrrr}",
        r"\toprule",
        r"Dataset & Pixels & $N\!\to\!L$ & Stride & Train & Validation & Test & Selected \\",
        r"\midrule",
    ]
    for name, label, pixels in [
        ("kth", "KTH", r"$64^2$, gray"),
        ("clevrer", "CLEVRER", r"$64^2$, RGB"),
    ]:
        d = datasets[name]
        c = d["split_counts"]
        table.append(
            f"{label} & {pixels} & ${d['history']}\\!\\to\\!{d['horizon']}$ & {d['frame_stride']} & {c['train']:,} & {c['validation']:,} & {c['test']:,} & {d['test_cases']} "
            + r"\\"
        )
    table += [r"\bottomrule", r"\end{tabular}"]
    (OUTPUT / "setup_table.tex").write_text("\n".join(table) + "\n")
    previews = [
        VIDEO / "notes/kth_data_preview.png",
        VIDEO / "notes/clevrer_data_preview.png",
    ]
    fig, axes = plt.subplots(2, 1, figsize=(10.2, 4.6))
    labels = [
        "(a) KTH: observed examples from six actions (different clips)",
        "(b) CLEVRER: observed frames of one clip (labels are original frame indices)",
    ]
    for axis, path, label in zip(axes, previews, labels):
        axis.imshow(Image.open(path))
        axis.axis("off")
        axis.set_title(label, loc="left", fontsize=10, pad=6)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.015, top=0.96, hspace=0.25)
    fig.savefig(OUTPUT / "dataset_previews.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT / "dataset_previews.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    (OUTPUT / "asset_provenance.json").write_text(
        json.dumps(
            {
                "purpose": "Dataset/setup figure and actual training-budget tables, not forecast results",
                "datasets": datasets,
                "preview_inputs": [
                    {"path": str(path.relative_to(ROOT)), "sha256": digest(path)}
                    for path in previews
                ],
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
