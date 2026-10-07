#!/usr/bin/env python3
"""Validate decoded dataset structure and save compact acquisition provenance."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8388608), b""):
            h.update(block)
    return h.hexdigest()


def audit(path, full_hash=False):
    m = json.loads(path.read_text())
    seen = set()
    members = {}
    lengths = {}
    people = {}
    actions = {}
    for c in m["clips"]:
        assert c["clip_id"] not in seen, f"Duplicate ID {c['clip_id']}"
        seen.add(c["clip_id"])
        split = c["split"]
        file = path.parent / c["path"]
        a = np.load(file, mmap_mode="r")
        assert a.dtype == np.uint8 and a.shape == (
            c["num_frames"],
            m["resolution"],
            m["resolution"],
            m["channels"],
        ), str(file)
        if full_hash:
            assert digest(file) == c["sha256"], f"Hash mismatch: {file}"
        if "source_frame_indices" in c:
            indices = c["source_frame_indices"]
            assert len(indices) == c["num_frames"]
            assert np.all(np.diff(indices) == m["frame_stride"])
            assert indices[0] == c["source_frame_start"]
            assert indices[-1] + 1 == c["source_frame_end_exclusive"]
        key = (c["source_archive"], c["source_member"])
        if key in members:
            assert members[key] == split, f"Source-video leakage: {key}"
        members[key] = split
        lengths.setdefault(split, []).append(c["num_frames"])
        if "person" in c:
            people.setdefault(split, set()).add(c["person"])
            actions.setdefault(split, Counter()).update([c["action"]])
    for a in people:
        for b in people:
            if a != b:
                assert people[a].isdisjoint(people[b]), f"Subject leakage: {a},{b}"
    annotation = path.parent.parent / "archives" / "kth" / "00sequences.txt"
    annotation_provenance = None
    if m["dataset"] == "kth" and annotation.is_file():
        annotation_provenance = {
            "url": "https://www.csc.kth.se/cvap/actions/00sequences.txt",
            "path": str(annotation),
            "sha256": digest(annotation),
        }
    history, horizon = (10, 30) if m["dataset"] == "kth" else (2, 14)
    return {
        "dataset": m["dataset"],
        "manifest_path": str(path),
        "manifest_sha256": digest(path),
        "resolution": m["resolution"],
        "channels": m["channels"],
        "frame_stride": m["frame_stride"],
        "spatial_transform": m["spatial_transform"],
        "normalization": m["normalization"],
        "source_videos": len(members),
        "clips": len(seen),
        "sources": m["sources"],
        "sequence_annotations": annotation_provenance,
        "decoded_file_hashes_checked": full_hash,
        "history": history,
        "forecast_horizon": horizon,
        "split_summary": {
            s: {
                "clips": len(v),
                "frames_min": min(v),
                "frames_max": max(v),
                "frames_total": sum(v),
                "eligible_evaluation_clips": sum(n >= history + horizon for n in v),
                "next_frame_windows_stride1": sum(max(0, n - history) for n in v),
                "subjects": sorted(people.get(s, [])),
                "actions": dict(actions.get(s, {})),
            }
            for s, v in lengths.items()
        },
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifests", nargs="+", type=Path)
    p.add_argument("--full-hash", action="store_true")
    p.add_argument(
        "--output",
        type=Path,
        default=Path("research/video_forecasting/notes/data_summary.json"),
    )
    a = p.parse_args()
    result = {
        "schema_version": 1,
        "ffmpeg_version": subprocess.check_output(
            ["ffmpeg", "-version"], text=True
        ).splitlines()[0],
        "preparation_script_sha256": digest(
            Path(__file__).with_name("prepare_videos.py")
        ),
        "datasets": [audit(path, a.full_hash) for path in a.manifests],
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                d["dataset"]: {"clips": d["clips"], "source_videos": d["source_videos"]}
                for d in result["datasets"]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
