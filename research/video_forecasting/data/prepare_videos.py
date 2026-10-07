#!/usr/bin/env python3
"""Download official videos and make versioned, file-backed pixel manifests.

Resumable downloads use curl; archive CRCs are verified during extraction.
Each decoded clip is written atomically. Archives are retained for provenance.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import zipfile
import numpy as np

ACTIONS = ("boxing", "handclapping", "handwaving", "jogging", "running", "walking")
KTH_URL = "https://www.csc.kth.se/cvap/actions/"
CLEVRER_URL = "https://data.csail.mit.edu/clevrer/videos/"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def download(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["curl", "-L", "--fail", "--retry", "5", "-C", "-", "-o", str(path), url],
        check=True,
    )


def decode(path, resolution, channels, stride, center_crop=False, diagnostics=None):
    # select operates on decoded frame indices, not timestamps; no frame duplication.
    filters = [f"select=not(mod(n\\,{stride}))"]
    if center_crop:
        filters += [
            f"scale={resolution}:{resolution}:force_original_aspect_ratio=increase:flags=bilinear",
            f"crop={resolution}:{resolution}",
        ]
    else:
        filters.append(f"scale={resolution}:{resolution}:flags=bilinear")
    filt = ",".join(filters)
    result = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-threads",
            "1",
            "-i",
            str(path),
            "-vf",
            filt,
            "-vsync",
            "0",
            "-threads",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray" if channels == 1 else "rgb24",
            "pipe:1",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    raw = result.stdout
    if result.stderr:
        message = re.sub(
            r"0x[0-9a-f]+", "<decoder>", result.stderr.decode(errors="replace")
        )
        print(f"Decoder diagnostics for {path.name}: {message}", flush=True)
        if diagnostics is not None:
            diagnostics.append(message)
    frame_size = resolution * resolution * channels
    if not raw or len(raw) % frame_size:
        raise ValueError(f"Invalid decoded video {path}")
    return np.frombuffer(raw, np.uint8).reshape(-1, resolution, resolution, channels)


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def save_array(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npy")
    np.save(temporary, array)
    temporary.replace(path)


def kth_segments(text):
    result = {}
    for match in re.finditer(r"(person\d+_\w+_d\d)\s+frames\s+([\d,\-\s]+)", text):
        result[match[1]] = [
            (int(a) - 1, int(b)) for a, b in re.findall(r"(\d+)-(\d+)", match[2])
        ]
    return result


def prepare_archive(archive, dataset, split, root, resolution, workers, segments):
    entries = []
    source_zip = zipfile.ZipFile(archive)
    names = sorted(
        n
        for n in source_zip.namelist()
        if n.lower().endswith(".avi" if dataset == "kth" else ".mp4")
    )

    def process(name):
        stem = Path(name).stem
        person = int(re.search(r"person(\d+)", stem)[1]) if dataset == "kth" else None
        actual_split = (
            ("train" if person <= 18 else "validation" if person <= 20 else "test")
            if person
            else split
        )
        stem = stem.removesuffix("_uncomp")
        # One source per task; zip read checks CRC. Never extract untrusted member paths.
        with tempfile.TemporaryDirectory(prefix="scisi_video_") as tmp:
            source = Path(tmp) / Path(name).name
            # ZipFile serializes shared file reads internally; decoding is parallel.
            source.write_bytes(source_zip.read(name))
            diagnostics = []
            frames = decode(
                source,
                resolution,
                1 if dataset == "kth" else 3,
                1 if dataset == "kth" else 5,
                dataset == "clevrer",
                diagnostics,
            )
        ranges = segments[stem] if dataset == "kth" else [(0, len(frames))]
        out = []
        for sequence, (start, end) in enumerate(ranges):
            # Official annotations occasionally exceed actual EOF: record the truncation.
            end_actual = min(end, len(frames))
            if end_actual <= start:
                continue
            clip_id = f"{stem}_s{sequence:02d}" if dataset == "kth" else stem
            rel = Path("frames") / actual_split / (clip_id + ".npy")
            array = frames[start:end_actual]
            dest = root / rel
            save_array(dest, array)
            record = {
                "clip_id": clip_id,
                "path": str(rel),
                "split": actual_split,
                "num_frames": len(array),
                "source_archive": archive.name,
                "source_member": name,
                "source_frame_start": start * (1 if dataset == "kth" else 5),
                "source_frame_end_exclusive": (end_actual - 1)
                * (1 if dataset == "kth" else 5)
                + 1,
                "sha256": sha256(dest),
            }
            if dataset == "clevrer":
                record["source_frame_indices"] = list(
                    range(start * 5, end_actual * 5, 5)
                )
            if diagnostics:
                record["source_decode_diagnostics"] = diagnostics
            if person:
                record.update(
                    person=person,
                    action=stem.split("_")[1],
                    scenario=stem.split("_")[2],
                    annotation_end_exclusive=end,
                )
            out.append(record)
        return out

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for i, out in enumerate(pool.map(process, names)):
                entries.extend(out)
                if (i + 1) % 100 == 0 or i + 1 == len(names):
                    print(f"{archive.name}: decoded {i+1}/{len(names)}", flush=True)
    finally:
        source_zip.close()
    return entries


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["kth", "clevrer"], required=True)
    p.add_argument("--root", type=Path, default=Path("data/video_forecasting"))
    p.add_argument("--resolution", type=int, default=64)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--skip-download", action="store_true")
    p.add_argument(
        "--manifest-name",
        default="manifest.json",
        help="Use a distinct filename for partial split preparation",
    )
    p.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    args = p.parse_args()
    archives = args.root / "archives" / args.dataset
    destination = args.root / f"{args.dataset}_{args.resolution}"
    destination.mkdir(parents=True, exist_ok=True)
    segments = {}
    if args.dataset == "kth":
        annotations = archives / "00sequences.txt"
        if not args.skip_download:
            download(KTH_URL + "00sequences.txt", annotations)
        segments = kth_segments(annotations.read_text())
        sources = [
            (action, KTH_URL + action + ".zip", archives / (action + ".zip"))
            for action in ACTIONS
        ]
    else:
        sources = [
            (
                split,
                CLEVRER_URL + split + "/video_" + split + ".zip",
                archives / ("video_" + split + ".zip"),
            )
            for split in args.splits
        ]
    clips = []
    provenance = []
    for split, url, archive in sources:
        if not args.skip_download:
            download(url, archive)
        print(f"Processing {archive}", flush=True)
        partial_path = destination / (archive.stem + ".json")
        if partial_path.exists():
            partial = json.loads(partial_path.read_text())
            if partial["archive_sha256"] != sha256(archive):
                raise ValueError("Archive changed since processing")
        else:
            partial = {
                "archive_sha256": sha256(archive),
                "clips": prepare_archive(
                    archive,
                    args.dataset,
                    split,
                    destination,
                    args.resolution,
                    args.workers,
                    segments,
                ),
            }
            atomic_json(partial_path, partial)
        clips.extend(partial["clips"])
        provenance.append(
            {
                "url": url,
                "archive": archive.name,
                "bytes": archive.stat().st_size,
                "sha256": partial["archive_sha256"],
            }
        )
    manifest = {
        "schema_version": 1,
        "dataset": args.dataset,
        "resolution": args.resolution,
        "channels": 1 if args.dataset == "kth" else 3,
        "frame_stride": 1 if args.dataset == "kth" else 5,
        "pixel_storage": "uint8 THWC",
        "normalization": "minus_one_one",
        "spatial_transform": (
            "bilinear resize square"
            if args.dataset == "kth"
            else "bilinear resize short side, center crop square"
        ),
        "splits": (
            "persons1-18train19-20validation21-25test"
            if args.dataset == "kth"
            else "official"
        ),
        "sources": provenance,
        "clips": sorted(clips, key=lambda c: c["clip_id"]),
    }
    atomic_json(destination / args.manifest_name, manifest)
    print(
        json.dumps(
            {
                "manifest": str(destination / args.manifest_name),
                "clips": len(clips),
                "split_counts": {
                    s: sum(c["split"] == s for c in clips)
                    for s in ["train", "validation", "test"]
                },
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
