#!/usr/bin/env python3
"""Download the declared pretrained video metrics, without forecasting weights."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import requests
import torch

I3D_URL = "https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1"
I3D_SHA256 = "bec6519f66ea534e953026b4ae2c65553c17bf105611c746d904657e5860a5e2"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--i3d-path", default="cache/video_metrics/i3d_torchscript.pt")
    parser.add_argument("--output", default="cache/video_metrics/manifest.json")
    args = parser.parse_args()
    path = Path(args.i3d_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        partial = path.with_suffix(".partial")
        with requests.get(I3D_URL, stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            with partial.open("wb") as stream:
                for block in response.iter_content(1024 * 1024):
                    stream.write(block)
        if sha256(partial) != I3D_SHA256:
            raise ValueError("Downloaded I3D does not match the declared Google-converted weights")
        partial.replace(path)
    if sha256(path) != I3D_SHA256:
        raise ValueError("Existing I3D weights have an unexpected hash; inspect before replacing")
    import lpips
    lpips.LPIPS(net="alex", version="0.1", verbose=False).eval()
    files = {
        "i3d_google_conversion": path,
        "alexnet_imagenet": Path(torch.hub.get_dir()) / "checkpoints/alexnet-owt-7be5be79.pth",
        "lpips_calibration": Path(lpips.__file__).parent / "weights/v0.1/alex.pth",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "i3d_source": I3D_URL,
        "versions": {package: importlib.metadata.version(package)
                     for package in ("torch", "torchvision", "lpips", "scikit-image", "numpy")},
        "weights": {key: {"path": str(value), "sha256": sha256(value),
                          "bytes": value.stat().st_size} for key, value in files.items()},
    }, indent=2) + "\n")
    print(f"Pretrained video metrics verified; provenance: {output}")


if __name__ == "__main__":
    main()
