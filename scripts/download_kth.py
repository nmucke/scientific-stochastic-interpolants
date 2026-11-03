"""Download the KTH human actions dataset and pack it into one npz file.

Follows the SRVP / RIVER protocol: every 160x120 video is squash-resized to
size x size (no crop, ffmpeg's default bicubic scaler) and kept as a single
uint8 grayscale channel. Whole videos are kept (00sequences.txt is downloaded
for reference but not used to cut them).

    python scripts/download_kth.py
    python scripts/download_kth.py --root data/KTH --size 64 --remove-zips

Output in <root>/:

    raw/<action>.zip, raw/00sequences.txt   downloaded files
    raw/avi/*.avi                           extracted videos (see --remove-avi)
    kth_<size>.npz                          frames uint8 [N_frames, size, size],
                                            offsets int64 [N_videos + 1],
                                            person int64, action <U12,
                                            scenario int64, names <U32

Video v is frames[offsets[v]:offsets[v + 1]]; videos are sorted by
(person, action, scenario). Every step is skipped when its output exists, so
re-running the script is cheap.
"""

import argparse
import logging
import re
import shutil
import subprocess
import urllib.request
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

BASE_URL = "https://www.csc.kth.se/cvap/actions"
ACTIONS = ("walking", "jogging", "running", "boxing", "handwaving", "handclapping")
SEQUENCES_FILE = "00sequences.txt"
NAME_PATTERN = re.compile(r"person(\d{2})_([a-z]+)_d(\d)_uncomp\.avi")
CHUNK_SIZE = 1 << 20


def find_ffmpeg() -> str:
    """Return a working ffmpeg executable: the system one, else imageio's."""
    exe = shutil.which("ffmpeg")
    if exe is not None:
        return exe
    try:
        import imageio_ffmpeg

        return str(imageio_ffmpeg.get_ffmpeg_exe())
    except ImportError as err:
        raise RuntimeError(
            "No ffmpeg found: install it system-wide or `pip install imageio-ffmpeg`."
        ) from err


def remote_size(url: str) -> int | None:
    """Content-Length of a URL, or None when the server does not report it."""
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=60) as response:
        length = response.headers.get("Content-Length")
    return int(length) if length is not None else None


def download(url: str, path: Path) -> None:
    """Stream url to path, skipping it when a complete copy already exists."""
    expected = remote_size(url)
    if path.exists() and (expected is None or path.stat().st_size == expected):
        logger.info(f"Already downloaded: {path.name}")
        return

    tmp_path = path.with_suffix(path.suffix + ".part")
    with (
        urllib.request.urlopen(url, timeout=60) as response,
        open(tmp_path, "wb") as f,
        tqdm(
            total=expected, unit="B", unit_scale=True, desc=path.name, leave=False
        ) as bar,
    ):
        while chunk := response.read(CHUNK_SIZE):
            f.write(chunk)
            bar.update(len(chunk))

    if expected is not None and tmp_path.stat().st_size != expected:
        raise RuntimeError(
            f"Incomplete download of {url}: got {tmp_path.stat().st_size} "
            f"of {expected} bytes."
        )
    tmp_path.rename(path)
    logger.info(f"Downloaded {path.name} ({path.stat().st_size / 1e6:.1f} MB)")


def extract_avis(zip_path: Path, avi_dir: Path) -> None:
    """Extract the .avi files of one zip into avi_dir (flat, existing files kept)."""
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            name = Path(member.filename).name
            if not NAME_PATTERN.fullmatch(name):
                continue
            target = avi_dir / name
            if target.exists() and target.stat().st_size == member.file_size:
                continue
            with archive.open(member) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)


def decode_video(ffmpeg: str, path: Path, size: int) -> np.ndarray:
    """Decode one avi into uint8 grayscale frames of shape [T, size, size]."""
    command = [
        ffmpeg,
        "-nostdin",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-vf",
        f"scale={size}:{size}:flags=bicubic",
        "-pix_fmt",
        "gray",
        "-f",
        "rawvideo",
        "-",
    ]
    result = subprocess.run(command, capture_output=True, check=True)
    raw = np.frombuffer(result.stdout, dtype=np.uint8)
    if raw.size == 0 or raw.size % (size * size) != 0:
        raise RuntimeError(f"Bad ffmpeg output for {path.name}: {raw.size} bytes.")
    return raw.reshape(-1, size, size)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", type=Path, default=Path("data/KTH"))
    parser.add_argument("--size", type=int, default=64, help="Output side length.")
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Use the zips already in <root>/raw without contacting the server.",
    )
    parser.add_argument(
        "--remove-zips", action="store_true", help="Delete the zips once extracted."
    )
    parser.add_argument(
        "--remove-avi",
        action="store_true",
        help="Delete the extracted avi files once the npz is written.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Rebuild the npz if it exists."
    )
    parser.add_argument(
        "--workers", type=int, default=16, help="Parallel ffmpeg processes."
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = parse_args()

    raw_dir = args.root / "raw"
    avi_dir = raw_dir / "avi"
    avi_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.root / f"kth_{args.size}.npz"

    if out_path.exists() and not args.overwrite:
        logger.info(f"{out_path} already exists; pass --overwrite to rebuild it.")
        return

    # Download and extract
    zip_paths = [raw_dir / f"{action}.zip" for action in ACTIONS]
    if not args.skip_download:
        download(f"{BASE_URL}/{SEQUENCES_FILE}", raw_dir / SEQUENCES_FILE)
        for zip_path in zip_paths:
            download(f"{BASE_URL}/{zip_path.name}", zip_path)
    for zip_path in zip_paths:
        if zip_path.exists():
            logger.info(f"Extracting {zip_path.name}")
            extract_avis(zip_path, avi_dir)
            if args.remove_zips:
                zip_path.unlink()
        elif not any(avi_dir.glob(f"*_{zip_path.stem}_*.avi")):
            raise FileNotFoundError(f"Missing {zip_path} and no extracted videos.")

    # Collect videos sorted by (person, action, scenario)
    videos = []
    for path in avi_dir.glob("*.avi"):
        match = NAME_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        person, action, scenario = int(match[1]), match[2], int(match[3])
        videos.append((person, ACTIONS.index(action), scenario, path))
    videos.sort(key=lambda v: v[:3])
    logger.info(f"Found {len(videos)} videos (599 expected)")

    # Decode
    ffmpeg = find_ffmpeg()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        clips = list(
            tqdm(
                pool.map(lambda v: decode_video(ffmpeg, v[3], args.size), videos),
                total=len(videos),
                desc="Decoding",
            )
        )

    lengths = np.array([len(clip) for clip in clips], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    persons = np.array([v[0] for v in videos], dtype=np.int64)
    actions = np.array([ACTIONS[v[1]] for v in videos], dtype="<U12")
    scenarios = np.array([v[2] for v in videos], dtype=np.int64)
    names = np.array(
        [f"person{v[0]:02d}_{ACTIONS[v[1]]}_d{v[2]}" for v in videos], dtype="<U32"
    )

    tmp_path = out_path.with_name(out_path.stem + ".tmp.npz")
    np.savez(
        tmp_path,
        frames=np.concatenate(clips, axis=0),
        offsets=offsets,
        person=persons,
        action=actions,
        scenario=scenarios,
        names=names,
    )
    tmp_path.rename(out_path)

    if args.remove_avi:
        shutil.rmtree(avi_dir)

    logger.info(
        f"Wrote {out_path} ({out_path.stat().st_size / 1e9:.2f} GB): "
        f"{len(videos)} videos, {offsets[-1]} frames, "
        f"length min/mean/max {lengths.min()}/{lengths.mean():.0f}/{lengths.max()}"
    )
    logger.info(f"Videos per person: {dict(sorted(Counter(persons.tolist()).items()))}")
    logger.info(f"Videos per action: {dict(Counter(actions.tolist()))}")


if __name__ == "__main__":
    main()
