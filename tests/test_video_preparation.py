"""Scientific preprocessing checks: frame indexing, channel order and stride."""

import importlib.util
from pathlib import Path
import shutil
import subprocess
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "prepare_videos",
    Path(__file__).parents[1] / "research/video_forecasting/data/prepare_videos.py",
)
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


def test_annotation_inclusive_one_based_ranges():
    text = "Header\nperson01_running_d1 frames 1-35, 95-130\n\nperson02_boxing_d4 frames 2-80, 100-200\n"
    assert prepare.kth_segments(text) == {
        "person01_running_d1": [(0, 35), (94, 130)],
        "person02_boxing_d4": [(1, 80), (99, 200)],
    }


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_decode_retains_exact_stride_and_rgb(tmp_path):
    frames = np.zeros((12, 8, 12, 3), dtype=np.uint8)
    for i in range(12):
        frames[i, ...] = [i * 10, 40, 90]
    path = tmp_path / "clip.mkv"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            "12x8",
            "-r",
            "25",
            "-i",
            "pipe:0",
            "-c:v",
            "ffv1",
            str(path),
        ],
        input=frames.tobytes(),
        check=True,
    )
    decoded = prepare.decode(path, 4, 3, 5, center_crop=True)
    assert decoded.shape == (3, 4, 4, 3)
    np.testing.assert_allclose(decoded[:, 2, 2, 0], [0, 50, 100], atol=1)
    np.testing.assert_allclose(decoded[..., 1], 40, atol=1)
    np.testing.assert_allclose(decoded[..., 2], 90, atol=1)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_archive_processing_records_raw_indices_and_split(tmp_path):
    import zipfile

    frames = np.zeros((12, 8, 12, 3), dtype=np.uint8)
    for i in range(12):
        frames[i] = i * 10
    source = tmp_path / "clip.mkv"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            "12x8",
            "-r",
            "25",
            "-i",
            "pipe:0",
            "-c:v",
            "ffv1",
            str(source),
        ],
        input=frames.tobytes(),
        check=True,
    )
    archive = tmp_path / "video_test.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for i in range(2):
            z.write(source, f"nested/video_{i:05d}.mp4")
    records = prepare.prepare_archive(archive, "clevrer", "test", tmp_path, 4, 2, {})
    assert len(records) == 2
    for record in records:
        assert record["source_frame_indices"] == [0, 5, 10]
        assert record["source_frame_end_exclusive"] == 11
        assert record["split"] == "test"
        array = np.load(tmp_path / record["path"])
        assert array.shape == (3, 4, 4, 3)
        np.testing.assert_allclose(array[:, 2, 2, 0], [0, 50, 100], atol=1)
