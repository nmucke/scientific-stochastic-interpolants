"""Latent KTH case: configs, models, the encoding script and the decoded plots.

The RIVER autoencoder (river_lib) is replaced by an average-pool / upsample mock.
"""

import importlib.util
import shutil
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from scisi.plotting.cases import plot_case
from scisi.sampling.sde_solvers import heun_step
from scisi.training.flow_trainer import FlowTrainer
from scisi.training.map_trainer import MapTrainer

ROOT = Path(__file__).resolve().parents[1]
L = 2


class MockAutoencoder(torch.nn.Module):
    """1x64x64 frames in [0, 1] <-> 4x8x8 latents (channel c is the pooled frame + c)."""

    def __init__(self, ckpt_path):
        super().__init__()
        self.ckpt_path = ckpt_path

    def encode(self, frames):
        assert frames.shape[1:] == (1, 64, 64) and frames.dtype == torch.float32
        assert 0 <= frames.min() and frames.max() <= 1
        pooled = torch.nn.functional.avg_pool2d(frames, 8)
        return torch.cat([pooled + c for c in range(4)], dim=1)

    def decode(self, latents):
        assert latents.shape[1:] == (4, 8, 8)
        return torch.nn.functional.interpolate(latents[:, :1], scale_factor=8)


@pytest.fixture(autouse=True)
def river_lib(monkeypatch):
    module = types.ModuleType("river_lib")
    module.KTHAutoencoder = MockAutoencoder
    monkeypatch.setitem(sys.modules, "river_lib", module)


@pytest.fixture
def data_dir(tmp_path):
    """A small kth_64.npz: persons 1, 2 (train) and 19 (not train), block-constant frames."""
    rng = np.random.default_rng(0)
    persons, lengths = [1, 2, 19], [12, 9, 11]
    blocks = rng.integers(0, 256, (sum(lengths), 8, 8), dtype=np.uint8)
    np.savez(
        tmp_path / "kth_64.npz",
        frames=np.kron(blocks, np.ones((1, 8, 8), dtype=np.uint8)),
        offsets=np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64),
        person=np.asarray(persons, dtype=np.int64),
        action=np.asarray(["walking"] * 3, dtype="<U12"),
        scenario=np.ones(3, dtype=np.int64),
        names=np.asarray([f"person{p:02d}_walking_d1" for p in persons], dtype="<U32"),
    )
    return tmp_path


def encode(data_dir, monkeypatch, *args):
    spec = importlib.util.spec_from_file_location(
        "encode_kth", ROOT / "scripts" / "encode_kth.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        sys,
        "argv",
        ["encode_kth.py", "--root", str(data_dir), "--device", "cpu", *args],
    )
    module.main()


def compose_case(data_dir, overrides=()):
    paths = [
        f"{split}.dataset.paths=[{data_dir}]" for split in ("train_data", "val_data")
    ]
    with initialize_config_dir(config_dir=str(ROOT / "config"), version_base=None):
        return compose(
            config_name="kth_latent",
            overrides=paths
            + [
                f"test_data.paths=[{data_dir}]",
                "train_data.num_workers=0",
                "val_data.num_workers=0",
                "train_data.batch_size=4",
                "val_data.batch_size=4",
                # The synthetic videos are short and have persons 1, 2 and 19 only.
                f"len_field_history={L}",
                "test_data.persons=[19]",
                "test_data.test_clip_length=6",
                "test_data.num_test_clips=3",
                # A small backbone keeps the CPU tests fast.
                "backbone.hidden_channels=[16,32,64]",
                "backbone.attention.embedding_dim=64",
                *overrides,
            ],
        )


def test_encode_script(data_dir, monkeypatch):
    encode(data_dir, monkeypatch, "--batch-size", "5")  # 32 frames: uneven batches
    with np.load(data_dir / "kth_64.npz") as data:
        pixels = {key: data[key] for key in data.files}
    with np.load(data_dir / "kth_64_latents.npz") as data:
        latents = {key: data[key] for key in data.files}
    assert set(latents) == set(pixels) | {"mean", "std"}
    for key in ("offsets", "person", "action", "scenario", "names"):
        assert np.array_equal(latents[key], pixels[key])
        assert latents[key].dtype == pixels[key].dtype
    frames = latents["frames"]
    assert frames.shape == (32, 4, 8, 8) and frames.dtype == np.float32
    # The mock latent channel c is the 8x8-pooled frame in [0, 1] plus c.
    expected = pixels["frames"][:, ::8, ::8] / 255.0
    for c in range(4):
        np.testing.assert_allclose(frames[:, c], expected + c, atol=1e-6)
    # Statistics over persons 1 and 2 only (the first 21 frames).
    assert latents["mean"].shape == latents["std"].shape == (4,)
    np.testing.assert_allclose(latents["mean"], frames[:21].mean((0, 2, 3)), atol=1e-6)
    np.testing.assert_allclose(latents["std"], frames[:21].std((0, 2, 3)), atol=1e-6)
    assert not np.allclose(latents["mean"], frames.mean((0, 2, 3)), atol=1e-4)


@pytest.mark.parametrize(
    "models, trainer_class",
    [("stochastic_interpolant", FlowTrainer), ("ito_map", MapTrainer)],
)
def test_config_trains_and_samples(
    data_dir, monkeypatch, tmp_path, models, trainer_class
):
    torch.set_num_threads(1)
    encode(data_dir, monkeypatch)
    cfg = compose_case(data_dir, [f"models={models}"])

    preprocesser = instantiate(cfg.preprocesser)
    train = instantiate(cfg.train_data, dataset={"preprocesser": preprocesser})
    val = instantiate(
        cfg.val_data, dataset={"preprocesser": preprocesser, "persons": [19]}
    )
    batch = next(iter(train))
    assert batch["field_history"].shape == (4, 4, 8, 8, L)
    assert batch["base"].shape == batch["target"].shape == (4, 4, 8, 8)

    # One optimizer step and one validation pass with the case's trainer.
    model = instantiate(cfg.model)
    is_map = trainer_class is MapTrainer
    trainer = instantiate(
        cfg.trainer,
        model=model,
        optimizer=instantiate(cfg.optimizer, params=model.parameters()),
        loss_fn=instantiate(cfg.loss_fn),
        train_dataloader=train,
        val_dataloader=val,
        device="cpu",
        checkpoint_dir=str(tmp_path / "run"),
        num_epochs=1,
        max_train_batches=1,
        max_val_batches=1,
        # ito_map: include the off-diagonal loss from the first epoch.
        **({"diagonal_epochs": 0, "warmup_epochs": 0} if is_map else {}),
    )
    assert isinstance(trainer, trainer_class)
    trainer.train()
    assert np.isfinite(trainer.history[-1]["train_loss"])
    if is_map:
        assert trainer.components["off_diagonal_loss"] > 0

    # Forecast a test clip as scripts/test.py does.
    test = instantiate(cfg.test_data)
    trajectory = test[0]["x"].unsqueeze(0)
    assert trajectory.shape == (1, 4, 8, 8, 6)
    init_data = preprocesser.transform(
        base=trajectory[..., L - 1], field_history=trajectory[..., :L], is_batch=True
    )
    model.eval()
    with torch.no_grad():
        predicted = model.sample_trajectory(
            base=init_data["base"],
            field_history=init_data["field_history"],
            batch_size=2,
            num_physical_steps=6,
            num_steps=2,
            **({} if is_map else {"stepper": heun_step}),
        )
    assert predicted.shape == (2, 4, 8, 8, 6)
    assert torch.isfinite(predicted).all()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_plotters_write_decoded_video(data_dir, monkeypatch, tmp_path):
    encode(data_dir, monkeypatch)
    test = instantiate(compose_case(data_dir).test_data)
    true = test[0]["x"]  # [4, 8, 8, 6]
    predicted = true + 0.01 * torch.randn(2, *true.shape)

    plot_case("kth_latent", true, predicted, test, str(tmp_path))
    assert (tmp_path / "video.mp4").stat().st_size > 0
    assert (tmp_path / "video.png").stat().st_size > 0
