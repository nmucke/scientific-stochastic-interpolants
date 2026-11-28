"""Checks of the WP1 step-1 report on synthetic eval outputs."""

import importlib.util
import json
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).parents[1] / "experiments" / "wp1_step1_report.py"
spec = importlib.util.spec_from_file_location("wp1_step1_report", SCRIPT)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def test_shell_mean_of_constant_power_is_constant():
    for size in (8, 9):
        power = np.full((1, size, size // 2 + 1), 3.0)
        np.testing.assert_allclose(report.shell_mean(power, size), 3.0)


def write_run(root, name, omitted_scale, spread_scale):
    rng = np.random.default_rng(len(name))
    for trajectory, origin in [(160, 5), (161, 5), (162, 5)]:
        out = root / name / "path_compression" / f"traj{trajectory}_origin{origin}"
        out.mkdir(parents=True)
        results = [
            {
                "num_path_features": k,
                "omitted_variance_fraction_estimate": omitted_scale / k,
                "within_variance_standard_error_across_groups": 0.001,
                "total_conditional_forecast_variance_estimate": 0.2,
                "within_coarse_path_variance": 0.2 * omitted_scale / k,
            }
            for k in (1, 4)
        ]
        meta = {"trajectory": trajectory, "origin": origin, "interpolation": "X", "noise_covariance": "Y",
                "teacher_steps": 8, "results": results}
        (out / "results.json").write_text(json.dumps(meta))
    euler = root / name / "euler_exact_noise"
    euler.mkdir(parents=True)
    (euler / "results.json").write_text(json.dumps({"ensemble_size": 4}))
    lines = []
    for trajectory in (160, 161, 162):
        for origin in (5, 25):
            steps = {}
            for n in (1, 2, 8):
                spread = (spread_scale if n < 8 else 1.0) * (1 + 0.1 * rng.random())
                steps[str(n)] = {"rms_spread": spread, "fair_crps": 0.3 / spread, "ensemble_mean_rmse": 1.0,
                                 "anomaly_shell_spectrum": list(spread**2 * np.ones(12))}
            lines.append(json.dumps({"trajectory": trajectory, "origin": origin, "steps": steps}))
    (euler / "cases.jsonl").write_text("\n".join(lines))


def test_report_on_synthetic_runs(tmp_path):
    write_run(tmp_path, "white", 0.2, 0.5)
    write_run(tmp_path, "col", 0.4, 0.5)
    log = tmp_path / "train.log"
    log.write_text("x\rEpoch 0 | lr 1e-4, train_loss 0.5, val_loss 0.4 | best\n"
                   "Epoch 1 | lr 1e-4, train_loss 0.3, val_loss 0.45 | best\nEarly stopping triggered at epoch 1\n")
    output = tmp_path / "report.json"
    report.main(["--eval-root", str(tmp_path), "--run", "white_t2=white", "--run", "col=col",
                 "--steps", "1", "2", "--reference-steps", "8", "--bootstrap", "50",
                 "--log", f"col={log}", "--output", str(output)])
    data = json.loads(output.read_text())
    pc = data["path_compression"]["col"]["K4"]
    np.testing.assert_allclose(pc["omitted_fraction_across_histories"]["mean"], 0.1)
    np.testing.assert_allclose(pc["paired_difference_vs_white_t2"]["mean"], 0.05)
    euler = data["euler_exact_noise"]
    assert euler["num_cases"] == 6 and euler["trajectories"] == [160, 161, 162]
    ratio = euler["models"]["white_t2"]["1"]["spread_ratio_to_reference"]
    assert 0.4 < ratio["estimate"] < 0.6 and ratio["ci95"][0] <= ratio["estimate"] <= ratio["ci95"][1]
    np.testing.assert_allclose(euler["models"]["white_t2"]["8"]["spread_ratio_to_reference"]["estimate"], 1.0)
    band = euler["models"]["white_t2"]["8"]["anomaly_band_power_ratio_k1-4"]["estimate"]
    np.testing.assert_allclose(band, 1.0)
    training = data["training"]["col"]
    assert training["best_epoch"] == 0 and training["early_stopped"] and training["num_epochs"] == 2


def test_spectrum_rescaling_matches_resolution(tmp_path):
    # Same physical spectrum on 8x8 and 16x16 grids: per-mode power scales with H*W.
    for size, name in ((8, "small"), (16, "large")):
        sigma = np.zeros((1, size, size // 2 + 1))
        sigma[0, 1, 0] = sigma[0, -1, 0] = size * size * 0.5
        np.savez(tmp_path / f"{name}.npz", sigma=sigma, q1=np.float64(1 / 3))
    out = report.spectrum_report(tmp_path / "small.npz", tmp_path / "large.npz")
    np.testing.assert_allclose(out["reference"]["per_mode_power_rescale_to_this_grid"], 0.25)
    np.testing.assert_allclose(out["reference"]["shell_ratio_this_over_rescaled_reference"][1], 1.0)
    np.testing.assert_allclose(out["total_residual_variance_per_pixel"], out["reference"]["total_residual_variance_per_pixel"])
