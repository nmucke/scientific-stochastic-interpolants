import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "assumed_ns_reference",
    Path(__file__).parents[3]
    / "research/ito_maps/experiments/simulate_assumed_ns_reference.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_saved_leads_include_initial_and_nondivisible_final():
    assert module.saved_leads(130, 64) == [0, 64, 128, 130]
    assert module.saved_leads(128, 64) == [0, 64, 128]
    assert module.saved_leads(1, 64) == [0, 1]
    with pytest.raises(ValueError):
        module.saved_leads(0, 2)


def test_initial_state_uses_last_observed_full_resolution_frame():
    states = np.arange(2 * 5 * 256 * 256, dtype=np.float32).reshape(2, 5, 256, 256)
    times = np.arange(5)
    initial = module.validate_initial_state(states, times, 1, 3, 128)
    np.testing.assert_array_equal(initial, states[1, 2, ::2, ::2])
    initial[:] = 0
    assert np.any(states[1, 2])  # Never mutate the original archive.
    with pytest.raises(ValueError):
        module.validate_initial_state(states, times, 1, 6, 128)
    with pytest.raises(ValueError):
        module.validate_initial_state(states, times[:2], 1, 3, 128)
