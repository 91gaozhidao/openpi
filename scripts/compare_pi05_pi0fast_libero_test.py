"""Tests for the pi05 vs pi0_fast comparison utilities."""

import numpy as np

from scripts.compare_pi05_pi0fast_libero import Args
from scripts.compare_pi05_pi0fast_libero import compare_results
from scripts.compare_pi05_pi0fast_libero import create_dummy_observation


def test_create_dummy_observation():
    obs = create_dummy_observation()
    assert "observation/state" in obs
    assert "observation/image" in obs
    assert "observation/wrist_image" in obs
    assert "prompt" in obs
    assert obs["observation/state"].shape == (8,)
    assert obs["observation/image"].shape == (224, 224, 3)


def test_compare_results_basic():
    """Test that compare_results computes correct metrics."""
    args = Args()

    # Create matching trajectories
    actions = np.random.rand(10, 7).astype(np.float32)
    way1_result = {"actions": actions}
    way2_result = {"actions": actions.copy()}

    metrics = compare_results(way1_result, way2_result, args)

    assert "trajectory" in metrics
    assert metrics["trajectory"]["mse"] == 0.0
    assert metrics["trajectory"]["max_abs_error"] == 0.0
    assert metrics["trajectory"]["mean_per_step_l2"] == 0.0


def test_compare_results_with_difference():
    """Test that compare_results detects differences between trajectories."""
    args = Args()

    way1_actions = np.zeros((10, 7), dtype=np.float32)
    way2_actions = np.ones((10, 7), dtype=np.float32)

    way1_result = {"actions": way1_actions}
    way2_result = {"actions": way2_actions}

    metrics = compare_results(way1_result, way2_result, args)

    assert metrics["trajectory"]["mse"] > 0
    assert metrics["trajectory"]["max_abs_error"] == 1.0
    assert len(metrics["trajectory"]["per_step_l2"]) == 10


def test_compare_results_with_tokens():
    """Test that compare_results handles token metrics."""
    args = Args()

    way1_result = {"actions": np.random.rand(10, 7).astype(np.float32)}
    way2_result = {
        "actions": np.random.rand(10, 7).astype(np.float32),
        "generated_tokens": np.array([100, 200, 300, 1, 0, 0, 0]),
        "decoded_text": "Action: some tokens|",
    }

    metrics = compare_results(way1_result, way2_result, args)

    assert "tokens" in metrics
    assert metrics["tokens"]["total_generated"] == 4  # 100, 200, 300, 1 (non-zero)
    assert metrics["tokens"]["contains_eos"] is True
    assert metrics["tokens"]["unique_tokens"] == 4


def test_compare_results_different_shapes():
    """Test compare_results aligns different action shapes."""
    args = Args()

    way1_result = {"actions": np.random.rand(10, 7).astype(np.float32)}
    way2_result = {"actions": np.random.rand(5, 7).astype(np.float32)}

    metrics = compare_results(way1_result, way2_result, args)

    # Should compare only min(10, 5) = 5 steps
    assert len(metrics["trajectory"]["per_step_l2"]) == 5


def test_args_defaults():
    """Test that Args has sensible defaults."""
    args = Args()
    assert args.pi05_config_name == "pi05_libero"
    assert args.fast_action_dim == 7
    assert args.fast_action_horizon == 10
    assert args.ar_max_decode_steps == 256
    assert args.ar_temperature == 0.0
    assert args.seed == 42
