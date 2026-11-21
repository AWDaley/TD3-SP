"""Behavioral checks for prediction quality and honest held-out evaluation."""

import numpy as np
import pytest
import torch

from state_prediction.data import TransitionDataset
from state_prediction.evaluation import (
    LinearDeltaPredictor,
    PersistencePredictor,
    evaluate_predictor,
)
from state_prediction.model import StatePredictor


@pytest.fixture(scope="module", autouse=True)
def single_threaded_torch():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def linear_episodes(episodes=18, steps=24, seed=4, action_sign=1.0):
    """A stable, fully observed system with a known effect of each action."""
    rng = np.random.default_rng(seed)
    states, actions, next_states, episode_ids, terminal = [], [], [], [], []
    matrix = np.array([[0.82, 0.30], [-0.15, 0.65]], dtype=np.float32)
    control = np.array([0.70, 0.35], dtype=np.float32) * action_sign
    for episode in range(episodes):
        state = rng.uniform(-1.0, 1.0, 2).astype(np.float32)
        for step in range(steps):
            action = rng.uniform(-1.0, 1.0, 1).astype(np.float32)
            next_state = (matrix @ state + control * action[0]).astype(np.float32)
            states.append(state.copy())
            actions.append(action)
            next_states.append(next_state.copy())
            episode_ids.append(episode)
            terminal.append(step == steps - 1)
            state = next_state
    return TransitionDataset(
        states, actions, next_states, episode_ids, terminated=terminal
    )


def increment_episodes(lengths=(4, 4, 4), final_partial=False):
    states, targets, ids, terminated, truncated = [], [], [], [], []
    for episode, length in enumerate(lengths):
        for step in range(length):
            value = float(episode * 100 + step)
            states.append([value])
            targets.append([value + 1])
            ids.append(episode)
            ended = step == length - 1
            if final_partial and episode == len(lengths) - 1:
                ended = False
            terminated.append(ended and episode % 2 == 0)
            truncated.append(ended and episode % 2 == 1)
    return TransitionDataset(
        states,
        np.zeros((len(states), 1), dtype=np.float32),
        targets,
        ids,
        terminated=terminated,
        truncated=truncated,
    )


def test_episode_splits_are_disjoint_complete_and_reproducible():
    dataset = linear_episodes()
    splits = dataset.split_by_episode(seed=17)
    repeated = dataset.split_by_episode(seed=17)
    groups = {
        name: set(part.episode_ids.tolist()) for name, part in splits.items()
    }
    assert set(groups) == {"train", "validation", "test"}
    assert all(groups.values())
    assert groups["train"].isdisjoint(groups["validation"])
    assert groups["train"].isdisjoint(groups["test"])
    assert groups["validation"].isdisjoint(groups["test"])
    assert set.union(*groups.values()) == set(dataset.episode_ids.tolist())
    assert sum(map(len, splits.values())) == len(dataset)
    for name in splits:
        assert splits[name].fingerprint() == repeated[name].fingerprint()
        # Splitting must retain complete trajectories, not just random rows.
        assert len(splits[name]) == 24 * len(groups[name])


def test_split_requires_three_independent_episodes():
    with pytest.raises(ValueError):
        linear_episodes(episodes=2).split_by_episode()
    splits = linear_episodes(episodes=3).split_by_episode()
    assert all(len(np.unique(part.episode_ids)) == 1 for part in splits.values())


def test_dataset_roundtrip_preserves_data_boundaries_and_fingerprint(tmp_path):
    dataset = increment_episodes(lengths=(2, 3, 4), final_partial=True)
    path = tmp_path / "transitions.npz"
    dataset.save(path)
    restored = TransitionDataset.load(path)
    assert dataset.fingerprint() == restored.fingerprint()
    for field in (
        "states", "actions", "next_states", "episode_ids", "terminated", "truncated"
    ):
        np.testing.assert_array_equal(getattr(dataset, field), getattr(restored, field))


def test_rollouts_never_cross_terminal_or_time_limit_resets():
    dataset = increment_episodes(lengths=(2, 3, 2), final_partial=True)
    starts, actions, targets = dataset.rollout_windows(2)
    # Episode lengths 2, 3, 2 supply 1 + 2 + 1 valid windows.
    assert starts.shape == (4, 1)
    assert actions.shape == (4, 2, 1)
    assert targets.shape == (4, 2, 1)
    np.testing.assert_array_equal(starts[:, 0], [0, 100, 101, 200])
    np.testing.assert_array_equal(targets[:, -1, 0], starts[:, 0] + 2)
    assert dataset.rollout_windows(3)[0].shape == (1, 1)
    assert dataset.rollout_windows(4)[0].shape == (0, 1)
    # A terminal next state is a valid prediction target at the final step.
    assert dataset.rollout_windows(1)[0].shape[0] == len(dataset)


def test_rollout_sampling_is_capped_and_reproducible():
    dataset = linear_episodes()
    first = dataset.rollout_windows(5, max_windows=7, seed=11)
    second = dataset.rollout_windows(5, max_windows=7, seed=11)
    assert first[0].shape[0] == 7
    for actual, expected in zip(first, second):
        np.testing.assert_array_equal(actual, expected)


def test_dataset_rejects_discontinuity_and_transitions_after_an_end():
    with pytest.raises(ValueError):
        TransitionDataset([[0], [50]], [[0], [0]], [[1], [51]], [0, 0])
    with pytest.raises(ValueError):
        TransitionDataset(
            [[0], [1]], [[0], [0]], [[1], [2]], [0, 0], truncated=[True, False]
        )
    with pytest.raises(ValueError):
        TransitionDataset(
            [[0], [100], [1]], [[0], [0], [0]], [[1], [101], [2]], [0, 1, 0]
        )


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_dataset_rejects_nonfinite_measurements(value):
    with pytest.raises(ValueError):
        TransitionDataset([[value]], [[0]], [[1]], [0])


def test_persistence_metrics_have_known_one_step_and_open_loop_errors():
    report = evaluate_predictor(
        PersistencePredictor(),
        increment_episodes(),
        state_scale=np.array([2.0]),
        horizons=(1, 2, 4, 8),
    )
    assert report["one_step"]["rmse"] == pytest.approx(1.0)
    assert report["one_step"]["mae"] == pytest.approx(1.0)
    assert report["one_step"]["bias"] == pytest.approx(-1.0)
    assert report["one_step"]["normalized_rmse"] == pytest.approx(0.5)
    assert report["one_step"]["per_dimension"][0]["rmse"] == pytest.approx(1.0)
    assert report["rollout"]["2"]["rmse"] == pytest.approx(2.0)
    assert report["rollout"]["4"]["rmse"] == pytest.approx(4.0)
    assert report["selection_score"] == pytest.approx((0.5 + 1.0 + 2.0) / 3)
    assert report["skipped_horizons"] == [8]


def test_linear_baseline_learns_known_dynamics_on_unseen_episodes():
    splits = linear_episodes().split_by_episode(seed=5)
    baseline = LinearDeltaPredictor.fit(splits["train"])
    report = evaluate_predictor(baseline, splits["test"], horizons=(1, 5))
    assert report["one_step"]["rmse"] < 1e-3
    assert report["rollout"]["5"]["rmse"] < 2e-3


def test_normalizers_use_training_data_even_with_shifted_validation():
    training = linear_episodes(episodes=4, seed=1)
    validation = linear_episodes(episodes=2, seed=9)
    # Shift validation to make accidental train+validation scaling conspicuous.
    validation = TransitionDataset(
        validation.states + 100,
        validation.actions,
        validation.next_states + 100,
        validation.episode_ids,
        terminated=validation.terminated,
    )
    model = StatePredictor(2, 1, hidden_dims=(16, 16), seed=8)
    model.fit(
        training, validation, epochs=1, batch_size=64, patience=1,
        horizons=(1,), max_windows=16, seed=8,
    )
    np.testing.assert_allclose(
        model.state_scale, np.std(training.states, axis=0), rtol=1e-5
    )
    assert np.max(model.state_scale) < 2


def test_constant_state_feature_has_finite_unit_normalization():
    data = TransitionDataset(
        [[0, 2], [1, 2], [2, 2]],
        [[0], [0], [0]],
        [[1, 2], [2, 2], [3, 2]],
        [0, 0, 0],
    )
    model = StatePredictor(2, 1, hidden_dims=(8,), seed=2)
    model.fit_normalizers(data)
    assert model.state_scale[1] == pytest.approx(1)
    assert np.isfinite(model.predict(data.states, data.actions)).all()


@pytest.fixture(scope="module")
def trained_predictor():
    splits = linear_episodes().split_by_episode(seed=3)
    model = StatePredictor(
        2, 1, hidden_dims=(32, 32), learning_rate=0.005, seed=12
    )
    training = model.fit(
        splits["train"], splits["validation"], epochs=45, batch_size=64,
        patience=10, seed=12, horizons=(1, 3, 5), max_windows=48,
    )
    return model, splits, training


def test_learning_improves_predictions_on_held_out_episodes(trained_predictor):
    model, splits, training = trained_predictor
    learned = evaluate_predictor(
        model, splits["test"], horizons=(1, 3, 5), max_windows=48
    )
    persistence = evaluate_predictor(
        PersistencePredictor(), splits["test"], state_scale=model.state_scale,
        horizons=(1, 3, 5), max_windows=48,
    )
    assert learned["selection_score"] < 0.4 * persistence["selection_score"]
    assert training["best_validation_score"] < training["initial_validation_score"]
    assert training["epochs_run"] > 0
    controls = np.array([[-0.8], [0.8]], dtype=np.float32)
    states = np.repeat([[0.2, -0.3]], 2, axis=0).astype(np.float32)
    predictions = model.predict(states, controls)
    # A predictor which merely copies the state cannot pass this check.
    assert predictions[1, 0] - predictions[0, 0] > 0.6


def test_rollout_feeds_predicted_states_back_into_model(trained_predictor):
    model, _, _ = trained_predictor
    initial = np.array([0.1, -0.2], dtype=np.float32)
    controls = np.array([[0.5], [-0.2], [0.8]], dtype=np.float32)
    expected = []
    state = initial.copy()
    for action in controls:
        state = model.predict(state, action)
        expected.append(state)
    actual = model.rollout(initial, controls)
    assert actual.shape == (3, 2)
    np.testing.assert_allclose(actual, np.asarray(expected), rtol=1e-5, atol=1e-6)
    batch = model.rollout(initial[None, :], controls[None, :, :])
    assert batch.shape == (1, 3, 2)
    np.testing.assert_allclose(batch[0], actual, atol=1e-6)


def test_checkpoint_roundtrip_preserves_forecasts_and_metadata(
    trained_predictor, tmp_path
):
    model, splits, _ = trained_predictor
    path = tmp_path / "predictor.pt"
    metadata = {"dataset_fingerprint": "test-data", "split_episode_ids": [1, 3]}
    model.save(path, metadata=metadata)
    restored = StatePredictor.load(path, device="cpu")
    states, controls, _ = splits["test"].rollout_windows(5, max_windows=4)
    np.testing.assert_allclose(
        restored.rollout(states, controls), model.rollout(states, controls), atol=1e-7
    )
    np.testing.assert_array_equal(restored.state_scale, model.state_scale)
    assert restored.metadata == metadata


def test_fine_tuning_restores_incumbent_if_training_degrades_validation(
    trained_predictor, tmp_path
):
    model, splits, _ = trained_predictor
    path = tmp_path / "incumbent.pt"
    model.save(path)
    candidate = StatePredictor.load(path, device="cpu")
    state_scale = candidate.state_scale.copy()
    initial = evaluate_predictor(
        candidate, splits["validation"], horizons=(1, 3), max_windows=32, seed=2
    )["selection_score"]
    probe_states = splits["validation"].states[:4]
    probe_actions = splits["validation"].actions[:4]
    initial_predictions = candidate.predict(probe_states, probe_actions)
    # Train against a deliberately incompatible actuator mapping. Validation
    # must keep the best checkpoint instead of blindly saving the final epoch.
    incompatible = linear_episodes(episodes=8, action_sign=-1, seed=22)
    result = candidate.fit(
        incompatible, splits["validation"], epochs=5, batch_size=32,
        patience=3, horizons=(1, 3), max_windows=32, seed=2,
        initialize_normalizers=False,
    )
    final = evaluate_predictor(
        candidate, splits["validation"], horizons=(1, 3), max_windows=32, seed=2
    )["selection_score"]
    assert final <= initial + 1e-6
    assert result["best_epoch"] == 0
    assert result["best_validation_score"] == pytest.approx(final, abs=1e-6)
    np.testing.assert_array_equal(
        candidate.predict(probe_states, probe_actions), initial_predictions
    )
    np.testing.assert_array_equal(candidate.state_scale, state_scale)


def test_multistep_training_produces_finite_forecasts_and_best_checkpoint():
    splits = linear_episodes(episodes=6, steps=12).split_by_episode(seed=3)
    model = StatePredictor(2, 1, hidden_dims=(16, 16), seed=6)
    result = model.fit(
        splits["train"], splits["validation"], epochs=2, batch_size=16,
        patience=2, horizons=(1, 3), max_windows=12, seed=6,
        rollout_loss_weight=0.2, rollout_horizon=3,
    )
    initial, controls, _ = splits["test"].rollout_windows(3)
    assert np.isfinite(model.rollout(initial, controls)).all()
    assert result["best_validation_score"] <= result["initial_validation_score"]
