"""End-to-end checks for reproducible reports and validation-gated updates."""

import hashlib
import json

import numpy as np
import pytest
import torch

from state_prediction.collection import TransitionRecorder, collect_environment
from state_prediction.data import TransitionDataset
from state_prediction.evaluation import evaluate_predictor
from state_prediction.model import StatePredictor
from state_prediction.toy import make_toy_dataset
from state_prediction.workflow import (
    evaluate_experiment,
    improve_experiment,
    restore_splits,
    train_experiment,
)


@pytest.fixture(scope="module", autouse=True)
def single_threaded_torch():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def trained_run(tmp_path_factory):
    directory = tmp_path_factory.mktemp("prediction-experiment")
    data = make_toy_dataset(episodes=12, steps=16, seed=4)
    data_path = directory / "data.npz"
    data.save(data_path)
    output = directory / "initial"
    report = train_experiment(
        data_path, output, hidden_dims=(16, 16), learning_rate=0.003,
        seed=17, epochs=4, batch_size=32, patience=4,
        horizons=(1, 3, 5), max_windows=3,
    )
    return data, data_path, output, report


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_saved_checkpoint_exactly_reproduces_held_out_report(trained_run, tmp_path):
    data, data_path, output, report = trained_run
    checkpoint = output / "model.pt"
    original_digest = digest(checkpoint)
    result_path = tmp_path / "evaluation.json"
    repeated = evaluate_experiment(data_path, checkpoint, output=result_path)
    assert repeated["split"] == "test"
    assert repeated["transitions"] == report["splits"]["test"]["transitions"]
    assert repeated["transitions"] < len(data)
    assert repeated["metrics"] == report["test"]["state_predictor"]
    assert json.loads(result_path.read_text()) == repeated
    assert json.loads((output / "report.json").read_text()) == report
    assert (output / "report.md").is_file()
    assert (output / "history.csv").is_file()
    assert digest(checkpoint) == original_digest
    saved_model = StatePredictor.load(checkpoint)
    validation = restore_splits(data, saved_model.metadata)["validation"]
    validation_score = evaluate_predictor(
        saved_model, validation, horizons=(1, 3, 5), max_windows=3, seed=17
    )["selection_score"]
    assert validation_score == pytest.approx(report["fit"]["best_validation_score"])


def test_changed_dataset_cannot_reuse_the_frozen_split(trained_run, tmp_path):
    data, _, output, _ = trained_run
    changed = TransitionDataset(
        data.states + 0.25, data.actions, data.next_states + 0.25,
        data.episode_ids, data.terminated, data.truncated,
    )
    path = tmp_path / "changed.npz"
    changed.save(path)
    checkpoint = output / "model.pt"
    with pytest.raises(ValueError, match="Dataset differs"):
        evaluate_experiment(path, checkpoint)
    with pytest.raises(ValueError, match="Dataset differs"):
        improve_experiment(path, checkpoint, tmp_path / "rejected")
    assert not (tmp_path / "rejected" / "model.pt").exists()


def test_external_flag_does_not_relabel_original_data_as_independent(trained_run):
    _, data_path, output, _ = trained_run
    with pytest.raises(ValueError, match="training dataset"):
        evaluate_experiment(data_path, output / "model.pt", external=True)


def test_new_external_data_is_explicitly_labeled_unverified(trained_run, tmp_path):
    _, _, output, _ = trained_run
    data = make_toy_dataset(episodes=3, steps=8, seed=234)
    path = tmp_path / "independent.npz"
    data.save(path)
    report = evaluate_experiment(path, output / "model.pt", external=True)
    assert report["split"] == "external_independence_not_verified"
    assert report["transitions"] == len(data)


def test_improve_uses_fixed_validation_windows_and_preserves_rejected_incumbent(
    trained_run, tmp_path
):
    data, data_path, original_output, original_report = trained_run
    checkpoint = original_output / "model.pt"
    before = digest(checkpoint)
    # Changing the minibatch seed must not change which validation windows are
    # scored. This cap is deliberately much smaller than the validation set.
    result = improve_experiment(
        data_path, checkpoint, tmp_path / "candidate", seed=93, epochs=3,
        batch_size=32, patience=3, learning_rate=0.001,
        rollout_loss_weight=0.2, rollout_horizon=3, min_improvement=1_000_000,
    )
    decision = result["promotion"]
    assert decision["decision"] == "kept_incumbent"
    assert result["fit"]["initial_validation_score"] == pytest.approx(
        decision["incumbent_validation_score"], abs=1e-12
    )
    assert result["fit"]["best_validation_score"] == pytest.approx(
        decision["candidate_validation_score"], abs=1e-12
    )
    assert result["evaluation"] == original_report["evaluation"]
    assert result["test"] == original_report["test"]
    assert digest(checkpoint) == before
    incumbent = StatePredictor.load(checkpoint)
    saved = StatePredictor.load(tmp_path / "candidate" / "model.pt")
    np.testing.assert_array_equal(
        saved.predict(data.states[:8], data.actions[:8]),
        incumbent.predict(data.states[:8], data.actions[:8]),
    )
    assert saved.metadata["last_improvement"]["decision"] == "kept_incumbent"


def test_improvement_decision_matches_validation_gate(trained_run, tmp_path):
    _, data_path, output, _ = trained_run
    checkpoint = output / "model.pt"
    before = digest(checkpoint)
    result = improve_experiment(
        data_path, checkpoint, tmp_path / "candidate", seed=31, epochs=5,
        batch_size=32, patience=3, learning_rate=0.002,
        rollout_loss_weight=0.1, rollout_horizon=3,
    )
    promotion = result["promotion"]
    accepted = (
        promotion["incumbent_validation_score"]
        - promotion["candidate_validation_score"] > 1e-12
    )
    assert promotion["decision"] == (
        "accepted_candidate" if accepted else "kept_incumbent"
    )
    expected_score = promotion[
        "candidate_validation_score" if accepted else "incumbent_validation_score"
    ]
    assert result["validation"]["selection_score"] == pytest.approx(expected_score)
    assert digest(checkpoint) == before


def test_existing_run_cannot_be_overwritten_by_an_improvement(trained_run):
    _, data_path, output, _ = trained_run
    checkpoint = output / "model.pt"
    before = digest(checkpoint)
    with pytest.raises(ValueError, match="already exists"):
        improve_experiment(data_path, checkpoint, output)
    assert digest(checkpoint) == before


@pytest.mark.parametrize("existing_output", ["checkpoint", "dataset", "report"])
def test_evaluation_cannot_overwrite_existing_data(trained_run, existing_output):
    _, data_path, output, _ = trained_run
    checkpoint = output / "model.pt"
    target = {
        "checkpoint": checkpoint,
        "dataset": data_path,
        "report": output / "report.json",
    }[existing_output]
    before = digest(target)
    with pytest.raises(ValueError, match="already exists"):
        evaluate_experiment(data_path, checkpoint, output=target)
    assert digest(target) == before


def test_markdown_report_discloses_unavailable_validation_and_test_horizons(tmp_path):
    data = make_toy_dataset(episodes=3, steps=2, seed=2)
    path = tmp_path / "short-episodes.npz"
    data.save(path)
    output = tmp_path / "short-run"
    report = train_experiment(
        path, output, hidden_dims=(8,), epochs=1, batch_size=4, patience=1,
        horizons=(1, 3, 5), max_windows=3,
    )
    assert report["validation"]["skipped_horizons"] == [3, 5]
    assert report["test"]["state_predictor"]["skipped_horizons"] == [3, 5]
    markdown = (output / "report.md").read_text()
    for split in ("validation", "test"):
        disclosure = next(line for line in markdown.splitlines()
                          if f"Unavailable {split} horizons" in line)
        assert "3, 5" in disclosure
        assert "too short" in disclosure


def test_wrapped_recorder_exports_chronological_partial_episodes():
    recorder = TransitionRecorder(1, 1, capacity=4)
    for episode in range(3):
        for step in range(3):
            state = episode * 100 + step
            recorder.add(
                [state], [0], [state + 1], episode,
                terminated=step == 2 and episode % 2 == 0,
                truncated=step == 2 and episode % 2 == 1,
            )
    data = recorder.dataset()
    np.testing.assert_array_equal(data.states[:, 0], [102, 200, 201, 202])
    np.testing.assert_array_equal(data.episode_ids, [1, 2, 2, 2])
    starts, _, targets = data.rollout_windows(2)
    np.testing.assert_array_equal(starts[:, 0], [200, 201])
    np.testing.assert_array_equal(targets[:, -1, 0], [202, 203])
    snapshot_hash = data.fingerprint()
    recorder.add([300], [0], [301], 3)
    assert data.fingerprint() == snapshot_hash
    np.testing.assert_array_equal(recorder.dataset().states[:, 0], [200, 201, 202, 300])


def test_collector_marks_its_own_cutoff_without_losing_final_state(monkeypatch):
    import gymnasium as gym
    import env_compat

    class EndlessEnvironment:
        observation_space = gym.spaces.Box(-1000, 1000, shape=(1,), dtype=np.float32)
        action_space = gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float32)
        closed = False

        def reset(self, seed=None):
            self.state = np.array([0], dtype=np.float32)
            return self.state.copy(), {}

        def step(self, action):
            self.state = self.state + 1
            return self.state.copy(), 0.0, False, False, {}

        def close(self):
            self.closed = True

    env = EndlessEnvironment()
    monkeypatch.setattr(env_compat, "make_env", lambda name: env)
    data = collect_environment("fake", episodes=3, max_steps_per_episode=2, seed=7)
    assert env.closed
    assert len(data) == 6
    assert data.episode_count == 3
    np.testing.assert_array_equal(data.truncated, [False, True] * 3)
    assert not data.terminated.any()
    np.testing.assert_array_equal(data.next_states[:, 0], [1, 2] * 3)
    assert data.rollout_windows(2)[0].shape[0] == 3
