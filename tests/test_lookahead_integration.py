"""Check the action actually executed/recorded and checkpoint compatibility."""

from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

import main
import TD3
from action_selection.configuration import load_prediction_model
from state_prediction.data import TransitionDataset
from state_prediction.model import StatePredictor


def test_actor_only_training_action_preserves_original_noise_path():
    policy = SimpleNamespace(select_action=lambda state: np.array([0.7, -0.5]))
    before = np.random.get_state()
    try:
        np.random.seed(19)
        expected = (policy.select_action(None) + np.random.normal(0, 0.6, size=2)).clip(-2, 2)
        np.random.seed(19)
        actual = main.select_training_action(policy, [1], 2, 2.0, 0.3)
        np.testing.assert_array_equal(actual, expected)
    finally:
        np.random.set_state(before)


def test_no_noise_is_added_after_selector(monkeypatch):
    received = []

    def select(state, exploration_std):
        received.append((state.copy(), exploration_std))
        return np.array([0.75], dtype=np.float32)

    def reject_noise(*args, **kwargs):
        raise AssertionError("Exploration must happen inside the selector")

    monkeypatch.setattr(np.random, "normal", reject_noise)
    actual = main.select_training_action(
        None, [1, 2, 3], 1, 2.0, 0.4, SimpleNamespace(select_action=select),
    )
    np.testing.assert_array_equal(actual, [0.75])
    np.testing.assert_array_equal(received[0][0], [1, 2, 3])
    assert received[0][1] == 0.4


def test_training_evaluation_replay_and_archive_use_selected_actions(monkeypatch, tmp_path):
    import action_selection.configuration as configuration
    import action_selection.tasks as tasks

    environments, selectors, replay_rows, policies = [], [], [], []

    class Environment:
        def __init__(self):
            self.observation_space = gym.spaces.Box(-10, 10, shape=(3,), dtype=np.float32)
            self.action_space = gym.spaces.Box(-2, 2, shape=(1,), dtype=np.float32)
            self.actions = []
            self.closed = False

        def reset(self, seed=None):
            self.t = 0
            return np.zeros(3, dtype=np.float32), {}

        def step(self, action):
            self.actions.append(np.array(action, copy=True))
            self.t += 1
            return np.full(3, self.t, dtype=np.float32), float(action[0]), False, self.t == 2, {}

        def close(self):
            self.closed = True

    def make_env(name):
        environment = Environment()
        environments.append(environment)
        return environment

    class Policy:
        def __init__(self, **kwargs):
            self.updates = 0
            policies.append(self)

        def select_action(self, state):
            raise AssertionError("Lookahead was bypassed")

        def train(self, replay, batch_size):
            self.updates += 1

    class Replay:
        def __init__(self, *args, **kwargs):
            pass

        def add(self, state, action, next_state, reward, done):
            replay_rows.append((np.array(action, copy=True), reward, done))

    class Selector:
        def __init__(self, config):
            self.config = config
            self.exploration = []
            self.last_info = {}

        def select_action(self, state, exploration_std=0.0):
            self.exploration.append(exploration_std)
            self.last_info = {"intervention": True, "fallback": False,
                              "invalid_candidates": 0, "elapsed_ms": 1.0}
            return np.array([0.75], dtype=np.float32)

    def factory(policy, env, config, predictor):
        selector = Selector(config)
        selectors.append(selector)
        return selector

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(main, "make_env", make_env)
    monkeypatch.setattr(main.TD3, "TD3", Policy)
    monkeypatch.setattr(main.utils, "ReplayBuffer", Replay)
    monkeypatch.setattr(configuration, "load_prediction_model", lambda *args: object())
    monkeypatch.setattr(configuration, "selector_for_env", factory)
    monkeypatch.setattr(tasks, "task_for_env", lambda env: object())
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"fixture for checkpoint provenance only")
    archive = tmp_path / "executed.npz"
    main.main([
        "--env", "Test-v0", "--action_selection", "lookahead",
        "--prediction_checkpoint", str(checkpoint), "--prediction_data", str(archive),
        "--max_timesteps", "4", "--start_timesteps", "0", "--eval_freq", "2",
        "--eval_episodes", "1", "--batch_size", "1", "--replay_size", "8",
        "--expl_noise", "0.4",
    ])
    assert policies[0].updates == 4
    assert len(environments[0].actions) == 4
    assert selectors[0].exploration == [0.4] * 4
    assert all(selector.exploration == [0.0, 0.0] for selector in selectors[1:])
    assert selectors[0].config.seed != selectors[1].config.seed
    assert all(env.closed for env in environments)
    for action, reward, terminal in replay_rows:
        np.testing.assert_array_equal(action, [0.75])
        assert reward == 0.75
        assert terminal == 0.0  # Time-limit truncation still bootstraps in TD3.
    data = TransitionDataset.load(archive)
    np.testing.assert_array_equal(data.actions, np.full((4, 1), 0.75))
    np.testing.assert_array_equal(data.terminated, [False] * 4)
    np.testing.assert_array_equal(data.truncated, [False, True, False, True])
    assert (tmp_path / "results/TD3_Test-v0_0_lookahead_k16_h1_selection.json").is_file()


def test_evaluation_closes_environment_when_selector_construction_fails(monkeypatch):
    closed = []
    env = SimpleNamespace(close=lambda: closed.append(True))
    monkeypatch.setattr(main, "make_env", lambda name: env)

    def invalid(env):
        raise ValueError("unsupported reward")

    with pytest.raises(ValueError, match="unsupported reward"):
        main.eval_policy(None, "unsupported", 0, selector_factory=invalid)
    assert closed == [True]


def test_evaluation_loader_needs_only_weights_and_updates_targets(tmp_path):
    policy = TD3.TD3(3, 1, 2.0)
    prefix = tmp_path / "policy"
    torch.save(policy.actor.state_dict(), str(prefix) + "_actor")
    torch.save(policy.critic.state_dict(), str(prefix) + "_critic")
    actor_weights = {name: value.clone() for name, value in policy.actor.state_dict().items()}
    with torch.no_grad():
        for value in policy.actor.parameters():
            value.zero_()
    parameter_ids = [id(value) for value in policy.actor.parameters()]
    policy.load_for_evaluation(prefix)
    assert [id(value) for value in policy.actor.parameters()] == parameter_ids
    for name, expected in actor_weights.items():
        torch.testing.assert_close(policy.actor.state_dict()[name], expected)
        torch.testing.assert_close(policy.actor_target.state_dict()[name], expected)
    for name, expected in policy.critic.state_dict().items():
        torch.testing.assert_close(policy.critic_target.state_dict()[name], expected)
    assert not Path(str(prefix) + "_actor_optimizer").exists()


def test_prediction_loader_rejects_unfitted_or_mismatched_models(tmp_path):
    env = gym.make("Pendulum-v1")
    try:
        policy = TD3.TD3(3, 1, 2.0)
        checkpoint = tmp_path / "predictor.pt"
        StatePredictor(3, 1, hidden_dims=(8,)).save(checkpoint)
        with pytest.raises(ValueError, match="fitted"):
            load_prediction_model(checkpoint, policy, env)
        StatePredictor(2, 1, hidden_dims=(8,)).save(checkpoint)
        with pytest.raises(ValueError, match="dimensions"):
            load_prediction_model(checkpoint, policy, env)
    finally:
        env.close()


@pytest.mark.parametrize("args, message", [
    (["--action_selection", "lookahead"], "requires --prediction_checkpoint"),
    (["--action_selection", "critic", "--policy", "DDPG"], "supported for --policy TD3"),
    (["--action_selection", "critic", "--planning_candidates", "1"], "at least two"),
    (["--prediction_checkpoint", "unused.pt"], "used only with --action_selection lookahead"),
])
def test_invalid_cli_combinations_fail_before_environment_creation(monkeypatch, capsys, args, message):
    def unexpected(name):
        raise AssertionError("Invalid arguments should fail before an environment is created")
    monkeypatch.setattr(main, "make_env", unexpected)
    with pytest.raises(SystemExit) as error:
        main.main(args)
    assert error.value.code == 2
    assert message in capsys.readouterr().err
