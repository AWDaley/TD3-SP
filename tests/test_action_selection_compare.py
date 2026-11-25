"""Behavioral checks for paired frozen-controller evaluation and report output."""

import copy
import hashlib
import json
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch import nn

import TD3
from action_selection import compare
from state_prediction.data import TransitionDataset
from state_prediction.model import StatePredictor


@pytest.fixture(scope="module", autouse=True)
def single_threaded_torch():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class ZeroActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1, bias=False)
        nn.init.zeros_(self.linear.weight)

    def forward(self, states):
        return self.linear(states)


class QuadraticCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, states, actions):
        value = -(states.square() + actions.square()) * self.scale
        return value, value + 1


class ExactPredictor(nn.Module):
    state_dim = action_dim = 1
    normalizers_fitted = True

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))

    def forward(self, states, actions):
        return states + actions + self.anchor


class ThreeStepEnvironment:
    def __init__(self):
        self.observation_space = gym.spaces.Box(-10.0, 10.0, shape=(1,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.seeds = []
        self.actions = []
        self.closed = False

    def reset(self, seed=None):
        self.seeds.append(seed)
        self.state = np.array([0.7 + 0.01 * (seed % 5)], dtype=np.float32)
        self.steps = 0
        return self.state.copy(), {}

    def step(self, action):
        assert np.all(action >= -1) and np.all(action <= 1)
        self.actions.append(action.copy())
        self.state += action
        reward = -float(self.state[0] ** 2 + 0.01 * action[0] ** 2)
        self.steps += 1
        return self.state.copy(), reward, False, self.steps == 3, {}

    def close(self):
        self.closed = True


def transition_task(states, actions, next_states):
    reward = -(next_states[:, 0].square() + 0.01 * actions[:, 0].square())
    return reward, torch.zeros_like(reward, dtype=torch.bool)


def policy_and_predictor():
    return SimpleNamespace(actor=ZeroActor(), critic=QuadraticCritic(), discount=0.9), ExactPredictor()


def run_small_comparison(**kwargs):
    policy, predictor = policy_and_predictor()
    environments = []

    def factory(name):
        assert name == "test-control"
        environment = ThreeStepEnvironment()
        environments.append(environment)
        return environment

    report = compare.compare_policy(
        policy, predictor, env_name="test-control", episodes=3, seed=11,
        candidates=16, noise_std=0.5, env_factory=factory,
        task_factory=lambda env: transition_task, **kwargs,
    )
    return report, environments


def test_comparison_pairs_episode_seeds_and_reports_observed_returns():
    report, environments = run_small_comparison()
    assert len(environments) == 3
    assert all(env.seeds == [11, 12, 13] for env in environments)
    assert all(env.closed for env in environments)
    assert all(len(env.actions) == 9 for env in environments)
    for mode in compare.MODES:
        result = report["modes"][mode]
        assert result["summary"]["total_steps"] == 9
        assert result["summary"]["latency_ms"]["p95"] >= 0
        assert result["summary"]["mean_return"] == pytest.approx(
            np.mean([episode["return"] for episode in result["episodes"]])
        )
        assert [e["candidate_seed"] for e in result["episodes"]] == report["settings"]["candidate_seeds"]
        assert all(e["truncated"] and not e["terminated"] for e in result["episodes"])
    # The real fixture dynamics/reward exactly match the model and task. The
    # actor stands still, while the one-step planner can move toward zero.
    assert report["paired_return_differences"]["lookahead_minus_actor"]["mean"] > 0
    assert report["paired_return_differences"]["critic_minus_actor"]["mean"] == pytest.approx(0)
    for key, values in report["paired_return_differences"].items():
        treatment, reference = key.split("_minus_")
        expected = [a["return"] - b["return"] for a, b in zip(
            report["modes"][treatment]["episodes"], report["modes"][reference]["episodes"])]
        assert values["per_episode"] == expected
        assert values["mean"] == pytest.approx(np.mean(expected))
    json.dumps(report, allow_nan=False)


def test_repeat_comparison_reproduces_actions_and_returns_without_global_rng_draws():
    torch.manual_seed(30)
    np.random.seed(31)
    # Constructing tiny policy fixtures uses PyTorch initialization; freeze its
    # random stream to isolate evaluation's behavior.
    with torch.random.fork_rng(devices=[]):
        first, first_envs = run_small_comparison(horizon=3)
        second, second_envs = run_small_comparison(horizon=3)
    numpy_before = np.random.get_state()
    run_small_comparison()
    numpy_after = np.random.get_state()
    np.testing.assert_array_equal(numpy_before[1], numpy_after[1])
    assert numpy_before[2:] == numpy_after[2:]
    for mode, env1, env2 in zip(compare.MODES, first_envs, second_envs):
        np.testing.assert_array_equal(env1.actions, env2.actions)
        assert [e["return"] for e in first["modes"][mode]["episodes"]] == [
            e["return"] for e in second["modes"][mode]["episodes"]]


def test_evaluation_preserves_weights_gradients_and_mixed_module_modes():
    policy, predictor = policy_and_predictor()
    policy.actor.train()
    policy.actor.linear.eval()
    policy.critic.eval()
    networks = (policy.actor, policy.critic, predictor)
    weights = [copy.deepcopy(network.state_dict()) for network in networks]
    modes = [[module.training for module in network.modules()] for network in networks]
    for network in networks:
        for parameter in network.parameters():
            parameter.grad = torch.ones_like(parameter)
    torch_before = torch.random.get_rng_state().clone()
    compare.compare_policy(policy, predictor, episodes=1,
                           env_factory=lambda name: ThreeStepEnvironment(),
                           task_factory=lambda env: transition_task)
    assert torch.equal(torch_before, torch.random.get_rng_state())
    for network, saved, training_modes in zip(networks, weights, modes):
        assert [module.training for module in network.modules()] == training_modes
        for key, value in network.state_dict().items():
            assert torch.equal(value, saved[key])
        for parameter in network.parameters():
            assert torch.equal(parameter.grad, torch.ones_like(parameter))


def test_environment_closes_and_modes_restore_when_evaluation_fails():
    policy, predictor = policy_and_predictor()
    environment = ThreeStepEnvironment()
    original_step = environment.step

    def bad_step(action):
        state, _, terminated, truncated, info = original_step(action)
        return state, float("nan"), terminated, truncated, info

    environment.step = bad_step
    with pytest.raises(ValueError, match="nonfinite reward"):
        compare.compare_policy(policy, predictor, episodes=1,
                               env_factory=lambda name: environment,
                               task_factory=lambda env: transition_task)
    assert environment.closed
    assert policy.actor.training and predictor.training


def test_unfitted_or_wrong_dimension_model_rejected_before_stepping():
    policy, predictor = policy_and_predictor()
    predictor.normalizers_fitted = False
    with pytest.raises(ValueError, match="fitted normalizers"):
        compare.compare_policy(policy, predictor, env_factory=lambda name: pytest.fail("must not create env"))
    predictor.normalizers_fitted = True
    predictor.state_dim = 2
    environment = ThreeStepEnvironment()
    with pytest.raises(ValueError, match="dimensions"):
        compare.compare_policy(policy, predictor, env_factory=lambda name: environment)
    assert environment.closed
    assert environment.seeds == []


def test_report_cannot_overwrite_existing_file(tmp_path):
    output = tmp_path / "comparison.json"
    output.write_text("preserve this", encoding="utf-8")
    with pytest.raises(ValueError, match="already exists"):
        compare.compare_checkpoints("missing-policy", "missing-model.pt", output)
    assert output.read_text(encoding="utf-8") == "preserve this"


def test_cli_loads_only_actor_and_critic_and_writes_reproducible_provenance(tmp_path, monkeypatch, capsys):
    prefix = tmp_path / "policy"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        policy = TD3.TD3(3, 1, 2.0)
    torch.save(policy.actor.state_dict(), str(prefix) + "_actor")
    torch.save(policy.critic.state_dict(), str(prefix) + "_critic")
    # No optimizer checkpoints are created: evaluation must not require them.
    model = StatePredictor(3, 1, hidden_dims=(8,), seed=7)
    states = np.array([[1, 0, 0], [0.99, 0.1, 0.2]], dtype=np.float32)
    data = TransitionDataset(states, np.zeros((2, 1), dtype=np.float32),
                             states.copy(), np.array([0, 1]))
    model.fit_normalizers(data)
    checkpoint = tmp_path / "predictor.pt"
    model.save(checkpoint)
    monkeypatch.setattr(compare, "make_env", lambda name: gym.make(name, max_episode_steps=4))
    output = tmp_path / "reports" / "comparison.json"
    report = compare.main([
        "compare", "--policy_checkpoint", str(prefix),
        "--prediction_checkpoint", str(checkpoint), "--output", str(output),
        "--episodes", "2", "--seed", "73", "--candidates", "4", "--threads", "1",
    ])
    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert report["settings"]["episode_seeds"] == [73, 74]
    assert all(v["summary"]["total_steps"] == 8 for v in report["modes"].values())
    for identity in report["checkpoints"].values():
        from pathlib import Path
        assert identity["sha256"] == hashlib.sha256(Path(identity["path"]).read_bytes()).hexdigest()
    assert "do not establish a training improvement" in capsys.readouterr().out


@pytest.mark.parametrize("option,value,expected", [
    ("--episodes", "0", "episodes"),
    ("--seed", "-1", "seed"),
    ("--horizon", "0", "horizon"),
    ("--threads", "0", "threads"),
])
def test_cli_rejects_invalid_settings_before_checkpoint_access(option, value, expected, tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        compare.main([
            "compare", "--policy_checkpoint", "missing-policy",
            "--prediction_checkpoint", "missing-predictor", "--output", str(tmp_path / "out.json"),
            option, value,
        ])
    assert error.value.code == 2
    assert expected in capsys.readouterr().err
