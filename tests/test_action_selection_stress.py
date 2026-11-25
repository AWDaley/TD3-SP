"""Independent dynamics oracles and adversarial control API regression tests."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from action_selection import ActionSelector, SelectionConfig
import TD3


@pytest.fixture(scope="module", autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class PointActor(nn.Module):
    def __init__(self, dimensions):
        super().__init__()
        self.dimensions = dimensions
        self.gain = nn.Parameter(torch.tensor(0.7, dtype=torch.float64))

    def forward(self, states):
        return -self.gain * states[:, :self.dimensions] - 0.2 * states[:, self.dimensions:]


class PointModel(nn.Module):
    def __init__(self, dimensions):
        super().__init__()
        self.state_dim, self.action_dim = 2 * dimensions, dimensions
        self.register_buffer("dt", torch.tensor(0.25, dtype=torch.float64))

    def forward(self, states, actions):
        positions, velocities = states[:, :self.action_dim], states[:, self.action_dim:]
        return torch.cat((positions + self.dt * velocities + self.dt.square() / 2 * actions,
                          0.9 * velocities + self.dt * actions), dim=1)


class PointCritic(nn.Module):
    def __init__(self, dimensions):
        super().__init__()
        self.dimensions = dimensions
        self.register_buffer("weight", torch.tensor(0.3, dtype=torch.float64))

    def forward(self, states, actions):
        first = (-self.weight * states[:, :self.dimensions].square().sum(dim=1)
                 - 0.2 * states[:, self.dimensions:].square().sum(dim=1)
                 - 0.1 * actions.square().sum(dim=1))
        return first[:, None], (first + 0.17 * actions.sum(dim=1))[:, None]


def point_task(states, actions, next_states):
    dimensions = actions.shape[1]
    positions = next_states[:, :dimensions]
    reward = (-positions.square().sum(dim=1) - 0.03 * actions.square().sum(dim=1)
              - 0.05 * states[:, dimensions:].square().sum(dim=1))
    return reward, torch.linalg.vector_norm(positions, dim=1) > 1.25


def point_selector(dimensions=2, horizon=3, discount=0.8, **kwargs):
    policy = SimpleNamespace(actor=PointActor(dimensions), critic=PointCritic(dimensions),
                             discount=discount)
    return ActionSelector(
        policy, np.linspace(-1.3, -0.7, dimensions), np.linspace(0.6, 1.2, dimensions),
        SelectionConfig(mode="lookahead", horizon=horizon, seed=19),
        predictor=PointModel(dimensions), task=kwargs.get("task", point_task),
    )


def numpy_point_return(state, first_action, low, high, horizon, discount):
    """Independent scalar physical-system calculation, without selector calls."""
    dimensions = len(first_action)
    position, velocity = state[:dimensions].copy(), state[dimensions:].copy()
    total = 0.0
    for step in range(horizon):
        action = first_action if step == 0 else -0.7 * position - 0.2 * velocity
        action = np.clip(action, low, high)
        next_position = position + 0.25 * velocity + 0.03125 * action
        next_velocity = 0.9 * velocity + 0.25 * action
        reward = -sum(next_position ** 2) - 0.03 * sum(action ** 2) - 0.05 * sum(velocity ** 2)
        total += discount ** step * reward
        if np.linalg.norm(next_position) > 1.25:
            return total
        position, velocity = next_position, next_velocity
    action = np.clip(-0.7 * position - 0.2 * velocity, low, high)
    first = -0.3 * sum(position ** 2) - 0.2 * sum(velocity ** 2) - 0.1 * sum(action ** 2)
    second = first + 0.17 * sum(action)
    return total + discount ** horizon * min(first, second)


@pytest.mark.parametrize("dimensions,horizon,discount", [
    (1, 1, 0.99), (1, 5, 0.0), (2, 4, 0.5), (3, 7, 1.0), (3, 3, 0.99),
])
def test_batched_scores_match_independent_physics_and_scalar_scoring(dimensions, horizon, discount):
    selector = point_selector(dimensions, horizon, discount)
    rng = np.random.default_rng(29)
    state = np.concatenate((np.linspace(1.1, -0.3, dimensions),
                            np.linspace(0.5, -0.2, dimensions)))
    candidates = rng.uniform(-3, 3, size=(11, dimensions))
    low, high = selector.low.numpy(), selector.high.numpy()
    expected = np.array([numpy_point_return(state, action, low, high, horizon, discount)
                         for action in candidates])
    generator_before = copy.deepcopy(selector.rng.bit_generator.state)
    actual = selector.score_candidates(state, candidates)
    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-12, atol=1e-12)
    serial = torch.cat([selector.score_candidates(state, action[None]) for action in candidates])
    torch.testing.assert_close(actual, serial, rtol=1e-12, atol=1e-12)
    permutation = rng.permutation(len(candidates))
    permuted = selector.score_candidates(state, candidates[permutation])
    torch.testing.assert_close(permuted, actual[permutation], rtol=1e-12, atol=1e-12)
    assert selector.rng.bit_generator.state == generator_before
    assert actual.dtype == torch.float64 and not actual.requires_grad


def test_float64_action_bounds_are_a_snapshot_of_caller_arrays():
    low, high = np.array([-1.0]), np.array([1.0])
    policy = SimpleNamespace(actor=PointActor(1), critic=PointCritic(1), discount=0.9)
    selector = ActionSelector(policy, low, high, SelectionConfig(mode="critic", seed=19))
    reference = ActionSelector(policy, [-1.0], [1.0], SelectionConfig(mode="critic", seed=19))
    low[0], high[0] = 10.0, 11.0
    np.testing.assert_array_equal(selector.low.numpy(), [-1.0])
    np.testing.assert_array_equal(selector.high.numpy(), [1.0])
    np.testing.assert_array_equal(selector.select_action([0.2, 0.1]),
                                  reference.select_action([0.2, 0.1]))


@pytest.mark.parametrize("source", ["reward", "critic"])
def test_complex_scores_cannot_hide_nonfinite_imaginary_components(source):
    def complex_task(states, actions, next_states):
        reward, done = point_task(states, actions, next_states)
        return torch.complex(reward, torch.full_like(reward, float("nan"))), done

    class ComplexCritic(PointCritic):
        def forward(self, states, actions):
            first, second = super().forward(states, actions)
            return torch.complex(first, torch.full_like(first, float("nan"))), second

    selector = point_selector(task=complex_task if source == "reward" else point_task)
    if source == "critic":
        selector.policy.critic = ComplexCritic(2)
    with pytest.raises(ValueError, match="real-valued"):
        selector.score_candidates([0.1, 0.2, 0.0, 0.0], [[0.0, 0.0]])


def test_mixed_terminal_times_keep_rewards_and_only_bootstrap_survivors():
    class Actor(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("one", torch.tensor(1.0))
            self.batch_sizes = []

        def forward(self, states):
            assert (states[:, 0] < 3).all(), "Actor was called after termination"
            self.batch_sizes.append(len(states))
            return self.one.expand(len(states), 1)

    class Model(nn.Module):
        state_dim = action_dim = 1

        def __init__(self):
            super().__init__()
            self.batch_sizes = []

        def forward(self, states, actions):
            assert (states[:, 0] < 3).all(), "Model was called after termination"
            self.batch_sizes.append(len(states))
            return states + actions

    class Critic(nn.Module):
        def forward(self, states, actions):
            torch.testing.assert_close(states, torch.tensor([[0.0]]))
            value = torch.full((len(states), 1), 7.0)
            return value, value + 1

    def task(states, actions, next_states):
        done = next_states[:, 0] >= 3
        return torch.where(done, 10.0, 1.0), done

    actor, model = Actor(), Model()
    selector = ActionSelector(
        SimpleNamespace(actor=actor, critic=Critic(), discount=0.8), [-5], [5],
        SelectionConfig(mode="lookahead", horizon=4), predictor=model, task=task,
    )
    score = selector.score_candidates([0.0], [[4.0], [2.0], [0.0], [-3.0]])
    expected = torch.tensor([10, 1 + 0.8 * 10, 1 + 0.8 + 0.8 ** 2 + 0.8 ** 3 * 10,
                             1 + 0.8 + 0.8 ** 2 + 0.8 ** 3 + 0.8 ** 4 * 7])
    torch.testing.assert_close(score, expected)
    assert model.batch_sizes == [4, 3, 2, 2]
    assert actor.batch_sizes == [3, 2, 2, 1]


@pytest.mark.parametrize("fault", ["predictor", "actor", "reward", "Q1", "Q2"])
def test_late_nonfinite_fault_excludes_only_its_own_branch(fault):
    # State is [clock, branch marker]. The marker is the first chosen action
    # and is retained thereafter, so every injected fault occurs AFTER step 1.
    class Actor(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("one", torch.tensor(1.0))

        def forward(self, states):
            actions = self.one.expand(len(states), 1).clone()
            if fault == "actor":
                bad = (states[:, 0] == 1) & (states[:, 1] > 0)
                actions[bad] = torch.nan
            return actions

    class Model(nn.Module):
        state_dim, action_dim = 2, 1

        def forward(self, states, actions):
            marker = torch.where(states[:, 0] == 0, actions[:, 0], states[:, 1])
            predicted = torch.stack((states[:, 0] + 1, marker), dim=1)
            if fault == "predictor":
                predicted[(states[:, 0] == 1) & (states[:, 1] > 0)] = torch.inf
            return predicted

    class Critic(nn.Module):
        def forward(self, states, actions):
            first = torch.full((len(states), 1), 4.0)
            second = torch.full_like(first, 5.0)
            if fault in {"Q1", "Q2"}:
                bad = (states[:, 0] == 3) & (states[:, 1] > 0)
                (first if fault == "Q1" else second)[bad] = torch.inf
            return first, second

    def task(states, actions, next_states):
        rewards = torch.ones(len(states))
        if fault == "reward":
            rewards[(states[:, 0] == 1) & (states[:, 1] > 0)] = torch.nan
        return rewards, torch.zeros(len(states), dtype=torch.bool)

    selector = ActionSelector(
        SimpleNamespace(actor=Actor(), critic=Critic(), discount=0.5), [-2], [2],
        SelectionConfig(mode="lookahead", horizon=3), predictor=Model(), task=task,
    )
    scores = selector.score_candidates([0.0, 0.0], [[1.0], [-1.0]])
    assert torch.isneginf(scores[0])
    assert float(scores[1]) == 2.25  # Three unit rewards plus the value-4 tail.


def test_selector_recovers_after_an_all_invalid_prediction_call():
    class SwitchableModel(PointModel):
        invalid = True

        def forward(self, states, actions):
            if self.invalid:
                return torch.full_like(states, torch.nan)
            return super().forward(states, actions)

    selector = point_selector()
    selector.predictor = SwitchableModel(2)
    state = np.array([0.1, 0.2, 0.0, 0.0])
    action = selector.select_action(state)
    actor_action = selector.policy.actor(torch.from_numpy(state)[None]).detach().numpy()[0]
    np.testing.assert_array_equal(action, actor_action)
    previous_info = dict(selector.last_info)
    assert previous_info["fallback"] and previous_info["selected_score"] is None
    assert previous_info["invalid_candidates"] == selector.config.candidates
    selector.predictor.invalid = False
    action = selector.select_action(state)
    assert np.isfinite(action).all()
    assert not selector.last_info["fallback"]
    assert selector.last_info["invalid_candidates"] == 0
    assert np.isfinite(selector.last_info["selected_score"])
    assert previous_info["fallback"]  # Previous diagnostics remain a snapshot.


@pytest.mark.parametrize("input_kind", ["readonly_numpy_views", "tensor_views"])
def test_scoring_does_not_mutate_observation_or_candidate_inputs(input_kind):
    selector = point_selector()
    if input_kind == "readonly_numpy_views":
        state = np.array([0.0, 0.1, 0.2, 0.3], dtype=np.float64)[::-1]
        candidates = np.arange(8, dtype=np.float64).reshape(4, 2)[:, ::-1]
        state.setflags(write=False)
        candidates.setflags(write=False)
        before_state, before_candidates = state.copy(), candidates.copy()
    else:
        state = torch.tensor([0.1, 0.0, 0.2, 0.0], dtype=torch.float64, requires_grad=True)
        candidates = torch.arange(8, dtype=torch.float64).reshape(2, 4).T.detach().requires_grad_()
        state.grad = torch.ones_like(state)
        candidates.grad = torch.full_like(candidates, 3.0)
        before_state, before_candidates = state.detach().clone(), candidates.detach().clone()
    scores = selector.score_candidates(state, candidates)
    assert not scores.requires_grad
    selector.select_action(state)
    if input_kind == "readonly_numpy_views":
        np.testing.assert_array_equal(state, before_state)
        np.testing.assert_array_equal(candidates, before_candidates)
        assert not state.flags.writeable and not candidates.flags.writeable
    else:
        torch.testing.assert_close(state, before_state)
        torch.testing.assert_close(candidates, before_candidates)
        torch.testing.assert_close(state.grad, torch.ones_like(state))
        torch.testing.assert_close(candidates.grad, torch.full_like(candidates, 3.0))


def assert_nested_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_nested_equal(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for item, reference in zip(actual, expected):
            assert_nested_equal(item, reference)
    else:
        assert actual == expected


@pytest.mark.parametrize("task_raises", [False, True])
def test_batchnorm_dropout_optimizer_and_rng_state_survive_scoring(task_raises):
    class Actor(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(2, 8), nn.BatchNorm1d(8), nn.ReLU(),
                                     nn.Dropout(0.4), nn.Linear(8, 1), nn.Tanh())

        def forward(self, states):
            return self.net(states)

    class Critic(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(3, 8), nn.BatchNorm1d(8),
                                     nn.Dropout(0.3), nn.Linear(8, 2))

        def forward(self, states, actions):
            result = self.net(torch.cat((states, actions), dim=1))
            return result[:, :1], result[:, 1:]

    class Model(nn.Module):
        state_dim, action_dim = 2, 1

        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(3, 8), nn.BatchNorm1d(8),
                                     nn.Dropout(0.2), nn.Linear(8, 2))

        def forward(self, states, actions):
            return states + 0.01 * self.net(torch.cat((states, actions), dim=1))

    class Task(nn.Module):
        def forward(self, states, actions, next_states):
            if task_raises:
                raise RuntimeError("injected task failure")
            return -next_states.square().sum(dim=1), torch.zeros(len(states), dtype=torch.bool)

    actor, critic, model, task = Actor(), Critic(), Model(), Task()
    networks = [actor, critic, model, task]
    parameters = [parameter for network in networks for parameter in network.parameters()]
    optimizer = torch.optim.Adam(parameters, lr=0.001)
    sum(parameter.square().sum() for parameter in parameters).backward()
    optimizer.step()  # Give Adam real, nonempty moment/step state to preserve.
    actor.net[1].eval()
    critic.eval()
    critic.net[2].train()
    modes = [(module, module.training) for network in networks for module in network.modules()]
    weights = [copy.deepcopy(network.state_dict()) for network in networks]
    gradients = [parameter.grad.clone() for parameter in parameters]
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    numpy_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()
    selector = ActionSelector(
        SimpleNamespace(actor=actor, critic=critic, discount=0.9), [-1], [1],
        SelectionConfig(mode="lookahead", horizon=3), predictor=model, task=task,
    )
    if task_raises:
        with pytest.raises(RuntimeError, match="injected task failure"):
            selector.select_action([0.1, 0.2])
    else:
        selector.select_action([0.1, 0.2], exploration_std=0.1)
    assert torch.is_grad_enabled()
    for module, training in modes:
        assert module.training == training
    for network, before in zip(networks, weights):
        assert_nested_equal(network.state_dict(), before)
    for parameter, before in zip(parameters, gradients):
        torch.testing.assert_close(parameter.grad, before, rtol=0, atol=0)
    assert_nested_equal(optimizer.state_dict(), optimizer_before)
    np_after = np.random.get_state()
    assert numpy_before[0] == np_after[0] and numpy_before[2:] == np_after[2:]
    np.testing.assert_array_equal(numpy_before[1], np_after[1])
    torch.testing.assert_close(torch.get_rng_state(), torch_before, rtol=0, atol=0)


@pytest.mark.parametrize("fault", ["wrong_shape", "nonfinite"])
def test_failed_evaluation_checkpoint_load_is_atomic(tmp_path, fault):
    policy = TD3.TD3(2, 1, 1.0)
    networks = [policy.actor, policy.critic, policy.actor_target, policy.critic_target]
    before = [copy.deepcopy(network.state_dict()) for network in networks]
    actor_state = {name: value + 1 for name, value in before[0].items()}
    critic_state = copy.deepcopy(before[1])
    if fault == "wrong_shape":
        critic_state["l1.weight"] = torch.zeros(1)
        error = RuntimeError
    else:
        critic_state["l1.weight"][0, 0] = torch.nan
        error = ValueError
    prefix = str(tmp_path / "bad-policy")
    torch.save(actor_state, prefix + "_actor")
    torch.save(critic_state, prefix + "_critic")
    parameter_ids = [id(parameter) for network in networks for parameter in network.parameters()]
    with pytest.raises(error):
        policy.load_for_evaluation(prefix)
    for network, expected in zip(networks, before):
        assert_nested_equal(network.state_dict(), expected)
    assert parameter_ids == [id(parameter) for network in networks for parameter in network.parameters()]


@pytest.mark.parametrize("fault", ["actor_shape", "model_shape", "critic_pair", "task_pair"])
def test_malformed_component_outputs_fail_explicitly_and_restore_modes(fault):
    selector = point_selector()
    original_actor = selector.policy.actor.forward
    original_model = selector.predictor.forward
    if fault == "actor_shape":
        selector.policy.actor.forward = lambda states: original_actor(states)[:, 0]
    elif fault == "model_shape":
        selector.predictor.forward = lambda states, actions: original_model(states, actions)[:, :1]
    elif fault == "critic_pair":
        selector.policy.critic.forward = lambda states, actions: torch.ones(len(states))
    else:
        selector.task = lambda states, actions, next_states: torch.ones(len(states))
    networks = [selector.policy.actor, selector.policy.critic, selector.predictor]
    for network in networks:
        network.train()
    with pytest.raises(ValueError):
        selector.select_action([0.1, 0.2, 0.0, 0.0])
    assert all(network.training for network in networks)
