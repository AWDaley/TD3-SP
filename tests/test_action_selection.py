"""Control semantics for actor proposals, predicted rewards and critic tails."""

from types import SimpleNamespace
import json

import numpy as np
import pytest
import torch
from torch import nn

from action_selection.selector import ActionSelector, SelectionConfig
from action_selection.tasks import PendulumTask, task_for_env


class LinearActor(nn.Module):
    def __init__(self, slope=0.0, bias=0.0):
        super().__init__()
        self.slope = nn.Parameter(torch.tensor(float(slope)))
        self.bias = nn.Parameter(torch.tensor(float(bias)))
        self.seen = []

    def forward(self, states):
        self.seen.append(states.detach().clone())
        return states[:, :1] * self.slope + self.bias


class LinearCritic(nn.Module):
    def __init__(self, state_weight=1.0, action_weight=1.0, offset=10.0):
        super().__init__()
        self.state_weight = nn.Parameter(torch.tensor(float(state_weight)))
        self.action_weight = nn.Parameter(torch.tensor(float(action_weight)))
        self.offset = float(offset)
        self.seen = []

    def forward(self, states, actions):
        self.seen.append((states.detach().clone(), actions.detach().clone()))
        first = states[:, :1] * self.state_weight + actions * self.action_weight
        return first, first + self.offset


class AdditiveModel(nn.Module):
    state_dim = 1
    action_dim = 1

    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(1.0))
        self.seen = []

    def forward(self, states, actions):
        self.seen.append((states.detach().clone(), actions.detach().clone()))
        return states + self.gain * actions


def reward_next_state(states, actions, next_states):
    return next_states[:, 0], torch.zeros(len(states), dtype=torch.bool, device=states.device)


def make_selector(*, actor=None, critic=None, model=None, task=reward_next_state,
                  mode="lookahead", horizon=1, candidates=16, noise_std=0.2,
                  low=-10.0, high=10.0, seed=7, discount=0.5):
    policy = SimpleNamespace(
        actor=actor if actor is not None else LinearActor(),
        critic=critic if critic is not None else LinearCritic(),
        discount=discount,
    )
    selector = ActionSelector(
        policy, np.array([low]), np.array([high]),
        SelectionConfig(mode=mode, horizon=horizon, candidates=candidates,
                        noise_std=noise_std, seed=seed),
        predictor=model if model is not None else AdditiveModel(), task=task,
    )
    return selector


def test_one_step_score_includes_immediate_reward_and_future_action():
    def task(states, actions, next_states):
        return 2 * states[:, 0] + 3 * actions[:, 0], torch.zeros(len(states), dtype=torch.bool)

    selector = make_selector(actor=LinearActor(slope=0.25),
                             critic=LinearCritic(10, 4), task=task)
    actual = selector.score_candidates(np.array([1.0]), torch.tensor([[-0.5], [0.5]]))
    # s'=[0.5,1.5], next actor actions=[0.125,0.375], Q1=[5.5,16.5].
    # r=[0.5,3.5]; r + 0.5*Q1 = [3.25,11.75].
    torch.testing.assert_close(actual, torch.tensor([3.25, 11.75]))
    future_states, future_actions = selector.policy.critic.seen[-1]
    torch.testing.assert_close(future_states[:, 0], torch.tensor([0.5, 1.5]))
    torch.testing.assert_close(future_actions[:, 0], torch.tensor([0.125, 0.375]))


def test_critic_search_uses_the_lower_of_both_critics():
    class DisagreeingCritics(LinearCritic):
        def forward(self, states, actions):
            return 100 * actions, -10 * actions

    selector = make_selector(critic=DisagreeingCritics(), mode="critic")
    scores = selector.score_candidates([0.0], torch.tensor([[0.0], [0.5], [-0.5]]))
    torch.testing.assert_close(scores, torch.tensor([0.0, -5.0, -50.0]))


def test_nonfinite_member_of_critic_pair_excludes_the_candidate():
    class PartlyInvalidCritics(LinearCritic):
        def forward(self, states, actions):
            first = torch.where(actions > 0, torch.full_like(actions, float("inf")), actions)
            return first, actions + 10

    selector = make_selector(critic=PartlyInvalidCritics(), mode="critic")
    scores = selector.score_candidates([0.0], [[-1.0], [1.0]])
    assert float(scores[0]) == -1
    assert torch.isneginf(scores[1])


def test_multistep_rollout_recomputes_actor_from_each_predicted_state():
    model = AdditiveModel()
    selector = make_selector(actor=LinearActor(slope=0.5), model=model, horizon=3)
    score = selector.score_candidates([1.0], torch.tensor([[2.0]]))
    # States 1 -> 3 -> 4.5 -> 6.75. Continuation actions 1.5,2.25,
    # tail actor=3.375; Q=10.125. Rewards are the three next states.
    torch.testing.assert_close(score, torch.tensor([8.203125]))
    assert [float(actions[0, 0]) for _, actions in model.seen] == [2.0, 1.5, 2.25]


def test_true_terminal_branches_receive_reward_then_stop_without_bootstrap():
    class NoTerminalSteps(AdditiveModel):
        def forward(self, states, actions):
            assert (states[:, 0] < 2).all(), "A terminal branch was stepped again"
            return super().forward(states, actions)

    class NoTailCritic(LinearCritic):
        def forward(self, states, actions):
            raise AssertionError("All branches terminate; no critic bootstrap is permitted")

    def task(states, actions, next_states):
        done = next_states[:, 0] >= 2
        return torch.where(done, 10.0, 1.0), done

    model = NoTerminalSteps()
    selector = make_selector(actor=LinearActor(bias=1), critic=NoTailCritic(),
                             model=model, task=task, horizon=4)
    scores = selector.score_candidates([0.0], torch.tensor([[1.0], [3.0]]))
    torch.testing.assert_close(scores, torch.tensor([6.0, 10.0]))
    assert [len(states) for states, _ in model.seen] == [2, 1]


def test_planning_horizon_does_not_turn_nonterminal_states_into_terminals():
    def task(states, actions, next_states):
        return torch.ones(len(states)), torch.zeros(len(states), dtype=torch.bool)

    class ConstantCritic(LinearCritic):
        def forward(self, states, actions):
            value = torch.full((len(states), 1), 8.0)
            return value, value + 1

    selector = make_selector(task=task, critic=ConstantCritic(), horizon=2)
    torch.testing.assert_close(selector.score_candidates([0.0], [[0.0]]), torch.tensor([3.5]))


def test_zero_discount_only_scores_immediate_reward():
    class NoTailCritic(LinearCritic):
        def forward(self, states, actions):
            raise AssertionError("Zero discount cannot use a critic tail")

    actor, model = LinearActor(), AdditiveModel()
    selector = make_selector(actor=actor, critic=NoTailCritic(), model=model,
                             discount=0.0, horizon=4)
    scores = selector.score_candidates([0.0], [[-1.0], [1.0]])
    torch.testing.assert_close(scores, torch.tensor([-1.0, 1.0]))
    assert len(model.seen) == 1
    assert not actor.seen


def test_original_actor_candidate_wins_ties_and_all_actions_obey_bounds():
    critic = LinearCritic(0, 0)
    selector = make_selector(actor=LinearActor(bias=0.75), critic=critic,
                             mode="critic", noise_std=20, low=-0.2, high=0.2)
    action = selector.select_action(np.array([0.0]), exploration_std=2.0)
    np.testing.assert_allclose(action, [0.2])
    assert selector.last_info["selected_index"] == 0
    candidates = critic.seen[-1][1]
    assert (candidates >= -0.2).all() and (candidates <= 0.2).all()
    torch.testing.assert_close(candidates[0], torch.tensor([0.2]))


def test_per_axis_asymmetric_bounds_are_respected():
    class TwoAxisActor(nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = nn.Parameter(torch.tensor([3.0, -4.0]))

        def forward(self, states):
            return self.bias.expand(len(states), -1)

    class TwoAxisCritic(LinearCritic):
        def forward(self, states, actions):
            self.seen.append((states.detach().clone(), actions.detach().clone()))
            zeros = torch.zeros(len(states), 1)
            return zeros, zeros

    critic = TwoAxisCritic()
    selector = ActionSelector(
        SimpleNamespace(actor=TwoAxisActor(), critic=critic, discount=0.5),
        [-0.2, -2.0], [0.8, 4.0],
        SelectionConfig(mode="critic", noise_std=10, seed=4),
    )
    np.testing.assert_allclose(selector.select_action([0.0]), [0.8, -2.0])
    actions = critic.seen[-1][1]
    assert (actions >= torch.tensor([-0.2, -2.0])).all()
    assert (actions <= torch.tensor([0.8, 4.0])).all()


def test_continuation_actions_are_clipped_before_model_and_tail_critic():
    model = AdditiveModel()
    critic = LinearCritic()
    selector = make_selector(actor=LinearActor(bias=100), critic=critic,
                             model=model, horizon=2, low=-1, high=1)
    selector.score_candidates([0.0], [[-5.0]])
    assert [float(actions[0, 0]) for _, actions in model.seen] == [-1.0, 1.0]
    assert float(critic.seen[-1][1][0, 0]) == 1.0


def test_selector_randomness_is_reproducible_and_does_not_consume_global_streams():
    first = make_selector(mode="critic", low=-1, high=1, seed=19)
    second = make_selector(mode="critic", low=-1, high=1, seed=19)
    np.random.seed(144)
    torch.manual_seed(144)
    numpy_before = np.random.get_state()
    torch_before = torch.random.get_rng_state().clone()
    first_actions = [first.select_action([0.0], exploration_std=0.2) for _ in range(3)]
    second_actions = [second.select_action([0.0], exploration_std=0.2) for _ in range(3)]
    np.testing.assert_array_equal(first_actions, second_actions)
    numpy_after = np.random.get_state()
    assert numpy_before[0] == numpy_after[0]
    np.testing.assert_array_equal(numpy_before[1], numpy_after[1])
    assert numpy_before[2:] == numpy_after[2:]
    torch.testing.assert_close(torch_before, torch.random.get_rng_state())
    first.reset(seed=19)
    np.testing.assert_array_equal(first.select_action([0.0], exploration_std=0.2), first_actions[0])


def test_invalid_prediction_only_excludes_affected_candidates():
    class SelectivelyInvalid(AdditiveModel):
        def forward(self, states, actions):
            value = super().forward(states, actions)
            return torch.where(actions > 0, torch.full_like(value, float("nan")), value)

    selector = make_selector(model=SelectivelyInvalid())
    score = selector.score_candidates([0.0], [[-1.0], [1.0]])
    assert torch.isfinite(score[0])
    assert torch.isneginf(score[1])


def test_all_invalid_predictions_fall_back_to_actor_with_finite_json_diagnostics():
    class InvalidModel(AdditiveModel):
        def forward(self, states, actions):
            return torch.full_like(states, float("nan"))

    selector = make_selector(actor=LinearActor(bias=0.3), model=InvalidModel())
    np.testing.assert_allclose(selector.select_action([0.0]), [0.3])
    assert selector.last_info["fallback"]
    json.dumps(selector.last_info, allow_nan=False)


def test_scoring_preserves_parameters_gradients_and_training_flags():
    selector = make_selector(actor=LinearActor(slope=0.5), horizon=3)
    modules = [selector.policy.actor, selector.policy.critic, selector.predictor]
    modules[0].train()
    modules[1].eval()
    modules[2].train()
    parameters = [parameter for module in modules for parameter in module.parameters()]
    for parameter in parameters:
        parameter.grad = torch.full_like(parameter, 7.0)
    before = [(parameter.detach().clone(), parameter.grad.clone()) for parameter in parameters]
    modes = [module.training for module in modules]
    scores = selector.score_candidates([1.0], [[-1.0], [1.0]])
    assert not scores.requires_grad
    selector.select_action([1.0])
    for parameter, (value, grad) in zip(parameters, before):
        torch.testing.assert_close(parameter, value)
        torch.testing.assert_close(parameter.grad, grad)
    assert [module.training for module in modules] == modes


@pytest.mark.parametrize("kwargs", [
    {"candidates": 0}, {"horizon": 0}, {"noise_std": -0.1},
    {"noise_std": float("nan")}, {"mode": "unknown"},
])
def test_invalid_search_configuration_is_rejected(kwargs):
    with pytest.raises((ValueError, TypeError)):
        make_selector(**kwargs)


@pytest.mark.parametrize("state", [[float("nan")], [float("inf")], [[0.0]], [0.0, 0.0]])
def test_invalid_states_do_not_reach_action_execution(state):
    selector = make_selector()
    with pytest.raises(ValueError):
        selector.select_action(state)


def test_lookahead_requires_an_explicit_task_and_predictor():
    policy = SimpleNamespace(actor=LinearActor(), critic=LinearCritic(), discount=0.5)
    with pytest.raises(ValueError):
        ActionSelector(policy, [-1], [1], SelectionConfig(mode="lookahead"),
                       predictor=AdditiveModel())
    with pytest.raises(ValueError):
        ActionSelector(policy, [-1], [1], SelectionConfig(mode="lookahead"),
                       task=reward_next_state)


def test_bad_reward_or_terminal_shape_fails_instead_of_broadcasting():
    def task(states, actions, next_states):
        return torch.ones(len(states), 2), torch.zeros(len(states), dtype=torch.bool)

    selector = make_selector(task=task)
    with pytest.raises(ValueError):
        selector.score_candidates([0.0], [[0.0], [1.0]])


def test_terminal_flags_must_be_boolean_not_probabilities():
    def task(states, actions, next_states):
        return torch.ones(len(states)), torch.full((len(states),), 0.3)

    selector = make_selector(task=task)
    with pytest.raises(ValueError, match="boolean"):
        selector.score_candidates([0.0], [[0.0], [1.0]])


def test_pendulum_task_matches_real_rewards_and_preserves_truncation_bootstrap():
    import gymnasium as gym

    env = gym.make("Pendulum-v1", max_episode_steps=1)
    try:
        task = task_for_env(env)
        for angle in [-np.pi, -0.3, 0.0, 0.8, np.pi]:
            for torque in [-5.0, 0.0, 1.5, 5.0]:
                env.reset(seed=2)
                env.unwrapped.state = np.array([angle, 1.25])
                state = env.unwrapped._get_obs()
                action = np.array([torque], dtype=np.float32)
                next_state, real_reward, terminated, truncated, _ = env.step(action)
                predicted_reward, predicted_terminal = task(
                    torch.tensor(state).unsqueeze(0), torch.tensor(action).unsqueeze(0),
                    torch.tensor(next_state).unsqueeze(0),
                )
                assert float(predicted_reward[0]) == pytest.approx(real_reward, abs=2e-6)
                assert terminated is False and truncated is True
                assert predicted_terminal.dtype == torch.bool
                assert not bool(predicted_terminal[0])
    finally:
        env.close()


def test_pendulum_reward_uses_current_state_and_clips_torque():
    task = PendulumTask()
    states = torch.tensor([[1.0, 0.0, 2.0]])
    reward, done = task(states, torch.tensor([[8.0]]), torch.tensor([[-1.0, 0.0, 8.0]]))
    torch.testing.assert_close(reward, torch.tensor([-0.404]))
    assert not done.any()


def test_task_factory_rejects_unknown_environment_and_custom_reward_wrappers():
    import gymnasium as gym

    class DoubleReward(gym.RewardWrapper):
        def reward(self, reward):
            return 2 * reward

    other = gym.make("MountainCarContinuous-v0")
    wrapped = DoubleReward(gym.make("Pendulum-v1"))
    try:
        with pytest.raises(ValueError, match="custom task"):
            task_for_env(other)
        with pytest.raises(ValueError, match="Custom environment wrappers"):
            task_for_env(wrapped)
    finally:
        other.close()
        wrapped.close()


def test_pendulum_task_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        PendulumTask()(torch.zeros(2, 3), torch.zeros(2, 2), torch.zeros(2, 3))
