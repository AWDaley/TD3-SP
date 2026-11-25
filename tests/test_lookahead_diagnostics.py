"""Validate the independent diagnostic oracle and finite-return reference."""

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch import nn

from experiments.lookahead_diagnostics import (
    ExactPendulum, actor_reference_returns, held_out_states, ranking_metrics,
    validate_actor_reference, validate_oracle,
)


def test_oracle_matches_installed_gym_at_nominal_and_shifted_gravity():
    report = validate_oracle(samples=80)
    assert report["max_absolute_observation_error"] <= 5e-7
    assert report["max_absolute_reward_error"] <= 2e-12


@pytest.mark.parametrize("gravity", [10.0, 15.0])
def test_recursive_oracle_matches_real_gym_with_identical_fixed_actions(gravity):
    env = gym.make("Pendulum-v1", g=gravity)
    rng = np.random.default_rng(781)
    oracle = ExactPendulum(gravity=gravity)
    try:
        env.reset(seed=92)
        for _ in range(6):
            theta, velocity = rng.uniform(-np.pi, np.pi), rng.uniform(-8, 8)
            env.unwrapped.state = np.array([theta, velocity])
            state = torch.tensor([[np.cos(theta), np.sin(theta), velocity]], dtype=torch.float64)
            for _ in range(25):
                action = rng.uniform(-2, 2, size=1)
                state = oracle(state, torch.as_tensor(action[None], dtype=torch.float64))
                observation, *_ = env.unwrapped.step(action)
                np.testing.assert_allclose(state.numpy()[0], observation, atol=5e-7, rtol=0)
    finally:
        env.close()


class ConstantActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.control = nn.Parameter(torch.tensor([0.23]))

    def forward(self, states):
        return self.control.expand(len(states), 1)


def test_batched_candidate_return_matches_independent_environment_rollout():
    actor = ConstantActor()
    states, _ = held_out_states(seed=123, states_per_group=1)
    candidates = np.array([[[-2.], [0.], [2.]]] * len(states))
    before = actor.control.detach().clone()
    expected = np.empty((len(states), 3))
    actual = actor_reference_returns(actor, states, candidates, horizon=12, discount=0.97)
    env = gym.make("Pendulum-v1")
    try:
        for row, state in enumerate(states):
            for column, first_action in enumerate(candidates[row]):
                env.reset(seed=1)
                env.unwrapped.state = np.array([np.arctan2(state[1], state[0]), state[2]])
                total = 0.0
                for step in range(12):
                    action = first_action if step == 0 else actor.control.detach().numpy()
                    _, reward, *_ = env.unwrapped.step(action)
                    total += 0.97 ** step * reward
                expected[row, column] = total
    finally:
        env.close()
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=0)
    validation = validate_actor_reference(actor, states, candidates, actual,
                                          horizon=12, discount=0.97)
    assert validation["max_absolute_return_difference"] < 2e-6
    torch.testing.assert_close(actor.control, before)
    assert actor.control.grad is None


def test_ranking_metrics_handle_ties_and_constant_scores():
    matching = ranking_metrics(np.array([1., 1., 3., 4.]), np.array([2., 2., 5., 9.]))
    assert matching["spearman"] == pytest.approx(1.0)
    assert matching["pairwise_order_agreement"] == 1.0
    assert matching["comparable_pairs"] == 5
    inverse = ranking_metrics(np.array([4., 3., 2., 1.]), np.array([1., 2., 3., 4.]))
    assert inverse["spearman"] == pytest.approx(-1.0)
    assert inverse["pairwise_order_agreement"] == 0.0
    flat = ranking_metrics(np.zeros(3), np.array([1., 2., 3.]))
    assert flat["spearman"] is None
    assert flat["pairwise_order_agreement"] == 0.5
    tied = ranking_metrics(np.zeros(3), np.zeros(3))
    assert tied["pairwise_order_agreement"] is None


def test_diagnostic_state_sampling_has_fixed_strata_without_global_rng_use():
    before = np.random.get_state()
    first, labels = held_out_states()
    second, _ = held_out_states()
    after = np.random.get_state()
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    assert first.shape == (96, 3)
    assert {str(label): int((labels == label).sum()) for label in np.unique(labels)} == {
        "high_speed": 32, "near_upright": 32, "nominal": 32}
    assert np.allclose(np.linalg.norm(first[:, :2], axis=1), 1)
    assert (np.abs(first[labels == "high_speed", 2]) >= 7).all()
