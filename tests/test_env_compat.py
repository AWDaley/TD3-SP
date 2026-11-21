"""Keep environment resets and TD bootstrapping semantics distinct."""

from types import SimpleNamespace

import numpy as np
import pytest

from env_compat import continuous_dimensions, reset_env, step_env


@pytest.mark.parametrize(
    "step_result, expected_terminal, expected_truncated, bootstrap",
    [
        (([1], 1.0, False, {}), False, False, 1.0),
        (([1], 1.0, True, {}), True, False, 0.0),
        (([1], 1.0, True, {"TimeLimit.truncated": True}), False, True, 1.0),
        (([1], 1.0, False, False, {}), False, False, 1.0),
        (([1], 1.0, True, False, {}), True, False, 0.0),
        (([1], 1.0, False, True, {}), False, True, 1.0),
        (([1], 1.0, True, True, {}), True, True, 0.0),
    ],
)
def test_legacy_and_modern_steps_preserve_bootstrap_semantics(
    step_result, expected_terminal, expected_truncated, bootstrap
):
    env = SimpleNamespace(step=lambda action: step_result)
    observation, reward, terminal, truncated, info = step_env(env, [0])
    assert terminal is expected_terminal
    assert truncated is expected_truncated
    assert 1.0 - float(terminal) == bootstrap
    assert observation == [1]
    assert reward == 1.0
    assert info == step_result[-1]


def test_modern_reset_extracts_observation_and_forwards_seed():
    calls = []

    def reset(seed=None):
        calls.append(seed)
        return np.array([1, 2]), {"info": "not an observation"}

    observation = reset_env(SimpleNamespace(reset=reset), seed=27)
    np.testing.assert_array_equal(observation, [1, 2])
    assert calls == [27]


def test_legacy_reset_calls_separate_seed_method():
    seeds = []
    env = SimpleNamespace(
        seed=lambda value: seeds.append(value),
        reset=lambda: np.array([3, 4]),
    )
    np.testing.assert_array_equal(reset_env(env, seed=12), [3, 4])
    assert seeds == [12]


def test_reset_without_a_seed_does_not_reseed_legacy_environment():
    seeds = []
    env = SimpleNamespace(
        seed=lambda value: seeds.append(value),
        reset=lambda: np.array([3, 4]),
    )
    np.testing.assert_array_equal(reset_env(env), [3, 4])
    assert seeds == []


@pytest.mark.parametrize("structured_side", ["observation", "action"])
def test_dict_spaces_fail_with_explanation_instead_of_shape_typeerror(structured_side):
    import gymnasium as gym

    vector = gym.spaces.Box(-1, 1, shape=(2,), dtype=np.float32)
    structured = gym.spaces.Dict({"position": vector})
    assert structured.shape is None
    env = SimpleNamespace(
        observation_space=structured if structured_side == "observation" else vector,
        action_space=structured if structured_side == "action" else vector,
    )
    with pytest.raises(ValueError, match=f"flat continuous {structured_side} space"):
        continuous_dimensions(env)
