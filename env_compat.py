"""Small adapters for single-environment Gymnasium and legacy Gym APIs."""

import numpy as np


def make_env(name):
    try:
        import gymnasium as gym
    except ImportError:
        try:
            import gym
        except ImportError as exc:
            raise ImportError("Install environments with: pip install -r requirements.txt") from exc
    return gym.make(name)


def reset_env(env, seed=None):
    if seed is None:
        result = env.reset()
    else:
        try:
            result = env.reset(seed=int(seed))
        except TypeError:
            # Gym before 0.26 supplies a separate seeding method.
            env.seed(int(seed))
            result = env.reset()
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
        return result[0]
    return result


def step_env(env, action):
    result = env.step(action)
    if len(result) == 5:
        observation, reward, terminated, truncated, info = result
    elif len(result) == 4:
        observation, reward, done, info = result
        truncated = bool(info.get("TimeLimit.truncated", False))
        terminated = bool(done) and not truncated
    else:
        raise ValueError("Expected a Gym step result with four or five values")
    return observation, reward, bool(terminated), bool(truncated), info


def continuous_dimensions(env, require_symmetric_actions=False):
    """Reject shapes the upstream vector-state TD3 architecture cannot consume."""
    observation = env.observation_space
    action = env.action_space
    observation_shape = getattr(observation, "shape", None)
    action_shape = getattr(action, "shape", None)
    if observation_shape is None or len(observation_shape) != 1 or not hasattr(observation, "low"):
        raise ValueError("Expected a flat continuous observation space")
    if action_shape is None or len(action_shape) != 1 or not hasattr(action, "low"):
        raise ValueError("Expected a flat continuous action space")
    if not np.isfinite(action.low).all() or not np.isfinite(action.high).all():
        raise ValueError("Action bounds must be finite")
    if require_symmetric_actions:
        if not (np.allclose(action.high, action.high[0]) and np.allclose(action.low, -action.high)):
            raise ValueError("Upstream TD3 requires uniform, symmetric action bounds")
    return observation.shape[0], action.shape[0]
