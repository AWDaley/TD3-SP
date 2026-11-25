"""Task rewards and true termination for imagined transitions.

A dynamics model supplies next observations, not rewards or episode endings.
Custom tasks implement ``task(states, actions, next_states)`` and return one
reward and one boolean *true termination* flag per batch row. Time limits and
the chosen planning horizon are not true terminations.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import math

import numpy as np
import torch


@dataclass(frozen=True)
class PendulumTask:
    """Exact built-in Pendulum reward in observation coordinates.

Observations are ``[cos(theta), sin(theta), angular_velocity]``. Gymnasium's
Pendulum charges the angle/velocity cost at the CURRENT state, and clips the
applied torque before charging its cost. Pendulum has no true terminal states;
the outer TimeLimit wrapper only truncates episodes.
"""

    max_torque: float = 2.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.max_torque) or self.max_torque <= 0:
            raise ValueError("max_torque must be finite and positive")

    def __call__(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if states.ndim != 2 or states.shape[1] != 3:
            raise ValueError("Pendulum states must have shape [batch, 3]")
        if actions.shape != (len(states), 1):
            raise ValueError("Pendulum actions must have shape [batch, 1]")
        if next_states.shape != states.shape:
            raise ValueError("Pendulum next_states must match states")
        if not (states.device == actions.device == next_states.device):
            raise ValueError("Task tensors must be on the same device")
        angle = torch.atan2(states[:, 1], states[:, 0])
        torque = actions[:, 0].clamp(-self.max_torque, self.max_torque)
        rewards = -(angle.square() + 0.1 * states[:, 2].square() + 0.001 * torque.square())
        terminated = torch.zeros(len(states), dtype=torch.bool, device=states.device)
        return rewards, terminated


def task_for_env(env) -> PendulumTask:
    """Return a verified built-in task adapter or fail without guessing.

Only unmodified Gymnasium/Gym Pendulum-v1 (and legacy Pendulum-v0) with the
standard environment-checking and time-limit wrappers is recognized. A custom
reward, action, observation, or termination wrapper needs an explicit custom
task callable passed to ``ActionSelector`` instead of this factory.
    """
    base = getattr(env, "unwrapped", env)
    package = type(base).__module__.split(".")[0]
    if package not in {"gymnasium", "gym"}:
        raise ValueError("No built-in planning task for this environment; provide a custom task callable")
    try:
        pendulum_type = importlib.import_module(
            f"{package}.envs.classic_control.pendulum"
        ).PendulumEnv
    except (ImportError, AttributeError) as exc:
        raise ValueError("The built-in Pendulum task is unavailable") from exc
    spec = getattr(base, "spec", None)
    if type(base) is not pendulum_type or getattr(spec, "id", None) not in {
        "Pendulum-v0", "Pendulum-v1"
    }:
        raise ValueError("No built-in planning task for this environment; provide a custom task callable")

    # Exact wrapper types avoid silently scoring a transformed observation,
    # action or reward using the original environment's reward function.
    allowed = set()
    wrapper_modules = (
        ("wrappers.common", ("TimeLimit", "OrderEnforcing", "PassiveEnvChecker")),
        ("wrappers.time_limit", ("TimeLimit",)),
        ("wrappers.order_enforcing", ("OrderEnforcing",)),
        ("wrappers.env_checker", ("PassiveEnvChecker",)),
    )
    for module_name, names in wrapper_modules:
        try:
            module = importlib.import_module(f"{package}.{module_name}")
        except ImportError:
            continue
        allowed.update(getattr(module, name) for name in names if hasattr(module, name))
    current = env
    while current is not base:
        if type(current) not in allowed:
            raise ValueError(
                "Custom environment wrappers need a custom planning task callable "
                "that matches their reward, action, observation and termination semantics"
            )
        current = current.env
    if "step" in base.__dict__:
        raise ValueError("A modified environment step needs a custom planning task callable")

    task = PendulumTask(float(base.max_torque))
    observation = env.observation_space
    action = env.action_space
    expected_high = np.array([1.0, 1.0, float(base.max_speed)])
    if (
        observation.shape != (3,)
        or action.shape != (1,)
        or not np.allclose(observation.high, expected_high)
        or not np.allclose(observation.low, -expected_high)
        or not np.allclose(action.high, [task.max_torque])
        or not np.allclose(action.low, [-task.max_torque])
    ):
        raise ValueError("Pendulum spaces do not match its reward/task adapter")
    return task
