"""Small synthetic navigation dynamics for a reproducible CPU demonstration.

This is a point-mass numerical example, not a validated hydrodynamic AUV model.
State: x, y, vx, vy, current_x, current_y. Actions: two bounded forces.
"""

import numpy as np

from .collection import TransitionRecorder


def make_toy_dataset(episodes=40, steps=80, seed=0):
    if episodes < 3 or steps < 1:
        raise ValueError("The demo requires at least three episodes and positive steps")
    rng = np.random.default_rng(seed)
    recorder = TransitionRecorder(6, 2, episodes * steps)
    dt = 0.2
    for episode in range(episodes):
        state = np.concatenate((rng.uniform(-2, 2, 2), rng.uniform(-1, 1, 2),
                                rng.uniform(-0.3, 0.3, 2))).astype(np.float32)
        action = np.zeros(2, dtype=np.float32)
        for step in range(steps):
            action = (0.5 * action + 0.5 * rng.uniform(-1, 1, 2)).astype(np.float32)
            velocity = state[2:4]
            acceleration = 0.8 * action + state[4:6] - 0.2 * velocity - 0.25 * velocity * np.abs(velocity)
            next_velocity = velocity + dt * acceleration
            next_state = np.concatenate((state[:2] + dt * next_velocity,
                                         next_velocity, state[4:6])).astype(np.float32)
            recorder.add(state, action, next_state, episode, truncated=step + 1 == steps)
            state = next_state
    return recorder.dataset()
