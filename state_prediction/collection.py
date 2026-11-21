"""Record actual transitions without mixing observations across resets."""

from pathlib import Path

import numpy as np

from .data import TransitionDataset


class TransitionRecorder:
    """Bounded chronological archive; keeps the newest ``capacity`` transitions.

    Unlike TD3's bootstrap mask, this preserves both termination and truncation.
    A ring-buffer cut may leave a partial first/last episode; windows still never
    cross an episode boundary.
    """

    def __init__(self, state_dim, action_dim, capacity=100000):
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self.states = np.empty((capacity, state_dim), dtype=np.float32)
        self.actions = np.empty((capacity, action_dim), dtype=np.float32)
        self.next_states = np.empty_like(self.states)
        self.episode_ids = np.empty(capacity, dtype=np.int64)
        self.terminated = np.empty(capacity, dtype=bool)
        self.truncated = np.empty(capacity, dtype=bool)
        self.count = 0

    def add(self, state, action, next_state, episode_id, terminated=False, truncated=False):
        i = self.count % self.capacity
        self.states[i] = state
        self.actions[i] = action
        self.next_states[i] = next_state
        self.episode_ids[i] = episode_id
        self.terminated[i] = terminated
        self.truncated[i] = truncated
        self.count += 1

    def dataset(self):
        if self.count == 0:
            raise ValueError("No transitions have been recorded")
        stop = self.count
        indices = np.arange(max(0, stop - self.capacity), stop) % self.capacity
        return TransitionDataset(
            self.states[indices], self.actions[indices], self.next_states[indices],
            self.episode_ids[indices], self.terminated[indices], self.truncated[indices],
        )


def collect_environment(env_name, episodes=30, seed=0, max_steps_per_episode=1000,
                        policy_checkpoint=None, exploration_noise=0.1, device="cpu"):
    """Collect seeded random actions or a saved TD3 actor with optional noise."""
    from env_compat import make_env, reset_env, step_env, continuous_dimensions

    if episodes < 1 or max_steps_per_episode < 1:
        raise ValueError("episodes and max_steps_per_episode must be positive")
    if exploration_noise < 0:
        raise ValueError("exploration_noise cannot be negative")
    env = make_env(env_name)
    try:
        state_dim, action_dim = continuous_dimensions(env, bool(policy_checkpoint))
        recorder = TransitionRecorder(state_dim, action_dim, episodes * max_steps_per_episode)
        rng = np.random.default_rng(seed)
        actor = None
        if policy_checkpoint:
            import torch
            from TD3 import Actor

            actor = Actor(state_dim, action_dim, float(env.action_space.high[0])).to(device)
            path = Path(str(policy_checkpoint) + "_actor")
            actor.load_state_dict(torch.load(path, map_location=device, weights_only=True))
            actor.eval()
        for episode in range(episodes):
            state = reset_env(env, seed=seed + episode)
            env.action_space.seed(seed + episode)
            for step in range(max_steps_per_episode):
                if actor is None:
                    action = env.action_space.sample()
                else:
                    import torch
                    with torch.inference_mode():
                        observation = torch.as_tensor(np.asarray(state), dtype=torch.float32, device=device)
                        action = actor(observation[None]).cpu().numpy()[0]
                    scale = (env.action_space.high - env.action_space.low) / 2
                    action = np.clip(action + rng.normal(size=action_dim) * exploration_noise * scale,
                                     env.action_space.low, env.action_space.high)
                next_state, _, terminated, truncated, _ = step_env(env, action)
                # Explicit collection cutoffs also delimit a trajectory.
                truncated = truncated or (step + 1 == max_steps_per_episode and not terminated)
                recorder.add(state, action, next_state, episode, terminated, truncated)
                state = next_state
                if terminated or truncated:
                    break
        return recorder.dataset()
    finally:
        env.close()
