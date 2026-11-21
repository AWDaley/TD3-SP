"""Validated, episode-aware transition data for dynamics learning.

Episode IDs describe contiguous trajectories, not randomly assigned split labels.
Both true terminations and time-limit truncations end a trajectory for forecasting.
The final transition of a trajectory can be used as a target, but a rollout never
continues into the reset state of the following episode.
"""

from __future__ import annotations

import hashlib
import json
import operator
import os
from pathlib import Path
import tempfile

import numpy as np


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _episode_ids(values, expected_length=None):
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.dtype.kind not in "iu":
        raise ValueError("episode_ids must be a one-dimensional integer array")
    if expected_length is not None and len(raw) != expected_length:
        raise ValueError("episode_ids must have one entry per transition")
    if raw.size and (np.any(raw < 0) or np.any(raw > np.iinfo(np.int64).max)):
        raise ValueError("episode_ids must be nonnegative signed 64-bit integers")
    return np.array(raw, dtype=np.int64, order="C", copy=True)


class TransitionDataset:
    """A collection of complete or partial, non-overlapping episode trajectories.

    Arrays are copied and made read-only after validation. States and actions are
    stored as float32. Continuity is checked with rtol=1e-5 and atol=1e-6 to allow
    ordinary serialization rounding. Episode IDs are retained when subsetting.
    """

    FORMAT_VERSION = 1

    def __init__(self, states, actions, next_states, episode_ids,
                 terminated=None, truncated=None):
        arrays = {}
        for name, values in (("states", states), ("actions", actions),
                             ("next_states", next_states)):
            try:
                array = np.array(values, dtype=np.float32, order="C", copy=True)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{name} must be a numeric two-dimensional array") from exc
            if array.ndim != 2 or min(array.shape) < 1:
                raise ValueError(f"{name} must be a nonempty two-dimensional array")
            if not np.isfinite(array).all():
                raise ValueError(f"{name} must contain only finite values")
            arrays[name] = array

        n = len(arrays["states"])
        if arrays["next_states"].shape != arrays["states"].shape:
            raise ValueError("states and next_states must have the same shape")
        if len(arrays["actions"]) != n:
            raise ValueError("actions must have one row per transition")
        arrays["episode_ids"] = _episode_ids(episode_ids, n)
        for name, values in (("terminated", terminated), ("truncated", truncated)):
            if values is None:
                array = np.zeros(n, dtype=bool)
            else:
                raw = np.asarray(values)
                if raw.shape != (n,) or not np.isin(raw, (False, True)).all():
                    raise ValueError(f"{name} must be a boolean vector of length {n}")
                array = np.array(raw, dtype=bool, order="C", copy=True)
            arrays[name] = array

        ids = arrays["episode_ids"]
        starts = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1].astype(np.int64)
        block_ids = ids[starts]
        if len(np.unique(block_ids)) != len(block_ids):
            raise ValueError("each episode ID must occur in one contiguous block")
        lengths = np.diff(np.r_[starts, n]).astype(np.int64)
        same_episode = ids[:-1] == ids[1:]
        ended = arrays["terminated"] | arrays["truncated"]
        if np.any(ended[:-1] & same_episode):
            raise ValueError("an episode cannot continue after termination or truncation")
        continuous = np.isclose(arrays["next_states"][:-1], arrays["states"][1:],
                                rtol=1e-5, atol=1e-6).all(axis=1)
        bad = np.flatnonzero(same_episode & ~continuous)
        if bad.size:
            row = int(bad[0])
            raise ValueError(
                f"trajectory continuity error after transition {row} "
                f"in episode {int(ids[row])}: next_state must match the following state"
            )

        for name, array in arrays.items():
            array.setflags(write=False)
            setattr(self, name, array)
        self._starts = starts
        self._lengths = lengths

    def __len__(self):
        return len(self.states)

    @property
    def state_dim(self):
        return self.states.shape[1]

    @property
    def action_dim(self):
        return self.actions.shape[1]

    @property
    def episode_count(self):
        return len(self._starts)

    @property
    def unique_episode_ids(self):
        """Episode IDs in recording order, as an independent array."""
        return self.episode_ids[self._starts].copy()

    def subset(self, episode_ids):
        """Select entire episodes while preserving their original order and IDs."""
        requested = _episode_ids(list(episode_ids))
        if not len(requested):
            raise ValueError("an episode subset cannot be empty")
        missing = np.setdiff1d(requested, self.unique_episode_ids)
        if missing.size:
            raise ValueError(f"unknown episode IDs: {missing.tolist()}")
        selected = np.isin(self.episode_ids, requested)
        return type(self)(self.states[selected], self.actions[selected],
                          self.next_states[selected], self.episode_ids[selected],
                          self.terminated[selected], self.truncated[selected])

    def split_by_episode(self, train_fraction=0.7, validation_fraction=0.15, seed=0):
        """Split whole episodes, leaving at least one episode in every split.

        Fractions refer to episode counts, so transition counts can differ when
        episode lengths vary. Only the training split should fit preprocessing.
        """
        if (not np.isfinite(train_fraction) or not np.isfinite(validation_fraction)
                or not 0 < train_fraction < 1 or not 0 < validation_fraction < 1
                or train_fraction + validation_fraction >= 1):
            raise ValueError("train and validation fractions must be positive and sum to less than one")
        if self.episode_count < 3:
            raise ValueError("at least three episodes are required for train/validation/test splitting")
        ids = np.random.default_rng(seed).permutation(self.unique_episode_ids)
        n = len(ids)
        n_train = min(n - 2, max(1, int(np.floor(n * train_fraction))))
        n_validation = min(n - n_train - 1,
                           max(1, int(np.floor(n * validation_fraction))))
        return {
            "train": self.subset(ids[:n_train]),
            "validation": self.subset(ids[n_train:n_train + n_validation]),
            "test": self.subset(ids[n_train + n_validation:]),
        }

    def rollout_windows(self, horizon, max_windows=None, seed=0):
        """Return initial states, action sequences and next-state sequences.

        Shapes are [B,D], [B,H,A], and [B,H,D]. Unavailable horizons return
        correctly shaped empty arrays. With a cap, starts are sampled uniformly
        without replacement before materializing any trajectory windows.
        """
        horizon = _positive_integer(horizon, "horizon")
        if max_windows is not None:
            max_windows = _positive_integer(max_windows, "max_windows")
        counts = np.maximum(self._lengths - horizon + 1, 0)
        cumulative = np.cumsum(counts, dtype=np.int64)
        total = int(cumulative[-1])
        if total == 0:
            return (np.empty((0, self.state_dim), dtype=np.float32),
                    np.empty((0, horizon, self.action_dim), dtype=np.float32),
                    np.empty((0, horizon, self.state_dim), dtype=np.float32))
        if max_windows is not None and max_windows < total:
            ranks = np.sort(np.random.default_rng(seed).choice(
                total, size=max_windows, replace=False))
        else:
            ranks = np.arange(total, dtype=np.int64)
        blocks = np.searchsorted(cumulative, ranks, side="right")
        preceding = np.r_[0, cumulative[:-1]]
        starts = self._starts[blocks] + ranks - preceding[blocks]
        indices = starts[:, None] + np.arange(horizon)[None, :]
        return self.states[starts], self.actions[indices], self.next_states[indices]

    def fingerprint(self):
        """Hash canonical array contents, independent of NPZ compression details."""
        digest = hashlib.sha256(b"TD3-SP-transition-dataset-v1\0")
        for name, dtype in (("states", "<f4"), ("actions", "<f4"),
                            ("next_states", "<f4"), ("episode_ids", "<i8"),
                            ("terminated", "?"), ("truncated", "?")):
            array = np.ascontiguousarray(getattr(self, name), dtype=dtype)
            digest.update(json.dumps([name, list(array.shape), dtype]).encode("utf-8"))
            digest.update(memoryview(array).cast("B"))
        return digest.hexdigest()

    def save(self, path):
        """Atomically write a compressed, pickle-free NPZ dataset."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".",
                                             suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                np.savez_compressed(
                    stream, format_version=np.array(self.FORMAT_VERSION, dtype=np.int64),
                    states=self.states, actions=self.actions, next_states=self.next_states,
                    episode_ids=self.episode_ids, terminated=self.terminated,
                    truncated=self.truncated,
                )
            os.replace(temporary, path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        return path

    @classmethod
    def load(cls, path):
        """Load and revalidate NPZ data; arbitrary Python objects are prohibited."""
        with np.load(path, allow_pickle=False) as data:
            required = ("states", "actions", "next_states", "episode_ids")
            missing = [name for name in required if name not in data]
            if missing:
                raise ValueError(f"dataset is missing arrays: {', '.join(missing)}")
            version = data["format_version"] if "format_version" in data else np.array(1)
            if version.shape != () or version.dtype.kind not in "iu" or int(version) != cls.FORMAT_VERSION:
                raise ValueError("unsupported transition dataset format version")
            return cls(*(data[name] for name in required),
                       terminated=data["terminated"] if "terminated" in data else None,
                       truncated=data["truncated"] if "truncated" in data else None)
