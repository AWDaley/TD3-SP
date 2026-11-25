"""Bounded action search using TD3 critics and optional short model rollouts.

This changes the behaviour policy only. It never changes TD3's loss, target
networks, replay labels, or optimizer state. All rollout continuations use the
current actor and the terminal value uses the current pair of critics.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
import time
from typing import Any

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class SelectionConfig:
    """Search settings; noise scales are fractions of each action half-range.

    ``horizon`` affects only lookahead. Candidate zero is always the unperturbed
    actor action. The other candidates share one explored centre during training
    and have independent search perturbations. No noise is added after ranking.
    """

    mode: str = "actor"
    candidates: int = 16
    noise_std: float = 0.2
    horizon: int = 1
    seed: int = 0

    def __post_init__(self) -> None:
        if self.mode not in ("actor", "critic", "lookahead"):
            raise ValueError("mode must be actor, critic, or lookahead")
        for name in ("candidates", "horizon"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
            object.__setattr__(self, name, int(value))
        if not math.isfinite(self.noise_std) or self.noise_std < 0:
            raise ValueError("noise_std must be finite and nonnegative")
        object.__setattr__(self, "noise_std", float(self.noise_std))
        if isinstance(self.seed, bool) or not isinstance(self.seed, (int, np.integer)) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        object.__setattr__(self, "seed", int(self.seed))


def _module_tensor(module: Any) -> torch.Tensor | None:
    if isinstance(module, nn.Module):
        return next(iter(module.parameters()), next(iter(module.buffers()), None))
    return None


class ActionSelector:
    """Select a bounded action with actor, critic-only, or model-based scoring.

    ``task(states, actions, next_states)`` must return batched reward tensors and
    boolean *true termination* tensors, each shaped ``[B]`` or ``[B, 1]``.
    A planning horizon or environment time limit is not a true termination.

    Actor/critic/model devices and dtypes must agree. This class deliberately
    does not move a live model or alter its optimizer. Invalid numerical scores
    exclude individual candidates; if all scores are invalid, the actor action
    is returned and ``last_info['fallback']`` is true. Shape/configuration errors
    raise instead of being silently treated as successful fallback.
    """

    def __init__(self, policy, action_low, action_high, config: SelectionConfig,
                 predictor=None, task=None) -> None:
        if not isinstance(config, SelectionConfig):
            raise TypeError("config must be a SelectionConfig")
        self.policy = policy
        self.config = config
        self.predictor = predictor
        self.task = task
        tensor = _module_tensor(policy.actor)
        self.device = tensor.device if tensor is not None else torch.device("cpu")
        self.dtype = tensor.dtype if tensor is not None else torch.float32
        if not self.dtype.is_floating_point:
            raise ValueError("actor must use a floating-point dtype")

        low = np.asarray(action_low, dtype=np.float64)
        high = np.asarray(action_high, dtype=np.float64)
        if low.ndim != 1 or high.shape != low.shape or low.size == 0:
            raise ValueError("action bounds must be equally sized nonempty vectors")
        if not (np.isfinite(low).all() and np.isfinite(high).all() and (high > low).all()):
            raise ValueError("action bounds must be finite with high greater than low")
        self.action_dim = low.size
        self.low = torch.as_tensor(low, dtype=self.dtype, device=self.device)
        self.high = torch.as_tensor(high, dtype=self.dtype, device=self.device)
        self.half_range = high / 2.0 - low / 2.0
        if not (torch.isfinite(self.low).all() and torch.isfinite(self.high).all()
                and (self.high > self.low).all()):
            raise ValueError("action bounds cannot be represented as distinct limits in the actor dtype")

        self.state_dim = getattr(predictor, "state_dim", None)
        actor_input = getattr(getattr(policy.actor, "l1", None), "in_features", None)
        if self.state_dim is not None and actor_input is not None and self.state_dim != actor_input:
            raise ValueError("predictor state dimension does not match the actor")
        if self.state_dim is None:
            self.state_dim = actor_input
        self.discount = float(getattr(policy, "discount", 1.0))
        if not math.isfinite(self.discount) or not 0 <= self.discount <= 1:
            raise ValueError("policy discount must lie between zero and one")
        if config.mode != "actor" and not callable(getattr(policy, "critic", None)):
            raise ValueError("critic and lookahead modes require twin critics")
        if config.mode == "lookahead":
            if not callable(predictor) or not callable(task):
                raise ValueError("lookahead requires a predictor and transition task")
            if getattr(predictor, "action_dim", self.action_dim) != self.action_dim:
                raise ValueError("predictor action dimension does not match action bounds")
        for name, module in (("critic", getattr(policy, "critic", None)),
                             ("predictor", predictor)):
            reference = _module_tensor(module)
            if reference is not None and (reference.device != self.device or reference.dtype != self.dtype):
                raise ValueError(f"{name} must share the actor's device and dtype")
        self.rng = np.random.default_rng(config.seed)
        self.last_info: dict[str, Any] = {}

    def reset(self, seed: int | None = None) -> None:
        """Restart this selector's private proposal stream, without global RNGs."""
        self.rng = np.random.default_rng(self.config.seed if seed is None else seed)
        self.last_info = {}

    @contextmanager
    def _inference(self):
        # Evaluation mode prevents BatchNorm buffers and Dropout RNG use from
        # changing training state. Preserve mixed per-submodule modes exactly.
        modules = {}
        for network in (self.policy.actor, getattr(self.policy, "critic", None),
                        self.predictor, self.task):
            if isinstance(network, nn.Module):
                for module in network.modules():
                    modules[id(module)] = (module, module.training)
        try:
            for module, _ in modules.values():
                module.training = False
            with torch.no_grad():
                yield
        finally:
            for module, training in modules.values():
                module.training = training

    def _tensor(self, value) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.detach().to(device=self.device, dtype=self.dtype)
        # Own the array: read-only observations and negative strides are valid.
        return torch.as_tensor(np.array(value, copy=True), device=self.device, dtype=self.dtype)

    def _state(self, state) -> torch.Tensor:
        result = self._tensor(state)
        if result.ndim != 1 or result.numel() == 0:
            raise ValueError("state must be a nonempty observation vector")
        if self.state_dim is not None and result.numel() != self.state_dim:
            raise ValueError("state dimension does not match the predictor or actor")
        if not torch.isfinite(result).all():
            raise ValueError("state must contain only finite values")
        return result

    def _clip(self, actions: torch.Tensor) -> torch.Tensor:
        return torch.maximum(self.low, torch.minimum(self.high, actions))

    def _actions(self, states: torch.Tensor) -> torch.Tensor:
        actions = self.policy.actor(states)
        if not isinstance(actions, torch.Tensor) or actions.shape != (len(states), self.action_dim):
            raise ValueError("actor must return actions shaped [batch, action_dim]")
        if actions.device != self.device or actions.dtype != self.dtype:
            raise ValueError("actor output must share its input device and dtype")
        return actions

    def _vector(self, value, batch: int, name: str, *, boolean=False) -> torch.Tensor:
        if not isinstance(value, torch.Tensor) or value.shape not in ((batch,), (batch, 1)):
            raise ValueError(f"{name} must be a tensor shaped [batch] or [batch, 1]")
        if value.device != self.device:
            raise ValueError(f"{name} must share the actor's device")
        if boolean and value.dtype != torch.bool:
            raise ValueError("terminated must have boolean dtype")
        return value.reshape(batch) if boolean else value.reshape(batch).to(self.dtype)

    def _values(self, states, actions):
        result = self.policy.critic(states, actions)
        if not isinstance(result, (tuple, list)) or len(result) != 2:
            raise ValueError("critic must return two batched Q-value tensors")
        q1 = self._vector(result[0], len(states), "Q1")
        q2 = self._vector(result[1], len(states), "Q2")
        # Taking the minimum first would conceal a +inf value in one critic.
        finite = torch.isfinite(q1) & torch.isfinite(q2)
        return torch.where(finite, torch.minimum(q1, q2), -torch.inf)

    def score_candidates(self, state, candidates) -> torch.Tensor:
        """Score an explicit action batch without consuming the proposal RNG.

        Actions are clipped to bounds. The returned vector stays on the actor's
        device and uses ``-inf`` for numerically invalid branches. This is also
        useful for testing fixed candidates and counterfactual diagnostics.
        """
        if self.config.mode == "actor":
            raise ValueError("actor mode does not score candidates")
        state_tensor = self._state(state)
        action_tensor = self._tensor(candidates)
        if action_tensor.ndim != 2 or action_tensor.shape[1] != self.action_dim or len(action_tensor) == 0:
            raise ValueError("candidates must have shape [positive batch, action_dim]")
        if not torch.isfinite(action_tensor).all():
            raise ValueError("candidate actions must be finite")
        with self._inference():
            return self._scores(state_tensor, self._clip(action_tensor))

    def _scores(self, state, actions):
        count = len(actions)
        states = state.unsqueeze(0).expand(count, -1).clone()
        if self.config.mode == "critic":
            scores = self._values(states, actions)
            return torch.where(torch.isfinite(scores), scores, -torch.inf)

        scores = torch.zeros(count, device=self.device, dtype=self.dtype)
        alive = torch.ones(count, device=self.device, dtype=torch.bool)
        valid = alive.clone()

        def keep(indices, finite):
            rejected = indices[~finite]
            alive[rejected] = False
            valid[rejected] = False
            return indices[finite]

        for step in range(self.config.horizon):
            indices = torch.nonzero(alive, as_tuple=True)[0]
            if len(indices) == 0:
                break
            controls = actions[indices] if step == 0 else self._actions(states[indices])
            finite = torch.isfinite(controls).all(dim=1)
            indices = keep(indices, finite)
            controls = self._clip(controls[finite])
            if len(indices) == 0:
                break
            predicted = self.predictor(states[indices], controls)
            if not isinstance(predicted, torch.Tensor) or predicted.shape != states[indices].shape:
                raise ValueError("predictor must return next states shaped [batch, state_dim]")
            if predicted.device != self.device or predicted.dtype != self.dtype:
                raise ValueError("predictor output must share the actor's device and dtype")
            finite = torch.isfinite(predicted).all(dim=1)
            indices = keep(indices, finite)
            controls, predicted = controls[finite], predicted[finite]
            if len(indices) == 0:
                break
            transition = self.task(states[indices], controls, predicted)
            if not isinstance(transition, (tuple, list)) or len(transition) != 2:
                raise ValueError("task must return (rewards, terminated)")
            reward = self._vector(transition[0], len(indices), "reward")
            terminal = self._vector(transition[1], len(indices), "terminated", boolean=True)
            finite = torch.isfinite(reward)
            indices = keep(indices, finite)
            predicted, reward, terminal = predicted[finite], reward[finite], terminal[finite]
            scores[indices] += (self.discount ** step) * reward
            states[indices] = predicted
            alive[indices] = ~terminal
            finite = torch.isfinite(scores[indices])
            keep(indices, finite)
            if self.discount == 0:
                break

        indices = torch.nonzero(alive, as_tuple=True)[0]
        if len(indices) and self.discount > 0:
            tail_actions = self._actions(states[indices])
            finite = torch.isfinite(tail_actions).all(dim=1)
            indices = keep(indices, finite)
            tail_actions = self._clip(tail_actions[finite])
            if len(indices):
                scores[indices] += (self.discount ** self.config.horizon) * self._values(states[indices], tail_actions)
        valid &= torch.isfinite(scores)
        return torch.where(valid, scores, -torch.inf)

    def select_action(self, state, exploration_std: float = 0.0) -> np.ndarray:
        """Choose the final action; callers must not add noise afterward."""
        started = time.perf_counter()
        if not math.isfinite(exploration_std) or exploration_std < 0:
            raise ValueError("exploration_std must be finite and nonnegative")
        if self.config.mode != "actor" and self.config.candidates == 1 and exploration_std > 0:
            raise ValueError("exploration with action search requires at least two candidates")
        state_tensor = self._state(state)
        with self._inference():
            raw = self._actions(state_tensor.unsqueeze(0))[0]
            if not torch.isfinite(raw).all():
                raise FloatingPointError("actor produced a nonfinite action")
            base = self._clip(raw)
            centre = base.cpu().numpy().astype(np.float64)
            if exploration_std:
                centre += self.rng.normal(size=self.action_dim) * exploration_std * self.half_range
            if self.config.mode == "actor":
                chosen = self._clip(self._tensor(centre))
                selected, fallback, invalid = 0, False, 0
                baseline_score = selected_score = None
                count = 1
            else:
                count = self.config.candidates
                perturbations = self.rng.normal(size=(count - 1, self.action_dim))
                others = centre + perturbations * self.config.noise_std * self.half_range
                candidates = torch.cat((base.unsqueeze(0), self._clip(self._tensor(others))), dim=0)
                scores = self._scores(state_tensor, candidates)
                finite = torch.isfinite(scores)
                invalid = int((~finite).sum().item())
                fallback = not bool(finite.any())
                selected = 0 if fallback else int(torch.argmax(scores).item())
                chosen = candidates[selected]
                baseline_score = float(scores[0].item()) if bool(finite[0]) else None
                selected_score = None if fallback else float(scores[selected].item())
            result = chosen.cpu().numpy().copy()
            intervention = self.config.mode != "actor" and not np.allclose(
                result, base.cpu().numpy(), rtol=0, atol=1e-7)
        self.last_info = {
            "selected_index": selected,
            "baseline_score": baseline_score,
            "selected_score": selected_score,
            "fallback": fallback,
            "fallback_reason": "no_finite_candidate_score" if fallback else None,
            "intervention": bool(intervention),
            "candidate_count": count,
            "horizon": self.config.horizon if self.config.mode == "lookahead" else 0,
            "invalid_candidates": invalid,
            "elapsed_ms": 1000.0 * (time.perf_counter() - started),
        }
        return result
