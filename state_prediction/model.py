"""A normalized residual dynamics model, separate from TD3's actor and critics.

The model learns ``next_state - state`` from recorded state/action pairs. Its
normalizers are fitted exclusively on training data. Validation trajectories
choose the retained checkpoint, including the incumbent before fine tuning.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import tempfile
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn


CHECKPOINT_VERSION = 1


def _scale(values: np.ndarray) -> np.ndarray:
    """Population standard deviation with a unit scale for constant features."""
    result = np.std(values.astype(np.float64), axis=0)
    return np.where(result > 1e-6, result, 1.0).astype(np.float32)


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or int(value) != value or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _numpy(value: Any, name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    result = np.asarray(value, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    result = np.ascontiguousarray(result)
    # Validated datasets may expose read-only arrays. PyTorch warns even when
    # inference only reads them, and it cannot wrap negative NumPy strides.
    return result if result.flags.writeable else result.copy()


def _cpu_copy(value: Any) -> Any:
    """Make portable checkpoint data without moving a live training model."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    return copy.deepcopy(value)


class StatePredictor(nn.Module):
    """Predict next observations from current observations and applied controls.

    ``predict`` and ``rollout`` expose NumPy inference. ``forward`` accepts
    tensors and remains differentiable for training on open-loop trajectories.
    The observations must contain sufficient motion information for the chosen
    environment; this class does not infer hidden state from sensor history.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Iterable[int] = (128, 128),
        learning_rate: float = 1e-3,
        device: str | torch.device = "cpu",
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.state_dim = _positive_integer(state_dim, "state_dim")
        self.action_dim = _positive_integer(action_dim, "action_dim")
        self.hidden_dims = tuple(
            _positive_integer(size, "hidden dimension") for size in hidden_dims
        )
        if not self.hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer")
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        self.learning_rate = float(learning_rate)
        self.seed = int(seed)
        self.metadata: dict[str, Any] = {}
        self.normalizers_fitted = False

        # Network construction is reproducible without consuming the TD3
        # agent's CPU random stream. All initialization happens on the CPU.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(self.seed)
            layers: list[nn.Module] = []
            previous = self.state_dim + self.action_dim
            for size in self.hidden_dims:
                layers.extend((nn.Linear(previous, size), nn.SiLU()))
                previous = size
            output = nn.Linear(previous, self.state_dim)
            nn.init.zeros_(output.weight)
            nn.init.zeros_(output.bias)
            layers.append(output)
            self.network = nn.Sequential(*layers)

        self.register_buffer("input_mean", torch.zeros(self.state_dim + self.action_dim))
        self.register_buffer("input_scale", torch.ones(self.state_dim + self.action_dim))
        self.register_buffer("delta_mean", torch.zeros(self.state_dim))
        self.register_buffer("delta_scale", torch.ones(self.state_dim))
        self.to(torch.device(device))
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    @property
    def device(self) -> torch.device:
        return self.input_mean.device

    @property
    def state_scale(self) -> np.ndarray:
        """Training-state feature scales used for comparable validation errors."""
        return self.input_scale[: self.state_dim].detach().cpu().numpy().copy()

    def _check_dataset(self, data: Any, name: str) -> None:
        if len(data) == 0:
            raise ValueError(f"{name} dataset is empty")
        if data.states.shape[1] != self.state_dim:
            raise ValueError(f"{name} state dimension does not match the model")
        if data.actions.shape[1] != self.action_dim:
            raise ValueError(f"{name} action dimension does not match the model")
        for field in ("states", "actions", "next_states"):
            if not np.isfinite(getattr(data, field)).all():
                raise ValueError(f"{name}.{field} contains nonfinite values")

    def fit_normalizers(self, train: Any) -> None:
        """Fit every preprocessing statistic on training transitions only."""
        self._check_dataset(train, "training")
        inputs = np.concatenate((train.states, train.actions), axis=1)
        deltas = train.next_states.astype(np.float64) - train.states.astype(np.float64)
        stats = {
            "input_mean": inputs.astype(np.float64).mean(axis=0).astype(np.float32),
            "input_scale": _scale(inputs),
            "delta_mean": deltas.mean(axis=0).astype(np.float32),
            "delta_scale": _scale(deltas),
        }
        with torch.no_grad():
            for name, value in stats.items():
                if not np.isfinite(value).all():
                    raise ValueError("Training data produced invalid normalization statistics")
                getattr(self, name).copy_(torch.as_tensor(value, device=self.device))
        self.normalizers_fitted = True

    def normalized_delta(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat((states, actions), dim=-1)
        return self.network((inputs - self.input_mean) / self.input_scale)

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return states + self.delta_mean + self.delta_scale * self.normalized_delta(states, actions)

    def predict(self, states: Any, actions: Any) -> np.ndarray:
        """Return predicted next states for one pair or an aligned batch."""
        state_array = _numpy(states, "states")
        action_array = _numpy(actions, "actions")
        single = state_array.ndim == 1
        if state_array.ndim not in (1, 2) or action_array.ndim != state_array.ndim:
            raise ValueError("states and actions must both be vectors or both be matrices")
        if state_array.shape[-1] != self.state_dim or action_array.shape[-1] != self.action_dim:
            raise ValueError("Prediction input feature dimensions do not match the model")
        if not single and state_array.shape[0] != action_array.shape[0]:
            raise ValueError("states and actions must have the same batch size")
        if single:
            state_array, action_array = state_array[None, :], action_array[None, :]
        with torch.no_grad():
            output = self(
                torch.as_tensor(state_array, device=self.device),
                torch.as_tensor(action_array, device=self.device),
            ).cpu().numpy()
        if not np.isfinite(output).all():
            raise FloatingPointError("State predictor produced nonfinite next states")
        return output[0] if single else output

    def _rollout_tensor(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        predictions = []
        for step in range(actions.shape[1]):
            states = self(states, actions[:, step])
            predictions.append(states)
        return torch.stack(predictions, dim=1)

    def rollout(self, initial_states: Any, action_sequences: Any) -> np.ndarray:
        """Predict an open-loop future; feed each prediction into the next step."""
        states = _numpy(initial_states, "initial_states")
        actions = _numpy(action_sequences, "action_sequences")
        single = states.ndim == 1
        if states.ndim not in (1, 2) or actions.ndim != states.ndim + 1:
            raise ValueError("Use shapes (D,), (H,A) or (B,D), (B,H,A) for rollouts")
        if states.shape[-1] != self.state_dim or actions.shape[-1] != self.action_dim:
            raise ValueError("Rollout input feature dimensions do not match the model")
        if actions.shape[-2] < 1:
            raise ValueError("A rollout needs at least one action")
        if not single and states.shape[0] != actions.shape[0]:
            raise ValueError("initial_states and action_sequences must have the same batch size")
        if single:
            states, actions = states[None, :], actions[None, :, :]
        with torch.no_grad():
            output = self._rollout_tensor(
                torch.as_tensor(states, device=self.device),
                torch.as_tensor(actions, device=self.device),
            ).cpu().numpy()
        if not np.isfinite(output).all():
            raise FloatingPointError("State predictor produced nonfinite rollout states")
        return output[0] if single else output

    def fit(
        self,
        train: Any,
        validation: Any,
        *,
        epochs: int = 50,
        batch_size: int = 256,
        patience: int = 10,
        seed: int = 0,
        validation_seed: int | None = None,
        horizons: Iterable[int] = (1, 5, 10),
        max_windows: int = 1000,
        rollout_loss_weight: float = 0.0,
        rollout_horizon: int = 5,
        initialize_normalizers: bool = True,
    ) -> dict[str, Any]:
        """Optimize on training data and retain the best validation snapshot.

        The selection score averages normalized *endpoint* error over the
        available requested horizons. Callers must reserve test data separately.
        For fine tuning, pass ``initialize_normalizers=False`` so the incumbent
        and candidate use identical preprocessing. Optional rollout loss follows
        the model's own predictions, with no ground-truth state substitutions.
        ``validation_seed`` fixes the sampled validation windows independently
        of training shuffle/window sampling; by default it follows ``seed``.
        """
        from .evaluation import evaluate_predictor

        epochs = _positive_integer(epochs, "epochs")
        batch_size = _positive_integer(batch_size, "batch_size")
        patience = _positive_integer(patience, "patience")
        rollout_horizon = _positive_integer(rollout_horizon, "rollout_horizon")
        max_windows = _positive_integer(max_windows, "max_windows")
        horizons = tuple(_positive_integer(h, "horizon") for h in horizons)
        if not horizons:
            raise ValueError("At least one validation horizon is required")
        if not math.isfinite(rollout_loss_weight) or rollout_loss_weight < 0:
            raise ValueError("rollout_loss_weight must be finite and nonnegative")
        self._check_dataset(train, "training")
        self._check_dataset(validation, "validation")
        if initialize_normalizers:
            self.fit_normalizers(train)
        elif not self.normalizers_fitted:
            raise ValueError("Fit normalizers before fine tuning an existing model")

        rng = np.random.default_rng(seed)
        evaluation_seed = seed if validation_seed is None else validation_seed

        def validation_score() -> float:
            self.eval()
            result = evaluate_predictor(
                self, validation, state_scale=self.state_scale, horizons=horizons,
                max_windows=max_windows, seed=evaluation_seed,
            )
            score = float(result["selection_score"])
            if not math.isfinite(score):
                raise FloatingPointError("Validation selection score is nonfinite")
            return score

        initial_score = validation_score()
        best_score = initial_score
        best_epoch = 0
        best_weights = _cpu_copy(self.state_dict())
        best_optimizer = _cpu_copy(self.optimizer.state_dict())
        history: list[dict[str, Any]] = []
        stale_epochs = 0
        stopped_reason = "epochs_completed"

        rollout_data = None
        if rollout_loss_weight > 0:
            windows = train.rollout_windows(
                rollout_horizon, max_windows=max_windows, seed=seed,
            )
            if len(windows[0]) == 0:
                raise ValueError(
                    f"No training episode supports rollout_horizon={rollout_horizon}; "
                    "choose a shorter horizon or collect longer episodes"
                )
            rollout_data = tuple(torch.as_tensor(item, device=self.device) for item in windows)

        try:
            for epoch in range(1, epochs + 1):
                self.train()
                indices = rng.permutation(len(train))
                totals = np.zeros(3, dtype=np.float64)
                examples = 0
                for offset in range(0, len(indices), batch_size):
                    batch = indices[offset: offset + batch_size]
                    states = torch.as_tensor(train.states[batch], device=self.device)
                    actions = torch.as_tensor(train.actions[batch], device=self.device)
                    next_states = torch.as_tensor(train.next_states[batch], device=self.device)
                    targets = (next_states - states - self.delta_mean) / self.delta_scale
                    prediction = self.normalized_delta(states, actions)
                    one_step_loss = torch.mean((prediction - targets) ** 2)
                    rollout_loss = one_step_loss.new_zeros(())
                    if rollout_data is not None:
                        sample_count = min(batch_size, len(rollout_data[0]))
                        selected = torch.as_tensor(
                            rng.integers(0, len(rollout_data[0]), size=sample_count),
                            dtype=torch.long, device=self.device,
                        )
                        initial, controls, expected = (item[selected] for item in rollout_data)
                        predicted = self._rollout_tensor(initial, controls)
                        scaled_error = (predicted - expected) / self.input_scale[: self.state_dim]
                        rollout_loss = torch.mean(scaled_error ** 2)
                    loss = one_step_loss + rollout_loss_weight * rollout_loss
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite prediction training loss; best model restored")
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.parameters(), max_norm=10.0, error_if_nonfinite=True,
                    )
                    self.optimizer.step()
                    size = len(batch)
                    totals += size * np.array([
                        loss.item(), one_step_loss.item(), rollout_loss.item(),
                    ])
                    examples += size

                try:
                    score = validation_score()
                except FloatingPointError as error:
                    history.append({
                        "epoch": epoch,
                        "training_loss": float(totals[0] / examples),
                        "one_step_training_loss": float(totals[1] / examples),
                        "rollout_training_loss": float(totals[2] / examples),
                        "validation_score": None,
                        "error": str(error),
                    })
                    stopped_reason = "nonfinite_validation"
                    break
                history.append({
                    "epoch": epoch,
                    "training_loss": float(totals[0] / examples),
                    "one_step_training_loss": float(totals[1] / examples),
                    "rollout_training_loss": float(totals[2] / examples),
                    "validation_score": score,
                })
                if score < best_score:
                    best_score = score
                    best_epoch = epoch
                    best_weights = _cpu_copy(self.state_dict())
                    best_optimizer = _cpu_copy(self.optimizer.state_dict())
                    stale_epochs = 0
                else:
                    stale_epochs += 1
                    if stale_epochs >= patience:
                        stopped_reason = "early_stopping"
                        break
        finally:
            # Restore weights and optimizer together, even if a training step
            # fails; this also keeps an incumbent when every candidate is worse.
            self.load_state_dict(best_weights)
            self.optimizer.load_state_dict(best_optimizer)
            self.eval()

        return {
            "history": history,
            "initial_validation_score": initial_score,
            "best_validation_score": best_score,
            "best_epoch": best_epoch,
            "epochs_run": len(history),
            "stopped_reason": stopped_reason,
            "validation_seed": int(evaluation_seed),
        }

    def save(self, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
        """Atomically save a CPU-portable checkpoint with safe-loadable data."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        saved_metadata = self.metadata if metadata is None else metadata
        # JSON-compatible provenance keeps checkpoints compatible with
        # torch.load(weights_only=True) and excludes arbitrary Python objects.
        saved_metadata = json.loads(json.dumps(saved_metadata, allow_nan=False))
        if not isinstance(saved_metadata, dict):
            raise ValueError("Checkpoint metadata must be a dictionary")
        checkpoint = {
            "format": "td3-sp-state-predictor",
            "version": CHECKPOINT_VERSION,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "hidden_dims": list(self.hidden_dims),
            "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
            "seed": self.seed,
            "normalizers_fitted": self.normalizers_fitted,
            "state_dict": _cpu_copy(self.state_dict()),
            "optimizer_state_dict": _cpu_copy(self.optimizer.state_dict()),
            "metadata": saved_metadata,
        }
        with tempfile.NamedTemporaryFile(
            prefix=destination.name + ".", suffix=".tmp",
            dir=destination.parent, delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        try:
            torch.save(checkpoint, temporary_path)
            temporary_path.replace(destination)
        finally:
            temporary_path.unlink(missing_ok=True)
        self.metadata = saved_metadata

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cpu") -> "StatePredictor":
        """Load tensor/primitive checkpoint data without arbitrary pickle code."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict) or checkpoint.get("format") != "td3-sp-state-predictor":
            raise ValueError("This file is not a TD3-SP state predictor checkpoint")
        if checkpoint.get("version") != CHECKPOINT_VERSION:
            raise ValueError("Unsupported state predictor checkpoint version")
        model = cls(
            checkpoint["state_dim"], checkpoint["action_dim"],
            hidden_dims=checkpoint["hidden_dims"],
            learning_rate=checkpoint["learning_rate"], device=device,
            seed=checkpoint["seed"],
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        model.normalizers_fitted = bool(checkpoint["normalizers_fitted"])
        model.metadata = checkpoint.get("metadata", {})
        if not isinstance(model.metadata, dict):
            raise ValueError("Invalid checkpoint metadata")
        for tensor in model.state_dict().values():
            if not torch.isfinite(tensor).all():
                raise ValueError("Checkpoint contains nonfinite model parameters")
        if (model.input_scale <= 0).any() or (model.delta_scale <= 0).any():
            raise ValueError("Checkpoint contains invalid feature scales")
        model.eval()
        return model
