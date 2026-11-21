"""Held-out forecast metrics and train-only fitted comparison baselines."""

from __future__ import annotations

import numpy as np

from .data import _positive_integer


def _scale(values, dimension):
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (dimension,) or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("state_scale must be a finite, nonnegative vector with one entry per state dimension")
    # Constant training features have no meaningful empirical standard deviation.
    return np.where(values > 1e-6, values, 1.0)


def _checked_prediction(values, shape):
    values = np.asarray(values, dtype=np.float64)
    if values.shape != shape:
        raise ValueError(f"prediction shape {values.shape} does not match expected shape {shape}")
    if not np.isfinite(values).all():
        raise FloatingPointError("predictor produced nonfinite values; forecast metrics are invalid")
    return values


class _MetricAccumulator:
    def __init__(self, state_scale):
        self.scale = state_scale
        self.count = 0
        self.squared = np.zeros_like(state_scale)
        self.absolute = np.zeros_like(state_scale)
        self.signed = np.zeros_like(state_scale)

    def update(self, predictions, targets):
        targets = np.asarray(targets, dtype=np.float64)
        predictions = _checked_prediction(predictions, targets.shape)
        error = predictions - targets
        with np.errstate(over="raise", invalid="raise"):
            self.squared += np.square(error).sum(axis=0)
            self.absolute += np.abs(error).sum(axis=0)
            self.signed += error.sum(axis=0)
        self.count += len(targets)

    def result(self):
        if not self.count:
            raise ValueError("cannot evaluate an empty set of predictions")
        mse = self.squared / self.count
        mae = self.absolute / self.count
        bias = self.signed / self.count
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            normalized_mse = mse / np.square(self.scale)
        per_dimension = [
            {"index": i, "mse": float(mse[i]), "rmse": float(np.sqrt(mse[i])),
             "mae": float(mae[i]), "bias": float(bias[i]),
             "normalized_rmse": float(np.sqrt(normalized_mse[i]))}
            for i in range(len(self.scale))
        ]
        return {
            "sample_count": self.count,
            "mse": float(mse.mean()), "rmse": float(np.sqrt(mse.mean())),
            "mae": float(mae.mean()), "bias": float(bias.mean()),
            "normalized_rmse": float(np.sqrt(normalized_mse.mean())),
            "per_dimension": per_dimension,
        }


def evaluate_predictor(model, dataset, state_scale=None, horizons=(1, 5, 10),
                       max_windows=1000, seed=0, batch_size=4096):
    """Evaluate without fitting on the supplied dataset.

    One-step errors use every transition. Longer-horizon errors compare the
    predicted endpoint with the recorded endpoint, applying the recorded action
    sequence to predicted states (no intermediate teacher forcing). Horizon 1
    always participates in selection and reuses all one-step predictions.

    selection_score averages normalized RMSE equally over the available horizons.
    The scale must come from training data; if omitted, model.state_scale is used,
    otherwise units are used. No statistic is fitted to validation/test data.
    """
    batch_size = _positive_integer(batch_size, "batch_size")
    if max_windows is not None:
        max_windows = _positive_integer(max_windows, "max_windows")
    requested = sorted({1, *(_positive_integer(h, "horizon") for h in horizons)})
    if state_scale is not None:
        scale_source = "provided"
    else:
        state_scale = getattr(model, "state_scale", None)
        scale_source = "model_training_scale" if state_scale is not None else "unit"
        if state_scale is None:
            state_scale = np.ones(dataset.state_dim)
    state_scale = _scale(state_scale, dataset.state_dim)
    one_step_accumulator = _MetricAccumulator(state_scale)
    for start in range(0, len(dataset), batch_size):
        stop = start + batch_size
        predictions = model.predict(dataset.states[start:stop], dataset.actions[start:stop])
        one_step_accumulator.update(predictions, dataset.next_states[start:stop])
    one_step = one_step_accumulator.result()
    rollouts = {"1": dict(one_step, kind="open_loop_endpoint")}
    skipped = []
    for horizon in requested:
        if horizon == 1:
            continue
        initial, actions, targets = dataset.rollout_windows(horizon, max_windows, seed)
        if not len(initial):
            skipped.append(horizon)
            continue
        accumulator = _MetricAccumulator(state_scale)
        for start in range(0, len(initial), batch_size):
            stop = start + batch_size
            predictions = _checked_prediction(
                model.rollout(initial[start:stop], actions[start:stop]),
                targets[start:stop].shape,
            )
            accumulator.update(predictions[:, -1], targets[start:stop, -1])
        rollouts[str(horizon)] = dict(accumulator.result(), kind="open_loop_endpoint")
    return {
        "transition_count": len(dataset), "episode_count": dataset.episode_count,
        "one_step": one_step, "rollout": rollouts,
        "requested_horizons": requested,
        "available_horizons": [h for h in requested if h not in skipped],
        "skipped_horizons": skipped,
        "selection_score": float(np.mean([m["normalized_rmse"] for m in rollouts.values()])),
        "normalization": scale_source, "state_scale": state_scale.tolist(),
    }


def _inputs(states, actions, state_dim=None, action_dim=None):
    states = np.asarray(states, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    single = states.ndim == 1
    if single:
        states = states[None, :]
        if actions.ndim == 1:
            actions = actions[None, :]
    if (states.ndim != 2 or actions.ndim != 2 or len(states) != len(actions)
            or states.shape[1] < 1 or actions.shape[1] < 1):
        raise ValueError("states and actions must be matching vectors or batches of vectors")
    if state_dim is not None and states.shape[1] != state_dim:
        raise ValueError("state dimension does not match the fitted predictor")
    if action_dim is not None and actions.shape[1] != action_dim:
        raise ValueError("action dimension does not match the fitted predictor")
    if not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise ValueError("states and actions must contain only finite values")
    return states, actions, single


class _RolloutMixin:
    def rollout(self, initial_states, action_sequences):
        states = np.asarray(initial_states, dtype=np.float64)
        actions = np.asarray(action_sequences, dtype=np.float64)
        single = states.ndim == 1
        if single:
            states = states[None, :]
            if actions.ndim == 2:
                actions = actions[None, :, :]
        if (states.ndim != 2 or actions.ndim != 3 or len(states) != len(actions)
                or actions.shape[1] < 1):
            raise ValueError("rollout requires states [B,D] and actions [B,H,A], or single equivalents")
        predictions = []
        for step in range(actions.shape[1]):
            states = self.predict(states, actions[:, step])
            predictions.append(states)
        result = np.stack(predictions, axis=1)
        return result[0] if single else result


class PersistencePredictor(_RolloutMixin):
    """The no-motion baseline: every future state equals the initial state."""

    def predict(self, states, actions):
        states, _, single = _inputs(states, actions)
        result = states.copy()
        return result[0] if single else result


class LinearDeltaPredictor(_RolloutMixin):
    """A ridge-fitted affine state-change model with train-only normalization."""

    @classmethod
    def fit(cls, dataset, ridge=1e-6):
        """Return a fitted baseline; call as LinearDeltaPredictor.fit(train)."""
        if not np.isfinite(ridge) or ridge <= 0:
            raise ValueError("ridge must be positive and finite")
        model = cls()
        model.state_dim = dataset.state_dim
        model.action_dim = dataset.action_dim
        model.input_mean = np.concatenate((dataset.states.mean(axis=0, dtype=np.float64),
                                           dataset.actions.mean(axis=0, dtype=np.float64)))
        input_scale = np.concatenate((dataset.states.std(axis=0, dtype=np.float64),
                                      dataset.actions.std(axis=0, dtype=np.float64)))
        model.input_scale = _scale(input_scale, len(input_scale))
        model.state_scale = _scale(input_scale[:model.state_dim], model.state_dim)
        dimensions = len(input_scale) + 1
        gram = np.zeros((dimensions, dimensions), dtype=np.float64)
        cross = np.zeros((dimensions, model.state_dim), dtype=np.float64)
        for start in range(0, len(dataset), 4096):
            states = dataset.states[start:start + 4096].astype(np.float64)
            actions = dataset.actions[start:start + 4096].astype(np.float64)
            inputs = (np.concatenate((states, actions), axis=1) - model.input_mean) / model.input_scale
            design = np.column_stack((inputs, np.ones(len(inputs))))
            delta = dataset.next_states[start:start + 4096].astype(np.float64) - states
            gram += design.T @ design
            cross += design.T @ delta
        regularization = np.eye(dimensions) * ridge
        regularization[-1, -1] = 0.0  # Do not penalize the intercept.
        model.coefficients = np.linalg.solve(gram / len(dataset) + regularization,
                                            cross / len(dataset))
        if not np.isfinite(model.coefficients).all():
            raise FloatingPointError("linear baseline fit produced nonfinite coefficients")
        return model

    def predict(self, states, actions):
        if not hasattr(self, "coefficients"):
            raise ValueError("fit the linear baseline on training data before prediction")
        states, actions, single = _inputs(states, actions, self.state_dim, self.action_dim)
        inputs = (np.concatenate((states, actions), axis=1) - self.input_mean) / self.input_scale
        design = np.column_stack((inputs, np.ones(len(inputs))))
        result = states + design @ self.coefficients
        result = _checked_prediction(result, states.shape)
        return result[0] if single else result
