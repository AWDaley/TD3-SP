"""Held-out Pendulum dynamics and first-action decision diagnostics.

The exact dynamics below are a diagnostic oracle, not a deployable learned model.
They are checked against Gymnasium before every recorded experiment. A candidate's
reference value is its 200-step discounted return followed by the frozen actor;
this is not a globally optimal value or a full replanning-controller benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import TD3
from action_selection import ActionSelector, SelectionConfig
from action_selection.tasks import PendulumTask
from state_prediction.model import StatePredictor


class ExactPendulum(nn.Module):
    """Pendulum-v1 equations in observation coordinates, preserving input dtype."""

    state_dim = 3
    action_dim = 1

    def __init__(self, *, gravity=10.0, dt=0.05, mass=1.0, length=1.0,
                 max_speed=8.0, max_torque=2.0):
        super().__init__()
        values = (gravity, dt, mass, length, max_speed, max_torque)
        if not all(math.isfinite(value) and value > 0 for value in values):
            raise ValueError("Pendulum parameters must be finite and positive")
        self.gravity, self.dt, self.mass, self.length = gravity, dt, mass, length
        self.max_speed, self.max_torque = max_speed, max_torque

    def forward(self, states, actions):
        if states.ndim != 2 or states.shape[1] != 3 or actions.shape != (len(states), 1):
            raise ValueError("Use states [batch, 3] and actions [batch, 1]")
        theta = torch.atan2(states[:, 1], states[:, 0])
        torque = actions[:, 0].clamp(-self.max_torque, self.max_torque)
        velocity = states[:, 2] + (
            3 * self.gravity / (2 * self.length) * torch.sin(theta)
            + 3 / (self.mass * self.length ** 2) * torque
        ) * self.dt
        velocity = velocity.clamp(-self.max_speed, self.max_speed)
        theta = theta + velocity * self.dt
        return torch.stack((torch.cos(theta), torch.sin(theta), velocity), dim=1)


def _digest(path):
    path = Path(path)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _runtime():
    import gymnasium
    return {"python": platform.python_version(), "torch": torch.__version__,
            "numpy": np.__version__, "gymnasium": gymnasium.__version__,
            "device": "cpu", "torch_threads": torch.get_num_threads()}


def held_out_states(seed=812345, states_per_group=32):
    """Balanced diagnostic strata, not a sample from the learned policy."""
    if states_per_group < 1:
        raise ValueError("states_per_group must be positive")
    rng = np.random.default_rng(seed)
    angle = np.concatenate((
        rng.uniform(-np.pi, np.pi, states_per_group),
        rng.uniform(-0.2, 0.2, states_per_group),
        rng.uniform(-np.pi, np.pi, states_per_group),
    ))
    velocity = np.concatenate((
        rng.uniform(-4.0, 4.0, states_per_group),
        rng.uniform(-1.0, 1.0, states_per_group),
        rng.uniform(7.0, 8.0, states_per_group)
        * rng.choice(np.array([-1.0, 1.0]), states_per_group),
    ))
    groups = np.repeat(["nominal", "near_upright", "high_speed"], states_per_group)
    states = np.stack((np.cos(angle), np.sin(angle), velocity), axis=1)
    return states, groups


def validate_oracle(seed=91842, samples=128):
    """Cross-check double-precision equations against installed Gymnasium."""
    import gymnasium as gym
    from gymnasium.envs.classic_control.pendulum import PendulumEnv
    rng = np.random.default_rng(seed)
    max_state_error = max_reward_error = 0.0
    for gravity in (10.0, 15.0):
        env = gym.make("Pendulum-v1", g=gravity)
        oracle = ExactPendulum(gravity=gravity)
        task = PendulumTask()
        try:
            env.reset(seed=seed)
            for _ in range(samples):
                theta, velocity = rng.uniform(-4 * np.pi, 4 * np.pi), rng.uniform(-8, 8)
                action = rng.uniform(-4, 4, size=1)
                env.unwrapped.state = np.array([theta, velocity], dtype=np.float64)
                expected_state, expected_reward, terminal, truncation, _ = env.unwrapped.step(action)
                state = torch.tensor([[np.cos(theta), np.sin(theta), velocity]], dtype=torch.float64)
                control = torch.as_tensor(action[None], dtype=torch.float64)
                actual_state = oracle(state, control)
                actual_reward, _ = task(state, control, actual_state)
                max_state_error = max(max_state_error, float(np.max(np.abs(
                    actual_state.numpy()[0] - expected_state))))
                max_reward_error = max(max_reward_error, abs(float(actual_reward[0]) - expected_reward))
                if terminal or truncation:
                    raise AssertionError("Unwrapped Pendulum must not terminate")
        finally:
            env.close()
    # Gym's observation is rounded to float32; the oracle remains float64.
    if max_state_error > 5e-7 or max_reward_error > 2e-12:
        raise AssertionError(f"Exact-dynamics validation failed: {max_state_error=}, {max_reward_error=}")
    return {
        "samples_per_gravity": samples, "gravities": [10.0, 15.0],
        "max_absolute_observation_error": max_state_error,
        "max_absolute_reward_error": max_reward_error,
        "observation_tolerance": 5e-7, "reward_tolerance": 2e-12,
        "explanation": "Observation differences include Gymnasium float32 output rounding.",
        "installed_step_source_sha256": hashlib.sha256(inspect.getsource(PendulumEnv.step).encode()).hexdigest(),
    }


def _prediction_metrics(predicted, actual, scales):
    delta = predicted - actual
    angle_delta = np.arctan2(predicted[:, 1], predicted[:, 0]) - np.arctan2(actual[:, 1], actual[:, 0])
    angle_delta = (angle_delta + np.pi) % (2 * np.pi) - np.pi
    radius_error = np.abs(np.linalg.norm(predicted[:, :2], axis=1) - 1.0)
    excess = np.maximum(np.abs(predicted) - np.array([1.0, 1.0, 8.0]), 0)
    per_feature_violations = excess > 1e-6
    out_of_bounds = per_feature_violations.any(axis=1)
    return {
        "states": len(predicted),
        "rmse": float(np.sqrt(np.mean(delta ** 2))),
        "normalized_rmse": float(np.sqrt(np.mean((delta / scales) ** 2))),
        "per_feature_rmse": np.sqrt(np.mean(delta ** 2, axis=0)).tolist(),
        "angle_rmse_radians": float(np.sqrt(np.mean(angle_delta ** 2))),
        "mean_unit_circle_radius_error": float(radius_error.mean()),
        "max_unit_circle_radius_error": float(radius_error.max()),
        "observation_bounds_violation_fraction": float(out_of_bounds.mean()),
        "per_feature_bounds_violation_fraction": per_feature_violations.mean(axis=0).tolist(),
        "per_feature_maximum_bound_excess": excess.max(axis=0).tolist(),
        "maximum_absolute_angular_velocity": float(np.abs(predicted[:, 2]).max()),
    }


def forecast_diagnostics(predictor, *, seed=812345, states_per_group=32, horizons=(1, 3, 5)):
    states, groups = held_out_states(seed, states_per_group)
    actions = np.random.default_rng(seed + 1).uniform(-2, 2, size=(len(states), max(horizons), 1))
    states_tensor = torch.as_tensor(states, dtype=torch.float32)
    scales = predictor.state_scale.astype(np.float64)
    result = {
        "settings": {"state_seed": seed, "states_per_group": states_per_group,
                     "horizons": list(horizons), "action_seed": seed + 1},
        "normalization": {"source": "saved predictor training-state scales", "scales": scales.tolist()},
        "interpretation": [
            "Balanced held-out state strata are not an estimate of the operating distribution.",
            "Every model and oracle receives identical fixed actions; this isolates forecast error.",
            "Gravity 15 is a deliberate dynamics mismatch; the predictor was fitted for gravity 10.",
            "No fitting or hyperparameter selection uses these states or outcomes.",
        ],
        "scenarios": {},
    }
    with torch.inference_mode():
        predicted = states_tensor.clone()
        forecasts = {}
        for step in range(max(horizons)):
            predicted = predictor(predicted, torch.as_tensor(actions[:, step], dtype=torch.float32))
            if not torch.isfinite(predicted).all():
                raise FloatingPointError("Forecast diagnostics produced nonfinite predictions")
            if step + 1 in horizons:
                forecasts[step + 1] = predicted.numpy().astype(np.float64)
        for gravity in (10.0, 15.0):
            oracle = ExactPendulum(gravity=gravity)
            actual = torch.as_tensor(states, dtype=torch.float64)
            records = {}
            for step in range(max(horizons)):
                actual = oracle(actual, torch.as_tensor(actions[:, step], dtype=torch.float64))
                horizon = step + 1
                if horizon not in horizons:
                    continue
                prediction = forecasts[horizon]
                truth = actual.numpy()
                records[str(horizon)] = {
                    "all": _prediction_metrics(prediction, truth, scales),
                    "groups": {group: _prediction_metrics(prediction[groups == group], truth[groups == group], scales)
                               for group in ("nominal", "near_upright", "high_speed")},
                }
            result["scenarios"][f"gravity_{gravity:g}"] = records
    result["states"] = states.tolist()
    result["groups"] = groups.tolist()
    result["actions"] = actions.tolist()
    return result


def actor_reference_returns(actor, states, candidates, *, horizon=200, discount=0.99):
    """Finite-horizon discounted return after each first action, then the actor."""
    if horizon < 1 or not 0 <= discount <= 1:
        raise ValueError("Invalid reference horizon or discount")
    states = np.asarray(states, dtype=np.float64)
    candidates = np.asarray(candidates, dtype=np.float64)
    if candidates.ndim != 3 or candidates.shape[0] != len(states) or candidates.shape[2] != 1:
        raise ValueError("Candidates must have shape [states, candidates, 1]")
    batch, count, _ = candidates.shape
    current = torch.as_tensor(np.repeat(states, count, axis=0), dtype=torch.float64)
    action = torch.as_tensor(candidates.reshape(-1, 1), dtype=torch.float64)
    returns = torch.zeros(len(current), dtype=torch.float64)
    dynamics, task = ExactPendulum(), PendulumTask()
    reference = next(actor.parameters())
    if reference.device.type != "cpu":
        raise ValueError("The exact reference diagnostic runs on CPU")
    with torch.inference_mode():
        for step in range(horizon):
            if step:
                # Gym observations and the trained actor are float32; latent
                # dynamics retain double precision between observations.
                action = actor(current.to(dtype=reference.dtype)).to(dtype=torch.float64).clamp(-2, 2)
            following = dynamics(current, action)
            reward, _ = task(current, action, following)
            returns += (discount ** step) * reward
            current = following
    return returns.numpy().reshape(batch, count)


def validate_actor_reference(actor, states, candidates, expected, *, horizon=200,
                             discount=0.99, samples=12):
    """Check complete batched return traces against scalar Gym/actor execution.

    The tolerance accommodates float32 neural-network batch/scalar roundoff.
    Individual discrepancies are retained so very small regrets can be assessed
    in light of numerical precision rather than interpreted as exact rankings.
    """
    import gymnasium as gym
    state_indices = np.linspace(0, len(states) - 1, min(samples, len(states)), dtype=int)
    actor_dtype = next(actor.parameters()).dtype
    env = gym.make("Pendulum-v1")
    records = []
    try:
        with torch.inference_mode():
            for sample, index in enumerate(state_indices):
                candidate = (sample * 3) % candidates.shape[1]
                env.reset(seed=1)
                env.unwrapped.state = np.array([
                    np.arctan2(states[index, 1], states[index, 0]), states[index, 2],
                ], dtype=np.float64)
                state = env.unwrapped._get_obs()
                total = 0.0
                for step in range(horizon):
                    action = candidates[index, candidate] if step == 0 else actor(
                        torch.as_tensor(state[None], dtype=actor_dtype)
                    ).numpy()[0].clip(-2, 2)
                    state, reward, _, _, _ = env.unwrapped.step(action)
                    total += discount ** step * reward
                records.append({"state_index": int(index), "candidate_index": int(candidate),
                                "batched_reference_return": float(expected[index, candidate]),
                                "gym_scalar_return": total,
                                "absolute_difference": abs(total - float(expected[index, candidate]))})
    finally:
        env.close()
    maximum = max(row["absolute_difference"] for row in records)
    tolerance = 0.05
    if maximum > tolerance:
        raise AssertionError(f"Batched actor reference differs from Gym: {maximum} > {tolerance}")
    return {"horizon": horizon, "samples": len(records), "max_absolute_return_difference": maximum,
            "tolerance": tolerance, "samples_detail": records,
            "note": "Float32 batch/scalar actor roundoff can affect tiny candidate-return differences."}


def _average_ranks(values):
    values = np.asarray(values)
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    starts = np.cumsum(counts) - counts
    return (starts + (counts - 1) / 2)[inverse]


def ranking_metrics(scores, reference):
    """Tie-aware Spearman and pair agreement against candidate returns."""
    if not (np.isfinite(scores).all() and np.isfinite(reference).all()):
        raise ValueError("Ranking metrics require finite values")
    rank_a, rank_b = _average_ranks(scores), _average_ranks(reference)
    rank_a, rank_b = rank_a - rank_a.mean(), rank_b - rank_b.mean()
    denominator = np.linalg.norm(rank_a) * np.linalg.norm(rank_b)
    correlation = float(np.dot(rank_a, rank_b) / denominator) if denominator > 0 else None
    left, right = np.triu_indices(len(scores), k=1)
    reference_diff = reference[left] - reference[right]
    comparable = np.abs(reference_diff) > 1e-8
    if comparable.any():
        truth = np.sign(reference_diff[comparable])
        prediction = np.sign((scores[left] - scores[right])[comparable])
        agreement = float(np.mean((prediction == truth) + 0.5 * (prediction == 0)))
    else:
        agreement = None
    return {"spearman": correlation, "pairwise_order_agreement": agreement,
            "comparable_pairs": int(comparable.sum())}


def _method_summary(rows):
    def average(key):
        values = [row[key] for row in rows if row.get(key) is not None]
        return float(np.mean(values)) if values else None
    return {
        "states": len(rows),
        "mean_selected_reference_return": average("selected_reference_return"),
        "mean_regret_to_best_tested_candidate": average("regret_to_best_tested_candidate"),
        "max_regret_to_best_tested_candidate": float(max(row["regret_to_best_tested_candidate"] for row in rows)),
        "mean_return_change_from_actor_candidate": average("return_change_from_actor_candidate"),
        "candidate_optimal_selection_fraction": float(np.mean([row["regret_to_best_tested_candidate"] <= 1e-8 for row in rows])),
        "mean_spearman": average("spearman"),
        "mean_pairwise_order_agreement": average("pairwise_order_agreement"),
        "mean_model_score_minus_exact_score_at_selected": average("model_score_minus_exact_score_at_selected"),
    }


def decision_diagnostics(policy, predictor, *, seed=812345, states_per_group=32,
                         candidates=16, noise_std=0.2, horizons=(1, 3, 5), reference_horizon=200):
    states, groups = held_out_states(seed, states_per_group)
    rng = np.random.default_rng(seed + 2)
    with torch.inference_mode():
        raw_actions = policy.actor(torch.as_tensor(states, dtype=torch.float32)).numpy().clip(-2, 2)
    proposals = np.repeat(raw_actions[:, None, :], candidates, axis=1)
    proposals[:, 1:, :] += rng.normal(size=(len(states), candidates - 1, 1)) * noise_std * 2.0
    proposals = proposals.clip(-2, 2)
    # Use the exact float32 first actions that the actual selector can execute.
    proposals = proposals.astype(np.float32)
    reference = actor_reference_returns(policy.actor, states, proposals,
                                         horizon=reference_horizon, discount=policy.discount)
    reference_validation = validate_actor_reference(
        policy.actor, states, proposals, reference,
        horizon=reference_horizon, discount=policy.discount,
    )
    selectors = {"critic": ActionSelector(
        policy, [-2], [2], SelectionConfig(mode="critic", candidates=candidates, seed=seed))}
    for horizon in horizons:
        for kind, model in (("learned", predictor), ("oracle", ExactPendulum())):
            selectors[f"{kind}_h{horizon}"] = ActionSelector(
                policy, [-2], [2], SelectionConfig(mode="lookahead", horizon=horizon, candidates=candidates, seed=seed),
                predictor=model, task=PendulumTask())
    scores = {name: np.stack([selector.score_candidates(state, proposal).cpu().numpy()
                              for state, proposal in zip(states, proposals)])
              for name, selector in selectors.items()}
    if any(not np.isfinite(value).all() for value in scores.values()):
        raise FloatingPointError("Nonfinite candidate score encountered in the decision diagnostic")
    result = {
        "settings": {"state_seed": seed, "states_per_group": states_per_group,
                     "candidates": candidates, "noise_std": noise_std,
                     "candidate_seed": seed + 2, "horizons": list(horizons),
                     "reference_horizon": reference_horizon, "discount": policy.discount},
        "interpretation": [
            "Reference values are finite-horizon discounted returns after one candidate action, then the frozen actor.",
            "Regret is relative to the best tested candidate, not a globally optimal controller.",
            "These balanced diagnostic states do not estimate a policy's operating distribution.",
            "Oracle lookahead still uses a learned critic tail and is not an optimal planner.",
            "Oracle planner scores use float32 like the actual selector; the long reference uses float64 dynamics and float32 actor observations.",
            "Model optimism compares learned and exact-dynamics scores at the same horizon; raw Q scores are not subtracted from finite-horizon returns.",
            "This is one fixed trained policy; state samples are not independent training replicates.",
        ],
        "methods": {},
        "reference_validation": reference_validation,
    }
    for method in ("actor", *selectors):
        rows = []
        for index in range(len(states)):
            chosen = 0 if method == "actor" else int(np.argmax(scores[method][index]))
            outcome = float(reference[index, chosen])
            row = {
                "state_index": index, "group": str(groups[index]), "selected_candidate": chosen,
                "selected_reference_return": outcome,
                "regret_to_best_tested_candidate": float(reference[index].max() - outcome),
                "return_change_from_actor_candidate": float(outcome - reference[index, 0]),
            }
            if method != "actor":
                row.update(ranking_metrics(scores[method][index], reference[index]))
            if method.startswith("learned_h"):
                exact_method = method.replace("learned", "oracle")
                row["model_score_minus_exact_score_at_selected"] = float(
                    scores[method][index, chosen] - scores[exact_method][index, chosen])
            rows.append(row)
        result["methods"][method] = {
            "summary": _method_summary(rows),
            "groups": {group: _method_summary([row for row in rows if row["group"] == group])
                       for group in ("nominal", "near_upright", "high_speed")},
            "states": rows,
        }
    result["inputs"] = {"states": states.tolist(), "groups": groups.tolist(), "candidates": proposals.tolist()}
    result["reference_returns"] = reference.tolist()
    result["candidate_scores"] = {name: values.tolist() for name, values in scores.items()}
    return result


def _write_new(path, report):
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction_checkpoint", required=True)
    parser.add_argument("--policy_checkpoint")
    parser.add_argument("--phase", choices=("forecast", "decision", "all"), default="all")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=812345)
    parser.add_argument("--states_per_group", type=int, default=32)
    parser.add_argument("--candidates", type=int, default=16)
    parser.add_argument("--noise_std", type=float, default=0.2)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--reference_horizon", type=int, default=200)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    if args.phase != "forecast" and not args.policy_checkpoint:
        parser.error("Decision diagnostics require --policy_checkpoint")
    if args.seed < 0 or min(args.states_per_group, args.candidates, args.reference_horizon, args.threads, *args.horizons) < 1:
        parser.error("Counts/horizons must be positive and seed nonnegative")
    if not math.isfinite(args.noise_std) or args.noise_std < 0 or not 0 <= args.discount <= 1:
        parser.error("Noise must be finite/nonnegative and discount in [0,1]")
    output = Path(args.output)
    phases = ("forecast", "decision") if args.phase == "all" else (args.phase,)
    for phase in phases:
        if (output / f"{phase}.json").exists():
            parser.error(f"Refusing to overwrite {output / f'{phase}.json'}")
    torch.set_num_threads(args.threads)
    predictor = StatePredictor.load(args.prediction_checkpoint)
    if (predictor.state_dim, predictor.action_dim) != (3, 1) or not predictor.normalizers_fitted:
        parser.error("Use a fitted Pendulum predictor with dimensions (3,1)")
    validation = validate_oracle()
    common = {
        "format": "td3-sp-lookahead-decision-diagnostics", "version": 1,
        "runtime": _runtime(), "oracle_validation": validation,
        "predictor": _digest(args.prediction_checkpoint),
        "source": {name: _digest(REPO_ROOT / name) for name in (
            "experiments/lookahead_diagnostics.py", "action_selection/selector.py",
            "action_selection/tasks.py", "TD3.py")},
    }
    if "forecast" in phases:
        started = time.perf_counter()
        report = {**common, "phase": "forecast", **forecast_diagnostics(
            predictor, seed=args.seed, states_per_group=args.states_per_group, horizons=tuple(args.horizons))}
        report["elapsed_seconds"] = time.perf_counter() - started
        _write_new(output / "forecast.json", report)
        print(json.dumps({"phase": "forecast", "output": str(output / "forecast.json"),
                          "normalized_rmse": {scenario: {horizon: values["all"]["normalized_rmse"] for horizon, values in rows.items()}
                                              for scenario, rows in report["scenarios"].items()}}, indent=2), flush=True)
    if "decision" in phases:
        # Only CPU diagnostics are supported even on a CUDA-equipped host.
        policy = TD3.TD3(3, 1, 2.0, discount=args.discount)
        policy.actor.to("cpu")
        policy.critic.to("cpu")
        policy.load_for_evaluation(args.policy_checkpoint)
        started = time.perf_counter()
        report = {**common, "phase": "decision",
                  "policy": {suffix: _digest(args.policy_checkpoint + "_" + suffix) for suffix in ("actor", "critic")},
                  **decision_diagnostics(
                      policy, predictor, seed=args.seed, states_per_group=args.states_per_group,
                      candidates=args.candidates, noise_std=args.noise_std, horizons=tuple(args.horizons),
                      reference_horizon=args.reference_horizon)}
        report["elapsed_seconds"] = time.perf_counter() - started
        _write_new(output / "decision.json", report)
        print(json.dumps({"phase": "decision", "output": str(output / "decision.json"),
                          "methods": {name: values["summary"] for name, values in report["methods"].items()}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
