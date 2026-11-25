"""Paired, frozen-policy comparisons of actor, critic search, and lookahead.

This is an evaluation harness, not a training benchmark. Each mode uses the
same checkpoint and reset seeds. Critic search and lookahead also receive the
same candidate-noise stream at each episode; their visited states can differ.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from env_compat import continuous_dimensions, make_env, reset_env, step_env
from state_prediction.model import StatePredictor

from .selector import ActionSelector, SelectionConfig
from .tasks import task_for_env


MODES = ("actor", "critic", "lookahead")


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or int(value) != value or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _latency_summary(values):
    return {
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


@contextmanager
def _evaluation_mode(policy, predictor):
    """Disable training behavior without changing any caller's module modes."""
    modules = {}
    for network in (policy.actor, policy.critic, predictor):
        for module in network.modules():
            modules[id(module)] = (module, module.training)
    try:
        for network in (policy.actor, policy.critic, predictor):
            network.eval()
        with torch.no_grad():
            yield
    finally:
        # Assign each stored flag individually, preserving mixed parent/child
        # modes rather than recursively overwriting them with .train().
        for module, training in modules.values():
            module.training = training


def compare_policy(
    policy,
    predictor,
    *,
    env_name="Pendulum-v1",
    episodes=10,
    seed=123,
    candidates=16,
    noise_std=0.2,
    horizon=1,
    env_factory=None,
    task_factory=None,
):
    """Evaluate a fixed policy three ways and return a JSON-compatible report.

    ``env_factory(name)`` and ``task_factory(env)`` permit other environments
    with an explicit reward/true-termination adapter. No updates, fitting, or
    exploration are performed. Evaluation episodes are paired observations,
    not independent training seeds.
    """
    episodes = _integer(episodes, "episodes")
    seed = _integer(seed, "seed", minimum=0)
    # Validate selector settings before creating or stepping an environment.
    SelectionConfig(mode="lookahead", candidates=candidates, noise_std=noise_std,
                    horizon=horizon, seed=seed)
    if not predictor.normalizers_fitted:
        raise ValueError("The prediction checkpoint must have fitted normalizers")
    env_factory = make_env if env_factory is None else env_factory
    task_factory = task_for_env if task_factory is None else task_factory
    episode_seeds = [seed + episode for episode in range(episodes)]
    # A separate stream makes environment randomness and proposal randomness
    # independent, and resets proposal sampling identically for both searches.
    candidate_seeds = [seed + 1_000_000 + episode for episode in range(episodes)]
    report = {
        "format": "td3-sp-action-selection-comparison",
        "version": 1,
        "environment": env_name,
        "settings": {
            "episodes": episodes,
            "episode_seeds": episode_seeds,
            "candidate_seeds": candidate_seeds,
            "candidates": int(candidates),
            "noise_std": float(noise_std),
            "horizon": int(horizon),
            "discount": float(policy.discount),
            "exploration_std": 0.0,
            "device": str(next(policy.actor.parameters()).device),
        },
        "interpretation": [
            "All modes use one frozen actor, critic pair, and predictor.",
            "Reset seeds and candidate-noise streams are paired; trajectories may diverge.",
            "Return differences are descriptive for these episodes, not evidence across training seeds.",
            "Latency is wall-clock action selection time and includes first-call overhead.",
        ],
        "modes": {},
    }
    with _evaluation_mode(policy, predictor):
        for mode in MODES:
            env = env_factory(env_name)
            try:
                state_dim, action_dim = continuous_dimensions(env, require_symmetric_actions=True)
                if (predictor.state_dim, predictor.action_dim) != (state_dim, action_dim):
                    raise ValueError("Prediction checkpoint dimensions do not match the environment")
                task = task_factory(env)
                episode_reports = []
                all_latencies = []
                for episode, episode_seed in enumerate(episode_seeds):
                    selector = ActionSelector(
                        policy, env.action_space.low, env.action_space.high,
                        SelectionConfig(
                            mode=mode, candidates=candidates, noise_std=noise_std,
                            horizon=horizon if mode == "lookahead" else 1,
                            seed=candidate_seeds[episode],
                        ),
                        predictor=predictor if mode == "lookahead" else None,
                        task=task if mode == "lookahead" else None,
                    )
                    state = reset_env(env, episode_seed)
                    total_reward = 0.0
                    steps = interventions = fallbacks = invalid_candidates = 0
                    latencies = []
                    terminated = truncated = False
                    while not (terminated or truncated):
                        start = time.perf_counter()
                        action = selector.select_action(np.asarray(state), exploration_std=0.0)
                        latencies.append((time.perf_counter() - start) * 1000.0)
                        state, reward, terminated, truncated, _ = step_env(env, action)
                        if not math.isfinite(float(reward)):
                            raise ValueError("Evaluation environment returned a nonfinite reward")
                        total_reward += float(reward)
                        steps += 1
                        interventions += int(selector.last_info["intervention"])
                        fallbacks += int(selector.last_info["fallback"])
                        invalid_candidates += int(selector.last_info["invalid_candidates"])
                    all_latencies.extend(latencies)
                    episode_reports.append({
                        "episode": episode,
                        "seed": episode_seed,
                        "candidate_seed": candidate_seeds[episode],
                        "return": total_reward,
                        "steps": steps,
                        "terminated": terminated,
                        "truncated": truncated,
                        "interventions": interventions,
                        "fallbacks": fallbacks,
                        "invalid_candidates": invalid_candidates,
                        "latency_ms": _latency_summary(latencies),
                    })
                returns = [episode["return"] for episode in episode_reports]
                total_steps = sum(episode["steps"] for episode in episode_reports)
                report["modes"][mode] = {
                    "episodes": episode_reports,
                    "summary": {
                        "mean_return": float(np.mean(returns)),
                        "std_return": float(np.std(returns)),
                        "total_steps": total_steps,
                        "intervention_rate": sum(e["interventions"] for e in episode_reports) / total_steps,
                        "fallback_rate": sum(e["fallbacks"] for e in episode_reports) / total_steps,
                        "invalid_candidates": sum(e["invalid_candidates"] for e in episode_reports),
                        "latency_ms": _latency_summary(all_latencies),
                    },
                }
            finally:
                env.close()
    report["paired_return_differences"] = {}
    for treatment, reference in (("critic", "actor"), ("lookahead", "actor"), ("lookahead", "critic")):
        differences = [
            treatment_episode["return"] - reference_episode["return"]
            for treatment_episode, reference_episode in zip(
                report["modes"][treatment]["episodes"], report["modes"][reference]["episodes"]
            )
        ]
        report["paired_return_differences"][f"{treatment}_minus_{reference}"] = {
            "per_episode": differences,
            "mean": float(np.mean(differences)),
        }
    # Fail rather than silently producing a nonstandard JSON report.
    json.dumps(report, allow_nan=False)
    return report


def _file_identity(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest()}


def compare_checkpoints(
    policy_checkpoint,
    prediction_checkpoint,
    output,
    *,
    env_name="Pendulum-v1",
    episodes=10,
    seed=123,
    candidates=16,
    noise_std=0.2,
    horizon=1,
    discount=0.99,
):
    """Load saved actor/critic weights and write a new frozen-comparison report."""
    import TD3

    output = Path(output)
    if output.exists():
        raise ValueError(f"Output already exists: {output}; choose a new report path")
    _integer(episodes, "episodes")
    _integer(seed, "seed", minimum=0)
    SelectionConfig(mode="lookahead", candidates=candidates, noise_std=noise_std,
                    horizon=horizon, seed=seed)
    if not math.isfinite(discount) or not 0 <= discount <= 1:
        raise ValueError("discount must be finite and between 0 and 1")
    identities = {
        "actor": _file_identity(str(policy_checkpoint) + "_actor"),
        "critic": _file_identity(str(policy_checkpoint) + "_critic"),
        "predictor": _file_identity(prediction_checkpoint),
    }
    env = make_env(env_name)
    try:
        state_dim, action_dim = continuous_dimensions(env, require_symmetric_actions=True)
        task_for_env(env)  # Reject unsupported reward semantics before loading.
        # Loading replaces every network weight. Preserve the caller's CPU RNG
        # despite the temporary actor/critic initialization done by TD3.
        with torch.random.fork_rng(devices=[]):
            policy = TD3.TD3(state_dim, action_dim, float(env.action_space.high[0]),
                             discount=discount)
        policy.load_for_evaluation(str(policy_checkpoint))
        predictor = StatePredictor.load(prediction_checkpoint,
                                        device=next(policy.actor.parameters()).device)
    finally:
        env.close()
    report = compare_policy(
        policy, predictor, env_name=env_name, episodes=episodes, seed=seed,
        candidates=candidates, noise_std=noise_std, horizon=horizon,
    )
    report["checkpoints"] = identities
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects an existing report even if another process
    # created it while the comparison was running.
    with output.open("x", encoding="utf-8") as destination:
        destination.write(payload)
    return report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    compare = commands.add_parser("compare", help="Compare three action selectors using frozen checkpoints")
    compare.add_argument("--env", default="Pendulum-v1")
    compare.add_argument("--policy_checkpoint", required=True,
                         help="Saved TD3 prefix (without _actor or _critic)")
    compare.add_argument("--prediction_checkpoint", required=True)
    compare.add_argument("--output", required=True, help="New JSON report path; existing files are preserved")
    compare.add_argument("--episodes", type=int, default=10)
    compare.add_argument("--seed", type=int, default=123)
    compare.add_argument("--candidates", type=int, default=16)
    compare.add_argument("--noise_std", type=float, default=0.2,
                         help="Candidate perturbation standard deviation as a fraction of action half-range")
    compare.add_argument("--horizon", type=int, default=1)
    compare.add_argument("--discount", type=float, default=0.99,
                         help="Discount used to train the saved TD3 checkpoint")
    compare.add_argument("--threads", type=int, default=None)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    previous_threads = torch.get_num_threads()
    try:
        if args.threads is not None:
            _integer(args.threads, "threads")
            torch.set_num_threads(args.threads)
        report = compare_checkpoints(
            args.policy_checkpoint, args.prediction_checkpoint, args.output,
            env_name=args.env, episodes=args.episodes, seed=args.seed,
            candidates=args.candidates, noise_std=args.noise_std,
            horizon=args.horizon, discount=args.discount,
        )
    except (ValueError, OSError, RuntimeError) as exc:
        parser.error(str(exc))
    finally:
        if torch.get_num_threads() != previous_threads:
            torch.set_num_threads(previous_threads)
    for mode in MODES:
        summary = report["modes"][mode]["summary"]
        print(f"{mode}: mean return {summary['mean_return']:.3f}; "
              f"action latency p95 {summary['latency_ms']['p95']:.3f} ms; "
              f"interventions {summary['intervention_rate']:.1%}")
    print("Frozen-checkpoint comparison; these episodes do not establish a training improvement.")
    print(f"Saved {args.output}")
    return report
