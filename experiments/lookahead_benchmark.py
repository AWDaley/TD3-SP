"""Matched-budget, multi-seed Pendulum tests for the optional TD3 controller.

Run from the repository root with ``python experiments/lookahead_benchmark.py run``.
The manifest fixes the protocol before workers start. No settings are selected
using the final evaluation episodes. Raw checkpoints and results live under runs/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))

import gymnasium
import numpy as np
import torch

import TD3
import utils
from action_selection import ActionSelector, SelectionConfig
from action_selection.tasks import task_for_env
from env_compat import make_env, reset_env, step_env
from state_prediction.data import TransitionDataset
from state_prediction.model import StatePredictor


MODES = ("actor", "critic", "lookahead")
SOURCE_FILES = (
    "TD3.py", "utils.py", "env_compat.py", "action_selection/selector.py",
    "action_selection/tasks.py", "state_prediction/model.py",
    "state_prediction/data.py", "experiments/lookahead_benchmark.py",
)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, data, *, replace=False):
    path = Path(path)
    if path.exists() and not replace:
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, indent=2, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=path.name + ".",
                                     suffix=".tmp", encoding="utf-8", delete=False) as stream:
        stream.write(payload)
        temporary = Path(stream.name)
    try:
        if path.exists() and not replace:
            raise FileExistsError(f"Refusing to overwrite {path}")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def network_hash(network):
    digest = hashlib.sha256()
    for name, tensor in sorted(network.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def latency_summary(values):
    return {"median_ms": float(np.median(values)),
            "p95_ms": float(np.percentile(values, 95)),
            "max_ms": float(np.max(values))}


def make_manifest(args):
    data_path = Path(args.data).resolve()
    prediction_path = Path(args.predictor).resolve()
    data = TransitionDataset.load(data_path)
    if (data.state_dim, data.action_dim) != (3, 1) or data.terminated.any():
        raise ValueError("Expected real standard Pendulum transitions with no true terminal states")
    if args.steps < 1 or not 0 <= args.warmup < args.steps:
        raise ValueError("Require 0 <= warmup < online steps")
    if args.batch_size < 1 or args.episodes < 1 or args.monitor_episodes < 1:
        raise ValueError("Batch size and evaluation counts must be positive")
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        raise ValueError("Training seeds must be unique and nonnegative")
    checkpoints = sorted({min(5000, args.steps), args.steps})
    monitor_steps = sorted({0, min(5000, args.steps), min(10000, args.steps), args.steps})
    return {
        "format": "td3-sp-extensive-benchmark", "version": 1,
        "protocol": {
            "environment": "Pendulum-v1", "training_seeds": args.seeds,
            "training_modes": list(MODES), "online_steps": args.steps,
            "prefill_transitions": len(data),
            "real_collection_budget_per_run": len(data) + args.steps,
            "warmup_steps": args.warmup, "batch_size": args.batch_size,
            "discount": 0.99, "tau": 0.005, "actor_learning_rate": 0.0003,
            "critic_learning_rate": 0.0003, "policy_noise_absolute": 0.4,
            "noise_clip_absolute": 1.0, "policy_freq": 2,
            "candidates": 16, "candidate_noise": 0.2,
            "exploration_noise": 0.1, "training_lookahead_horizon": 1,
            "monitor_steps": monitor_steps,
            "monitor_reset_seeds": list(range(200000, 200000 + args.monitor_episodes)),
            "final_reset_seeds": list(range(900000, 900000 + args.episodes)),
            "evaluation_candidate_seed_offset": 10000000,
            "training_selector_seed_offset": 1000000,
            "frozen_horizons": [1, 3, 5], "checkpoint_steps": checkpoints,
            "torch_threads_per_worker": 1,
        },
        "data": {"path": str(data_path), "sha256": file_hash(data_path),
                 "fingerprint": data.fingerprint(), "episodes": data.episode_count},
        "predictor": {"path": str(prediction_path), "sha256": file_hash(prediction_path)},
        "sources": {name: file_hash(REPOSITORY / name) for name in SOURCE_FILES},
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True).strip(),
        "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                    "numpy": np.__version__, "gymnasium": gymnasium.__version__, "device": "cpu"},
        "interpretation": [
            "Primary control endpoint is horizon-1 lookahead; horizons3/5 are fixed diagnostics.",
            "Every training arm receives the same archived real transition prefill.",
            "Archived Pendulum rewards are reconstructed analytically from recorded observations/actions.",
            "The predictor is fixed and final/monitor evaluation data never enters replay or fitting.",
            "Training seed is the replication unit; reset episodes within one policy are not independent training runs.",
            "All modes use private selector RNGs so candidate sampling does not consume TD3 replay/target-noise streams.",
            "Reported online training cost excludes earlier shared dataset collection and predictor fitting.",
            "Shared-host timings include scheduling contention; they are descriptive rather than deployment guarantees.",
        ],
    }


def check_manifest_files(manifest):
    for field in ("data", "predictor"):
        item = manifest[field]
        if file_hash(item["path"]) != item["sha256"]:
            raise ValueError(f"{field} changed after the protocol was fixed")
    for name, digest in manifest["sources"].items():
        if file_hash(REPOSITORY / name) != digest:
            raise ValueError(f"Source changed after the protocol was fixed: {name}")


def build_selector(policy, predictor, env, mode, horizon, seed, protocol):
    return ActionSelector(
        policy, env.action_space.low, env.action_space.high,
        SelectionConfig(mode=mode, candidates=protocol["candidates"],
                        noise_std=protocol["candidate_noise"], horizon=horizon, seed=seed),
        predictor=predictor if mode == "lookahead" else None,
        task=task_for_env(env) if mode == "lookahead" else None,
    )


def evaluate(policy, predictor, mode, horizon, episode_seeds, protocol):
    started = time.perf_counter()
    env = make_env(protocol["environment"])
    rows, latencies = [], []
    before = (network_hash(policy.actor), network_hash(policy.critic), network_hash(predictor))
    numpy_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()
    try:
        for episode_seed in episode_seeds:
            state = reset_env(env, episode_seed)
            selector = build_selector(
                policy, predictor, env, mode, horizon,
                episode_seed + protocol["evaluation_candidate_seed_offset"], protocol,
            )
            rewards, timings = [], []
            interventions = fallbacks = invalid = 0
            done = False
            while not done:
                action = selector.select_action(state)
                nxt, reward, terminated, truncated, _ = step_env(env, action)
                if not np.isfinite(reward):
                    raise FloatingPointError("Environment emitted a nonfinite reward")
                rewards.append(float(reward))
                timings.append(selector.last_info["elapsed_ms"])
                interventions += int(selector.last_info["intervention"])
                fallbacks += int(selector.last_info["fallback"])
                invalid += selector.last_info["invalid_candidates"]
                state, done = nxt, terminated or truncated
            latencies.extend(timings)
            rows.append({"seed": episode_seed, "return": float(sum(rewards)),
                         "steps": len(rewards), "terminated": terminated, "truncated": truncated,
                         "interventions": interventions, "fallbacks": fallbacks,
                         "invalid_candidates": invalid, "latency": latency_summary(timings)})
    finally:
        env.close()
    after = (network_hash(policy.actor), network_hash(policy.critic), network_hash(predictor))
    if before != after:
        raise AssertionError("Evaluation changed network parameters or buffers")
    numpy_after = np.random.get_state()
    if (numpy_before[0] != numpy_after[0] or not np.array_equal(numpy_before[1], numpy_after[1])
            or numpy_before[2:] != numpy_after[2:] or not torch.equal(torch_before, torch.get_rng_state())):
        raise AssertionError("Evaluation consumed global TD3 random streams")
    steps = sum(row["steps"] for row in rows)
    return {
        "mode": mode, "horizon": horizon if mode == "lookahead" else 0,
        "episodes": rows, "mean_return": float(np.mean([row["return"] for row in rows])),
        "std_return_across_episodes": float(np.std([row["return"] for row in rows])),
        "steps": steps, "intervention_rate": sum(row["interventions"] for row in rows) / steps,
        "fallbacks": sum(row["fallbacks"] for row in rows),
        "invalid_candidates": sum(row["invalid_candidates"] for row in rows),
        "latency": latency_summary(latencies), "evaluation_seconds": time.perf_counter() - started,
        "network_and_global_rng_unchanged": True,
    }


def save_checkpoint(policy, prefix, step):
    prefix = Path(prefix)
    identities = {}
    for name in ("actor", "critic"):
        network = getattr(policy, name)
        if any(not torch.isfinite(value).all() for value in network.state_dict().values()):
            raise FloatingPointError(f"Nonfinite {name} weights at step {step}")
        path = Path(str(prefix) + "_" + name)
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
        temporary = Path(str(path) + ".tmp")
        torch.save(network.state_dict(), temporary)
        temporary.replace(path)
        identities[name] = {"path": str(path), "sha256": file_hash(path)}
    marker = {"online_step": step, "td3_training_calls": policy.total_it,
              "actor_updates": policy.total_it // policy.policy_freq, "weights": identities}
    write_json(str(prefix) + ".json", marker)
    return marker


def train_worker(output, seed, mode):
    output = Path(output).resolve()
    manifest = json.loads((output / "manifest.json").read_text())
    check_manifest_files(manifest)
    protocol = manifest["protocol"]
    if seed not in protocol["training_seeds"] or mode not in protocol["training_modes"]:
        raise ValueError("Worker is outside the registered protocol")
    destination = output / f"seed_{seed}" / mode
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "result.json").exists() or (destination / "progress.json").exists():
        raise FileExistsError(f"Worker directory already has results/progress: {destination}")
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        raise ValueError("This registered benchmark is CPU-only; select a CPU runtime")
    torch.manual_seed(seed)
    np.random.seed(seed)
    data = TransitionDataset.load(manifest["data"]["path"])
    predictor = StatePredictor.load(manifest["predictor"]["path"], device="cpu")
    if not predictor.normalizers_fitted or (predictor.state_dim, predictor.action_dim) != (3, 1):
        raise ValueError("A fitted Pendulum predictor is required")
    model_initial = network_hash(predictor)
    policy = TD3.TD3(3, 1, 2.0, discount=protocol["discount"], tau=protocol["tau"],
                     policy_noise=protocol["policy_noise_absolute"],
                     noise_clip=protocol["noise_clip_absolute"], policy_freq=protocol["policy_freq"])
    initial_policy = {"actor": network_hash(policy.actor), "critic": network_hash(policy.critic)}
    replay = utils.ReplayBuffer(3, 1, max_size=len(data) + protocol["online_steps"] + 1)
    env = make_env(protocol["environment"])
    task = task_for_env(env)
    with torch.no_grad():
        rewards = task(torch.tensor(data.states.copy()), torch.tensor(data.actions.copy()),
                       torch.tensor(data.next_states.copy()))[0].numpy()
    for index in range(len(data)):
        replay.add(data.states[index], data.actions[index], data.next_states[index],
                   float(rewards[index]), float(data.terminated[index]))
    selector = build_selector(policy, predictor, env, mode, 1,
                              seed + protocol["training_selector_seed_offset"], protocol)
    monitor, checkpoints, training_returns = [], [], []
    training_seconds = 0.0
    training_stats = {"selections": 0, "interventions": 0, "fallbacks": 0, "invalid_candidates": 0}
    worker_start = time.perf_counter()
    write_json(destination / "progress.json", {"phase": "initial_evaluation", "online_step": 0}, replace=True)
    try:
        initial_eval = evaluate(policy, predictor, mode, 1, protocol["monitor_reset_seeds"], protocol)
        monitor.append({"online_step": 0, "real_collection_budget": len(data),
                        "training_seconds": 0.0, "evaluation": initial_eval})
        state = reset_env(env, seed)
        env.action_space.seed(seed)
        episode_return = 0.0
        for step in range(1, protocol["online_steps"] + 1):
            tick = time.perf_counter()
            if step <= protocol["warmup_steps"]:
                action = env.action_space.sample()
            else:
                action = selector.select_action(state, exploration_std=protocol["exploration_noise"])
                training_stats["selections"] += 1
                training_stats["interventions"] += int(selector.last_info["intervention"])
                training_stats["fallbacks"] += int(selector.last_info["fallback"])
                training_stats["invalid_candidates"] += selector.last_info["invalid_candidates"]
            nxt, reward, terminated, truncated, _ = step_env(env, action)
            if not (np.isfinite(action).all() and np.isfinite(nxt).all() and np.isfinite(reward)):
                raise FloatingPointError("Nonfinite training transition")
            replay.add(state, action, nxt, reward, float(terminated))
            episode_return += reward
            state = nxt
            if step > protocol["warmup_steps"]:
                policy.train(replay, protocol["batch_size"])
            if terminated or truncated:
                training_returns.append({"ending_step": step, "return": float(episode_return)})
                episode_return = 0.0
                state = reset_env(env)
            training_seconds += time.perf_counter() - tick
            if step in protocol["checkpoint_steps"]:
                marker = save_checkpoint(policy, destination / f"checkpoint_{step:05d}", step)
                checkpoints.append(marker)
                print(json.dumps({"checkpoint": str(destination / f"checkpoint_{step:05d}"),
                                  "seed": seed, "mode": mode, "online_step": step}), flush=True)
            if step in protocol["monitor_steps"]:
                evaluation = evaluate(policy, predictor, mode, 1, protocol["monitor_reset_seeds"], protocol)
                monitor.append({"online_step": step, "real_collection_budget": len(data) + step,
                                "training_seconds": training_seconds, "evaluation": evaluation})
                write_json(destination / "monitor.json", monitor, replace=True)
            if step % 1000 == 0 or step == protocol["online_steps"]:
                write_json(destination / "progress.json", {
                    "phase": "training", "seed": seed, "mode": mode, "online_step": step,
                    "training_seconds": training_seconds, "elapsed_seconds": time.perf_counter() - worker_start,
                    "td3_training_calls": policy.total_it, "selection": training_stats,
                }, replace=True)
        expected_updates = protocol["online_steps"] - protocol["warmup_steps"]
        if policy.total_it != expected_updates or replay.size != len(data) + protocol["online_steps"]:
            raise AssertionError("Training update or real replay budget mismatch")
        write_json(destination / "progress.json", {"phase": "final_evaluation", "seed": seed,
                   "mode": mode, "online_step": protocol["online_steps"]}, replace=True)
        final = {}
        final["deployed"] = evaluate(policy, predictor, mode, 1, protocol["final_reset_seeds"], protocol)
        final["actor_only"] = (final["deployed"] if mode == "actor" else
            evaluate(policy, predictor, "actor", 1, protocol["final_reset_seeds"], protocol))
        if mode == "actor":
            final["frozen_selectors"] = {"actor": final["deployed"]}
            final["frozen_selectors"]["critic"] = evaluate(
                policy, predictor, "critic", 1, protocol["final_reset_seeds"], protocol)
            for horizon in protocol["frozen_horizons"]:
                final["frozen_selectors"][f"lookahead_h{horizon}"] = evaluate(
                    policy, predictor, "lookahead", horizon, protocol["final_reset_seeds"], protocol)
        if network_hash(predictor) != model_initial or file_hash(manifest["predictor"]["path"]) != manifest["predictor"]["sha256"]:
            raise AssertionError("Fixed predictor changed during training")
        result = {
            "format": "td3-sp-extensive-run", "version": 1, "seed": seed, "training_mode": mode,
            "manifest_sha256": file_hash(output / "manifest.json"),
            "online_steps": protocol["online_steps"], "prefill_transitions": len(data),
            "real_collection_budget": len(data) + protocol["online_steps"],
            "td3_training_calls": policy.total_it, "actor_updates": policy.total_it // policy.policy_freq,
            "initial_policy": initial_policy, "checkpoints": checkpoints,
            "training_selection": training_stats, "training_episode_returns": training_returns,
            "online_training_seconds": training_seconds, "total_worker_seconds": time.perf_counter() - worker_start,
            "monitor": monitor, "final": final,
            "verification": {"predictor_unchanged": True, "replay_contains_real_transitions_only": True,
                             "evaluation_excluded_from_replay": True, "global_rng_preserved_during_evaluation": True},
        }
        write_json(destination / "result.json", result)
        write_json(destination / "progress.json", {"phase": "complete", "seed": seed, "mode": mode,
                   "online_step": protocol["online_steps"], "training_seconds": training_seconds}, replace=True)
        print(json.dumps({"complete": str(destination), "seed": seed, "mode": mode,
                          "deployed_return": final["deployed"]["mean_return"],
                          "actor_return": final["actor_only"]["mean_return"],
                          "online_training_seconds": training_seconds}), flush=True)
        return result
    finally:
        env.close()


def aggregate(output, *, require_complete=True):
    output = Path(output).resolve()
    manifest = json.loads((output / "manifest.json").read_text())
    results, missing = {}, []
    for seed in manifest["protocol"]["training_seeds"]:
        for mode in MODES:
            path = output / f"seed_{seed}" / mode / "result.json"
            if path.exists():
                result = json.loads(path.read_text())
                if result["manifest_sha256"] != file_hash(output / "manifest.json"):
                    raise ValueError(f"Result manifest mismatch: {path}")
                results[f"{seed}/{mode}"] = result
            else:
                missing.append(f"{seed}/{mode}")
    if require_complete and missing:
        raise ValueError(f"Incomplete benchmark: {missing}")
    summary = {"format": "td3-sp-extensive-summary", "version": 1,
               "manifest": manifest, "completed_runs": len(results), "missing_runs": missing,
               "per_seed": {}, "aggregate": {}, "paired_differences": {}}
    if not missing:
        seeds = manifest["protocol"]["training_seeds"]
        for seed in seeds:
            summary["per_seed"][str(seed)] = {
                mode: {"deployed_return": results[f"{seed}/{mode}"]["final"]["deployed"]["mean_return"],
                       "actor_only_return": results[f"{seed}/{mode}"]["final"]["actor_only"]["mean_return"],
                       "online_training_seconds": results[f"{seed}/{mode}"]["online_training_seconds"]}
                for mode in MODES
            }
        for mode in MODES:
            values = [summary["per_seed"][str(seed)][mode]["deployed_return"] for seed in seeds]
            actors = [summary["per_seed"][str(seed)][mode]["actor_only_return"] for seed in seeds]
            summary["aggregate"][mode] = {"per_seed_deployed_returns": values,
                "mean_deployed_return": float(np.mean(values)), "std_between_training_seeds": float(np.std(values, ddof=1)) if len(values)>1 else 0.0,
                "mean_actor_only_return": float(np.mean(actors)),
                "mean_online_training_seconds": float(np.mean([results[f"{seed}/{mode}"]["online_training_seconds"] for seed in seeds]))}
        for treatment, reference in (("lookahead", "actor"), ("lookahead", "critic"), ("critic", "actor")):
            diffs = [summary["per_seed"][str(seed)][treatment]["deployed_return"]
                     - summary["per_seed"][str(seed)][reference]["deployed_return"] for seed in seeds]
            summary["paired_differences"][f"{treatment}_minus_{reference}"] = {
                "per_training_seed": diffs, "mean": float(np.mean(diffs)),
                "positive_seeds": int(np.count_nonzero(np.array(diffs) > 0)), "seed_count": len(seeds)}
        summary["frozen_selectors"] = {}
        for label in ("actor", "critic", "lookahead_h1", "lookahead_h3", "lookahead_h5"):
            values = [results[f"{seed}/actor"]["final"]["frozen_selectors"][label]["mean_return"] for seed in seeds]
            summary["frozen_selectors"][label] = {"per_training_seed": values,
                                                "mean_return": float(np.mean(values))}
        summary["monitor_curves"] = {mode: [
            {"online_step": point["online_step"], "real_collection_budget": point["real_collection_budget"],
             "per_training_seed": [results[f"{seed}/{mode}"]["monitor"][index]["evaluation"]["mean_return"] for seed in seeds],
             "mean_return": float(np.mean([results[f"{seed}/{mode}"]["monitor"][index]["evaluation"]["mean_return"] for seed in seeds]))}
            for index, point in enumerate(results[f"{seeds[0]}/{mode}"]["monitor"])] for mode in MODES}
    write_json(output / "summary.json", summary, replace=True)
    return summary


def run(args):
    output = Path(args.output).resolve()
    desired = make_manifest(args)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        if not args.resume:
            raise FileExistsError("Output has a manifest; use --resume only for the identical protocol")
        manifest = json.loads(manifest_path.read_text())
        if manifest != desired:
            raise ValueError("Resume protocol/source/data differs from the saved manifest")
    else:
        write_json(manifest_path, desired)
    jobs = [(seed, mode) for seed in args.seeds for mode in MODES]
    pending = []
    for seed, mode in jobs:
        directory = output / f"seed_{seed}" / mode
        if (directory / "result.json").exists():
            continue
        if (directory / "progress.json").exists():
            raise ValueError(f"Incomplete worker {directory}; preserve it and start a new output directory")
        pending.append((seed, mode))
    active, completed = [], len(jobs) - len(pending)
    try:
        while pending or active:
            while pending and len(active) < args.workers:
                seed, mode = pending.pop(0)
                directory = output / f"seed_{seed}" / mode
                directory.mkdir(parents=True, exist_ok=True)
                log = (directory / "worker.log").open("x", encoding="utf-8")
                process = subprocess.Popen(
                    [sys.executable, "-u", str(Path(__file__).resolve()), "worker", "--output", str(output),
                     "--seed", str(seed), "--mode", mode], cwd=REPOSITORY, stdout=log, stderr=subprocess.STDOUT,
                    env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
                )
                active.append((process, log, seed, mode))
                print(json.dumps({"started": {"seed": seed, "mode": mode}, "completed": completed,
                                  "total_runs": len(jobs)}), flush=True)
            for item in active[:]:
                process, log, seed, mode = item
                code = process.poll()
                if code is None:
                    continue
                log.close()
                active.remove(item)
                if code != 0:
                    raise RuntimeError(f"Worker failed: seed={seed}, mode={mode}; inspect its worker.log")
                completed += 1
                aggregate(output, require_complete=False)
                print(json.dumps({"completed_run": {"seed": seed, "mode": mode},
                                  "completed": completed, "total_runs": len(jobs)}), flush=True)
            if active:
                time.sleep(1)
    finally:
        for process, log, _, _ in active:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=30)
            log.close()
    return aggregate(output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    launch = commands.add_parser("run")
    launch.add_argument("--output", default="runs/lookahead-extensive")
    launch.add_argument("--data", default="runs/pendulum-data.npz")
    launch.add_argument("--predictor", default="runs/pendulum-improved/model.pt")
    launch.add_argument("--steps", type=int, default=20000)
    launch.add_argument("--warmup", type=int, default=1000)
    launch.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33, 44, 55])
    launch.add_argument("--batch_size", type=int, default=64)
    launch.add_argument("--episodes", type=int, default=20)
    launch.add_argument("--monitor_episodes", type=int, default=5)
    launch.add_argument("--workers", type=int, default=3)
    launch.add_argument("--resume", action="store_true")
    worker = commands.add_parser("worker")
    worker.add_argument("--output", required=True)
    worker.add_argument("--seed", type=int, required=True)
    worker.add_argument("--mode", choices=MODES, required=True)
    summary = commands.add_parser("aggregate")
    summary.add_argument("--output", default="runs/lookahead-extensive")
    args = parser.parse_args(argv)
    if args.command == "run":
        if not 1 <= args.workers <= 3:
            parser.error("--workers must be between one and three")
        result = run(args)
    elif args.command == "worker":
        result = train_worker(args.output, args.seed, args.mode)
    else:
        result = aggregate(args.output)
    print(json.dumps({key: result[key] for key in ("completed_runs", "aggregate", "paired_differences")
                      if key in result}, indent=2), flush=True)


if __name__ == "__main__":
    main()
