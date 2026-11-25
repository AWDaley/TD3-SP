"""Secondary post-hoc evaluation of early TD3 checkpoints on fresh resets.

The 5,000-step checkpoint was selected after observing the first two training
seeds' monitoring curves. Fresh evaluation resets provide an independent scenario
check, but do not make this a preregistered primary result. No training, model
fitting, or changes to the original benchmark protocol occur here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

import numpy as np
import torch

import TD3
from experiments.lookahead_benchmark import (
    check_manifest_files, evaluate, file_hash, network_hash, write_json,
)
from state_prediction.model import StatePredictor


CONFIGURATIONS = (
    ("actor", "actor"), ("actor", "critic"), ("actor", "lookahead"),
    ("critic", "critic"), ("critic", "actor"),
    ("lookahead", "lookahead"), ("lookahead", "actor"),
)


def configuration_name(training_mode, evaluation_mode):
    return f"trained_{training_mode}_eval_{evaluation_mode}"


def _identity(path):
    return {"path": str(Path(path).resolve()), "sha256": file_hash(path)}


def _checkpoint(benchmark, seed, mode, step):
    prefix = benchmark / f"seed_{seed}" / mode / f"checkpoint_{step:05d}"
    marker_path = Path(str(prefix) + ".json")
    if not marker_path.exists():
        return None
    marker = json.loads(marker_path.read_text())
    if marker["online_step"] != step:
        raise ValueError(f"Unexpected checkpoint step in {marker_path}")
    for network in ("actor", "critic"):
        path = Path(str(prefix) + "_" + network)
        if file_hash(path) != marker["weights"][network]["sha256"]:
            raise ValueError(f"Checkpoint no longer matches its completion marker: {path}")
    return prefix, marker_path, marker


def make_manifest(benchmark, output, args, primary):
    reset_seeds = list(range(args.reset_seed, args.reset_seed + args.episodes))
    protocol = primary["protocol"]
    if args.checkpoint not in protocol["checkpoint_steps"]:
        raise ValueError("The requested checkpoint was not saved by the primary protocol")
    if set(reset_seeds) & set(protocol["monitor_reset_seeds"] + protocol["final_reset_seeds"]):
        raise ValueError("Secondary evaluation reset seeds must be new")
    return {
        "format": "td3-sp-secondary-early-checkpoint", "version": 1,
        "classification": "SECONDARY_POSTHOC",
        "selection_reason": "The 5000-step checkpoint was selected after observing seed 11/22 monitoring curves.",
        "primary_manifest": _identity(benchmark / "manifest.json"),
        "source": _identity(Path(__file__)),
        "predictor": primary["predictor"],
        "protocol": {
            "environment": protocol["environment"],
            "training_seeds": protocol["training_seeds"],
            "checkpoint_online_step": args.checkpoint,
            "training_data_budget_at_checkpoint": protocol["prefill_transitions"] + args.checkpoint,
            "reset_seeds": reset_seeds,
            "candidate_seed_offset": protocol["evaluation_candidate_seed_offset"],
            "candidates": protocol["candidates"], "candidate_noise": protocol["candidate_noise"],
            "lookahead_horizon": 1, "discount": protocol["discount"],
            "torch_threads": 1,
            "configurations": [{"training_mode": train, "evaluation_mode": test}
                               for train, test in CONFIGURATIONS],
        },
        "interpretation": [
            "This secondary checkpoint choice was not preregistered; keep it distinct from the primary 20000-step endpoint.",
            "Fresh reset seeds were not used for training, predictor fitting, or the monitoring observations that prompted this analysis.",
            "Training seed is the replication unit; the twenty reset episodes are not independent trained policies.",
            "Actor-only evaluation separates changes learned by the actor from planning required during deployment.",
            "Frozen actor-trained checkpoint evaluations isolate deployment search from assisted-training effects.",
            "All results reuse the same unchanged primary benchmark evaluator and candidate-seed rules.",
            "Shared-host latency is descriptive and not a deployment timing guarantee.",
        ],
    }


def summarize(output, manifest):
    seeds = manifest["protocol"]["training_seeds"]
    results = {}
    for train, test in CONFIGURATIONS:
        name = configuration_name(train, test)
        rows = []
        for seed in seeds:
            report = json.loads((output / f"seed_{seed}" / f"{name}.json").read_text())
            rows.append({"training_seed": seed, "mean_return": report["evaluation"]["mean_return"],
                         "episodes": len(report["evaluation"]["episodes"]),
                         "fallbacks": report["evaluation"]["fallbacks"],
                         "invalid_candidates": report["evaluation"]["invalid_candidates"],
                         "evaluation_steps": report["evaluation"]["steps"],
                         "network_and_global_rng_unchanged": report["evaluation"]["network_and_global_rng_unchanged"]})
        values = np.array([row["mean_return"] for row in rows])
        results[name] = {"per_training_seed": rows, "mean_return": float(values.mean()),
                         "sample_std_across_training_seeds": float(values.std(ddof=1)) if len(values) > 1 else None,
                         "min_seed_mean_return": float(values.min()),
                         "max_seed_mean_return": float(values.max())}
    comparisons = (
        ("assisted_training_and_deployment", "trained_lookahead_eval_lookahead", "trained_actor_eval_actor"),
        ("critic_training_and_deployment", "trained_critic_eval_critic", "trained_actor_eval_actor"),
        ("lookahead_vs_critic_training_and_deployment", "trained_lookahead_eval_lookahead", "trained_critic_eval_critic"),
        ("lookahead_trained_actor_alone", "trained_lookahead_eval_actor", "trained_actor_eval_actor"),
        ("critic_trained_actor_alone", "trained_critic_eval_actor", "trained_actor_eval_actor"),
        ("frozen_actor_lookahead_deployment", "trained_actor_eval_lookahead", "trained_actor_eval_actor"),
        ("frozen_actor_critic_deployment", "trained_actor_eval_critic", "trained_actor_eval_actor"),
        ("frozen_actor_lookahead_vs_critic", "trained_actor_eval_lookahead", "trained_actor_eval_critic"),
    )
    paired = {}
    for label, treatment, reference in comparisons:
        differences = [a["mean_return"] - b["mean_return"] for a, b in zip(
            results[treatment]["per_training_seed"], results[reference]["per_training_seed"])]
        paired[label] = {
            "treatment": treatment, "reference": reference,
            "per_training_seed": [{"training_seed": seed, "difference": delta}
                                  for seed, delta in zip(seeds, differences)],
            "mean_difference": float(np.mean(differences)),
            "sample_std_difference": float(np.std(differences, ddof=1)) if len(differences) > 1 else None,
            "positive_training_seeds": sum(value > 0 for value in differences),
        }
    return {
        "format": "td3-sp-secondary-early-summary", "version": 1,
        "classification": "SECONDARY_POSTHOC", "manifest": _identity(output / "manifest.json"),
        "protocol": manifest["protocol"], "interpretation": manifest["interpretation"],
        "configurations": results, "paired_comparisons": paired,
        "evaluation_steps": sum(row["evaluation_steps"] for item in results.values() for row in item["per_training_seed"]),
        "evaluation_episodes": sum(row["episodes"] for item in results.values() for row in item["per_training_seed"]),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark_dir", default="runs/lookahead-extensive")
    parser.add_argument("--output", default="runs/lookahead-extensive-early")
    parser.add_argument("--checkpoint", type=int, default=5000)
    parser.add_argument("--reset_seed", type=int, default=910000)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--wait_seconds", type=float, default=1800.0)
    args = parser.parse_args(argv)
    if min(args.checkpoint, args.episodes) < 1 or args.reset_seed < 0 or args.wait_seconds < 0:
        parser.error("Use positive checkpoint/episodes and nonnegative seed/wait")
    if torch.cuda.is_available():
        parser.error("This secondary protocol is CPU-only")
    torch.set_num_threads(1)
    benchmark, output = Path(args.benchmark_dir).resolve(), Path(args.output).resolve()
    primary = json.loads((benchmark / "manifest.json").read_text())
    check_manifest_files(primary)
    manifest = make_manifest(benchmark, output, args, primary)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Existing secondary manifest differs; use a new output directory")
    else:
        write_json(manifest_path, manifest)
    manifest_hash = file_hash(manifest_path)
    predictor = StatePredictor.load(primary["predictor"]["path"], device="cpu")
    predictor_start = network_hash(predictor)
    waiting_started = time.perf_counter()
    pending = [(seed, train) for seed in manifest["protocol"]["training_seeds"]
               for train in ("actor", "critic", "lookahead")]
    while pending:
        made_progress = False
        for seed, training_mode in list(pending):
            checkpoint = _checkpoint(benchmark, seed, training_mode, args.checkpoint)
            if checkpoint is None:
                continue
            prefix, marker_path, marker = checkpoint
            evaluations = [test for train, test in CONFIGURATIONS if train == training_mode]
            with torch.random.fork_rng(devices=[]):
                policy = TD3.TD3(3, 1, 2.0, discount=primary["protocol"]["discount"])
            policy.load_for_evaluation(prefix)
            for evaluation_mode in evaluations:
                name = configuration_name(training_mode, evaluation_mode)
                path = output / f"seed_{seed}" / f"{name}.json"
                if path.exists():
                    existing = json.loads(path.read_text())
                    if existing["manifest_sha256"] != manifest_hash or existing["checkpoint_marker"]["sha256"] != file_hash(marker_path):
                        raise ValueError(f"Existing result differs from frozen inputs: {path}")
                    print(json.dumps({"reused": str(path)}), flush=True)
                    continue
                record = {
                    "classification": "SECONDARY_POSTHOC", "manifest_sha256": manifest_hash,
                    "training_seed": seed, "training_mode": training_mode,
                    "evaluation_mode": evaluation_mode, "checkpoint_marker": _identity(marker_path),
                    "checkpoint": marker, "predictor": primary["predictor"],
                    "evaluation": evaluate(
                        policy, predictor, evaluation_mode, 1,
                        manifest["protocol"]["reset_seeds"], primary["protocol"]),
                }
                write_json(path, record)
                print(json.dumps({"completed": name, "training_seed": seed,
                                  "mean_return": record["evaluation"]["mean_return"],
                                  "evaluation_seconds": record["evaluation"]["evaluation_seconds"]}), flush=True)
            pending.remove((seed, training_mode))
            made_progress = True
        if pending:
            if time.perf_counter() - waiting_started > args.wait_seconds:
                raise TimeoutError(f"Checkpoints not available before timeout: {pending}")
            if not made_progress:
                print(json.dumps({"waiting_for": pending}), flush=True)
                time.sleep(5)
    if network_hash(predictor) != predictor_start:
        raise AssertionError("Secondary evaluation altered the predictor")
    check_manifest_files(primary)
    summary = summarize(output, manifest)
    summary_path = output / "summary.json"
    if summary_path.exists():
        if json.loads(summary_path.read_text()) != summary:
            raise ValueError("Existing summary differs from completed results")
    else:
        write_json(summary_path, summary)
    print(json.dumps({"summary": str(summary_path), "evaluation_steps": summary["evaluation_steps"],
                      "configuration_means": {name: data["mean_return"] for name, data in summary["configurations"].items()}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
