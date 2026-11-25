"""Archive and summarize a COMPLETE, fixed-protocol lookahead experiment.

The replication unit is a policy-training seed. Episode means are computed
within seeds before paired Student-t intervals; aliases of the same evaluation
are explicitly excluded from interaction and timing totals. This script never
trains models, chooses checkpoints, or changes benchmark source files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil

import numpy as np
from scipy.stats import t as student_t


REPOSITORY = Path(__file__).resolve().parents[1]
MODES = ("actor", "critic", "lookahead")
FROZEN = ("actor", "critic", "lookahead_h1", "lookahead_h3", "lookahead_h5")
LABELS = {"actor": "Actor", "critic": "Critic search", "lookahead": "Lookahead H1",
          "lookahead_h1": "H1", "lookahead_h3": "H3", "lookahead_h5": "H5"}
COLORS = {"actor": "#4D4D4D", "critic": "#0072B2", "lookahead": "#D55E00",
          "lookahead_h1": "#D55E00", "lookahead_h3": "#009E73", "lookahead_h5": "#CC79A7"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def normalize_svg_whitespace(path):
    """Keep Matplotlib SVG output clean for repository whitespace checks."""
    path = Path(path)
    path.write_text("\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()) + "\n",
                    encoding="utf-8")


def summary(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Summaries require at least two finite seed values")
    return {"values": values.tolist(), "n": len(values), "mean": float(values.mean()),
            "sd": float(values.std(ddof=1)), "minimum": float(values.min()), "maximum": float(values.max())}


def paired_interval(treatment, reference):
    treatment, reference = np.asarray(treatment), np.asarray(reference)
    if treatment.shape != reference.shape:
        raise ValueError("Paired groups must contain the same ordered seeds")
    result = summary(treatment - reference)
    half_width = float(student_t.ppf(0.975, result["n"] - 1)) * result["sd"] / math.sqrt(result["n"])
    result.update({"degrees_of_freedom": result["n"] - 1,
                   "ci95": [result["mean"] - half_width, result["mean"] + half_width],
                   "positive_seeds": int(np.count_nonzero(treatment > reference)),
                   "negative_seeds": int(np.count_nonzero(treatment < reference)),
                   "method": "unadjusted paired Student-t interval on training-seed episode means"})
    return result


def unique_evaluations(run):
    """Yield evaluations executed once, excluding serialized actor aliases."""
    for monitor in run["monitor"]:
        yield "monitor", f"step_{monitor['online_step']}", monitor["evaluation"]
    final = run["final"]
    if run["training_mode"] == "actor":
        if not final["deployed"] == final["actor_only"] == final["frozen_selectors"]["actor"]:
            raise ValueError("Expected actor-only/deployed/frozen actor evaluation aliases to match")
        for mode in FROZEN:
            yield "final", mode, final["frozen_selectors"][mode]
    else:
        yield "final", "deployed", final["deployed"]
        yield "final", "actor_only", final["actor_only"]


def episode_mean(evaluation):
    value = float(np.mean([episode["return"] for episode in evaluation["episodes"]]))
    if not math.isclose(value, evaluation["mean_return"], rel_tol=1e-12, abs_tol=1e-10):
        raise ValueError("Saved evaluation mean does not match its episode returns")
    return value


def load_complete(input_directory):
    root = Path(input_directory)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        manifest_path = root / "raw/manifest.json"
    manifest = read_json(manifest_path)
    protocol = manifest["protocol"]
    seeds = protocol["training_seeds"]
    if len(seeds) != 5 or tuple(protocol["training_modes"]) != MODES:
        raise ValueError("This report expects the declared five-seed, three-method protocol")
    paths, missing = {}, []
    for seed in seeds:
        for mode in MODES:
            path = root / f"seed_{seed}" / mode / "result.json"
            if not path.exists():
                path = root / "raw" / f"seed_{seed}_{mode}.json"
            if path.exists():
                paths[(seed, mode)] = path
            else:
                missing.append(f"seed={seed}, mode={mode}")
    if missing:
        raise ValueError("Refusing to summarize an incomplete benchmark: " + "; ".join(missing))
    runs = {key: read_json(path) for key, path in paths.items()}
    manifest_hash = digest(manifest_path)
    expected_calls = protocol["online_steps"] - protocol["warmup_steps"]
    for (seed, mode), run in runs.items():
        if (run["seed"], run["training_mode"]) != (seed, mode):
            raise ValueError("Run identity disagrees with its path")
        if run["manifest_sha256"] != manifest_hash:
            raise ValueError("A worker used a different manifest")
        if run["td3_training_calls"] != expected_calls or run["actor_updates"] != expected_calls // protocol["policy_freq"]:
            raise ValueError("Mismatched policy gradient budgets")
        if run["real_collection_budget"] != protocol["real_collection_budget_per_run"]:
            raise ValueError("Mismatched real interaction budgets")
        if not all(run["verification"].values()):
            raise ValueError("A worker reported a failed invariant")
        if [point["online_step"] for point in run["monitor"]] != protocol["monitor_steps"]:
            raise ValueError("Monitoring checkpoints differ from the declared protocol")
        for phase, _, evaluation in unique_evaluations(run):
            expected_seeds = protocol["monitor_reset_seeds"] if phase == "monitor" else protocol["final_reset_seeds"]
            if [episode["seed"] for episode in evaluation["episodes"]] != expected_seeds:
                raise ValueError("Evaluation episode seeds are not the declared paired seeds")
            if sum(episode["steps"] for episode in evaluation["episodes"]) != evaluation["steps"]:
                raise ValueError("Evaluation step accounting is inconsistent")
            if not evaluation["network_and_global_rng_unchanged"]:
                raise ValueError("Evaluation altered a frozen network or global random stream")
            episode_mean(evaluation)
    for seed in seeds:
        if any(runs[(seed, mode)]["initial_policy"] != runs[(seed, "actor")]["initial_policy"] for mode in MODES):
            raise ValueError("Compared methods did not begin with identical weights within a seed")
    return manifest_path, manifest, paths, runs


def aggregate(manifest, runs):
    protocol = manifest["protocol"]
    seeds = protocol["training_seeds"]
    budgets = np.array(protocol["monitor_steps"]) + protocol["prefill_transitions"]
    result = {
        "format": "td3-sp-extensive-analysis", "version": 1,
        "training_seeds": seeds, "replication_unit": "policy-training seed",
        "inference_scope": "conditional on the shared fixed predictor/dataset and the fixed final reset set",
        "statistical_method": "sample SD and unadjusted 95% paired Student-t intervals, n=5 and df=4",
        "per_seed": {}, "trained": {}, "frozen": {}, "primary_paired": {}, "secondary_paired": {},
        "learning_curves": {"real_interaction_budgets": budgets.tolist(), "methods": {}},
    }
    for seed in seeds:
        row = {"trained": {}, "frozen": {}}
        for mode in MODES:
            run = runs[(seed, mode)]
            curve = np.array([episode_mean(point["evaluation"]) for point in run["monitor"]])
            auc = float(np.sum((curve[1:] + curve[:-1]) / 2 * np.diff(budgets)))
            row["trained"][mode] = {
                "deployed_return": episode_mean(run["final"]["deployed"]),
                "actor_only_return": episode_mean(run["final"]["actor_only"]),
                "online_training_seconds": run["online_training_seconds"],
                "unique_evaluation_seconds": sum(e["evaluation_seconds"] for _, _, e in unique_evaluations(run)),
                "total_worker_seconds": run["total_worker_seconds"],
                "sparse_monitor_auc_return_steps": auc,
                "normalized_sparse_monitor_auc": auc / (budgets[-1] - budgets[0]),
                "training_intervention_rate": run["training_selection"]["interventions"] / run["training_selection"]["selections"],
            }
        for controller in FROZEN:
            row["frozen"][controller] = episode_mean(runs[(seed, "actor")]["final"]["frozen_selectors"][controller])
        result["per_seed"][str(seed)] = row
    for mode in MODES:
        result["trained"][mode] = {
            field: summary([result["per_seed"][str(seed)]["trained"][mode][field] for seed in seeds])
            for field in result["per_seed"][str(seeds[0])]["trained"][mode]
        }
        curves = np.array([[episode_mean(point["evaluation"]) for point in runs[(seed, mode)]["monitor"]] for seed in seeds])
        result["learning_curves"]["methods"][mode] = {"per_seed": curves.tolist(), "mean": curves.mean(axis=0).tolist(),
                                                        "sample_sd": curves.std(axis=0, ddof=1).tolist()}
    for controller in FROZEN:
        evaluations = [runs[(seed, "actor")]["final"]["frozen_selectors"][controller] for seed in seeds]
        result["frozen"][controller] = {
            "return": summary([episode_mean(e) for e in evaluations]),
            "run_median_latency_ms": summary([e["latency"]["median_ms"] for e in evaluations]),
            "run_p95_latency_ms": summary([e["latency"]["p95_ms"] for e in evaluations]),
            "maximum_observed_latency_ms": max(e["latency"]["max_ms"] for e in evaluations),
            "intervention_rate": summary([e["intervention_rate"] for e in evaluations]),
        }
    frozen_values = lambda key: result["frozen"][key]["return"]["values"]
    trained_values = lambda mode, field="deployed_return": result["trained"][mode][field]["values"]
    result["primary_paired"] = {
        "frozen_h1_minus_actor": paired_interval(frozen_values("lookahead_h1"), frozen_values("actor")),
        "frozen_h1_minus_critic": paired_interval(frozen_values("lookahead_h1"), frozen_values("critic")),
        "trained_lookahead_minus_actor": paired_interval(trained_values("lookahead"), trained_values("actor")),
        "trained_lookahead_minus_critic": paired_interval(trained_values("lookahead"), trained_values("critic")),
    }
    result["secondary_paired"] = {
        "frozen_h3_minus_actor": paired_interval(frozen_values("lookahead_h3"), frozen_values("actor")),
        "frozen_h5_minus_actor": paired_interval(frozen_values("lookahead_h5"), frozen_values("actor")),
        "lookahead_trained_actor_minus_actor_trained_actor": paired_interval(trained_values("lookahead", "actor_only_return"), trained_values("actor", "actor_only_return")),
        "lookahead_deployed_minus_own_actor": paired_interval(trained_values("lookahead"), trained_values("lookahead", "actor_only_return")),
    }
    evaluations = [(phase, evaluation) for run in runs.values() for phase, _, evaluation in unique_evaluations(run)]
    training_stats = [run["training_selection"] for run in runs.values()]
    result["accounting"] = {
        "completed_runs": len(runs), "missing_runs": 0,
        "new_training_interactions": sum(run["online_steps"] for run in runs.values()),
        "unique_shared_collection_interactions": protocol["prefill_transitions"],
        "unique_training_collection_total": sum(run["online_steps"] for run in runs.values()) + protocol["prefill_transitions"],
        "total_per_run_exposure_including_reused_prefill": sum(run["real_collection_budget"] for run in runs.values()),
        "monitor_evaluation_interactions": sum(e["steps"] for phase, e in evaluations if phase == "monitor"),
        "final_evaluation_interactions": sum(e["steps"] for phase, e in evaluations if phase == "final"),
        "unique_evaluation_interactions": sum(e["steps"] for _, e in evaluations),
        "unique_evaluation_episodes": sum(len(e["episodes"]) for _, e in evaluations),
        "td3_training_calls": sum(run["td3_training_calls"] for run in runs.values()),
        "delayed_actor_updates": sum(run["actor_updates"] for run in runs.values()),
        "training_fallbacks": sum(stat["fallbacks"] for stat in training_stats),
        "training_invalid_candidates": sum(stat["invalid_candidates"] for stat in training_stats),
        "evaluation_fallbacks": sum(e["fallbacks"] for _, e in evaluations),
        "evaluation_invalid_candidates": sum(e["invalid_candidates"] for _, e in evaluations),
        "sum_online_training_worker_seconds": sum(run["online_training_seconds"] for run in runs.values()),
        "sum_unique_evaluation_worker_seconds": sum(e["evaluation_seconds"] for _, e in evaluations),
        "sum_total_worker_seconds": sum(run["total_worker_seconds"] for run in runs.values()),
        "all_recorded_invariants_passed": True,
        "note": "Concurrent worker wall times overlap; their sum is not elapsed campaign time. Prior predictor fitting is excluded. Diagnostics are accounted separately.",
    }
    return result


def load_early(input_directory, primary_manifest_path, primary_manifest, runs):
    """Validate the separately declared post-hoc follow-up without pooling it."""
    root = Path(input_directory)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists() or read_json(manifest_path).get("classification") != "SECONDARY_POSTHOC":
        manifest_path = root / "raw/early_manifest.json"
    manifest = read_json(manifest_path)
    if manifest["classification"] != "SECONDARY_POSTHOC":
        raise ValueError("Early follow-up must retain its secondary post-hoc classification")
    if manifest["primary_manifest"]["sha256"] != digest(primary_manifest_path):
        raise ValueError("Early follow-up used a different primary manifest")
    if manifest["predictor"]["sha256"] != primary_manifest["predictor"]["sha256"]:
        raise ValueError("Early follow-up used a different predictor")
    protocol = manifest["protocol"]
    seeds = protocol["training_seeds"]
    if seeds != primary_manifest["protocol"]["training_seeds"]:
        raise ValueError("Early training seeds differ from primary seeds")
    if set(protocol["reset_seeds"]) & set(primary_manifest["protocol"]["final_reset_seeds"] + primary_manifest["protocol"]["monitor_reset_seeds"]):
        raise ValueError("Early post-hoc evaluations reused primary/monitor reset seeds")
    paths, records, missing = {}, {}, []
    for pair in protocol["configurations"]:
        training, evaluation = pair["training_mode"], pair["evaluation_mode"]
        name = f"trained_{training}_eval_{evaluation}"
        for seed in seeds:
            path = root / f"seed_{seed}" / f"{name}.json"
            if not path.exists():
                path = root / "raw" / f"early_seed_{seed}_{name}.json"
            if not path.exists():
                missing.append(f"{seed}/{name}")
                continue
            row = read_json(path)
            if row["manifest_sha256"] != digest(manifest_path) or row["classification"] != "SECONDARY_POSTHOC":
                raise ValueError("Early record provenance/classification mismatch")
            expected = next(checkpoint for checkpoint in runs[(seed, training)]["checkpoints"]
                            if checkpoint["online_step"] == protocol["checkpoint_online_step"])
            if any(row["checkpoint"]["weights"][network]["sha256"] != expected["weights"][network]["sha256"] for network in ("actor", "critic")):
                raise ValueError("Early evaluation checkpoint is not the saved primary checkpoint")
            result = row["evaluation"]
            if [episode["seed"] for episode in result["episodes"]] != protocol["reset_seeds"]:
                raise ValueError("Early evaluation did not use the fresh declared reset seeds")
            if not result["network_and_global_rng_unchanged"]:
                raise ValueError("Early evaluation changed a network or global RNG")
            episode_mean(result)
            paths[(seed, name)], records[(seed, name)] = path, row
    if missing:
        raise ValueError("Early follow-up incomplete: " + "; ".join(missing))
    configurations = {}
    for pair in protocol["configurations"]:
        name = f"trained_{pair['training_mode']}_eval_{pair['evaluation_mode']}"
        configurations[name] = summary([episode_mean(records[(seed, name)]["evaluation"]) for seed in seeds])
    contrasts = {
        "trained_lookahead_minus_actor": ("trained_lookahead_eval_lookahead", "trained_actor_eval_actor"),
        "trained_lookahead_minus_critic": ("trained_lookahead_eval_lookahead", "trained_critic_eval_critic"),
        "lookahead_trained_actor_minus_actor": ("trained_lookahead_eval_actor", "trained_actor_eval_actor"),
        "critic_trained_actor_minus_actor": ("trained_critic_eval_actor", "trained_actor_eval_actor"),
        "frozen_h1_minus_actor": ("trained_actor_eval_lookahead", "trained_actor_eval_actor"),
        "frozen_h1_minus_critic": ("trained_actor_eval_lookahead", "trained_actor_eval_critic"),
        "frozen_critic_minus_actor": ("trained_actor_eval_critic", "trained_actor_eval_actor"),
    }
    result = {
        "classification": "SECONDARY_POSTHOC", "selection_reason": manifest["selection_reason"],
        "protocol": protocol, "configurations": configurations,
        "paired": {name: paired_interval(configurations[treatment]["values"], configurations[reference]["values"])
                   for name, (treatment, reference) in contrasts.items()},
        "evaluation_interactions": sum(row["evaluation"]["steps"] for row in records.values()),
        "evaluation_episodes": sum(len(row["evaluation"]["episodes"]) for row in records.values()),
        "evaluation_seconds": sum(row["evaluation"]["evaluation_seconds"] for row in records.values()),
        "fallbacks": sum(row["evaluation"]["fallbacks"] for row in records.values()),
        "invalid_candidates": sum(row["evaluation"]["invalid_candidates"] for row in records.values()),
        "manifest_sha256": digest(manifest_path),
    }
    return manifest_path, paths, result


def archive_json(source, destination, artifacts, *, exact=False):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if exact:
        shutil.copyfile(source, destination)
    else:
        destination.write_text(json.dumps(read_json(source), separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
    artifacts.append({"source": str(source), "source_sha256": digest(source),
                      "archive": destination.name, "archive_sha256": digest(destination),
                      "bytes": destination.stat().st_size,
                      "transformation": "byte-exact copy" if exact else "JSON whitespace removed; all fields retained"})


def create_figures(analysis, forecast, decisions, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.titlesize": 13, "axes.labelsize": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.hashsalt": "td3-sp-extensive-lookahead"})
    jitter = np.linspace(-0.13, 0.13, 5)
    fig, axs = plt.subplots(2, 2, figsize=(14, 10.8), layout="constrained")
    fig.suptitle("TD3-SP: five-seed Pendulum comparison", fontsize=19, fontweight="bold")
    ax = axs[0, 0]
    for index, mode in enumerate(FROZEN):
        values = analysis["frozen"][mode]["return"]
        ax.scatter(index + jitter, values["values"], color=COLORS[mode], s=38, zorder=3)
        ax.errorbar(index, values["mean"], yerr=values["sd"], fmt="D", color="black", capsize=4, markersize=5, zorder=4)
    ax.set(xticks=range(len(FROZEN)), xticklabels=[LABELS[x] for x in FROZEN],
           ylabel="Undiscounted episode return (higher is better)", title="A. Frozen actor-trained policies: change only selector")
    ax.grid(axis="y", alpha=0.2)
    ax.text(0.02, 0.02, "Dots: five seed means; black diamonds: mean ± sample SD", transform=ax.transAxes, fontsize=9)
    ax = axs[0, 1]
    for index, mode in enumerate(MODES):
        deployed = analysis["trained"][mode]["deployed_return"]
        actor = analysis["trained"][mode]["actor_only_return"]
        ax.scatter(index + jitter - 0.07, deployed["values"], color=COLORS[mode], s=40)
        ax.scatter(index + jitter + 0.07, actor["values"], facecolors="none", edgecolors=COLORS[mode], s=40)
        ax.plot([index - 0.19, index + 0.05], [deployed["mean"]] * 2, color="black", lw=2)
        ax.plot([index - 0.05, index + 0.19], [actor["mean"]] * 2, color="black", lw=2, linestyle=":")
    ax.set(xticks=range(3), xticklabels=["Actor-trained", "Critic-trained", "H1-trained"],
           ylabel="Undiscounted episode return", title="B. Matched training: final deployed and actor-only")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(handles=[Line2D([], [], marker="o", linestyle="none", color="#444444", label="Deployed selector"),
                       Line2D([], [], marker="o", linestyle="none", color="#444444", markerfacecolor="none", label="Actor alone")],
              fontsize=9, loc="best")
    ax = axs[1, 0]
    x = np.asarray(analysis["learning_curves"]["real_interaction_budgets"]) / 1000
    for mode in MODES:
        curve = analysis["learning_curves"]["methods"][mode]
        mean, sd = np.array(curve["mean"]), np.array(curve["sample_sd"])
        ax.plot(x, mean, "o-", color=COLORS[mode], label=LABELS[mode], lw=2)
        ax.fill_between(x, mean - sd, mean + sd, color=COLORS[mode], alpha=0.14)
    ax.set(xticks=x, xlabel="Real training interactions per run (thousands; includes 6k prefill)",
           ylabel="Monitoring return (five episodes/checkpoint)", title="C. Learning curves: mean ± sample SD across five seeds")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=9)
    ax = axs[1, 1]
    for index, mode in enumerate(FROZEN):
        values = analysis["frozen"][mode]["run_p95_latency_ms"]
        ax.scatter(index + jitter, values["values"], color=COLORS[mode], s=38)
        ax.plot([index - 0.16, index + 0.16], [values["mean"]] * 2, color="black", lw=2)
    ax.set(xticks=range(len(FROZEN)), xticklabels=[LABELS[x] for x in FROZEN],
           ylabel="Per-run p95 action-selection latency (ms; log scale)", yscale="log",
           title="D. Frozen-controller latency on a shared CPU host")
    ax.grid(axis="y", which="both", alpha=0.2)
    for extension in ("png", "svg"):
        fig.savefig(output / f"control_results.{extension}", dpi=180, metadata={"Creator": "TD3-SP reproducible summarizer", "Date": analysis["provenance"]["measurement_date"]})
        if extension == "svg":
            normalize_svg_whitespace(output / f"control_results.{extension}")
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(14, 5.5), layout="constrained")
    fig.suptitle("Forecast and decision diagnostics: 96 balanced test states", fontsize=18, fontweight="bold")
    groups = {"nominal": ("Ordinary states", "#0072B2"), "near_upright": ("Near upright", "#009E73"),
              "high_speed": ("High angular speed", "#D55E00")}
    horizons = forecast["settings"]["horizons"]
    ax = axs[0]
    for scenario, style in (("gravity_10", "-"), ("gravity_15", "--")):
        for group, (label, color) in groups.items():
            values = [forecast["scenarios"][scenario][str(h)]["groups"][group]["normalized_rmse"] for h in horizons]
            ax.plot(horizons, values, marker="o", linestyle=style, color=color, lw=2,
                    label=label if scenario == "gravity_10" else None)
    ax.set(xticks=horizons, xlabel="Forecast horizon (steps)", ylabel="Normalized endpoint RMSE (log scale)",
           yscale="log", title="A. Prediction error by state group and gravity")
    ax.grid(which="both", alpha=0.18)
    ax.legend(fontsize=9, loc="best")
    ax.text(0.02, 0.98, "Solid: training dynamics g=10\nDashed: changed dynamics g=15", transform=ax.transAxes,
            va="top", fontsize=9, bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none"})
    ax = axs[1]
    methods = ("actor", "critic", "learned_h1", "oracle_h1", "learned_h3", "oracle_h3", "learned_h5", "oracle_h5")
    values = [decisions["methods"][method]["summary"]["mean_regret_to_best_tested_candidate"] for method in methods]
    colors = ["#4D4D4D", "#0072B2"] + ["#D55E00", "#009E73"] * 3
    ax.bar(range(len(methods)), values, color=colors)
    ax.axhline(values[0], color="#4D4D4D", linestyle=":", lw=1.5)
    ax.set(xticks=range(len(methods)), xticklabels=["Actor", "Critic", "Learned\nH1", "Exact\nH1", "Learned\nH3", "Exact\nH3", "Learned\nH5", "Exact\nH5"],
           ylabel="Mean discounted-return regret (lower is better)",
           title="B. Best-of-16 first-action regret, then frozen actor")
    ax.tick_params(axis="x", labelsize=9)
    ax.set_ylim(0, max(values) * 1.5)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.text(0.02, 0.98, "One seed-11 policy; 200-step reference\nExact dynamics still uses a learned critic tail", transform=ax.transAxes,
            va="top", fontsize=9)
    for extension in ("png", "svg"):
        fig.savefig(output / f"diagnostic_results.{extension}", dpi=180, metadata={"Creator": "TD3-SP reproducible summarizer", "Date": analysis["provenance"]["measurement_date"]})
        if extension == "svg":
            normalize_svg_whitespace(output / f"diagnostic_results.{extension}")
    plt.close(fig)


def report_text(analysis, manifest, forecast, decisions, *, artifact_prefix, test_count, measurement_date):
    seeds = analysis["training_seeds"]
    p = analysis["primary_paired"]
    a = analysis["accounting"]
    link = lambda name: f"{artifact_prefix}/{name}"
    def ci(item):
        return f"{item['mean']:+.3f} [{item['ci95'][0]:+.3f}, {item['ci95'][1]:+.3f}]"
    def ms(item):
        return f"{item['mean']:.3f} ± {item['sd']:.3f}"
    frozen_rows = "\n".join(f"| {seed} | " + " | ".join(f"{analysis['per_seed'][str(seed)]['frozen'][mode]:.3f}" for mode in FROZEN) + " |" for seed in seeds)
    trained_rows = "\n".join(f"| {seed} | " + " | ".join(f"{analysis['per_seed'][str(seed)]['trained'][mode]['deployed_return']:.3f}" for mode in MODES) + " |" for seed in seeds)
    frozen_aggregate = " | ".join(ms(analysis["frozen"][mode]["return"]) for mode in FROZEN)
    trained_aggregate = " | ".join(ms(analysis["trained"][mode]["deployed_return"]) for mode in MODES)
    actor_rows = "\n".join(f"| {LABELS[mode]} | {ms(analysis['trained'][mode]['actor_only_return'])} | {ms(analysis['trained'][mode]['normalized_sparse_monitor_auc'])} |" for mode in MODES)
    runtime_rows = "\n".join(f"| {LABELS[mode]} | {ms(analysis['trained'][mode]['online_training_seconds'])} | {ms(analysis['trained'][mode]['unique_evaluation_seconds'])} |" for mode in MODES)
    latency_rows = "\n".join(f"| {LABELS[mode]} | {ms(analysis['frozen'][mode]['run_p95_latency_ms'])} | {analysis['frozen'][mode]['maximum_observed_latency_ms']:.3f} | {analysis['frozen'][mode]['intervention_rate']['mean']:.1%} |" for mode in FROZEN)
    forecast_rows = "\n".join(f"| {h} | {forecast['scenarios']['gravity_10'][str(h)]['all']['normalized_rmse']:.5f} | {forecast['scenarios']['gravity_15'][str(h)]['all']['normalized_rmse']:.5f} |" for h in forecast["settings"]["horizons"])
    decision_rows = "\n".join(f"| {mode} | {row['summary']['mean_regret_to_best_tested_candidate']:.5f} | " +
        ("—" if row['summary']['mean_pairwise_order_agreement'] is None else f"{row['summary']['mean_pairwise_order_agreement']:.3f}") + " |"
        for mode, row in decisions["methods"].items())
    intervals_cross = [name for name, item in p.items() if item["ci95"][0] <= 0 <= item["ci95"][1]]
    opening = ("The four primary paired intervals include zero, so these five seeds do not establish an average H1 advantage over either comparison in either experiment."
               if len(intervals_cross) == 4 else
               f"{len(intervals_cross)} of the four primary paired intervals include zero; the individual endpoints and all seed values are reported below.")
    high_speed = forecast["scenarios"]["gravity_10"]["5"]["groups"]["high_speed"]
    oracle = decisions["methods"]["oracle_h1"]["summary"]
    learned = decisions["methods"]["learned_h1"]["summary"]
    early = analysis["early_posthoc"]
    early_names = ("trained_actor_eval_actor", "trained_critic_eval_critic", "trained_lookahead_eval_lookahead",
                   "trained_critic_eval_actor", "trained_lookahead_eval_actor")
    early_rows = "\n".join(f"| {seed} | " + " | ".join(f"{early['configurations'][name]['values'][index]:.3f}" for name in early_names) + " |"
                           for index, seed in enumerate(seeds))
    early_average = " | ".join(ms(early["configurations"][name]) for name in early_names)
    early_contrast_labels = {
        "trained_lookahead_minus_actor": "H1-trained deployed minus actor-trained",
        "trained_lookahead_minus_critic": "H1-trained deployed minus critic-trained",
        "lookahead_trained_actor_minus_actor": "H1-trained actor alone minus actor-trained",
        "critic_trained_actor_minus_actor": "Critic-trained actor alone minus actor-trained",
        "frozen_h1_minus_actor": "Frozen early policy: H1 minus actor",
        "frozen_h1_minus_critic": "Frozen early policy: H1 minus critic search",
        "frozen_critic_minus_actor": "Frozen early policy: critic search minus actor",
    }
    early_contrasts = "\n".join(f"| {label} | {ci(early['paired'][name])} | {early['paired'][name]['positive_seeds']}/5 |"
                                for name, label in early_contrast_labels.items())
    return f"""# Extensive TD3-SP lookahead results

Measured on **{measurement_date}**. The [predeclared protocol](LOOKAHEAD_TEST_PROTOCOL.md),
[analysis JSON]({link('analysis.json')}), and [artifact index]({link('artifact_index.json')})
record the settings, calculations, complete episode outcomes, and source hashes.

## Main result

All 15 planned training runs completed. Frozen H1 minus actor was **{ci(p['frozen_h1_minus_actor'])}**
return points; H1 minus critic search was **{ci(p['frozen_h1_minus_critic'])}**.
After matched training, the deployed H1-trained controller minus actor-trained
controller was **{ci(p['trained_lookahead_minus_actor'])}**, and minus critic-trained
controller was **{ci(p['trained_lookahead_minus_critic'])}**. Brackets are nominal
95% paired Student-t intervals on **five training-seed means**, with four degrees
of freedom. {opening}

These results are conditional on standard Pendulum, one fixed predictor, one
shared dataset, and the declared reset set. The earlier five-episode smoke check
used a 650-step policy; it is not pooled with these 20,000-step results.

![Control results]({link('control_results.png')})

## Design and interaction accounting

Seeds were {', '.join(map(str, seeds))}. Each method received the same 6,000 real
archived transitions and 20,000 new interactions, with 1,000 random warm-up steps,
19,000 TD3 training calls, and 9,500 delayed actor updates. Every seed began with
identical actor/critic weights across methods. The 16-candidate, H1 configuration
was fixed before outcomes were examined; H3/H5 were secondary diagnostics.
All behavior modes used private selector random streams.

Twenty final reset seeds, 900000–900019, were held fixed across methods. Monitoring
used five different seeds at 0/5k/10k/20k new interactions. The final checkpoint
was always step 20k; monitoring did not select a best checkpoint. Evaluation
data never entered replay or predictor fitting.

There were **{a['new_training_interactions']:,} new training interactions**, plus
**{a['unique_shared_collection_interactions']:,} unique shared collection transitions**
({a['unique_training_collection_total']:,} unique training collection interactions).
Counting reuse of the same prefill across 15 runs gives {a['total_per_run_exposure_including_reused_prefill']:,}
transition exposures; it does not mean that many unique transitions were collected.
Evaluation excluded from training executed **{a['unique_evaluation_interactions']:,} interactions**:
{a['monitor_evaluation_interactions']:,} monitoring and {a['final_evaluation_interactions']:,} final.
The repeated serialized actor/deployed/frozen entries are counted once.
Diagnostics below are a separate designed-state analysis.
The separate post-hoc early-checkpoint follow-up added {early['evaluation_interactions']:,}
evaluation interactions, bringing benchmark plus follow-up evaluation to
{a['unique_evaluation_interactions'] + early['evaluation_interactions']:,}. It added no training data.

## Frozen policies: does adding a selector help?

Each row uses one actor-trained actor/critic checkpoint with every selector.
Entries are actual undiscounted 200-step returns averaged over 20 reset episodes.
Higher is better.

| Training seed | Actor | Critic search | H1 | H3 | H5 |
| --- | ---: | ---: | ---: | ---: | ---: |
{frozen_rows}
| Mean ± sample SD | {frozen_aggregate} |

H1 exceeded actor return in {p['frozen_h1_minus_actor']['positive_seeds']}/5 seeds and
critic search in {p['frozen_h1_minus_critic']['positive_seeds']}/5. Pairing improves
comparability but does not make 100 evaluation episodes into 100 independently
trained policies. The t intervals assume reasonably well-behaved independent
seed differences; with five seeds they are sensitive to outliers. They are
unadjusted, and no multiple-comparison-adjusted superiority claim is made.

## Matched training: does the behavior policy improve learning?

These policies were independently trained from the matched initial weights and
interaction budgets. Each is evaluated with its deployed controller.

| Training seed | Actor-trained | Critic-trained | H1-trained |
| --- | ---: | ---: | ---: |
{trained_rows}
| Mean ± sample SD | {trained_aggregate} |

H1-trained deployed return exceeded actor-trained return in
{p['trained_lookahead_minus_actor']['positive_seeds']}/5 seeds and critic-trained
return in {p['trained_lookahead_minus_critic']['positive_seeds']}/5. The actor-only
evaluation below removes planning at deployment. Its endpoint can differ from
the combined controller even though both use the same trained actor weights.

| Training method | Final actor-only return, mean ± SD | Normalized sparse monitoring AUC, mean ± SD |
| --- | ---: | ---: |
{actor_rows}

The AUC integrates the four monitored returns with trapezoids from 6k to 26k real
training interactions, then divides by the 20k interval. It is a coarse average
monitoring return, not a densely measured learning curve or an integral over
the unobserved initial 0–6k interval. It is secondary to the fixed final endpoints.

## Secondary post-hoc check: the 5,000-step checkpoint

After observing the first two seeds' monitoring curves, we selected the saved
5k checkpoint for a follow-up on **new** reset seeds 910000–910019. This choice
was made after seeing results, so these findings are explicitly **secondary and
post-hoc**. Fresh reset seeds prevent reuse of the monitoring scenarios; they do
not turn this checkpoint choice into a preregistered primary endpoint. Training
and model settings were unchanged. The checkpoint had 11,000 real-transition
exposures including prefill.

| Seed | Actor-trained | Critic-trained deployed | H1-trained deployed | Critic-trained actor alone | H1-trained actor alone |
| --- | ---: | ---: | ---: | ---: | ---: |
{early_rows}
| Mean ± sample SD | {early_average} |

| Secondary early contrast | Mean paired difference [95% t interval] | Positive seeds |
| --- | ---: | ---: |
{early_contrasts}

These intervals are exploratory and do not account for selecting the 5k
checkpoint after observing monitoring outcomes.

The early frozen actor-trained controller means were
{ms(early['configurations']['trained_actor_eval_actor'])} for actor-only,
{ms(early['configurations']['trained_actor_eval_critic'])} for critic search, and
{ms(early['configurations']['trained_actor_eval_lookahead'])} for H1. This separates
an effect of planning at deployment from an effect learned through assisted
training. Any early advantage should be confirmed by a new experiment that
predeclares the early checkpoint. The primary 20k results remain the main endpoint.
All 35 additional evaluations preserved network/RNG invariants and recorded
{early['fallbacks']} fallbacks and {early['invalid_candidates']} invalid candidates.

## Compute and numerical reliability

| Training method | Online training seconds, mean ± SD | Unique evaluation seconds per worker, mean ± SD |
| --- | ---: | ---: |
{runtime_rows}

Training time includes interaction, selection, replay insertion, and TD3 updates.
It excludes earlier predictor fitting, shared data collection, checkpoint I/O,
and evaluation. Actor-trained workers perform more final selector comparisons,
so their evaluation workload is intentionally larger. Up to three CPU workers
ran concurrently with one PyTorch thread each. The summed online worker time was
{a['sum_online_training_worker_seconds']:.1f}s and summed unique evaluation time
{a['sum_unique_evaluation_worker_seconds']:.1f}s; these overlapping worker wall times
are not elapsed campaign time or isolated CPU benchmarks.

| Frozen controller | Mean ± SD of five run-level p95 latencies (ms) | Largest observed latency (ms) | Mean intervention rate |
| --- | ---: | ---: | ---: |
{latency_rows}

These p95 statistics are summarized across runs, not a pooled p95 reconstructed
from percentile summaries. Shared-host scheduling and first-call overhead remain
in the measurements. There were **{a['training_fallbacks']} training and {a['evaluation_fallbacks']} evaluation fallbacks**,
and **{a['training_invalid_candidates']} training and {a['evaluation_invalid_candidates']} evaluation invalid candidates**.
All recorded invariants passed: fixed predictor, real-only replay, evaluation
excluded from learning, frozen weights/buffers, and unchanged global random streams.
Zero numerical failures does not establish physically valid predictions.

## Forecast and action-ranking diagnostics

The independent diagnostic set contains 32 ordinary, 32 near-upright, and 32
high-angular-speed states. These balanced strata are not an estimate of the
controller's operating distribution. Training scales normalize forecast error.
The predictor is unchanged, and no outcome here is used for fitting or tuning.

![Forecast and decision diagnostics]({link('diagnostic_results.png')})

| Forecast horizon | Normalized RMSE, original gravity 10 | Normalized RMSE, changed gravity 15 |
| --- | ---: | ---: |
{forecast_rows}

At H5 with original dynamics, normalized RMSE was
{forecast['scenarios']['gravity_10']['5']['groups']['nominal']['normalized_rmse']:.5f} for ordinary states,
{forecast['scenarios']['gravity_10']['5']['groups']['near_upright']['normalized_rmse']:.5f} near upright, and
{high_speed['normalized_rmse']:.5f} at high speed. In the high-speed stratum,
{high_speed['per_feature_bounds_violation_fraction'][2]:.1%} of predicted endpoint velocities exceeded
the physical ±8 limit; the largest magnitude was {high_speed['maximum_absolute_angular_velocity']:.3f}.
{high_speed['observation_bounds_violation_fraction']:.1%} violated at least one observation bound.
Finite-value checks do not detect these errors. Gravity 15 is a deliberately
changed plant, exposing dynamics mismatch rather than measuring routine accuracy.

The decision test uses only the final seed-11 actor-trained policy and the same
16 candidate first actions at each state. Reference values are discounted
200-step returns after that first action followed by the frozen actor. Regret
is relative to the best tested candidate, not a globally optimal action or the
full replanning controller. Its units differ from the undiscounted episodic
returns above.

| First-action selector | Mean regret to best tested candidate | Mean pairwise ranking agreement |
| --- | ---: | ---: |
{decision_rows}

Learned H1 regret was {learned['mean_regret_to_best_tested_candidate']:.5f}; exact-dynamics H1 regret
was {oracle['mean_regret_to_best_tested_candidate']:.5f}. Exact lookahead still uses a learned critic
at its tail. The actor already has little local improvement headroom in this
diagnostic, and replacing learned dynamics with exact dynamics alone does not
make the H1 score select better actions here. This supports investigating critic
tail accuracy and scoring, alongside model error, before increasing model size
or planning horizon.

Exact dynamics matched installed Gymnasium observations within
{forecast['oracle_validation']['max_absolute_observation_error']:.3g} and rewards within
{forecast['oracle_validation']['max_absolute_reward_error']:.3g} over 256 one-step cases.
Twelve independently stepped 200-step reference traces matched the vectorized
return calculation within {decisions['reference_validation']['max_absolute_return_difference']:.3g}.
Complete states, candidate scores, reference returns, and group metrics are
preserved in the raw diagnostic records.

## Software findings and next steps

**{test_count} tests passed** in the expanded suite. Stress tests found two API
defects: float64 action bounds could alias mutable caller arrays, and complex
critic/reward outputs could discard imaginary NaNs during casting. Bounds are
now copied and complex scores rejected. Both fixes preceded the full benchmark;
normal finite float32 scoring is unchanged. The suite also verifies independent
return calculations, mixed terminal times, late numerical faults, fallback
recovery, immutable inputs, optimizer/RNG/mode restoration after exceptions,
atomic checkpoint loading, and exact-dynamics diagnostic calculations.

Keep the H1 feature experimental and compare against both existing controllers
for each new task. Prioritize valid-state prediction at speed limits and under
dynamics changes, then inspect candidate ordering and critic-tail error. Further
tests should use independently fitted predictors, more policy seeds, and a task
with meaningful planning headroom. AUV conclusions require representative
vehicle dynamics, reward/termination rules, disturbance scenarios, and direct
navigation outcomes. GPU hardware was not tested.

## Reproduction and artifacts

Base revision was `{manifest['source_commit'][:7]}`. The manifest's individual
source SHA256 values identify the actual tested working tree, including the two
defensive fixes and benchmark runner; the unmodified base revision alone is not
the tested code. Original and archived JSON hashes are retained in the artifact
index. Raw result JSON is minified without dropping any episode or candidate data;
the manifest is copied byte-for-byte. Large checkpoints are not committed.

```bash
python -m pip install -r requirements-experiments.txt
python experiments/lookahead_benchmark.py run --output runs/lookahead-extensive --data runs/pendulum-data.npz --predictor runs/pendulum-improved/model.pt --steps 20000 --warmup 1000 --seeds 11 22 33 44 55 --batch_size 64 --episodes 20 --monitor_episodes 5 --workers 3
python experiments/lookahead_diagnostics.py --phase all --prediction_checkpoint runs/pendulum-improved/model.pt --policy_checkpoint runs/lookahead-extensive/seed_11/actor/checkpoint_20000 --output runs/lookahead-extensive-diagnostics/final --threads 1
python experiments/lookahead_early_checkpoint.py --benchmark_dir runs/lookahead-extensive --output runs/lookahead-extensive-early --checkpoint 5000 --reset_seed 910000 --episodes 20
python experiments/summarize_lookahead.py --input runs/lookahead-extensive --diagnostics runs/lookahead-extensive-diagnostics/final --early runs/lookahead-extensive-early --output docs/results/lookahead_extensive --report docs/LOOKAHEAD_EXTENSIVE_RESULTS.md --test-count {test_count} --measurement-date {measurement_date}
```

Use fresh output paths when reproducing; scripts preserve existing results.
The shared dataset/predictor workflow is documented in
[the prediction results](STATE_PREDICTION_RESULTS.md). Rebuilding figures from
committed records requires no retraining: use `--input docs/results/lookahead_extensive`
and `--diagnostics docs/results/lookahead_extensive/raw` plus
`--early docs/results/lookahead_extensive` with new output/report paths.
"""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="runs/lookahead-extensive")
    parser.add_argument("--diagnostics", default="runs/lookahead-extensive-diagnostics/final")
    parser.add_argument("--early", default="runs/lookahead-extensive-early")
    parser.add_argument("--output", default="docs/results/lookahead_extensive")
    parser.add_argument("--report", default="docs/LOOKAHEAD_EXTENSIVE_RESULTS.md")
    parser.add_argument("--test-count", type=int, required=True)
    parser.add_argument("--measurement-date", default="2026-11-25")
    args = parser.parse_args(argv)
    output, report_path = Path(args.output), Path(args.report)
    if output.exists() or report_path.exists():
        parser.error("Choose new output/report paths; existing results are preserved")
    try:
        manifest_path, manifest, paths, runs = load_complete(args.input)
        diagnostics = Path(args.diagnostics)
        forecast, decisions = read_json(diagnostics / "forecast.json"), read_json(diagnostics / "decision.json")
        for diagnostic in (forecast, decisions):
            if diagnostic["predictor"]["sha256"] != manifest["predictor"]["sha256"]:
                raise ValueError("Diagnostic predictor differs from the policy benchmark")
            for source, identity in diagnostic["source"].items():
                if source in manifest["sources"] and identity["sha256"] != manifest["sources"][source]:
                    raise ValueError("Diagnostic source differs from benchmark source: " + source)
        expected = runs[(manifest["protocol"]["training_seeds"][0], "actor")]["checkpoints"][-1]["weights"]
        if any(decisions["policy"][name]["sha256"] != expected[name]["sha256"] for name in ("actor", "critic")):
            raise ValueError("Decision diagnostics did not use the declared final seed-11 policy")
        analysis = aggregate(manifest, runs)
        early_manifest, early_paths, early_analysis = load_early(args.early, manifest_path, manifest, runs)
        analysis["early_posthoc"] = early_analysis
        source_matches = {name: (REPOSITORY / name).exists() and digest(REPOSITORY / name) == sha for name, sha in manifest["sources"].items()}
        analysis["provenance"] = {"measurement_date": args.measurement_date, "base_source_commit": manifest["source_commit"],
                                   "manifest_sha256": digest(manifest_path), "tested_source_sha256": manifest["sources"],
                                   "current_sources_match_manifest": source_matches, "summarizer_sha256": digest(__file__),
                                   "test_count": args.test_count}
        output.mkdir(parents=True)
        artifacts = []
        archive_json(manifest_path, output / "raw/manifest.json", artifacts, exact=True)
        for (seed, mode), path in paths.items():
            archive_json(path, output / "raw" / f"seed_{seed}_{mode}.json", artifacts)
            log = path.parent / "worker.log"
            if log.exists():
                shutil.copyfile(log, output / "raw" / f"seed_{seed}_{mode}.log")
        for name in ("forecast", "decision"):
            archive_json(diagnostics / f"{name}.json", output / "raw" / f"{name}.json", artifacts)
        archive_json(early_manifest, output / "raw/early_manifest.json", artifacts, exact=True)
        for (seed, name), path in early_paths.items():
            archive_json(path, output / "raw" / f"early_seed_{seed}_{name}.json", artifacts)
        (output / "artifact_index.json").write_text(json.dumps(artifacts, indent=2) + "\n", encoding="utf-8")
        (output / "analysis.json").write_text(json.dumps(analysis, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        create_figures(analysis, forecast, decisions, output)
        prefix = Path(os.path.relpath(output, report_path.parent)).as_posix()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report_text(analysis, manifest, forecast, decisions, artifact_prefix=prefix,
                                           test_count=args.test_count, measurement_date=args.measurement_date), encoding="utf-8")
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    print(json.dumps({"report": str(report_path), "analysis": str(output / "analysis.json"),
                      "primary_paired": analysis["primary_paired"], "accounting": analysis["accounting"]}, indent=2))
    return analysis


if __name__ == "__main__":
    main()
