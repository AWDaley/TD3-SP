"""Reproducible experiment workflow with a frozen, episode-level holdout."""

import csv
import json
from pathlib import Path

import numpy as np

from .data import TransitionDataset
from .evaluation import evaluate_predictor, PersistencePredictor, LinearDeltaPredictor
from .model import StatePredictor


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _prepare_run(output):
    output = Path(output)
    for name in ("model.pt", "report.json", "report.md", "history.csv"):
        if (output / name).exists():
            raise ValueError(f"{output / name} already exists; choose a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    return output


def _split_metadata(dataset, splits, *, seed, horizons, max_windows, state_names=None):
    return {
        "workflow_version": 1,
        "dataset_fingerprint": dataset.fingerprint(),
        "split_episode_ids": {name: np.unique(part.episode_ids).astype(int).tolist()
                              for name, part in splits.items()},
        "evaluation": {"seed": int(seed), "horizons": list(horizons), "max_windows": int(max_windows)},
        "state_names": state_names or [f"state_{i}" for i in range(dataset.state_dim)],
    }


def restore_splits(dataset, metadata):
    """Only the exact original data and split can be used for model selection."""
    if metadata.get("workflow_version") != 1:
        raise ValueError("Checkpoint has no supported experiment split; train with this workflow first")
    if dataset.fingerprint() != metadata.get("dataset_fingerprint"):
        raise ValueError("Dataset differs from the checkpoint's frozen split. Use the original dataset, "
                         "or explicitly use evaluate --external for a new independent dataset.")
    ids = metadata.get("split_episode_ids", {})
    if set(ids) != {"train", "validation", "test"}:
        raise ValueError("Checkpoint has an invalid split manifest")
    all_ids = [int(item) for name in ids for item in ids[name]]
    if len(all_ids) != len(set(all_ids)) or set(all_ids) != set(map(int, np.unique(dataset.episode_ids))):
        raise ValueError("Checkpoint splits overlap or do not cover the dataset")
    if any(not ids[name] for name in ids):
        raise ValueError("Each saved split must contain at least one episode")
    return {name: dataset.subset(ids[name]) for name in ("train", "validation", "test")}


def _evaluate(model, data, metadata):
    settings = metadata["evaluation"]
    return evaluate_predictor(model, data, state_scale=model.state_scale,
                              horizons=settings["horizons"], max_windows=settings["max_windows"],
                              seed=settings["seed"])


def _summary(model, splits, metadata, fit_result, promotion=None):
    # The test set is used only here, AFTER fit restored the validation-selected
    # weights and AFTER the improve command made its acceptance decision.
    models = {
        "state_predictor": model,
        "persistence": PersistencePredictor(),
        "linear_delta": LinearDeltaPredictor.fit(splits["train"]),
    }
    settings = metadata["evaluation"]
    test_metrics = {
        name: evaluate_predictor(candidate, splits["test"], state_scale=model.state_scale,
                                 horizons=settings["horizons"], max_windows=settings["max_windows"],
                                 seed=settings["seed"])
        for name, candidate in models.items()
    }
    return {
        "dataset_fingerprint": metadata["dataset_fingerprint"],
        "scope": "State prediction only; no predicted states enter TD3 actions or critic targets.",
        "selection_rule": "Lowest mean normalized endpoint RMSE across available validation horizons.",
        "normalization": "Training-state standard deviation; constant dimensions use a scale of 1.",
        "splits": {name: {"transitions": len(part), "episodes": int(len(np.unique(part.episode_ids))),
                          "episode_ids": metadata["split_episode_ids"][name]}
                   for name, part in splits.items()},
        "state_names": metadata["state_names"],
        "evaluation": settings,
        "fit": fit_result,
        "validation": _evaluate(model, splits["validation"], metadata),
        "test": test_metrics,
        "promotion": promotion,
    }


def _report_markdown(report):
    names = report["state_names"]
    lines = ["# State prediction accuracy", "", report["scope"], "",
             "Training, validation, and test contain separate episodes. Test scores do not select checkpoints.",
             "", "## Data", "", "| Split | Episodes | Transitions |", "|---|---:|---:|"]
    for name, split in report["splits"].items():
        lines.append(f"| {name} | {split['episodes']} | {split['transitions']} |")
    lines += ["", "## Held-out test accuracy", "",
              "Lower errors are better. Raw aggregate errors mix state units; compare normalized errors or individual features.",
              "", "| Model | One-step RMSE | One-step normalized RMSE | Mean rollout normalized RMSE |",
              "|---|---:|---:|---:|"]
    for name, metrics in report["test"].items():
        one = metrics["one_step"]
        lines.append(f"| {name} | {one['rmse']:.6g} | {one['normalized_rmse']:.6g} | {metrics['selection_score']:.6g} |")
    lines += ["", "## Open-loop endpoint accuracy", "",
              "Every rollout starts from one real state, then feeds predicted states back into the model with recorded actions.",
              "", "| Horizon (steps) | Neural normalized RMSE | Persistence | Linear delta | Windows |",
              "|---:|---:|---:|---:|---:|"]
    rollouts = report["test"]["state_predictor"]["rollout"]
    for horizon, metrics in rollouts.items():
        baseline = report["test"]["persistence"]["rollout"][horizon]
        linear = report["test"]["linear_delta"]["rollout"][horizon]
        lines.append(f"| {horizon} | {metrics['normalized_rmse']:.6g} | {baseline['normalized_rmse']:.6g} | {linear['normalized_rmse']:.6g} | {metrics['sample_count']} |")
    for label, metrics in (("validation", report["validation"]),
                           ("test", report["test"]["state_predictor"])):
        if metrics["skipped_horizons"]:
            lines += ["", f"Unavailable {label} horizons (episodes too short): "
                      + ", ".join(map(str, metrics["skipped_horizons"])) + "."]
    lines += ["", "## Neural one-step error by state dimension", "",
              "Bias is predicted minus observed. Use this table to locate systematic drift.", "",
              "| Feature | RMSE | MAE | Bias | Normalized RMSE |", "|---|---:|---:|---:|---:|"]
    for row in report["test"]["state_predictor"]["one_step"]["per_dimension"]:
        label = str(names[row["index"]]).replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {label} | {row['rmse']:.6g} | {row['mae']:.6g} | {row['bias']:.6g} | {row['normalized_rmse']:.6g} |")
    if report.get("promotion"):
        promotion = report["promotion"]
        lines += ["", "## Improvement decision", "",
                  f"Decision: **{promotion['decision']}**.", "",
                  f"Incumbent validation score: {promotion['incumbent_validation_score']:.6g}. "
                  f"Candidate validation score: {promotion['candidate_validation_score']:.6g}.", "",
                  "The original checkpoint is preserved. This decision was made before calculating test metrics."]
    lines += ["", "## Interpretation", "",
              "- Compare with both baselines; a small error by itself does not establish useful learning.",
              "- Increasing rollout error exposes compounding prediction mistakes.",
              "- Validation chooses training epochs and improvement candidates. Reserve test results for reporting.",
              "- These errors are not calibrated uncertainty estimates or evidence of safer vehicle control.", ""]
    return "\n".join(lines)


def _save_run(output, model, metadata, report):
    model.save(output / "model.pt", metadata=metadata)
    _write_json(output / "report.json", report)
    (output / "report.md").write_text(_report_markdown(report), encoding="utf-8")
    history = report["fit"].get("history", [])
    if history:
        keys = list(dict.fromkeys(key for row in history for key in row))
        with (output / "history.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(history)
    print(f"Saved predictor and accuracy reports to {output}")
    for name, result in report["test"].items():
        print(f"  {name}: test one-step RMSE={result['one_step']['rmse']:.6g}; "
              f"mean normalized rollout RMSE={result['selection_score']:.6g}")


def train_experiment(data_path, output, *, hidden_dims=(128, 128), learning_rate=1e-3,
                     device="cpu", seed=0, epochs=60, batch_size=256, patience=12,
                     horizons=(1, 5, 10), max_windows=1000, rollout_loss_weight=0.0,
                     rollout_horizon=5, train_fraction=0.7, validation_fraction=0.15,
                     state_names=None):
    data = TransitionDataset.load(data_path)
    if state_names is not None and len(state_names) != data.state_dim:
        raise ValueError("Provide exactly one state name per observation dimension")
    splits = data.split_by_episode(train_fraction=train_fraction,
                                   validation_fraction=validation_fraction, seed=seed)
    output = _prepare_run(output)
    metadata = _split_metadata(data, splits, seed=seed, horizons=horizons,
                               max_windows=max_windows, state_names=state_names)
    model = StatePredictor(data.state_dim, data.action_dim, hidden_dims=hidden_dims,
                           learning_rate=learning_rate, device=device, seed=seed)
    fit_result = model.fit(splits["train"], splits["validation"], epochs=epochs,
                           batch_size=batch_size, patience=patience, seed=seed,
                           horizons=horizons, max_windows=max_windows,
                           rollout_loss_weight=rollout_loss_weight, rollout_horizon=rollout_horizon)
    metadata["training"] = {"epochs": epochs, "batch_size": batch_size, "patience": patience,
                             "learning_rate": learning_rate, "rollout_loss_weight": rollout_loss_weight,
                             "rollout_horizon": rollout_horizon, "best_epoch": fit_result["best_epoch"]}
    report = _summary(model, splits, metadata, fit_result)
    _save_run(output, model, metadata, report)
    return report


def improve_experiment(data_path, checkpoint, output, *, device="cpu", epochs=40,
                       batch_size=256, patience=10, seed=1, learning_rate=3e-4,
                       rollout_loss_weight=0.2, rollout_horizon=5, min_improvement=0.0):
    if not np.isfinite(min_improvement) or min_improvement < 0:
        raise ValueError("min_improvement must be finite and nonnegative")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    data = TransitionDataset.load(data_path)
    model = StatePredictor.load(checkpoint, device=device)
    metadata = dict(model.metadata)
    splits = restore_splits(data, metadata)
    output = _prepare_run(output)
    incumbent_score = _evaluate(model, splits["validation"], metadata)["selection_score"]
    model.learning_rate = learning_rate
    for group in model.optimizer.param_groups:
        group["lr"] = learning_rate
    settings = metadata["evaluation"]
    fit_result = model.fit(splits["train"], splits["validation"], epochs=epochs,
                           batch_size=batch_size, patience=patience, seed=seed,
                           horizons=settings["horizons"], max_windows=settings["max_windows"],
                           rollout_loss_weight=rollout_loss_weight, rollout_horizon=rollout_horizon,
                           initialize_normalizers=False, validation_seed=settings["seed"])
    candidate_score = _evaluate(model, splits["validation"], metadata)["selection_score"]
    accepted = incumbent_score - candidate_score > max(min_improvement, 1e-12)
    if not accepted:
        model = StatePredictor.load(checkpoint, device=device)
    promotion = {
        "decision": "accepted_candidate" if accepted else "kept_incumbent",
        "incumbent_validation_score": incumbent_score,
        "candidate_validation_score": candidate_score,
        "min_improvement": min_improvement,
        "source_checkpoint": str(checkpoint),
        "training_seed": seed,
        "learning_rate": learning_rate,
        "rollout_loss_weight": rollout_loss_weight,
        "rollout_horizon": rollout_horizon,
    }
    metadata["last_improvement"] = promotion
    # Preserve the original evaluation window seed for fair comparisons even
    # when the training minibatch seed changes between improvement attempts.
    report = _summary(model, splits, metadata, fit_result, promotion)
    _save_run(output, model, metadata, report)
    print(f"Improvement decision: {promotion['decision']}")
    return report


def evaluate_experiment(data_path, checkpoint, *, output=None, device="cpu", external=False):
    if output and Path(output).exists():
        raise ValueError("Evaluation output already exists; choose a new path to preserve existing data")
    data = TransitionDataset.load(data_path)
    model = StatePredictor.load(checkpoint, device=device)
    metadata = model.metadata
    if not external:
        selected = restore_splits(data, metadata)["test"]
        split_name = "test"
    else:
        if data.fingerprint() == metadata.get("dataset_fingerprint"):
            raise ValueError("This is the training dataset; omit --external to evaluate its saved test split")
        selected = data
        split_name = "external_independence_not_verified"
    if "evaluation" not in metadata:
        raise ValueError("Checkpoint does not contain evaluation settings")
    result = {"split": split_name, "dataset_fingerprint": data.fingerprint(),
              "transitions": len(selected), "metrics": _evaluate(model, selected, metadata)}
    if output:
        _write_json(output, result)
        print(f"Saved evaluation to {output}")
    print(json.dumps(result, indent=2, allow_nan=False))
    return result
