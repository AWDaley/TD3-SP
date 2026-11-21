# Verified state prediction results

Measured on 2026-11-21 using the commands below. Full metrics, dataset fingerprints, episode split IDs, per-dimension errors, and promotion decisions are in [the machine-readable record](results/state_prediction_benchmarks.json).

Runtime: Python 3.12.14, PyTorch 2.5.1+cpu, NumPy 2.3.5, Gymnasium 1.4.0; CPU, two PyTorch threads.

## Held-out forecast accuracy

The score is the mean normalized endpoint RMSE at 1, 5, and 10 steps; lower is better. Each experiment fits its scales on its own training data, so compare models within a row.

| Dataset | Persistence | Linear delta | Trained neural | After improve |
|---|---:|---:|---:|---:|
| synthetic_navigation | 0.212334 | 0.025299 | 0.015015 | 0.015015 |
| pendulum | 0.689156 | 0.404434 | 0.008002 | 0.006719 |

## Improvement decisions

- **synthetic_navigation: kept_incumbent**. Validation score 0.01257192 → 0.01257192; selected-model test error decreased by 0.00%.
- **pendulum: accepted_candidate**. Validation score 0.02112839 → 0.01939506; selected-model test error decreased by 16.03%.

The synthetic candidate failed to beat the incumbent, so its saved predictions remain unchanged. The Pendulum candidate passed the validation gate; its test score also improved. Test results were calculated after each acceptance decision and were not used to choose a candidate.

## Reproduce

```bash
python -m state_prediction demo --output runs/toy-baseline --epochs 60 --seed 0
python -m state_prediction improve --data runs/toy-baseline/data.npz --checkpoint runs/toy-baseline/model.pt --output runs/toy-improved --epochs 40 --seed 1
python -m state_prediction collect --env Pendulum-v1 --episodes 30 --output runs/pendulum-data.npz --seed 17
python -m state_prediction train --data runs/pendulum-data.npz --output runs/pendulum-baseline --epochs 60 --seed 0 --state-names cos_theta sin_theta angular_velocity
python -m state_prediction improve --data runs/pendulum-data.npz --checkpoint runs/pendulum-baseline/model.pt --output runs/pendulum-improved --epochs 40 --seed 1
```

## Verification

- 43 behavioral tests passed, covering data separation, boundary handling, exact metrics, learned accuracy, checkpoint roundtrips, and acceptance/rejection behavior.
- A 650-step TD3 Pendulum run trained its policy and exported 650 transitions in four complete/partial episodes.
- Standalone collection loaded that saved actor and recorded another 30 transitions in three episodes.
- The evaluation command reproduced the saved predictor test report without training.

## Limits

These are single-seed software and prediction experiments. The toy system is not a validated underwater vehicle model. The Pendulum run measures predictive dynamics, not AUV transfer or TD3 control improvement. A prediction loss or validation gate is not a calibrated safety guarantee. Multi-seed comparisons and independent AUV trajectories are the next evidence needed for research claims.
