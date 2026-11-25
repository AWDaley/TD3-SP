# Lookahead integration checks

For the subsequent five-seed study, forecast stress cases, and regression fixes,
see [extensive validation results](LOOKAHEAD_EXTENSIVE_RESULTS.md). The measurements
below preserve the original small integration check rather than a full benchmark.

Measured on **2026-11-25** with Python 3.12.14, PyTorch 2.5.1+cpu,
NumPy 2.3.5, Gymnasium 1.4.0, CPU, one PyTorch thread.

## Behavioral and integration tests

**94 tests passed locally.** The suite includes the prior prediction/data tests
and new checks for exact return arithmetic, twin critics, terminal branch
stopping, actor feedback, bounds, numerical fallback, RNG/gradient isolation,
Pendulum reward parity, frozen comparisons, checkpoint loading, and the actual
action recorded by the training loop.

```bash
python -m pytest -q
```

## Frozen-policy comparison

This uses the existing **650-step** TD3 smoke checkpoint, a fitted improved
Pendulum predictor, 16 candidates, horizon 1, discount 0.99, and five matched
reset seeds 123–127. Each mode executes 1,000 real environment steps. The actor,
critics, and predictor are fixed throughout this comparison.

| Mode | Mean actual return | Median action latency (ms) | p95 latency (ms) | Intervention rate |
| --- | ---: | ---: | ---: | ---: |
| actor | -1735.9515 | 0.1371 | 0.3269 | 0.0% |
| critic | -1689.4137 | 0.6234 | 1.1530 | 98.5% |
| lookahead | -1604.0371 | 1.4548 | 3.5046 | 99.8% |

All modes recorded **zero fallbacks and zero invalid candidates**. Higher return
is better. Lookahead exceeded actor-only return by
131.9144 and critic-only search by
85.3766 on average in these episodes.

These are descriptive smoke results from **one lightly trained policy**, not
evidence across independent training seeds or proof of improved learning.
Latency includes first-call overhead and shared-host scheduling; maximum outliers
are retained in the [full frozen comparison JSON](results/lookahead_frozen_smoke.json).
That record includes individual episode returns, paired seeds, and checkpoint
SHA256 identities. Predictor training commands are in
[the earlier prediction results](STATE_PREDICTION_RESULTS.md).

```bash
python -m action_selection compare --env Pendulum-v1 --policy_checkpoint models/TD3_Pendulum-v1_0 --prediction_checkpoint runs/pendulum-improved/model.pt --episodes 5 --seed 123 --candidates 16 --noise_std 0.2 --horizon 1 --discount 0.99 --threads 1 --output runs/lookahead-integration-h1.json
```

Checkpoints are generated locally rather than committed to this repository;
retraining can change measured returns and timings. The hashes identify the
specific artifacts used for this recorded check.

## Training-loop check

A separate fresh TD3 run used seed 271, 650 real transitions, 100 random warm-up
steps, 16 candidates, and a **three-step** lookahead horizon. It completed all
550 TD3 training calls after warm-up, saved and reloaded the policy,
and exported 650 transitions in four complete/partial episodes.

The controller made 550 training selections, with 544 interventions,
0 fallbacks, and 0 invalid candidates. Recorded actions were finite
and within the environment bounds. The predictor's checksum remained unchanged.
Monitoring returns were -1438.6992 initially and
-678.2459 after training; those two monitoring episodes per checkpoint are
not a held-out performance claim or a comparison with ordinary TD3 training.

```bash
python main.py --env Pendulum-v1 --seed 271 --max_timesteps 650 --start_timesteps 100 --eval_freq 650 --eval_episodes 2 --batch_size 32 --replay_size 1000 --threads 1 --action_selection lookahead --prediction_checkpoint runs/pendulum-improved/model.pt --planning_candidates 16 --planning_horizon 3 --prediction_data runs/lookahead-training-h3.npz --save_model
```

Settings and verification are recorded in
[the training smoke JSON](results/lookahead_training_smoke.json).
The implementation was checked on CPU; GPU hardware was not exercised.

## What remains to establish

A research comparison needs multiple independently trained seeds, matched real
interaction budgets including predictor data, consistent gradient-update counts,
compute/latency reporting, and final evaluation scenarios held out from tuning.
AUV results additionally require an appropriate simulator/data and its exact
reward/termination adapter. These checks establish a working integration, not
vehicle-level reliability or safety.
