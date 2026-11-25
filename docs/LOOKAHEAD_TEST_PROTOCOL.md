# Extended lookahead test protocol

Prepared on **2026-11-25**, before the extended benchmark's final results.
The short runtime pilot determines the feasible compute budget; its returns do
not select settings. The primary lookahead horizon stays at one step.

## Questions

1. Does lookahead improve decisions when actor and critic weights are fixed?
2. Does using lookahead during data collection improve TD3 at an equal real
   interaction budget, and does any benefit remain in the actor alone?
3. How do runtime, horizon, prediction error, and action-ranking error affect
   the result?
4. Do numerical faults or unusual API inputs break controller invariants?

## Shared environment and data

Use standard `Pendulum-v1`, CPU, one PyTorch thread per worker, and at most three
concurrent policy-training workers. GPU behavior and AUV transfer are outside
this experiment.

All methods receive the same **6,000 real transitions** from 30 previously
recorded random-control episodes in `runs/pendulum-data.npz`. These populate
their real replay buffers; rewards are reconstructed from Pendulum's known
reward function and recorded state/action, and true terminal flags come from
the archive. All methods can learn from this common data.

The lookahead controller uses the existing fixed
`runs/pendulum-improved/model.pt` predictor. Its training/validation/test episode
partition remains unchanged. It is not refitted using policy evaluation data.
Its separate original forecasting test split does not serve as the policy test
set. New policy evaluations use independent reset seeds.

The shared predictor is held fixed across policy-training seeds. Therefore,
variation across runs estimates policy-training variation conditional on that
predictor and dataset; it does not estimate variation across independently
trained dynamics models. Record all artifact and source hashes.

## Matched training comparison

Train each of these methods from scratch for each seed in
**11, 22, 33, 44, 55**:

| Method | Training behavior |
| --- | --- |
| Actor | TD3 actor with Gaussian exploration |
| Critic | Nearby candidates ranked by the twin-critic minimum |
| Lookahead H1 | Nearby candidates ranked by one predicted reward plus critic continuation value |

Each of the 15 runs gets **20,000 new environment interactions**, including
1,000 initial random-action steps. Including common data collection, the real
transition budget is **26,000 per method/run**. The common dataset is collected
once and reused; report both unique physical collection and per-run exposure.
Evaluation interactions are counted separately and never enter replay.

Fixed settings:

- Batch size 64; one TD3 training call per new interaction after warm-up.
- Actor and critics retain their existing two 256-unit hidden layers.
- Discount 0.99; target interpolation 0.005; actor/critic learning rate 0.0003.
- Delayed actor update every two training calls.
- Target-action noise 0.4 and clip 1.0 in Pendulum's action units.
- Sixteen search candidates, including the original actor action.
- Candidate noise 0.2 and exploration noise 0.1 as fractions of action half-range.
- Use private selector randomness for every behavior mode, including actor,
  so action generation does not consume replay-sampling or target-noise streams.

Monitor five fixed reset seeds **200000–200004** at new-interaction steps
0, 5,000, 10,000, and 20,000. Save 5,000-step and 20,000-step weights. The final
20,000-step checkpoint is the evaluation checkpoint regardless of monitor return;
monitoring does not select a best checkpoint or configuration.

## Final held-out evaluations

Use the same 20 reset seeds **900000–900019** for final evaluation, with
exploration disabled. For each trained policy, evaluate its deployed selector.
For policies trained with critic search or lookahead, also evaluate the actor
alone. Reuse the actor evaluation when it is already the deployed controller.

For each of the five actor-trained final checkpoints, hold actor/critics fixed
and evaluate:

- Actor-only action selection.
- Critic-only candidate search.
- Learned lookahead at horizons 1, 3, and 5.

Reuse identical evaluations rather than count them twice. Pair environment reset
seeds and candidate-noise seeds across search modes. States and candidate actions
can diverge after controllers make different decisions. Horizons 3 and 5 are
secondary diagnostics, not a search for a replacement primary endpoint.

## Primary endpoints and interpretation

The primary frozen-controller comparisons are **H1 minus actor** and
**H1 minus critic search** on the same actor-trained checkpoints. The primary
training comparisons are the deployed **H1-trained controller minus actor-trained
controller** and **H1-trained controller minus critic-trained controller** at the
same real interaction budget.

First average the 20 episodes within each training seed, then compare the five
seed means. Report every seed, the mean paired difference, and its spread. If a
95% Student-t interval is reported, use the five paired seed differences as the
replications (four degrees of freedom), state its assumptions and small sample
size, and do not treat 100 evaluation episodes as 100 independent training runs.
No multiple-comparison-adjusted significance or broad superiority claim is made
from secondary horizon, group, or diagnostic comparisons.

Also report monitoring learning curves, actor-only endpoints, actual training
calls, wall-clock training/evaluation time, action latency median/p95/max,
intervention rates, invalid-candidate counts, and fallback rates. Runtime is
measured on a shared host with concurrent workers; it is not a hardware latency
guarantee. Prior predictor-training compute is outside the newly timed training
runs and must be identified separately.

## Forecast and action-ranking diagnostics

Use a fixed manually generated test set with seed **812345**: 32 ordinary
states, 32 near-upright states, and 32 high-angular-speed states. Compare forecast
errors at horizons 1, 3, and 5 using identical bounded action sequences in the
learned model and known dynamics. Normalize using the predictor's training
scales, never test statistics. Repeat under a deliberately changed gravity
parameter (`g=15` versus the training environment's `g=10`) to expose dynamics
mismatch. This is a designed stress test, not an estimate of routine error.

Using the final actor-trained seed-11 checkpoint, compare critic-only and learned
or exact-dynamics lookahead with the **same 16 candidate actions** at each test
state. The reference outcome is a 200-step discounted real-dynamics return after
the candidate first action followed by the frozen actor. Report ranking
agreement and regret relative to the best tested candidate, not a globally
optimal policy. Exact lookahead still uses learned critics at its tail.

Any vectorized exact-dynamics implementation must first match installed
Gymnasium transitions/rewards on independent random cases. Report these
diagnostics as conditional on one policy. Do not retrain or tune the predictor
against these held-out outcomes.

## Software robustness

Extend the existing 94-test suite with independent scalar reference calculations,
different terminal times, later-step numerical faults, immutable input checks,
float64 operation, exception restoration of module modes/gradients/randomness,
and atomic checkpoint-load failures. Fix concrete implementation defects and
freeze source hashes before full benchmark execution. Do not silently combine
results produced before and after a behavior-changing fix.

## Reproducibility and reporting

Preserve a machine-readable manifest before runs, per-run progress/checkpoints,
all final evaluation episodes, failure records, source/artifact hashes, and
aggregates. Existing completed runs must not be overwritten under a different
configuration. Commit scripts, tests, protocol, and compact measured reports;
large generated checkpoints remain reproducible local run artifacts.

## Secondary follow-up added during execution

After the first two training seeds completed, their predeclared five-episode
monitoring suggested a larger lookahead benefit at 5,000 steps than at the
20,000-step endpoint. Before running additional evaluations, the following
secondary check was specified. It is **post hoc**, not a replacement for the
primary endpoints above.

Evaluate every already saved 5,000-step checkpoint on **new reset seeds
910000–910019**. Settings, weights, predictor, and training remain unchanged.
At this checkpoint each method has received 11,000 real transitions including
the shared prefill, and completed 4,000 TD3 training calls.

For each of the five training seeds, evaluate the three deployed training
methods, actor-only selection for the two assisted methods, and critic-only/H1
selection on the actor-trained checkpoint. Reuse the actor baseline evaluation.
That is seven unique controller evaluations per seed, 700 episodes, and
**140,000 additional evaluation interactions**. These data never enter training.

Report individual seeds, paired seed-mean differences, actor-only endpoints, and
frozen-controller results separately from the original final-checkpoint study.
Fresh scenarios help distinguish an early-learning effect from the monitoring
scenario selection, but they do not remove the post-hoc choice of checkpoint.
Do not tune a horizon, action-search setting, or model against this follow-up.
