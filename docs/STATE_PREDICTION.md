# State prediction: learning, evaluating, and improving dynamics

## Scope

This feature learns a numerical next-state predictor from recorded transitions.
For an AUV, state features might include position, velocity, heading, angular
velocity, estimated current, and tracked obstacle information. The current
repository contains generic TD3 and demonstration environments; it does not
contain a hydrodynamically validated AUV simulator or a real-vehicle dataset.

The predictor trains separately from TD3's actor and twin critics. Training it
does not automatically change the controller. The optional [lookahead selector](LOOKAHEAD.md)
uses a saved predictor to compare candidate actions; TD3 losses, replay labels,
and Bellman targets remain unchanged. Prediction improvements alone do **not**
establish improved navigation, collision avoidance, or policy reward. The
lookahead comparison command measures actual return with frozen policy networks.

## Architecture

`state_prediction/model.py` implements a two-layer MLP with 128 units per hidden
layer and SiLU activations by default. It receives the concatenated state and
applied action, predicts a normalized state change, reverses that normalization,
and adds the result to the input state:

`predicted_next = state + delta_mean + delta_scale * network(normalized_inputs)`

All input and delta statistics are computed on the training episodes. The
normalizers are saved with the weights. Standard deviations of constant features
use a scale of one. The model initially predicts the average training delta.

The one-step loss is mean squared error in normalized delta space. Training
uses Adam, gradient clipping, validation-based early stopping, and restoration
of the best model and optimizer state, including the initial model if no update
improves validation. Nonfinite inputs and predictions fail explicitly.

An optional rollout loss repeatedly feeds predicted states back into the network
with recorded actions. It compares the predicted sequence with the recorded
sequence using training-state feature scales. Gradients flow through the whole
predicted sequence; real intermediate states are not substituted. The combined
loss is `one_step_loss + rollout_loss_weight * rollout_loss`.

## Reproducible data separation

The default split is approximately 70% training, 15% validation, and 15% test by
**episode count**, with at least one episode per split. Different episode
lengths can produce different transition proportions. Seeded episode IDs and a
SHA-256 fingerprint of the exact dataset are stored in each checkpoint.

- Training episodes fit normalizers, neural weights, and the linear baseline.
- Validation episodes choose the checkpoint and decide whether an improvement
  candidate is accepted.
- Test episodes are evaluated only after that decision. Their errors are for
  reporting and do not enter optimizer updates or checkpoint selection.

Randomly splitting neighboring transitions would put highly related parts of
the same trajectory on both sides of the split. This workflow keeps episodes
together instead. It cannot detect whether a user imported duplicate or closely
related trajectories under different episode IDs; use genuinely independent
episodes and preserve the original recording boundaries.

The `improve` command preserves the original dataset, split, normalizers, horizon
settings, and validation-window sampling seed. Changing its training seed only
changes training minibatch/window sampling. The data fingerprint blocks accidental
reuse of the manifest with a changed archive.

## Understanding the accuracy report

`report.md` is the readable result. `report.json` includes all per-dimension and
per-horizon values, split IDs, training history, and the improvement decision.
`history.csv` makes it easy to plot training loss and validation score.

| Metric | Meaning |
|---|---|
| RMSE | Root mean squared prediction error; more sensitive to large errors |
| MAE | Mean absolute error |
| Bias | Mean predicted-minus-observed error; exposes systematic drift |
| Per-dimension errors | Errors for each individual state feature |
| Normalized RMSE | Error divided by training-state feature scale before aggregation |
| Open-loop endpoint error | Error after recursively predicting a sequence of steps |

Raw aggregate errors mix state units, so inspect individual features or normalized
metrics when comparing position, velocity, angles, and other quantities.
Normalization makes numerical comparisons easier; it does not encode the safety
importance of a particular state dimension.

The default forecast horizons are 1, 5, and 10 steps. A step is one recorded
control interval, not necessarily one second. Horizon 1 uses all transitions.
Longer horizons use up to 1,000 reproducibly sampled windows each. All compared
models receive the same initial states and recorded action sequences.

The selection score is the arithmetic mean of normalized endpoint RMSE across
available requested horizons, always including horizon 1. Unsupported horizons
are listed explicitly. A short-episode dataset cannot substantiate long-horizon
accuracy. No window crosses an episode reset, true termination, or time-limit
truncation, although the final transition may be its last prediction target.

Two baselines put the results in context:

1. **Persistence:** predict that the state remains unchanged throughout the
   forecast. This is often deceptively competitive at very short intervals.
2. **Linear delta:** fit an affine relationship from normalized state/action to
   state change using ridge regression on training episodes only.

The neural model is useful only to the extent that its measured accuracy and
other characteristics justify the added complexity. A linear model may win on a
simple environment. Reports retain that result instead of hiding it.

## Improve a saved predictor

```bash
python -m state_prediction improve --data datasets/pendulum.npz --checkpoint runs/pendulum/model.pt --output runs/pendulum-improved
```

Defaults are 40 additional epochs, learning rate 0.0003, and a five-step rollout
loss with weight 0.2. The candidate starts from the original weights and optimizer
state. Its best validation checkpoint competes against the incumbent on exactly
the same validation windows. `--min-improvement` sets a required absolute decrease
in the normalized validation score; its default is zero.

If the candidate fails to improve, the output contains the incumbent and reports
`kept_incumbent`. Otherwise it reports `accepted_candidate`. The input checkpoint
is always preserved. Improved validation accuracy may still coincide with worse
test accuracy; the report exposes both. It is not a generalization guarantee.

Use the validation errors to decide the next change:

| Observed problem | Experiment to try |
|---|---|
| High training and validation error | Collect more varied trajectories or try a larger network in a new training run |
| Good one-step but poor longer forecasts | Enable or adjust `--rollout-loss-weight` and `--rollout-horizon` |
| Validation worsens while training loss falls | Reduce training, increase patience only with a reason, or collect more independent data |
| One feature has large bias | Check its units, action timing, representation, and whether relevant inputs are missing |
| Linear baseline wins | Consider using the simpler model, or investigate the source of neural generalization error |
| Errors concentrate in unfamiliar situations | Collect representative data from those conditions and start a new experiment |

For example, you can explicitly strengthen long-horizon training:

```bash
python -m state_prediction improve --data datasets/pendulum.npz --checkpoint runs/pendulum/model.pt --output runs/longer-rollouts --rollout-horizon 10 --rollout-loss-weight 1.0 --learning-rate 0.0001
```

Candidate evaluation is deterministic for the saved data/window settings on a
given runtime. GPU kernels and different library/hardware versions may introduce
small differences. Repeated tuning can overfit validation; use fresh, independent
final evaluation episodes for a larger research comparison. Do not choose model
changes by repeatedly optimizing the reported test score.

Adding more data requires a new `train` experiment and a clearly identified new
split. The current `improve` workflow intentionally uses the exact original data
so that its validation comparison remains meaningful.

## Import AUV or custom simulator data

Use the same observation representation consistently before and after each
control. Record the action actually applied, its resulting next observation, and
episode boundaries. Do not replace the last observation with a reset observation.

```python
import numpy as np
from state_prediction.data import TransitionDataset

dataset = TransitionDataset(
    states=np.asarray(states, dtype=np.float32),         # [N, state_dim]
    actions=np.asarray(actions, dtype=np.float32),       # [N, action_dim]
    next_states=np.asarray(next_states, dtype=np.float32),
    episode_ids=np.asarray(episode_ids, dtype=np.int64), # [N]
    terminated=np.asarray(terminated, dtype=bool),       # true terminal
    truncated=np.asarray(truncated, dtype=bool),         # time/data cutoff
)
dataset.save("datasets/auv.npz")
```

Each episode ID must form one contiguous block. Within an episode, a row's
`next_state` must match the following row's `state`. Complete and partial episodes
are supported. Missing termination flags are allowed for imported data, but
correct episode IDs are then essential. Arrays are validated and serialized
without Python pickle objects.

For angles, consider continuous representations such as sine/cosine instead of a
discontinuity at a wrap boundary. Include velocity and other information needed
to make the next state predictable. A feed-forward predictor cannot recover
unobserved current or obstacle motion from an isolated, insufficient observation;
history-based state estimation would be a separate extension.

Train and evaluate:

```bash
python -m state_prediction train --data datasets/auv.npz --output runs/auv --horizons 1 5 10 20
python -m state_prediction evaluate --data datasets/auv.npz --checkpoint runs/auv/model.pt --output runs/auv-test.json
```

`evaluate` selects the saved test episodes automatically. For a new archive of
independently collected evaluation episodes, use `--external`. This evaluates
all supplied episodes and labels their independence as unverified; the command
cannot establish their provenance automatically. It rejects the original
training archive in external mode. Existing output files are never overwritten.

## Use the model from Python

```python
from state_prediction.model import StatePredictor

model = StatePredictor.load("runs/auv/model.pt", device="cpu")
next_state = model.predict(state, action)
trajectory = model.rollout(state, proposed_action_sequence)
```

For single inputs, shapes are `[state_dim]`, `[action_dim]`, and
`[horizon, action_dim]`, producing `[state_dim]` and `[horizon, state_dim]`.
Batched inputs add a leading batch dimension.

These are deterministic forecasts. The model does not currently provide an
ensemble, calibrated confidence intervals, or collision guarantees. The optional
[lookahead selector](LOOKAHEAD.md) combines these forecasts with task rewards and
TD3 critics. Its later rollout actions are recomputed by the actor at predicted
states; the fixed-action ``rollout`` helper above is a separate evaluation API.

## References

- [TD3 paper](https://arxiv.org/abs/1802.09477): actor/critic algorithm retained by this repository.
- [Neural Network Dynamics for Model-Based Deep RL](https://arxiv.org/abs/1708.02596): learning normalized state changes and evaluating predicted trajectories.
- [PETS](https://arxiv.org/abs/1805.12114): learned dynamics, trajectory forecasting, and uncertainty-aware planning; this implementation does not claim to reproduce PETS.
- [Gymnasium migration guide](https://gymnasium.farama.org/introduction/migration_guide/): separate termination and truncation semantics.
