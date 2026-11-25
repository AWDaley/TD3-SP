# Short lookahead action selection for TD3

## What changes

This feature adds an optional controller around the TD3 actor. It proposes a
small batch of nearby actions, predicts their consequences, scores them, and
executes the best first action. At the next step it replans from the real state.

Three modes are available:

| Mode | Score | Required networks |
| --- | --- | --- |
| `actor` | Execute the actor proposal | Actor |
| `critic` | `min(Q1(state, action), Q2(state, action))` | Actor and twin critics |
| `lookahead` | Predicted rewards plus a discounted terminal critic value | Actor, twin critics, predictor |

`actor` remains the default. TD3's actor/critic losses, delayed updates, target
policy smoothing, and replay learning remain unchanged in every mode. The
predictor is loaded from a fitted checkpoint and stays fixed during a TD3 run.
There is no synthetic replay, distillation, or online dynamics fitting.

The built-in command-line reward adapter supports standard Gymnasium
`Pendulum-v1` and legacy Gym `Pendulum-v0`. The selector itself accepts explicit
custom task callables, so it is not restricted to Pendulum when used from Python.

## The return being scored

For one-step lookahead:

\[
\hat s' = f(s,a), \qquad
J(a) = r(s,a,\hat s') + \gamma(1-\hat d)
\min_i Q_i(\hat s',\mu(\hat s')).
\]

For a nonterminal horizon of `H` steps:

\[
J = \sum_{k=0}^{H-1}\gamma^k r(\hat s_k,a_k,\hat s_{k+1})
+ \gamma^H\min_i Q_i(\hat s_H,\mu(\hat s_H)).
\]

The candidate supplies `a_0`. Every later action is the current actor evaluated
at the current predicted state. The final value also uses the current actor and
current critics; target networks are used only by the unchanged TD3 training
algorithm. Search does not add target-policy smoothing noise.

True termination includes the terminal transition reward, then stops that
branch without a critic bootstrap. Reaching the planning horizon or an
environment time limit does not count as true termination. No actor/model/critic
calls are made for a branch after its terminal transition.

This is local first-action search with actor continuation, rather than joint
optimization of every action in a long sequence. Longer horizons increase
compute and can accumulate model error; one step is the default.

## Candidates, exploration, and invalid predictions

Candidate zero is the clipped, unperturbed actor action. With the default
`--planning_candidates 16`, there are 15 additional candidates. Their Gaussian
perturbations use standard deviation `--planning_noise` times each action
dimension's half-range. Every candidate and continuation action is clipped.

During training, `--expl_noise` first shifts the centre of those additional
candidates. The original actor candidate remains present. The selector then
scores the actual bounded candidates; no exploration noise is added to the
selected action afterward. Ranking can reduce how much exploration is actually
executed, so this behavior policy differs from ordinary additive-noise TD3.
Random warm-up actions remain unchanged. Search with nonzero exploration
requires at least two candidates.

Equal scores prefer the earlier candidate, so an exact tie retains the actor
action. A nonfinite predicted state, reward, continuation action, or either
critic value invalidates that candidate. If every candidate is invalid, the
controller returns the original clipped actor action and records a fallback.
Incorrect shapes/configuration raise errors. Fallback is a numerical recovery
mechanism, not a safety guarantee.

Candidate generation uses a private random generator. Evaluation constructs a
separate selector with no exploration, leaving the training candidate stream
unchanged. Scoring is batched, disables gradients, and preserves network weights,
existing gradients, and train/evaluation flags.

## Reproducible command-line workflow

Install the dependencies in the README, then run these commands from the repo.
Choose new dataset/report paths when repeating collection or comparison.

### 1. Collect data and train a predictor

```bash
python -m state_prediction collect --env Pendulum-v1 --episodes 30 --seed 17 --output datasets/lookahead-pendulum.npz --threads 1
python -m state_prediction train --data datasets/lookahead-pendulum.npz --output runs/lookahead-predictor --epochs 60 --threads 1 --state-names cos_theta sin_theta angular_velocity
```

The predictor needs the same observations, actions, timestep, and dynamics as
the controller environment. Dimension checks cannot establish semantic
compatibility. Training data for the predictor is additional real experience
and must be counted when making sample-efficiency comparisons.

### 2. Train ordinary TD3 and save a reference checkpoint

```bash
python main.py --env Pendulum-v1 --seed 0 --max_timesteps 10000 --start_timesteps 1000 --eval_episodes 5 --threads 1 --save_model
```

This saves `models/TD3_Pendulum-v1_0_actor` and
`models/TD3_Pendulum-v1_0_critic`, plus optimizer files. The short budget here is
an example workflow, not a convergence claim.

### 3. Compare the same checkpoint in all three modes

```bash
python -m action_selection compare --env Pendulum-v1 --policy_checkpoint models/TD3_Pendulum-v1_0 --prediction_checkpoint runs/lookahead-predictor/model.pt --episodes 10 --seed 123 --candidates 16 --noise_std 0.2 --horizon 1 --discount 0.99 --threads 1 --output runs/lookahead-comparison.json
```

The comparison loads actor and critic weights without requiring optimizers.
It performs no fitting or policy updates. Each episode uses the same explicit
reset seed across all three modes; the two searches also use matched candidate
noise seeds. Their states and resulting candidates can diverge after different
actions are executed. `--discount` must match the discount used to train the
checkpoint; legacy TD3 checkpoint files do not store that setting.

The new JSON report contains:

- Actual returns and episode lengths, including individual paired differences.
- Action latency median, p95, and maximum, including first-call overhead.
- Intervention/fallback rates and invalid-candidate counts.
- Environment, seeds, settings, device, and checkpoint SHA256 identities.

The output refuses to overwrite an existing file. These are paired evaluation
episodes from one trained policy, not independent training seeds. Compare
lookahead against **both** actor-only and critic-only search. A gain over only
actor-only selection does not isolate the benefit of dynamics forecasts.

### 4. Train TD3 with lookahead enabled

```bash
python main.py --env Pendulum-v1 --seed 0 --action_selection lookahead --prediction_checkpoint runs/lookahead-predictor/model.pt --planning_candidates 16 --planning_noise 0.2 --planning_horizon 1 --max_timesteps 10000 --start_timesteps 1000 --eval_episodes 5 --threads 1 --save_model --prediction_data datasets/lookahead-executed.npz
```

Training and evaluation share the same selector. Replay and the optional
prediction archive receive the final executed action and real environment
outcome. Time-limit truncations end recorded trajectories while TD3 continues
to bootstrap through them.

Saved files use `TD3_Pendulum-v1_0_lookahead_k16_h1` as their prefix. A sidecar
`results/TD3_Pendulum-v1_0_lookahead_k16_h1_selection.json` records settings,
predictor identity, monitoring returns, and training selection counters.
Critic-only runs use the suffix `_critic_k16` and require no predictor:

```bash
python main.py --env Pendulum-v1 --action_selection critic --planning_candidates 16 --max_timesteps 10000 --start_timesteps 1000 --eval_episodes 5 --threads 1 --save_model
```

`--load_model default` loads the original base checkpoint
`models/TD3_<environment>_<seed>`. To load an assisted run, pass its saved basename
explicitly, for example `--load_model TD3_Pendulum-v1_0_lookahead_k16_h1`.
As in upstream TD3, loading does not restore replay, environment state, or the
training-step counter. It is not an exact resumable training snapshot.

Training outputs are overwritten when their prefix is reused. Prefixes include
mode, candidate count, and lookahead horizon, but not noise or predictor identity.
Use separate working directories to preserve experiments that differ only in
those settings. Monitoring evaluations repeatedly use the same reset-seed
sequence; choose independent final evaluation seeds after tuning.

## Connect another environment from Python

The reward/termination adapter is explicit because a state predictor alone
cannot tell whether a predicted outcome is desirable or terminal. The built-in
Pendulum adapter uses the current angle and angular velocity plus the applied
torque penalty, exactly matching the environment. Pendulum has no true terminal
event; its episode length is a time limit. Modified reward/observation/action
wrappers and unsupported environments are rejected by the built-in factory.

For another environment, supply a callable with this contract:

```python
# states: [batch, state_dim], actions: [batch, action_dim]
# next_states: [batch, state_dim], all tensors on the actor device
def task(states, actions, next_states):
    rewards = your_reward_function(states, actions, next_states)  # float [batch]
    terminated = your_terminal_function(states, actions, next_states)  # bool [batch]
    return rewards, terminated
```

Use it with the reusable selector:

```python
from action_selection import ActionSelector, SelectionConfig
from state_prediction.model import StatePredictor

device = next(policy.actor.parameters()).device
predictor = StatePredictor.load("runs/your-task/model.pt", device=device)
selector = ActionSelector(
    policy, env.action_space.low, env.action_space.high,
    SelectionConfig(mode="lookahead", candidates=16, horizon=1, seed=0),
    predictor=predictor, task=task,
)
action = selector.select_action(state, exploration_std=0.0)
# Execute this action unchanged, then observe and repeat.
```

The adapter must reproduce the simulator's actual reward and true-termination
rules from the available state. Do not reuse a recorded reward or terminal flag
when evaluating a different action. If those rules depend on missing state,
extend the observation/model or provide the necessary task context explicitly.
The selector supports per-axis action bounds, but this repository's upstream
TD3 actor and CLI still require uniform symmetric bounds.

For AUV experiments, use representative dynamics data and the real navigation
objective, including any relevant vehicle/obstacle state. This repository does
not provide an AUV reward adapter or calibrated risk/uncertainty model.

## Verification and interpretation

See [measured integration checks](LOOKAHEAD_RESULTS.md) and their linked JSON
records. Unit/integration checks cover analytical returns, terminal branches,
actor feedback, clipping, invalid predictions, RNG isolation, checkpoint loading,
and the executed-action training path.

An implementation check establishes that the controller runs and uses the
intended score. A performance claim needs independent training seeds, fixed real
interaction budgets, compute accounting, and final held-out scenarios. Evaluate
the actor alone as well as the combined selector to determine whether deployment
still needs planning.

## Source layout

- `action_selection/selector.py`: candidate generation, scoring, and diagnostics.
- `action_selection/tasks.py`: reward/true-termination contract and Pendulum adapter.
- `action_selection/configuration.py`: fitted-model/environment validation.
- `action_selection/compare.py`: paired frozen-controller evaluation and CLI.
- `main.py`: optional shared controller in training and monitoring evaluation.
- `TD3.py`: unchanged learning updates; weight-only evaluation loading added.

The general short-horizon reward-plus-terminal-value design is related to
[TD-MPC](https://proceedings.mlr.press/v162/hansen22a.html). This implementation is
an action-selection extension of TD3, not a reproduction of TD-MPC.
