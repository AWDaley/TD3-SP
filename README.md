# TD3-SP: TD3 with state prediction

TD3-SP adds an action-conditioned state prediction model to the original TD3
implementation. It learns the change in observation from a real transition:

`(current state, applied action) -> predicted next state`

The predictor trains as a separate experiment. Optional **short lookahead action
selection** now uses it to compare nearby actions before executing one. Ordinary
TD3 remains the default; its critic targets and training losses are unchanged.

## What is included

- A normalized residual neural network with save/load and recursive forecasts.
- Transition recording from the TD3 training loop or a standalone environment.
- Separate **training, validation, and test episodes**, including correct handling
  of terminal states and time limits.
- One-step and multi-step RMSE, MAE, bias, and errors by state dimension.
- Persistence and fitted linear dynamics baselines.
- Early stopping and best-validation checkpoint selection.
- An `improve` command with an optional multi-step training loss. A candidate is
  accepted only if its validation score improves; the original file is preserved.
- Readable Markdown, JSON, and training-history CSV reports.
- Optional actor-only, critic-only search, and model lookahead controllers.
- Batched action candidates, actor feedback during short rollouts, and correct
  reward/termination handling with a built-in Pendulum adapter.
- Frozen-controller comparisons with paired seeds, observed returns, latency,
  interventions, fallback counts, and checkpoint identities.

## Install

Use Python 3.10 or newer. From this repository:

```bash
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` on Windows PowerShell, or
`source .venv/bin/activate` on Linux/macOS. Then install:

```bash
python -m pip install -r requirements.txt
```

For a CPU-only PyTorch installation, install it first with
`python -m pip install "torch>=2.5,<3" --index-url https://download.pytorch.org/whl/cpu`,
then install the requirements. MuJoCo is optional; neither the demo nor Pendulum
requires it.

## Try the complete workflow

```bash
python -m state_prediction demo --output runs/demo
python -m state_prediction evaluate --data runs/demo/data.npz --checkpoint runs/demo/model.pt
python -m state_prediction improve --data runs/demo/data.npz --checkpoint runs/demo/model.pt --output runs/demo-improved
```

The demo uses a small synthetic point-mass navigation system with velocity,
nonlinear drag, and a current term. **It is not a validated AUV simulator.**
Each output directory contains `model.pt`, `report.md`, `report.json`, and
`history.csv`. The demo also saves `data.npz`. Use a new output directory for each
run so previous results remain available.

The improvement command may report `kept_incumbent`. That means its extra
training did not beat the existing model on the frozen validation set.

## Use an actual Gymnasium environment

```bash
python -m state_prediction collect --env Pendulum-v1 --episodes 30 --output datasets/pendulum.npz
python -m state_prediction train --data datasets/pendulum.npz --output runs/pendulum --state-names cos_theta sin_theta angular_velocity
```

You can also record transitions while TD3 learns:

```bash
python main.py --env Pendulum-v1 --max_timesteps 10000 --start_timesteps 1000 --prediction_data datasets/td3-pendulum.npz --save_model
python -m state_prediction train --data datasets/td3-pendulum.npz --output runs/td3-predictor
```

`--prediction_capacity` limits the archive to the newest 100,000 transitions by
default. TD3's replay buffer is separate. Evaluation episodes are not added to
the prediction archive. At least three recorded episodes are required to train;
more episodes give a more useful assessment.

To collect from a saved actor without retraining TD3:

```bash
python -m state_prediction collect --env Pendulum-v1 --episodes 30 --policy-checkpoint models/TD3_Pendulum-v1_0 --output datasets/actor-pendulum.npz
```

See **[the state prediction guide](docs/STATE_PREDICTION.md)** for the architecture,
accuracy metrics, improvement procedure, and integration with your own AUV data.
See **[verified example results](docs/STATE_PREDICTION_RESULTS.md)** for measured runs.

## Use predictions to select actions

After training a predictor for Pendulum, enable lookahead explicitly:

```bash
python main.py --env Pendulum-v1 --action_selection lookahead --prediction_checkpoint runs/td3-predictor/model.pt --planning_candidates 16 --planning_horizon 1 --max_timesteps 10000 --start_timesteps 1000 --eval_episodes 5 --threads 1 --save_model
```

The actor's original action is one candidate. Other candidates are nearby bounded
actions. Each is scored using predicted immediate rewards plus the lower of the
two critics' estimates at the end of the rollout. Only the first action executes;
the next decision starts from the real observation. The predictor stays fixed
during this training run. Exploration is applied before scoring, with no noise
added afterward. `--action_selection critic` runs candidate search without a model.

Compare the same saved actor and critics in all three modes:

```bash
python -m action_selection compare --env Pendulum-v1 --policy_checkpoint models/TD3_Pendulum-v1_0 --prediction_checkpoint runs/td3-predictor/model.pt --episodes 10 --seed 123 --candidates 16 --horizon 1 --threads 1 --output runs/frozen-comparison.json
```

The built-in task adapter supports standard Pendulum. Other environments need an
explicit reward/true-termination callable; the CLI rejects unsupported tasks.
This is an experimental controller, without calibrated uncertainty or vehicle
safety guarantees. See **[the lookahead guide](docs/LOOKAHEAD.md)** for formulas,
training/evaluation commands, custom-task integration, and measured smoke results.

**[Extended validation results](docs/LOOKAHEAD_EXTENSIVE_RESULTS.md)** now cover
five training seeds per mode, 300,000 new training interactions, 380,000 evaluation
interactions, and 125 software tests. They show an early benefit from planning
on the tested Pendulum checkpoints, without establishing a final-return advantage
after 20,000 training steps. The report separates the predeclared final endpoints
from a secondary early-checkpoint follow-up and includes forecast failure cases.

## Tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

## Original TD3 implementation and attribution

The sections below describe the upstream research implementation and historical
results. Its original Python/PyTorch/Gym versions are historical, not the current
requirements above. The runner accepts both Gymnasium and the legacy Gym API;
historical `*-v2` MuJoCo IDs require their corresponding legacy environment.

### Addressing Function Approximation Error in Actor-Critic Methods

PyTorch implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3). If you use our code or data please cite the [paper](https://arxiv.org/abs/1802.09477).

Method is tested on [MuJoCo](http://www.mujoco.org/) continuous control tasks in [OpenAI gym](https://github.com/openai/gym). 
Networks are trained using [PyTorch 1.2](https://github.com/pytorch/pytorch) and Python 3.7. 

### Usage
The paper results can be reproduced by running:
```
./run_experiments.sh
```
Experiments on single environments can be run by calling:
```
python main.py --env HalfCheetah-v2
```

Hyper-parameters can be modified with different arguments to main.py. We include an implementation of DDPG (DDPG.py), which is not used in the paper, for easy comparison of hyper-parameters with TD3. This is not the implementation of "Our DDPG" as used in the paper (see OurDDPG.py). 

Algorithms which TD3 compares against (PPO, TRPO, ACKTR, DDPG) can be found at [OpenAI baselines repository](https://github.com/openai/baselines). 

### Results
Code is no longer exactly representative of the code used in the paper. Minor adjustments to hyperparamters, etc, to improve performance. Learning curves are still the original results found in the paper.

Learning curves found in the paper are found under /learning_curves. Each learning curve are formatted as NumPy arrays of 201 evaluations (201,), where each evaluation corresponds to the average total reward from running the policy for 10 episodes with no exploration. The first evaluation is the randomly initialized policy network (unused in the paper). Evaluations are peformed every 5000 time steps, over a total of 1 million time steps. 

Numerical results can be found in the paper, or from the learning curves. Video of the learned agent can be found [here](https://youtu.be/x33Vw-6vzso). 

### Bibtex

```
@inproceedings{fujimoto2018addressing,
  title={Addressing Function Approximation Error in Actor-Critic Methods},
  author={Fujimoto, Scott and Hoof, Herke and Meger, David},
  booktitle={International Conference on Machine Learning},
  pages={1582--1591},
  year={2018}
}
```
