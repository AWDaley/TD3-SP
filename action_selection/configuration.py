"""Construct optional TD3 selectors without changing the learning algorithm."""

from env_compat import continuous_dimensions
from state_prediction.model import StatePredictor

from .selector import ActionSelector
from .tasks import task_for_env


def load_prediction_model(checkpoint, policy, env):
    """Load a fitted predictor on the actor's device and check feature sizes.

    Matching dimensions cannot prove that a checkpoint describes this task.
    The caller must use a model trained for the same observations, actions,
    timestep, and environment dynamics.
    """
    if not checkpoint:
        raise ValueError("Lookahead requires --prediction_checkpoint")
    state_dim, action_dim = continuous_dimensions(env)
    device = next(policy.actor.parameters()).device
    model = StatePredictor.load(checkpoint, device=device)
    if (model.state_dim, model.action_dim) != (state_dim, action_dim):
        raise ValueError("Predictor dimensions do not match the environment")
    if not model.normalizers_fitted:
        raise ValueError("Lookahead requires a fitted predictor; train the model first")
    return model


def selector_for_env(policy, env, config, predictor=None):
    """Create the same controller for training and evaluation environments.

    The CLI uses explicitly supported task adapters. For another task, pass
    your own reward/true-termination callable directly to ``ActionSelector``.
    """
    state_dim, action_dim = continuous_dimensions(env)
    task = None
    if config.mode == "lookahead":
        task = task_for_env(env)
        if predictor is None:
            raise ValueError("Lookahead requires a fitted state predictor")
        if (predictor.state_dim, predictor.action_dim) != (state_dim, action_dim):
            raise ValueError("Predictor dimensions do not match the environment")
        if not predictor.normalizers_fitted:
            raise ValueError("Lookahead requires a fitted predictor; train the model first")
    return ActionSelector(
        policy, env.action_space.low, env.action_space.high, config,
        predictor=predictor, task=task,
    )
