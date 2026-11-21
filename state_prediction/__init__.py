"""Action-conditioned next-state prediction, training and held-out evaluation.

The predictor is an auxiliary model: TD3's actions and critic targets do not use
its predictions. Run ``python -m state_prediction --help`` for the workflow.
"""

from .data import TransitionDataset

__all__ = ["TransitionDataset"]
