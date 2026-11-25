"""Action-conditioned next-state prediction, training and held-out evaluation.

The predictor trains separately from TD3. Optional ``action_selection``
lookahead uses its forecasts to select actions; TD3 critic targets stay unchanged.
Run ``python -m state_prediction --help`` for the prediction workflow.
"""

from .data import TransitionDataset

__all__ = ["TransitionDataset"]
