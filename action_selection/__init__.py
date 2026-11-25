"""Optional action selection for TD3, with actor and critic-only baselines."""

from .selector import ActionSelector, SelectionConfig

__all__ = ["ActionSelector", "SelectionConfig"]
