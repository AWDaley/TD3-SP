"""Command-line collection, training, evaluation, and validation-gated improvement."""

import argparse
from pathlib import Path


def _runtime(parser):
    parser.add_argument("--device", default="cpu", help="PyTorch device, e.g. cpu or cuda")
    parser.add_argument("--threads", type=int, default=2, help="CPU threads used by PyTorch")


def _training(parser, improvement=False):
    _runtime(parser)
    parser.add_argument("--output", required=True, help="New directory for checkpoint and reports")
    parser.add_argument("--seed", type=int, default=1 if improvement else 0)
    parser.add_argument("--epochs", type=int, default=40 if improvement else 60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=3e-4 if improvement else 1e-3)
    parser.add_argument("--rollout-loss-weight", type=float, default=0.2 if improvement else 0.0,
                        help="Extra differentiable open-loop loss; zero trains only one-step predictions")
    parser.add_argument("--rollout-horizon", type=int, default=5)
    if not improvement:
        parser.add_argument("--hidden", type=int, nargs="+", default=[128, 128])
        parser.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 10])
        parser.add_argument("--max-windows", type=int, default=1000)
        parser.add_argument("--train-fraction", type=float, default=0.7)
        parser.add_argument("--validation-fraction", type=float, default=0.15)
        parser.add_argument("--state-names", nargs="+", default=None)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Run a small synthetic navigation example on CPU")
    _training(demo)
    demo.add_argument("--episodes", type=int, default=40)
    demo.add_argument("--episode-length", type=int, default=80)
    collect = commands.add_parser("collect", help="Record random controls or a saved TD3 actor")
    _runtime(collect)
    collect.add_argument("--env", default="Pendulum-v1")
    collect.add_argument("--episodes", type=int, default=30)
    collect.add_argument("--max-steps-per-episode", type=int, default=1000)
    collect.add_argument("--seed", type=int, default=0)
    collect.add_argument("--output", required=True, help="New .npz data archive")
    collect.add_argument("--policy-checkpoint", default=None, help="TD3 save prefix, without _actor")
    collect.add_argument("--exploration-noise", type=float, default=0.1)
    train = commands.add_parser("train", help="Train with episode-disjoint train/validation/test splits")
    _training(train)
    train.add_argument("--data", required=True)
    improve = commands.add_parser("improve", help="Fine-tune and accept only validation improvements")
    _training(improve, improvement=True)
    improve.add_argument("--data", required=True, help="Original dataset used to train this checkpoint")
    improve.add_argument("--checkpoint", required=True)
    improve.add_argument("--min-improvement", type=float, default=0.0,
                         help="Required absolute decrease in mean validation normalized RMSE")
    evaluate = commands.add_parser("evaluate", help="Evaluate a saved predictor without training")
    _runtime(evaluate)
    evaluate.add_argument("--data", required=True)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--output", default=None, help="Optional JSON report filename")
    evaluate.add_argument("--external", action="store_true",
                          help="Evaluate all of a NEW dataset; you must ensure it is independent")
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive")
    try:
        import torch
        torch.set_num_threads(args.threads)
        from .workflow import train_experiment, improve_experiment, evaluate_experiment

        if args.command == "collect":
            from .collection import collect_environment
            if Path(args.output).exists():
                raise ValueError("Output dataset already exists; choose a new path")
            data = collect_environment(args.env, args.episodes, args.seed, args.max_steps_per_episode,
                                       args.policy_checkpoint, args.exploration_noise, args.device)
            data.save(args.output)
            print(f"Saved {len(data)} transitions from {data.episode_count} episodes to {args.output}")
        elif args.command == "evaluate":
            evaluate_experiment(args.data, args.checkpoint, output=args.output,
                                 device=args.device, external=args.external)
        elif args.command == "improve":
            improve_experiment(args.data, args.checkpoint, args.output, device=args.device,
                                epochs=args.epochs, batch_size=args.batch_size, patience=args.patience,
                                seed=args.seed, learning_rate=args.learning_rate,
                                rollout_loss_weight=args.rollout_loss_weight,
                                rollout_horizon=args.rollout_horizon, min_improvement=args.min_improvement)
        else:
            if args.command == "demo":
                from .toy import make_toy_dataset
                args.data = str(Path(args.output) / "data.npz")
                if Path(args.data).exists():
                    raise ValueError("Demo data already exists; choose a new output directory")
                data = make_toy_dataset(args.episodes, args.episode_length, args.seed)
                data.save(args.data)
                if args.state_names is None:
                    args.state_names = ["x", "y", "vx", "vy", "current_x", "current_y"]
                print("Synthetic point-mass demonstration; not a validated AUV simulation.")
            train_experiment(args.data, args.output, hidden_dims=args.hidden,
                              learning_rate=args.learning_rate, device=args.device, seed=args.seed,
                              epochs=args.epochs, batch_size=args.batch_size, patience=args.patience,
                              horizons=args.horizons, max_windows=args.max_windows,
                              rollout_loss_weight=args.rollout_loss_weight,
                              rollout_horizon=args.rollout_horizon, train_fraction=args.train_fraction,
                              validation_fraction=args.validation_fraction, state_names=args.state_names)
    except (ValueError, FileNotFoundError, ImportError, FloatingPointError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
