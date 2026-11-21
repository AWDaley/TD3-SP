import numpy as np
import torch
import argparse
import os

from env_compat import make_env, reset_env, step_env, continuous_dimensions

import utils
import TD3
import OurDDPG
import DDPG


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10):
	eval_env = make_env(env_name)

	avg_reward = 0.
	try:
		for episode in range(eval_episodes):
			state = reset_env(eval_env, seed + 100 if episode == 0 else None)
			done = False
			while not done:
				action = policy.select_action(np.array(state))
				state, reward, terminated, truncated, _ = step_env(eval_env, action)
				done = terminated or truncated
				avg_reward += reward
	finally:
		eval_env.close()

	avg_reward /= eval_episodes

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("---------------------------------------")
	return avg_reward


if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--policy", default="TD3", choices=("TD3", "DDPG", "OurDDPG"))
	parser.add_argument("--env", default="HalfCheetah-v2")          # OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--start_timesteps", default=25000, type=int)# Time steps initial random policy is used
	parser.add_argument("--eval_freq", default=5000, type=int)      # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=1000000, type=int)# Max time steps to run environment
	parser.add_argument("--expl_noise", default=0.1, type=float)    # Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99, type=float)     # Discount factor
	parser.add_argument("--tau", default=0.005, type=float)         # Target network update rate
	parser.add_argument("--policy_noise", default=0.2, type=float)  # Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.5, type=float)    # Range to clip target policy noise
	parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
	parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
	parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
	parser.add_argument("--eval_episodes", default=10, type=int)
	parser.add_argument("--replay_size", default=1000000, type=int)
	parser.add_argument("--threads", default=None, type=int, help="Optional CPU thread limit")
	parser.add_argument("--prediction_data", default="", help="Save real transitions to this new .npz file")
	parser.add_argument("--prediction_capacity", default=100000, type=int,
		help="Maximum newest transitions retained in the prediction archive")
	args = parser.parse_args()
	for name in ("max_timesteps", "eval_freq", "eval_episodes", "batch_size", "replay_size", "prediction_capacity", "policy_freq"):
		if getattr(args, name) < 1: parser.error(f"--{name} must be positive")
	if args.start_timesteps < 0: parser.error("--start_timesteps cannot be negative")
	if args.threads is not None:
		if args.threads < 1: parser.error("--threads must be positive")
		torch.set_num_threads(args.threads)
	if args.prediction_data and os.path.exists(args.prediction_data):
		parser.error("Prediction archive already exists; choose a new --prediction_data path")

	file_name = f"{args.policy}_{args.env}_{args.seed}"
	print("---------------------------------------")
	print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
	print("---------------------------------------")

	if not os.path.exists("./results"):
		os.makedirs("./results")

	if args.save_model and not os.path.exists("./models"):
		os.makedirs("./models")

	env = make_env(args.env)

	# Set seeds
	env.action_space.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	
	state_dim, action_dim = continuous_dimensions(env, require_symmetric_actions=True)
	max_action = float(env.action_space.high[0])

	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": args.discount,
		"tau": args.tau,
	}

	# Initialize policy
	if args.policy == "TD3":
		# Target policy smoothing is scaled wrt the action scale
		kwargs["policy_noise"] = args.policy_noise * max_action
		kwargs["noise_clip"] = args.noise_clip * max_action
		kwargs["policy_freq"] = args.policy_freq
		policy = TD3.TD3(**kwargs)
	elif args.policy == "OurDDPG":
		policy = OurDDPG.DDPG(**kwargs)
	elif args.policy == "DDPG":
		policy = DDPG.DDPG(**kwargs)

	if args.load_model != "":
		policy_file = file_name if args.load_model == "default" else args.load_model
		policy.load(f"./models/{policy_file}")

	replay_buffer = utils.ReplayBuffer(state_dim, action_dim, max_size=args.replay_size)
	prediction_recorder = None
	if args.prediction_data:
		from state_prediction.collection import TransitionRecorder
		prediction_recorder = TransitionRecorder(state_dim, action_dim, args.prediction_capacity)
	
	# Evaluate untrained policy
	evaluations = [eval_policy(policy, args.env, args.seed, args.eval_episodes)]

	state, done = reset_env(env, args.seed), False
	episode_reward = 0
	episode_timesteps = 0
	episode_num = 0

	for t in range(int(args.max_timesteps)):
		
		episode_timesteps += 1

		# Select action randomly or according to policy
		if t < args.start_timesteps:
			action = env.action_space.sample()
		else:
			action = (
				policy.select_action(np.array(state))
				+ np.random.normal(0, max_action * args.expl_noise, size=action_dim)
			).clip(-max_action, max_action)

		# Perform action
		next_state, reward, terminated, truncated, _ = step_env(env, action)
		done = terminated or truncated
		# Time limits end a trajectory but TD3 should still bootstrap through them.
		done_bool = float(terminated)

		# Store data in replay buffer
		replay_buffer.add(state, action, next_state, reward, done_bool)
		if prediction_recorder is not None:
			collection_cut = t + 1 == args.max_timesteps and not terminated
			prediction_recorder.add(state, action, next_state, episode_num, terminated, truncated or collection_cut)

		state = next_state
		episode_reward += reward

		# Train agent after collecting sufficient data
		if t >= args.start_timesteps:
			policy.train(replay_buffer, args.batch_size)

		if done: 
			# +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
			print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
			# Reset environment
			state, done = reset_env(env), False
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1 

		# Evaluate episode
		if (t + 1) % args.eval_freq == 0:
			evaluations.append(eval_policy(policy, args.env, args.seed, args.eval_episodes))
			np.save(f"./results/{file_name}", evaluations)
			if args.save_model: policy.save(f"./models/{file_name}")

	if prediction_recorder is not None:
		dataset = prediction_recorder.dataset()
		dataset.save(args.prediction_data)
		print(f"Saved {len(dataset)} transitions from {dataset.episode_count} episodes to {args.prediction_data}")
		if prediction_recorder.count > args.prediction_capacity:
			print(f"Archive capacity reached: retained the newest {args.prediction_capacity} transitions.")
	np.save(f"./results/{file_name}", evaluations)
	if args.save_model: policy.save(f"./models/{file_name}")
	env.close()
