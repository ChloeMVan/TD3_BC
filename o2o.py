import numpy as np
import torch
import gym
import argparse
import os
import d4rl
import d4rl.locomotion

import utils
import TD3_BC
import TD3


def eval_online_policy(policy, env_name, seed, eval_episodes=10):
	eval_env = gym.make(env_name)
	eval_env.seed(seed + 100)

	avg_reward = 0.
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		while not done:
			action = policy.select_action(np.array(state))
			state, reward, done, _ = eval_env.step(action)
			avg_reward += reward

	avg_reward /= eval_episodes

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("---------------------------------------")
	return avg_reward


# Runs policy for X episodes and returns D4RL score
# A fixed seed is used for the eval environment
def eval_offline_policy(policy, env_name, seed, mean, std, seed_offset=100, eval_episodes=10):
	eval_env = gym.make(env_name)
	eval_env.seed(seed + seed_offset)

	avg_reward = 0.
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		while not done:
			state = (np.array(state).reshape(1,-1) - mean)/std
			action = policy.select_action(state)
			state, reward, done, _ = eval_env.step(action)
			avg_reward += reward

	avg_reward /= eval_episodes
	d4rl_score = eval_env.get_normalized_score(avg_reward) * 100

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}, D4RL score: {d4rl_score:.3f}")
	print("---------------------------------------")
	return d4rl_score


if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	# Experiment
	parser.add_argument("--policy", default="TD3_BC")               # Policy name
	parser.add_argument("--offline_env", default="hopper-medium-v0")        # OpenAI gym environment name
	parser.add_argument("--online_env", default="Hopper-v3")        # online environemnt 
	parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=15000, type=int)   # Max time steps to run environment
	parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
	parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
	# TD3
	parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99)                 # Discount factor
	parser.add_argument("--tau", default=0.005)                     # Target network update rate
	parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
	parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
	parser.add_argument("--start_timesteps", default=0, type=int)# Time steps initial random policy is used
	# TD3 + BC
	parser.add_argument("--alpha", default=2.5)
	parser.add_argument("--normalize", default=True)
	args = parser.parse_args()

	file_name = f"{args.policy}_{args.offline_env}_{args.seed}"
	print("---------------------------------------")
	print(f"Policy: {args.policy}, Env: {args.offline_env}, Seed: {args.seed}")
	print("---------------------------------------")

	if not os.path.exists("./results"):
		os.makedirs("./results")

	if args.save_model and not os.path.exists("./models"):
		os.makedirs("./models")

	env = gym.make(args.offline_env)

	# Set seeds
	env.seed(args.seed)
	env.action_space.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0] 
	max_action = float(env.action_space.high[0])

	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": args.discount,
		"tau": args.tau,
		# TD3
		"policy_noise": args.policy_noise * max_action,
		"noise_clip": args.noise_clip * max_action,
		"policy_freq": args.policy_freq,
		# TD3 + BC
		"alpha": args.alpha
	}

	# Initialize policy
	policy = TD3_BC.TD3_BC(**kwargs)

	if args.load_model != "":
		policy_file = file_name if args.load_model == "default" else args.load_model
		policy.load(f"./models/{policy_file}")

	replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
	replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))
	if args.normalize:
		mean,std = replay_buffer.normalize_states() 
	else:
		mean,std = 0,1
	
	# ----------------------- offline -----------------------------------
	print("STARTING OFFLINE TRAINING")
	evaluations = []
	for t in range(int(args.max_timesteps)):
		policy.train(replay_buffer, args.batch_size)
		# Evaluate episode
		if (t + 1) % args.eval_freq == 0:
			print(f"[Offline] Time steps: {t+1}")
			evaluations.append(eval_offline_policy(policy, args.offline_env, args.seed, mean, std))
			np.save(f"./results/{file_name}", evaluations)
			if args.save_model: 
				policy.save(f"./models/{file_name}")
	
	# ------------------------- online add here-----------------------------
	print("STARTING ONLINE TRAINING") 

	env = gym.make(args.online_env)
	online_buffer = utils.ReplayBuffer(state_dim, action_dim)
	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": args.discount,
		"tau": args.tau,
		# TD3
		"policy_noise": args.policy_noise * max_action,
		"noise_clip": args.noise_clip * max_action,
		"policy_freq": args.policy_freq,
	}

	td3policy = TD3.TD3(**kwargs) # initialize td3 policy

	td3policy.actor.load_state_dict(policy.actor.state_dict())
	td3policy.critic.load_state_dict(policy.critic.state_dict())

	td3policy.actor_target.load_state_dict(policy.actor_target.state_dict())
	td3policy.critic_target.load_state_dict(policy.critic_target.state_dict())

	evaluations = [eval_online_policy(td3policy, args.online_env, args.seed)]

	state, done = env.reset(), False
	episode_reward = 0
	episode_timesteps = 0
	episode_num = 0

	for t in range(int(args.max_timesteps)):
		
		episode_timesteps += 1

		# Select action according to policy
		if t < args.start_timesteps:
			action = env.action_space.sample()
		else:
			norm_state = (np.array(state).reshape(1,-1) - mean) / std
			action = (
				td3policy.select_action(np.array(state))
				+ np.random.normal(0, max_action * args.expl_noise, size=action_dim)
			).clip(-max_action, max_action)

		# Perform action
		next_state, reward, done, _ = env.step(action) 
		done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

		# Store data in replay buffer
		norm_next_state = (np.array(next_state).reshape(1,-1) - mean) / std
		online_buffer.add(state, action, next_state, reward, done_bool)

		state = next_state
		episode_reward += reward

		# Train agent 
		if t >= args.start_timesteps:
			td3policy.train(online_buffer, args.batch_size)

		if done: 
			# +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
			print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
			# Reset environment
			state, done = env.reset(), False
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1 

		# Evaluate episode
		if (t + 1) % args.eval_freq == 0:
			evaluations.append(eval_online_policy(td3policy, args.online_env, args.seed))
			np.save(f"./results/{file_name}", evaluations)
			if args.save_model: td3policy.save(f"./models/{file_name}")