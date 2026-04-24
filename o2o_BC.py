import copy
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


# Runs policy for X episodes and returns D4RL score
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, mean, std, seed_offset=100, eval_episodes=5):
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
	parser.add_argument("--env", default="hopper-expert-v0")        # OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=500_000, type=int)   # Max time steps to run environment
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
	# TD3 + BC
	parser.add_argument("--alpha", default=2.5)
	parser.add_argument("--normalize", default=True)
	parser.add_argument("--param_num", default=0, type=float)
	parser.add_argument("--file_tag")
	parser.add_argument("--function_type")
	parser.add_argument("--plateau_threshold", default=0.5, type=float)  # Min D4RL improvement over window to avoid plateau
	args = parser.parse_args()

	file_name = f"{args.file_tag}_{args.policy}_{args.env}_{args.seed}"
	print("---------------------------------------")
	print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
	print("---------------------------------------")

	if not os.path.exists("./results"):
		os.makedirs("./results")

	if not os.path.exists("./lossresults"):
		os.makedirs("./lossresults")
	
	if not os.path.exists("./q_results"):
		os.makedirs("./q_results")

	if args.save_model and not os.path.exists("./models"):
		os.makedirs("./models")

	env = gym.make(args.env)

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
		"alpha": args.alpha,
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
	# Plateau detection: rolling window of 100k steps worth of evals.
	# Once the window is full, compare mean of second half vs first half.
	# If improvement < plateau_threshold, switch to online early.
	plateau_window_steps = 100_000
	plateau_window_evals = max(2, plateau_window_steps // int(args.eval_freq))  # number of evals in window
	half = plateau_window_evals // 2

	print("STARTING OFFLINE TRAINING")
	evaluations = []
	actorlosses = []
	criticlosses = []
	target_Q1s = []
	target_Q2s = []
	target_Qs = []
	current_Q1s = []
	current_Q2s = []
	offline_steps_taken = 0
	for t in range(int(args.max_timesteps)):
		critic_loss, actor_loss, target_Q1, target_Q2, target_Q, current_Q1, current_Q2 = policy.train(replay_buffer, args.batch_size)
		if actor_loss is not None:
			# actorlosses.append(actor_loss)
			actorlosses.append(actor_loss.detach().cpu().numpy())
		elif actorlosses:
			actorlosses.append(actorlosses[-1])
		else:
			actorlosses.append(0)

		# criticlosses.append(critic_loss)
		criticlosses.append(critic_loss.detach().cpu().numpy())
		target_Q1s.append(target_Q1.detach().cpu().numpy())
		target_Q2s.append(target_Q2.detach().cpu().numpy())
		target_Qs.append(target_Q.detach().cpu().numpy())
		current_Q1s.append(current_Q1.detach().cpu().numpy())
		current_Q2s.append(current_Q2.detach().cpu().numpy())
		np.save(f"./lossresults/{file_name}_actor", actorlosses)
		np.save(f"./lossresults/{file_name}_critic", criticlosses)
		np.save(f"./q_results/{file_name}_target_Q1", target_Q1s)
		np.save(f"./q_results/{file_name}_target_Q2", target_Q2s)
		np.save(f"./q_results/{file_name}_target_Qs", target_Qs)
		np.save(f"./q_results/{file_name}_current_Q1", current_Q1s)
		np.save(f"./q_results/{file_name}_current_Q2", current_Q2s)

		offline_steps_taken = t + 1
		# Evaluate episode
		if (t + 1) % args.eval_freq == 0:
			print(f"[Offline] Time steps: {t+1}")
			evaluations.append(eval_policy(policy, args.env, args.seed, mean, std))
			np.save(f"./results/{file_name}", evaluations)
			if args.save_model:
				policy.save(f"./models/{file_name}")

			# Check for plateau once we have a full rolling window
			# if len(evaluations) >= plateau_window_evals:
			# 	window = evaluations[-plateau_window_evals:]
			# 	first_half_mean = np.mean(window[:half])
			# 	second_half_mean = np.mean(window[half:])
			# 	improvement = second_half_mean - first_half_mean
			# 	print(f"[Plateau check] Window improvement: {improvement:.3f} (threshold: {args.plateau_threshold})")
			# 	if improvement < args.plateau_threshold:
			# 		print(f"[Offline] Plateau detected at step {t+1}, switching to online training early.")
			# 		break
	
	# ------------------------- online ------------------------------------
	# print(f"STARTING ONLINE TRAINING (offline ran for {offline_steps_taken} / {args.max_timesteps} steps)")
	print(f"STARTING ONLINE TRAINING")

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
	total_online_steps = int(args.max_timesteps)
	state, done = env.reset(), False
	state = (np.array(state).reshape(1,-1) - mean)/std
    
	# td3policy = TD3.TD3(**kwargs) # initialize td3 policy

	# td3policy.actor.load_state_dict(policy.actor.state_dict())
	# td3policy.critic.load_state_dict(policy.critic.state_dict())

	# td3policy.actor_target.load_state_dict(policy.actor_target.state_dict())
	# td3policy.critic_target.load_state_dict(policy.critic_target.state_dict())
	
    # we use policy instead from offline
	param_num = args.param_num
	function_type = args.function_type

	for t in range(total_online_steps):
		# print((pow(t, 1.0/param_num)), param_num)
		if function_type == 'exp':
			beta = max(0.0, 1.0/(pow(t+1, 1.0/param_num)))  # linear decay from 1 to 0
		elif function_type == "linear":
			beta = max(0.0, (-1/249999) * t + (250000/249999))
		elif function_type == "log":
			beta = max(0.0,  np.log(500001 - t) / np.log(500001 - 1))
		else:
			print("functype unrecognized", function_type)
			print(1/0)


		# Select action
		action = policy.select_action(state)
		
		action = (
			action
			+ np.random.normal(0, max_action * args.expl_noise, size=action_dim)
		).clip(-max_action, max_action)

		next_state, reward, done, _ = env.step(action)
		next_state_norm = (np.array(next_state).reshape(1,-1) - mean)/std

		# Store new transition
		replay_buffer.add(state, action, next_state_norm, reward, done)

		# Train policy only once we have enough transitions
		if replay_buffer.size >= args.batch_size:
			critic_loss, actor_loss, target_Q1, target_Q2, target_Q, current_Q1, current_Q2 = policy.train(replay_buffer, args.batch_size, beta=beta)
			# criticlosses.append(critic_loss)
			criticlosses.append(critic_loss.detach().cpu().numpy())
			if actor_loss is not None:
				# actorlosses.append(actor_loss)
				actorlosses.append(actor_loss.detach().cpu().numpy())
			elif actorlosses:
				actorlosses.append(actorlosses[-1])
			else:
				actorlosses.append(0)

			target_Q1s.append(target_Q1.detach().cpu().numpy())
			target_Q2s.append(target_Q2.detach().cpu().numpy())
			target_Qs.append(target_Q.detach().cpu().numpy())
			current_Q1s.append(current_Q1.detach().cpu().numpy())
			current_Q2s.append(current_Q2.detach().cpu().numpy())
			np.save(f"./q_results/{file_name}_target_Q1", target_Q1s)
			np.save(f"./q_results/{file_name}_target_Q2", target_Q2s)
			np.save(f"./q_results/{file_name}_target_Qs", target_Qs)
			np.save(f"./q_results/{file_name}_current_Q1", current_Q1s)
			np.save(f"./q_results/{file_name}_current_Q2", current_Q2s)
			np.save(f"./lossresults/{file_name}_actor", actorlosses)
			np.save(f"./lossresults/{file_name}_critic", criticlosses)
			

		state = next_state_norm

		if done:
			state, done = env.reset(), False
			state = (np.array(state).reshape(1,-1) - mean)/std

		# Evaluation
		if (t + 1) % args.eval_freq == 0:
			print(f"[Online] Time steps: {t+1}")
			evaluations.append(eval_policy(policy, args.env, args.seed, mean, std))
			np.save(f"./results/{file_name}", evaluations)

			if args.save_model:
				policy.save(f"./models/{file_name}_online")