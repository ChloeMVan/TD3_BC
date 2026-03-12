import numpy as np
import matplotlib.pyplot as plt

# Load scores (same eval_freq you used when running the experiment)
scores = np.load("results/TD3_BC_hopper-random-v0_0.npy")
eval_freq = 100  # or 5000, etc. — must match the run

steps = (np.arange(len(scores)) + 1) * eval_freq  # step at each evaluation

plt.figure(figsize=(8, 5))
plt.plot(steps, scores)
plt.xlabel("Step")
plt.ylabel("D4RL score")
plt.title("D4RL score vs step")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/score_vs_step.png")  # optional
plt.show()