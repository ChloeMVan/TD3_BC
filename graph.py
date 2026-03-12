import sys
import numpy as np
import matplotlib.pyplot as plt

# File from command line: python graph.py results/TD3_BC_hopper-random-v0_0.npy
filename = sys.argv[1] if len(sys.argv) > 1 else input("Path to .npy file: ").strip()

# Load scores (same eval_freq you used when running the experiment)
scores = np.load(filename)
eval_freq = 5000 # or 5000, etc. — must match the run

steps = (np.arange(len(scores)) + 1) * eval_freq  # step at each evaluation

# Print data to terminal
print(f"\nData from {filename}")
print("-" * 30)
print(f"{'Step':>10}  {'D4RL score':>12}")
print("-" * 30)
for step, score in zip(steps, scores):
	print(f"{step:>10}  {score:>12.3f}")
print("-" * 30)
print(f"Total evaluations: {len(scores)}\n")

plt.figure(figsize=(8, 5))
plt.plot(steps, scores)
plt.xlabel("Step")
plt.ylabel("D4RL score")
plt.title("D4RL score vs step")
plt.grid(True, alpha=0.3)
plt.tight_layout()
out_path = filename.replace(".npy", "_plot.png")  # save next to file, or use "results/score_vs_step.png"
plt.savefig(out_path)
plt.show()