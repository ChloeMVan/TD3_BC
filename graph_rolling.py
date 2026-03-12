import sys
import numpy as np
import matplotlib.pyplot as plt

# File from command line: python graph_rolling.py results/TD3_BC_hopper-random-v0_0.npy [--window 10]
filename = sys.argv[1] if len(sys.argv) > 1 else input("Path to .npy file: ").strip()

# Optional: --window 10, --eval_freq 5000
window = 10
eval_freq = 5000
if "--window" in sys.argv:
	idx = sys.argv.index("--window")
	if idx + 1 < len(sys.argv):
		window = int(sys.argv[idx + 1])
if "--eval_freq" in sys.argv:
	idx = sys.argv.index("--eval_freq")
	if idx + 1 < len(sys.argv):
		eval_freq = int(sys.argv[idx + 1])

# Load scores
scores = np.load(filename)

steps = (np.arange(len(scores)) + 1) * eval_freq  # step at each evaluation

# Rolling average (valid = only where full window is available)
rolling = np.convolve(scores, np.ones(window) / window, mode="valid")
steps_rolling = steps[window - 1 :]  # align: rolling[i] = mean(scores[i:i+window])

# Print data to terminal
print(f"\nData from {filename} (rolling window = {window})")
print("-" * 40)
print(f"{'Step':>12}  {'D4RL score':>12}  {'Rolling avg':>12}")
print("-" * 40)
for i in range(len(scores)):
	step = steps[i]
	score = scores[i]
	roll = rolling[i - window + 1] if i >= window - 1 else np.nan
	roll_str = f"{roll:.3f}" if not np.isnan(roll) else "   —"
	print(f"{step:>12}  {score:>12.3f}  {roll_str:>12}")
print("-" * 40)
print(f"Total evaluations: {len(scores)}\n")

plt.figure(figsize=(8, 5))
plt.plot(steps, scores, alpha=0.4, label="Raw score")
plt.plot(steps_rolling, rolling, linewidth=2, label=f"Rolling avg (window={window})")
plt.xlabel("Step")
plt.ylabel("D4RL score")
plt.title("D4RL score vs step (rolling average)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
out_path = filename.replace(".npy", "_rolling_plot.png")
plt.savefig(out_path)
plt.show()
