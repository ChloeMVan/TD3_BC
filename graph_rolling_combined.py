import os
import sys
import numpy as np
import matplotlib.pyplot as plt

OUTPUT_DIR = "results/graphs"

# Usage: python graph_rolling_combined.py file1.npy file2.npy file3.npy [--window 10] [--eval_freq 5000]
if len(sys.argv) < 4:
    print("Usage: python graph_rolling_combined.py file1.npy file2.npy file3.npy [--window 10] [--eval_freq 5000]")
    sys.exit(1)

filenames = sys.argv[1:4]

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

ENV_COLORS = {
    "cheetah": ("#1f77b4", "#aec7e8"),  # blue, pale blue
    "hopper":  ("#d62728", "#f7a8a8"),  # red, pale red
    "walker":  ("#2ca02c", "#98df8a"),  # green, pale green
}

def get_env_style(filename):
    base = os.path.basename(filename).lower()
    for env, colors in ENV_COLORS.items():
        if env in base:
            return env.capitalize(), colors
    return os.path.basename(filename).replace(".npy", ""), ("#888888", "#cccccc")

plt.figure(figsize=(8, 5))

for filename in filenames:
    scores = np.load(filename)
    steps = (np.arange(len(scores)) + 1) * eval_freq

    effective_window = min(window, len(scores))
    rolling = np.convolve(scores, np.ones(effective_window) / effective_window, mode="valid")
    steps_rolling = steps[effective_window - 1:]

    label, (solid_color, pale_color) = get_env_style(filename)
    plt.plot(steps, scores, color=pale_color, alpha=0.6)
    plt.plot(steps_rolling, rolling, color=solid_color, linewidth=2, label=label)

plt.axvline(x=500_000, color="red", linestyle="--", linewidth=1.5, label="Offline → Online")
plt.xlabel("Step")
plt.ylabel("D4RL score")
plt.title("Medium-Expert Environment D4RL Scores vs Timesteps")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

os.makedirs(OUTPUT_DIR, exist_ok=True)
out_path = os.path.join(OUTPUT_DIR, "combined_rolling_plot.png")
plt.savefig(out_path)
plt.show()
