import os
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = "results"
OUTPUT_DIR = "results/graphs"

def compute_stability_scores(results_dir):
    stability_scores = {}
    for file in os.listdir(results_dir):
        if file.endswith('.npy'):
            filepath = os.path.join(results_dir, file)
            try:
                scores = np.load(filepath)
                if len(scores) > 0:
                    mean_score = np.mean(scores)
                    std_score = np.std(scores)
                    # Coefficient of variation as stability metric (lower is more stable)
                    stability = np.var(np.diff(scores[len(scores)//2:]))
                    stability_scores[file] = {
                        'mean': mean_score,
                        'std': std_score,
                        'cv': stability,
                        'num_evals': len(scores)
                    }
                else:
                    print(f"Warning: {file} is empty")
            except Exception as e:
                print(f"Error loading {file}: {e}")
    return stability_scores

def plot_stability_bar(scores, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    files = list(scores.keys())
    cvs = [scores[f]['cv'] for f in files]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(files)), cvs, color='skyblue')
    plt.xticks(range(len(files)), [f.replace('.npy', '').replace('_', '\n') for f in files], rotation=45, ha='right')
    plt.ylabel('Stability Score (Coefficient of Variation)')
    plt.title('Learning Stability Scores for Each Run')
    plt.tight_layout()
    
    # Add value labels on bars
    for bar, cv in zip(bars, cvs):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{cv:.3f}', ha='center', va='bottom')
    
    plt.savefig(os.path.join(output_dir, 'stability_bar.png'), dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    scores = compute_stability_scores(RESULTS_DIR)
    print("Learning Stability Scores (Coefficient of Variation: std/mean)")
    print("=" * 60)
    for file, metrics in sorted(scores.items()):
        print(f"{file}:")
        print(f"  Mean D4RL Score: {metrics['mean']:.3f}")
        print(f"  Std Dev: {metrics['std']:.3f}")
        print(f"  Stability (CV): {metrics['cv']:.3f}")
        print(f"  Num Evaluations: {metrics['num_evals']}")
        print()
    
    plot_stability_bar(scores, OUTPUT_DIR)