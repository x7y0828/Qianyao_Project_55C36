import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# Load statistical results
with open("results/summary_results.csv",
          "r", encoding="utf-8") as f:
    summary = list(csv.DictReader(f))

print("Number of summary records:", len(summary))

def plot_metric(metric, ylabel, title):
    rows = [r for r in summary if r["metric"] == metric]

    sigma = [float(r["sigma"]) for r in rows]

    baseline_mean = [float(r["baseline_mean"]) for r in rows]
    baseline_std = [float(r["baseline_std"]) for r in rows]

    adaptive_mean = [float(r["adaptive_mean"]) for r in rows]
    adaptive_std = [float(r["adaptive_std"]) for r in rows]

    x = np.arange(len(sigma))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.bar(x - width/2, baseline_mean, width,
           yerr=baseline_std, capsize=4, label="Baseline")

    ax.bar(x + width/2, adaptive_mean, width,
           yerr=adaptive_std, capsize=4, label="Adaptive")

    ax.set_xticks(x)
    ax.set_xticklabels(sigma)
    ax.set_xlabel("Noise standard deviation (sigma)")
    ax.set_ylabel(ylabel)
    ax.set_title(title + " (Mean ± SD)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    fig.savefig(output_dir / f"{metric}_comparison.png",
                dpi=300, bbox_inches="tight")

plot_metric("mse", "MSE", "MSE Comparison")

plot_metric("psnr", "PSNR (dB)", "PSNR Comparison")

plot_metric("gradient_mse", "Gradient MSE",
            "Gradient MSE Comparison")

plot_metric("runtime", "Runtime (s)", "Runtime Comparison")

plt.show()