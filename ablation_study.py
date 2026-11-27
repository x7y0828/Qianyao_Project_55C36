import numpy as np

from src.data import make_image
from src.baseline import chambolle
from src.modified import adaptive_chambolle
from src.metrics import mse, psnr, gradient_mse

import csv
from pathlib import Path

# 不再使用 MAD 估计 σ，而是直接使用已知的真实 σ
def oracle_chambolle(noisy, sigma, alpha=15, max_iter=200):
    lam = max(alpha * sigma**2, 1e-6)

    denoised = chambolle(
        noisy, lam=lam, max_iter=max_iter
    )

    return denoised, lam

def main():
    noise_levels = [0.05, 0.1, 0.2]

    ablation_results = []

    for sigma in noise_levels:
        scores = {
            "Baseline": [],
            "Adaptive": [],
            "Oracle": []
        }

        for seed in range(200, 220):
            clean, noisy = make_image(seed=seed, sigma=sigma)

            baseline = chambolle(
                noisy, lam=0.15, max_iter=200
            )

            adaptive, _ = adaptive_chambolle(
                noisy, alpha=15, max_iter=200
            )

            oracle, _ = oracle_chambolle(
                noisy, sigma=sigma, alpha=15, max_iter=200
            )

            scores["Baseline"].append(psnr(clean, baseline))
            scores["Adaptive"].append(psnr(clean, adaptive))
            scores["Oracle"].append(psnr(clean, oracle))

        print("--------------------")
        print("Noise sigma:", sigma)

        for name, values in scores.items():
            print(name, "mean PSNR:", np.mean(values))

        ablation_results.append({
            "sigma": sigma,
            "baseline_psnr": np.mean(scores["Baseline"]),
            "adaptive_psnr": np.mean(scores["Adaptive"]),
            "oracle_psnr": np.mean(scores["Oracle"])
        })

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "ablation_results.csv","w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ablation_results[0].keys())
        writer.writeheader()
        writer.writerows(ablation_results)

if __name__ == "__main__":
    main()