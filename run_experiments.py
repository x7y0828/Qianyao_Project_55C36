import numpy as np
import time

from src.data import make_image
from src.baseline import chambolle
from src.modified import adaptive_chambolle
from src.metrics import mse, psnr, gradient_mse

# final experiment settings
noise_levels = [0.05, 0.1, 0.2]
test_seeds = range(200, 220)

best_lambda = 0.15
best_alpha = 15.0
max_iter = 200

results = []

# paired experiments 实现配对实验
for sigma in noise_levels:
    for seed in test_seeds:
        clean, noisy = make_image(seed=seed, sigma=sigma)

        # Baseline
        denoised_baseline = chambolle(
            noisy, lam=best_lambda, max_iter=max_iter
        )

        # Adaptive
        denoised_adaptive, lam = adaptive_chambolle(
            noisy, alpha=best_alpha, max_iter=max_iter
        )

        # Record evaluation metrics
        results.append({
            "seed": seed,
            "sigma": sigma,
            "baseline_lambda": best_lambda,
            "adaptive_lambda": lam,

            "noisy_mse": mse(noisy, clean),
            "baseline_mse": mse(denoised_baseline, clean),
            "adaptive_mse": mse(denoised_adaptive, clean),

            "noisy_psnr": psnr(noisy, clean),
            "baseline_psnr": psnr(denoised_baseline, clean),
            "adaptive_psnr": psnr(denoised_adaptive, clean),

            "noisy_gradient_mse": gradient_mse(noisy, clean),
            "baseline_gradient_mse": gradient_mse(
                denoised_baseline, clean
            ),
            "adaptive_gradient_mse": gradient_mse(
                denoised_adaptive, clean
            )
        })

    print("Completed noise level:", sigma)

print("Total paired cases:", len(results))