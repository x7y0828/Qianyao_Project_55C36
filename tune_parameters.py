import numpy as np
from src.data import make_image
from src.baseline import chambolle
from src.modified import adaptive_chambolle
from src.metrics import psnr

# 261126
# validation settings
noise_levels = [0.05, 0.1, 0.2]
validation_seeds = range(100, 110)

lambda_candidates = [0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.4]
alpha_candidates = [2.5, 5, 7.5, 10, 15, 20, 40]

max_iter = 200

# baseline parameter tuning
baseline_results = {}

for lam in lambda_candidates:           # λ-7个候选值
    scores = []

    for sigma in noise_levels:          # σ-3个噪声水平
        for seed in validation_seeds:   # seed-10个随机种子
            clean, noisy = make_image(seed=seed, sigma=sigma)

            denoised = chambolle(
                noisy, lam=lam, max_iter=max_iter
            )

            scores.append(psnr(denoised, clean))

    baseline_results[lam] = np.mean(scores)

    print("Lambda:", lam,
          "Mean PSNR:", baseline_results[lam])

best_lambda = max(baseline_results, key=baseline_results.get)

print("Best lambda:", best_lambda)
print("Best mean PSNR:", baseline_results[best_lambda])

# adaptive parameter tuning
adaptive_results = {}

for alpha in alpha_candidates:
    scores = []

    for sigma in noise_levels:
        for seed in validation_seeds:
            clean, noisy = make_image(seed=seed, sigma=sigma)

            denoised, lam = adaptive_chambolle(
                noisy, alpha=alpha, max_iter=max_iter
            )

            scores.append(psnr(denoised, clean))

    adaptive_results[alpha] = np.mean(scores)

    print("Alpha:", alpha,
          "Mean PSNR:", adaptive_results[alpha])

best_alpha = max(adaptive_results, key=adaptive_results.get)

print("Best alpha:", best_alpha)
print("Best mean PSNR:", adaptive_results[best_alpha])

# parameter tuning summary
print("\nParameter tuning summary")
print("Best baseline lambda:", best_lambda)
print("Best adaptive alpha:", best_alpha)
print("Baseline validation PSNR:", baseline_results[best_lambda])
print("Adaptive validation PSNR:", adaptive_results[best_alpha])