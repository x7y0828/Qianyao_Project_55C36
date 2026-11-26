import numpy as np
import time

from src.data import make_image
from src.baseline import chambolle
from src.modified import adaptive_chambolle
from src.metrics import mse, psnr, gradient_mse

import csv
from pathlib import Path

# final experiment settings
noise_levels = [0.05, 0.1, 0.2]
test_seeds = range(200, 220)

best_lambda = 0.15
best_alpha = 15.0
max_iter = 200

results = []

# save data
def save_csv(results, filename):
    output_dir = Path("results")        #指定结果保存目录
    output_dir.mkdir(exist_ok=True)     #如果文件夹不存在则自动创建

    filepath = output_dir / filename

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(        #将字典数据写入CSV
            f, fieldnames=results[0].keys()
        )
        writer.writeheader()            #写入表头
        writer.writerows(results)       #写入所有实验数据

    print("Results saved to:", filepath)

# Runtime measurement with warm-up  自行设定预热1次、正式测量5次，取平均，减少单次测量偶然误差
def measure_runtime(func, *args, repeats=5, **kwargs):
    # Warm-up
    func(*args, **kwargs)

    times = []

    for _ in range(repeats):
        start = time.perf_counter()
        func(*args, **kwargs)
        elapsed = time.perf_counter() - start

        times.append(elapsed)

    return np.mean(times)

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

        # Runtime measurement
        runtime_baseline = measure_runtime(
            chambolle, noisy,
            lam=best_lambda, max_iter=max_iter
        )

        runtime_adaptive = measure_runtime(
            adaptive_chambolle, noisy,
            alpha=best_alpha, max_iter=max_iter
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

            "baseline_runtime": runtime_baseline,
            "adaptive_runtime": runtime_adaptive,
            "baseline_iterations": max_iter,
            "adaptive_iterations": max_iter,

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

save_csv(results, "raw_results.csv")

summary_results = []    #create a list of statistical results

# statistical analysis
for sigma in noise_levels:
    cases = [r for r in results if r["sigma"] == sigma]

    print("\nNoise level:", sigma)
    print("Number of cases:", len(cases))

    # for metric in ["mse", "psnr", "gradient_mse"]:
    for metric in ["mse", "psnr", "gradient_mse", "runtime"]:
        baseline = np.array([
            r["baseline_" + metric] for r in cases
        ])

        adaptive = np.array([
            r["adaptive_" + metric] for r in cases
        ])

        print("\nMetric:", metric)

        print("Baseline mean:", np.mean(baseline))
        print("Baseline std:", np.std(baseline, ddof=1))

        print("Adaptive mean:", np.mean(adaptive))
        print("Adaptive std:", np.std(adaptive, ddof=1))

        # Paired improvement
        if metric == "psnr":
            difference = adaptive - baseline
        else:
            difference = baseline - adaptive

        mean_diff = np.mean(difference)
        std_diff = np.std(difference, ddof=1)

        # 95% confidence interval (n = 20)
        margin = 2.093 * std_diff / np.sqrt(len(difference))

        print("Mean improvement:", mean_diff)
        print("95% CI:",
              mean_diff - margin,
              mean_diff + margin)

        # record statistical result
        summary_results.append({
            "sigma": sigma,
            "metric": metric,
            "n_cases": len(cases),
            "baseline_mean": np.mean(baseline),
            "baseline_std": np.std(baseline, ddof=1),
            "adaptive_mean": np.mean(adaptive),
            "adaptive_std": np.std(adaptive, ddof=1),
            "mean_improvement": mean_diff,
            "ci_lower": mean_diff - margin,
            "ci_upper": mean_diff + margin,
            "adaptive_win_rate": np.mean(difference > 0) * 100
        })

        print("Adaptive win rate:",
              np.mean(difference > 0) * 100, "%")

save_csv(summary_results, "summary_results.csv")