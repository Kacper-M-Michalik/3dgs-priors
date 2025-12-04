import argparse
import numpy as np
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run significance testing between two sets of scores.")
    parser.add_argument("baseline", type=str, help="Path to the baseline scores file.")
    parser.add_argument("comparison", type=str, help="Path to the comparison scores file.")

    args = parser.parse_args()

    psnr_baseline = []
    ssim_baseline = []
    lpips_baseline = []
    psnr_comparison = []
    ssim_comparison = []
    lpips_comparison = []

    for file in [args.baseline, args.comparison]:
        with open(file, "r") as f:
            for line in f:
                [example, psnr_str, ssim_str, lpips_str] = line.split(" ")
                psnr = float(psnr_str)
                ssim = float(ssim_str)
                lpips = float(lpips_str)
                if file == args.baseline:
                    psnr_baseline.append(psnr)
                    ssim_baseline.append(ssim)
                    lpips_baseline.append(lpips)
                else:
                    psnr_comparison.append(psnr)
                    ssim_comparison.append(ssim)
                    lpips_comparison.append(lpips)

    filename = args.comparison.split("/")[-1].split(".")[0]
    
    if os.path.exists("stats") is False:
        os.makedirs("stats")
    
    for (metric_name, baseline_scores, comparison_scores) in [("PSNR", psnr_baseline, psnr_comparison), ("SSIM", ssim_baseline, ssim_comparison), ("LPIPS", lpips_baseline, lpips_comparison)]:
        baseline_mean = np.mean(baseline_scores)
        comparison_mean = np.mean(comparison_scores)
        observed_diff = comparison_mean - baseline_mean

        # parametric
        synthetic_datasets = [np.random.normal(loc=np.mean(baseline_scores), scale=np.std(baseline_scores), size=len(baseline_scores)) for _ in range(300000)]
        synthetic_diffs = [np.mean(dataset) - baseline_mean for dataset in synthetic_datasets]
        parametric_p_value = np.mean(np.where(synthetic_diffs >= observed_diff, 1, 0))
        with open(f"stats/parametric_significance_{metric_name.lower()}_{filename}.txt", "w") as sig_file:
            sig_file.write(f"Observed Difference: {observed_diff:.7f}\n")
            sig_file.write(f"P-value: {parametric_p_value:.7f}\n")

        # non-parametric
        combined_scores = np.array(baseline_scores + comparison_scores)
        n_baseline = len(baseline_scores)
        n_comparison = len(comparison_scores)

        count = 0
        n_resamples = 300000
        for _ in range(n_resamples):
            np.random.shuffle(combined_scores)
            new_baseline = combined_scores[:n_baseline]
            new_comparison = combined_scores[n_baseline:]
            new_diff = np.mean(new_comparison) - np.mean(new_baseline)
            if new_diff >= observed_diff:
                count += 1
        
        p_value = count / n_resamples

        with open(f"stats/non_parametric_significance_{metric_name.lower()}_{filename}.txt", "w") as sig_file:
            sig_file.write(f"Observed Difference: {observed_diff:.7f}\n")
            sig_file.write(f"P-value: {p_value:.7f}\n")