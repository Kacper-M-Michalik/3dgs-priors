import argparse
import matplotlib.pyplot as plt
import numpy as np
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate statistics from a scores file.")
    parser.add_argument("input_file", type=str, help="Path to the input scores file.")

    args = parser.parse_args()

    all_psnr = []
    all_ssim = []
    all_lpips = []

    with open(args.input_file, "r") as f:
        for line in f:
            [example, psnr_str, ssim_str, lpips_str] = line.split(" ")
            psnr = float(psnr_str)
            ssim = float(ssim_str)
            lpips = float(lpips_str)
            all_psnr.append(psnr)
            all_ssim.append(ssim)
            all_lpips.append(lpips)

    filename = args.input_file.split("/")[-1].split(".")[0]
    
    if os.path.exists("stats") is False:
        os.makedirs("stats")

    with open(f"stats/{filename}_stats.txt", "w") as stats_file:
        for (metric_name, all_values) in [("PSNR", all_psnr), ("SSIM", all_ssim), ("LPIPS", all_lpips)]:
            fig, ax = plt.subplots()
            plt.violinplot(all_values, showmeans=True)
            ax.set_xticks([1], labels=[""])
            ax.set_ylabel(metric_name)
            ax.set_title(f"Distribution of {metric_name} Scores")
            plt.savefig(f"stats/{filename}_{metric_name.lower()}_distribution.png")
            mean = np.mean(all_values)
            std = np.std(all_values)
            stats_file.write(f"{metric_name}: Mean = {mean:.7f}, Std = {std:.7f}\n")