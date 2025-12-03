import matplotlib.pyplot as plt
import numpy as np
import os
filenames = ["reference_scores", "depth_only_scores", "normals_only_scores", "both_scores"]

combined_psnrs = []
combined_ssims = []
combined_lpips = []

for filename in filenames:
    all_psnr = []
    all_ssim = []
    all_lpips = []

    with open(f"scores/{filename}.txt", "r") as f:
        for line in f:
            [example, psnr_str, ssim_str, lpips_str] = line.split(" ")
            psnr = float(psnr_str)
            ssim = float(ssim_str)
            lpips = float(lpips_str)
            all_psnr.append(psnr)
            all_ssim.append(ssim)
            all_lpips.append(lpips)
    
    combined_psnrs.append(all_psnr)
    combined_ssims.append(all_ssim)
    combined_lpips.append(all_lpips)

if not os.path.exists("stats"):
    os.makedirs("stats")

for (metric_name, combined_values) in [("PSNR", combined_psnrs), ("SSIM", combined_ssims), ("LPIPS", combined_lpips)]:
    fig, ax = plt.subplots()
    violin = plt.violinplot(combined_values, showmeans=True)
    ax.set_xticks(np.arange(1, 5), labels=[ "None", "Depth Only", "Normal Only", "Both" ])
    ax.set_xlim(0.25, 4 + 0.75)
    ax.set_ylabel(metric_name)
    ax.set_title(f"Distribution of {metric_name} Scores")
    plt.savefig(f"stats/Comparison_{metric_name.lower()}_distribution.png")