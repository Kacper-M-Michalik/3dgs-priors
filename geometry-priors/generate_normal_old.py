import torch
from torch.utils.data import DataLoader
import numpy as np
import cv2
import os
import argparse
import glob
import shutil
import sys
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_PATH = os.path.join(SCRIPT_DIR, "surface_normal_uncertainty")

if REPO_PATH not in sys.path:
    sys.path.append(REPO_PATH)

from models.NNET import NNET
from data.dataloader_custom import CustomLoadPreprocess
import utils.utils as utils


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NNET(args).to(device).eval()

    model = utils.load_checkpoint(args.checkpoint, model)
    model.eval()


    in_path = os.path.join(args.in_folder, "srn_cars")
    out_path = os.path.join(args.out_folder, "srn_cars_normal_priors")

    if not os.path.exists(out_path):
        shutil.copytree(in_path, out_path)

    for set_name in ["test", "train", "val"]:
        subset = f"cars_{set_name}"
        search_path = os.path.join(in_path, subset)
        dest_path = os.path.join(out_path, subset)

        intrins = sorted(glob.glob(os.path.join(search_path, "*", "intrinsics.txt")))

        resume_index = 0
        for i, intrin in enumerate(intrins):
            out_dir = os.path.join(dest_path, os.path.basename(os.path.dirname(intrin)), "normal")
            if not os.path.exists(out_dir):
                resume_index = max(0, i - 1)
                break

        print("Resuming from index {}".format(resume_index))

    
        for i, intrin in enumerate(tqdm(intrins[resume_index:], initial=resume_index)):
            folder_path = os.path.dirname(intrin)
            normal_out_path = os.path.join(dest_path, os.path.basename(folder_path), "normal")

            os.makedirs(normal_out_path, exist_ok=True)

            rgb_folder = os.path.join(folder_path, "rgb")

            # Use repo’s own preprocessing (identical to test.py)
            dataset = CustomLoadPreprocess(args, rgb_folder)
            loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

            with torch.no_grad():
                for sample in loader:
                    img = sample["img"].to(device)        
                    img_name = sample["img_name"][0] + ".png"

                    print(img_name)
                    break
                    # Forward pass logic from test.py
                    norm_out_list, _, _ = model(img)
                    norm_out = norm_out_list[-1]
                    pred_norm = norm_out[:, :3, :, :]

                    arr = pred_norm.detach().cpu().permute(0, 2, 3, 1).numpy()

                    arr = ((arr + 1.0) * 0.5) * 255.0
                    arr = np.clip(arr, 0, 255).astype(np.uint8)

                    out_file = os.path.join(normal_out_path, img_name)
                    img_rgb = arr[0]           
                    img_bgr = img_rgb[:, :, ::-1]
                    cv2.imwrite(out_file, img_bgr)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate normal map priors')

    parser.add_argument('--in_folder', type=str, required=True, help='Input folder containing images')
    parser.add_argument('--out_folder', type=str, required=True, help='Output folder to save normal priors')

    parser.add_argument(
        '--checkpoint',
        type=str,
        default=os.path.join(REPO_PATH, "checkpoints", "scannet.pt")
    ) # Checkpoints to be downloaded separately.

    # Arguments required by NNET/Decoder
    parser.add_argument('--sampling_ratio', type=float, default=0.4)
    parser.add_argument('--importance_ratio', type=float, default=0.7)

    # Model arguments
    parser.add_argument('--input_height', type=int, default=512)
    parser.add_argument('--input_width', type=int, default=512)

    parser.add_argument('--architecture', type=str, default="BN")
    parser.add_argument('--pretrained', type=str, default="scannet", help = "Checkpoints")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    print("Processing cars dataset with normal priors")
    print("Input folder:", args.in_folder)
    print("Output folder:", args.out_folder)

    main(args)
