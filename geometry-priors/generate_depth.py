import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse
import json
import os
import glob

def pred_depth(device, model, transform, rgb):
    inp = transform(rgb).to(device)
    with torch.no_grad():
        pred = model(inp).cpu().numpy()[0]

    # Min-max scaling to [0,1]
    pred = pred - pred.min()
    pred = pred / (pred.max() + 1e-8)
    return pred

def main(args):
    base_path = os.path.join(args.in_folder, "srn_cars_priors/")
    train_path = os.path.join(base_path, "cars_test")
    test_path = os.path.join(args.in_folder, "cars_train")
    val_path = os.path.join(args.in_folder, "cars_val")
    print(base_path)
    print(train_path)
    print(test_path)
    print(val_path)

    intrins = sorted(glob.glob(os.path.join(base_path, "*", "intrinsics.txt")))
    print(intrins)
    print(len(intrins))

    #is_chair = "chair" in cfg.data.category
    #    if is_chair and dataset_name == "train":
    #        # Ugly thing from SRN's public dataset
    #        tmp = os.path.join(self.base_path, "chairs_2.0_train")
    #        if os.path.exists(tmp):
    #            self.base_path = tmp

    #    self.intrins = sorted(
    #        glob.glob(os.path.join(self.base_path, "*", "intrinsics.txt"))
    #    )
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
    # # Select large model
    # midas_model_type = "DPT_Large"  
    # model = torch.hub.load("intel-isl/MiDaS", midas_model_type).to(device).eval()
    # transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    # transform = transforms.dpt_transform

    # # TODO PERFORM GLOBBING AND FOLDER GENERATION HERE
    # images = sorted([f for f in os.listdir(IMG_DIR)])
    
    # for img in images:
    #     path = os.path.join(IMG_DIR, img)
    #     rgb = cv2.imread(path)[:, :, ::-1]
    #     # Produce depth map array
    #     depth = pred_depth(device, model, transform, rgb) 
    #     # Convert numpy array back to tensor
    #     pred_tensor = torch.from_numpy(depth).float() 

    #     # torch.save(pred_tensor, output_path)

    #     # The following is needed only if edge operator results are required
    #     # visualize_edges(rgb, depth, img)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Evaluate model')
    parser.add_argument('--in_folder', type=str, default='out', required = True, help='Inut folder to process')
    parser.add_argument('--out_folder', type=str, default='out', required = True, help='Output folder to save resultsing files to')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    dataset_name = args.dataset_name
    print("Processing cars dataset with depth priors")
    print("In folder: {}".format(args.in_folder))
    print("Out folder: {}".format(args.out_folder))

    main(args)