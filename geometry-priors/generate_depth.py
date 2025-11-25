import sys
sys.path.insert(0, '/content/models/depth/lib/python3.12/site-packages')

import torch
import numpy as np
import shutil
import pandas as pd
import cv2
import os
import argparse
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
    # Select large model
    midas_model_type = "DPT_Large"  
    model = torch.hub.load("intel-isl/MiDaS", midas_model_type).to(device).eval()
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = transforms.dpt_transform

    in_path = os.path.join(args.in_folder, "srn_cars")    
    out_path = os.path.join(args.out_folder, "srn_cars_prior")  
    print(in_path)
    print(out_path)

    for set in ["test", "train", "val"]:
        subset = "cars_{set}"
        search_path = os.path.join(in_path, subset)
        dest_path = os.path.join(out_path, subset)

        intrins = sorted(glob.glob(os.path.join(search_path, "*", "intrinsics.txt")))
        for intrin in intrins:
            folder_name = os.path.dirname(intrin)
            folder_path = os.path.basepath(intrin)
            print(folder_name)
            print(folder_path)
            break
            
            intrin_out_path = os.path.join(dest_path, folder_name)
            print(intrin_out_path)
            #if not os.path.exists(intrin_out_path):
            #    shutil.copytree(folder_path, dest_path)

            #os.mkdir()
            # add "rgb"
            #create output dir + "depth"
            #get all pictures
                #loop through all, call pred_depth
                    #rgb = cv2.imread(path)[:, :, ::-1]
                    #     # Produce depth map array
                    #     depth = pred_depth(device, model, transform, rgb) 
                    #     # Convert numpy array back to tensor
                    #     pred_tensor = torch.from_numpy(depth).float() 

                    #     # torch.save(pred_tensor, output_path)

                    #     # The following is needed only if edge operator results are required
                    #     # visualize_edges(rgb, depth, img)
                          #save picture in depth

def parse_arguments():
    parser = argparse.ArgumentParser(description='Evaluate model')
    parser.add_argument('--in_folder', type=str, default='out', required = True, help='Inut folder to process')
    parser.add_argument('--out_folder', type=str, default='out', required = True, help='Output folder to save resultsing files to')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    print("Processing cars dataset with depth priors")
    print("In folder: {}".format(args.in_folder))
    print("Out folder: {}".format(args.out_folder))

    main(args)