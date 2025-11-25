import torch
import shutil
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
    # Select large model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    midas_model_type = "DPT_Large"  
    model = torch.hub.load("intel-isl/MiDaS", midas_model_type).to(device).eval()
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = transforms.dpt_transform

    in_path = os.path.join(args.in_folder, "srn_cars")    
    out_path = os.path.join(args.out_folder, "srn_cars_prior")  
    print(in_path)
    print(out_path)

    for set in ["test", "train", "val"]:
        subset = "cars_{}".format(set)
        search_path = os.path.join(in_path, subset)
        dest_path = os.path.join(out_path, subset)

        intrins = sorted(glob.glob(os.path.join(search_path, "*", "intrinsics.txt")))
        for intrin in intrins:
            folder_path = os.path.dirname(intrin)       
            intrin_out_path = os.path.join(dest_path, os.path.basename(folder_path))

            if not os.path.exists(intrin_out_path):
                shutil.copytree(folder_path, intrin_out_path)
        
            depth_out_path = os.path.join(intrin_out_path, "depth") 
            if not os.path.exists(depth_out_path):
                os.mkdir(depth_out_path)

            rgbs = glob.glob(os.path.join(folder_path, "rgb", "*.png"))
            for rgb_path in rgbs:
                depth_file_id = os.path.basename(rgb_path).split('.')[0] + ".pt"
                final_output_path = os.path.join(depth_out_path, depth_file_id)
                rgb = cv2.imread(rgb_path)[:, :, ::-1]
                depth = pred_depth(device, model, transform, rgb)
                pred_tensor = torch.from_numpy(depth).float() 
                torch.save(pred_tensor, final_output_path)    

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