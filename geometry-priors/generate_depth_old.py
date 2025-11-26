import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import shutil
import cv2
import os
import argparse
import os
import glob

class ImageDataset(Dataset):
    def __init__(self, file_paths, transform):
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        rgb_path = self.file_paths[idx]        
        rgb = cv2.imread(rgb_path)[:, :, ::-1]         
        file_id = os.path.basename(rgb_path) # .split('.')[0] + ".pt"        
        # Return processed tensor and filename
        return self.transform(rgb).squeeze(0), file_id

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
    
    if not os.path.exists(out_path):
        shutil.copytree(in_path, out_path)

    batch_size = 16
    workers = 4

    # Needed for downsize
    target_size = (128, 128) 

    for set in ["test", "train", "val"]:
        subset = "cars_{}".format(set)
        search_path = os.path.join(in_path, subset)
        dest_path = os.path.join(out_path, subset)

        # Allows us to resume from previous progress
        resume_index = 0
        intrins = sorted(glob.glob(os.path.join(search_path, "*", "intrinsics.txt")))

        for i, intrin in enumerate(intrins):
            if not os.path.exists(os.path.join(dest_path, os.path.basename(os.path.dirname(intrin)), "depth")):
                resume_index = max(0, i-1)
                break
        
        print("Resuming from index {}".format(resume_index))

        for i, intrin in enumerate(intrins[resume_index:], start=resume_index):
            folder_path = os.path.dirname(intrin)           
            depth_out_path = os.path.join(dest_path, os.path.basename(folder_path), "depth") 
            if not os.path.exists(depth_out_path):
                os.mkdir(depth_out_path)

            rgbs = glob.glob(os.path.join(folder_path, "rgb", "*.png"))
            dataset = ImageDataset(rgbs, transform)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)

            print("Intrin Num:{}, RGBs found: {}".format(i, len(rgbs)))

            with torch.no_grad():
                for batch_imgs, batch_filenames in loader:      
                    # Prediction              
                    batch_imgs = batch_imgs.to(device)            
                    preds = model(batch_imgs)

                    # Batched downsize
                    preds = torch.nn.functional.interpolate(
                        preds.unsqueeze(1),
                        size=target_size,
                        mode="bicubic",
                        align_corners=False
                    ).squeeze(1)

                    # Batched 0 to 1 normalization
                    batch_flat = preds.flatten(start_dim=1)
                    min_val, max_val = torch.aminmax(batch_flat, dim=1, keepdim=True)
                    min_val = min_val.view(preds.size(0), 1, 1)
                    max_val = max_val.view(preds.size(0), 1, 1)
                    preds_normalized = (preds - min_val) / (max_val - min_val + 1e-8)
                    # Save as tensor
                    #preds_cpu = preds_normalized.cpu()   
                    #for j, filename in enumerate(batch_filenames):
                    #    final_output_path = os.path.join(depth_out_path, filename)
                    #    torch.save(preds_cpu[j].clone(), final_output_path)         

                    # Save as image
                    #preds_uint8 = preds_normalized.mul(255).byte().cpu().numpy()
                    #for j, filename in enumerate(batch_filenames):    
                    #    final_output_path = os.path.join(depth_out_path, filename)
                    #    cv2.imwrite(final_output_path, preds_uint8[j])
                    
                    # Save as iamge to dataframe
                    preds_uint8 = preds_normalized.mul(255).byte().cpu().numpy()
                    for j, filename in enumerate(batch_filenames):    
                        final_output_path = os.path.join(depth_out_path, filename)
                        cv2.imwrite(final_output_path, preds_uint8[j])

                    

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