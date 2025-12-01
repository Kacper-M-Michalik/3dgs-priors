import torch
from torch.utils.data import Dataset, DataLoader
from typing import NamedTuple
import pandas as pd
import numpy as np
import cv2
import os
import argparse
import os
import glob

class Subset(NamedTuple):
    name: str
    resume_index: 0

class ProcessedImage(NamedTuple):
    uuid: str
    file_id: int
    image: np.array

class ImageDataset(Dataset):
    def __init__(self, file_paths, transform):
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        rgb_path = self.file_paths[idx]        
        rgb = cv2.imread(rgb_path)[:, :, ::-1]         
        file_id = int(os.path.basename(rgb_path).split('.')[0]) # + ".pt"        
        # Return processed tensor and filename
        return self.transform(rgb).squeeze(0), file_id

def save(path, df):
    df.to_parquet(path)

def main(args):  
    # Select large model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    midas_model_type = "DPT_Large"  
    model = torch.hub.load("intel-isl/MiDaS", midas_model_type).to(device).eval()
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = transforms.dpt_transform

    batch_size = 16
    workers = 4
    # Needed for downsize
    target_size = (128, 128) 

    in_path = os.path.join(args.in_folder, "srn_cars")    
    out_path = os.path.join(args.out_folder, "srn_cars_depths.parquet")  
    print(in_path)
    print(out_path)

    set_names = ["test", "train", "val"] 
    sets = []

    df = pd.DataFrame()
    data = []
    if os.path.exists(out_path):
        df = pd.read_parquet(out_path)
        for set in set_names:
            processed_uuids = df.loc[df['split'] == set, 'uuid'].unique()
            processed_uuids.sort()
            sets.append(Subset(name=set, resume_index=len(processed_uuids)))
    else:
      for set in set_names:
          sets.append(Subset(name=set, resume_index=0))

    for set in sets:
        subset = "cars_{}".format(set.name)
        search_path = os.path.join(in_path, subset)

        # Allows us to resume from previous progress
        intrins = sorted(glob.glob(os.path.join(search_path, "*", "intrinsics.txt")))
        
        print("Split: {}, Resuming from index {}".format(set.name, set.resume_index))

        for i, intrin in enumerate(intrins[set.resume_index:], start=set.resume_index):
            folder_path = os.path.dirname(intrin)   
            uuid = os.path.basename(folder_path)        

            rgbs = glob.glob(os.path.join(folder_path, "rgb", "*.png"))
            dataset = ImageDataset(rgbs, transform)
            loader = DataLoader(dataset, batch_size=batch_size, drop_last=False, shuffle=False, num_workers=workers)

            print("Intrin Num:{}, RGBs found: {}".format(i, len(rgbs)))

            # Only append a fully completed batch, allows us to safely resume a run thats been stopped
            full_batch = []
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
                    preds_uint8 = preds_normalized.mul(255).byte().cpu().numpy()

                    # H W format -> 128 * 128
                    for j, file_id in enumerate(batch_filenames):   
                        full_batch.append(ProcessedImage(uuid=uuid, file_id=file_id.item(), image=preds_uint8[j].tobytes()))

            for entry in full_batch:
                data.append({
                    "split": set.name,
                    "uuid": entry.uuid,
                    "frame_id": entry.file_id,                    
                    "depth": entry.image
                })

            # Do occasional save if requested
            if args.save_iter != -1 and ((i-1) % args.save_iter) == 0:
                save(out_path, pd.concat([df, pd.DataFrame(data)], ignore_index=True))
        
        # Save on subset completion
        save(out_path, pd.concat([df, pd.DataFrame(data)], ignore_index=True))                  

def parse_arguments():
    parser = argparse.ArgumentParser(description='Evaluate model')
    parser.add_argument('--in_folder', type=str, default='out', required = True, help='Input folder to process')
    parser.add_argument('--out_folder', type=str, default='out', required = True, help='Output folder to save resulting files to')
    parser.add_argument('--save_iter', type=int, default=-1, help='How often to make an intermediate save')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    print("Processing cars dataset with depth priors")
    print("In folder: {}".format(args.in_folder))
    print("Out folder: {}".format(args.out_folder))

    main(args)