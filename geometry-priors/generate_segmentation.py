import torch
from torch.utils.data import Dataset, DataLoader
from typing import NamedTuple
import pandas as pd
import numpy as np
import os
import argparse
import glob
from PIL import Image
from transparent_background import Remover # <--- NEW LIBRARY
from torchvision import transforms

class Subset(NamedTuple):
    name: str
    resume_index: int

class ProcessedImage(NamedTuple):
    uuid: str
    file_id: int
    image: np.array

class ImageDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        rgb_path = self.file_paths[idx]
        rgb = Image.open(rgb_path).convert("RGB")
        file_id = int(os.path.basename(rgb_path).split('.')[0])
        return rgb, file_id

def save(path, df):
    folder = os.path.dirname(path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    df.to_parquet(path)

def main(args):  
    device = "cuda" if torch.cuda.is_available() else "cpu"
    remover = Remover(mode='base', device=device) 

    batch_size = 1 
    workers = 0   
    
    target_size = (128, 128) 

    in_path = os.path.join(args.in_folder, "srn_cars")    
    out_path = os.path.join(args.out_folder, "srn_cars_segmentation.parquet")  
    print(f"Input path: {in_path}")
    print(f"Output path: {out_path}")

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

    for set_info in sets:
        subset = "cars_{}".format(set_info.name)
        search_path = os.path.join(in_path, subset)
        intrins = sorted(glob.glob(os.path.join(search_path, "*", "intrinsics.txt")))
        
        print(f"Split: {set_info.name}, Resuming from index {set_info.resume_index}")

        for i, intrin in enumerate(intrins[set_info.resume_index:], start=set_info.resume_index):
            folder_path = os.path.dirname(intrin)   
            uuid = os.path.basename(folder_path)        

            rgbs = sorted(glob.glob(os.path.join(folder_path, "rgb", "*.png")))
            dataset = ImageDataset(rgbs)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)

            full_batch = []
            
            for batch_imgs, batch_filenames in loader:      
                for k in range(len(batch_filenames)):
                    img_tensor = batch_imgs[k] 
                    file_id = batch_filenames[k]

                    img_np = img_tensor.numpy() # For pre-processing differences
                    img_pil = Image.fromarray(img_np.astype('uint8'), 'RGB')
                    out = remover.process(img_pil) # Returns PIL Image with alpha channel
                    out_np = np.array(out)
                    alpha = out_np[:, :, 3] # (H, W)
                    
                    # Resizing
                    alpha_img = Image.fromarray(alpha)
                    alpha_resized = alpha_img.resize(target_size, resample=Image.NEAREST)
                    
                    mask_np = np.array(alpha_resized)
                    mask_binary = (mask_np > 127).astype(np.uint8) * 255
                    
                    # Not batched, so no loop. 
                    full_batch.append(ProcessedImage(uuid=uuid, file_id=file_id.item(), image=preds_uint8[j].tobytes()))


            for entry in full_batch:
                data.append({
                    "split": set_info.name,
                    "uuid": entry.uuid,
                    "frame_id": entry.file_id,                    
                    "segmentation": entry.image 
                })

            if args.save_iter != -1 and ((i-1) % args.save_iter) == 0:
                print(f"Auto-saving at iteration {i}...")
                save(out_path, pd.concat([df, pd.DataFrame(data)], ignore_index=True))
        
        # Save on subset completion
        save(out_path, pd.concat([df, pd.DataFrame(data)], ignore_index=True))    

def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate segmentation priors using InSPyReNet')
    parser.add_argument('--in_folder', type=str, required=True, help='Input folder containing images')
    parser.add_argument('--out_folder', type=str, required=True, help='Output folder to save segmentation priors')
    parser.add_argument('--save_iter', type=int, default=50, help='How often to make an intermediate save')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    print("Processing cars dataset with segmentation priors (InSPyReNet)")
    main(args)