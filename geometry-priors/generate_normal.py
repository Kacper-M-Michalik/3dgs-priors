import torch
from torch.utils.data import Dataset, DataLoader
from typing import NamedTuple
import pandas as pd
import numpy as np
import os
import argparse
import os
import glob
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_PATH = os.path.join(SCRIPT_DIR, "surface_normal_uncertainty")
if REPO_PATH not in sys.path:
    sys.path.append(REPO_PATH)

from models.NNET import NNET
from data.dataloader_custom import CustomLoadPreprocess

class Subset(NamedTuple):
    name: str
    resume_index: 0

class ProcessedImage(NamedTuple):
    uuid: str
    file_id: int
    image: np.array

def save(path, df):
    df.to_parquet(path)

def main(args):  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NNET(args).to(device).eval()
    ckpt = torch.load(args.checkpoint, map_location='cpu')['model']
    load_dict = {}
    for k, v in ckpt.items():
        if k.startswith('module.'):
            k_ = k.replace('module.', '')
            load_dict[k_] = v
        else:
            load_dict[k] = v
    model.load_state_dict(load_dict)    
    model.eval()

    batch_size = 8
    workers = 4
    # Needed for downsize
    target_size = (128, 128) 

    in_path = os.path.join(args.in_folder, "srn_cars")    
    out_path = os.path.join(args.out_folder, "srn_cars_normals.parquet")  
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
            
            rgb_folder = os.path.join(folder_path, "rgb")
            
            # Use the custom loader, batchsize other than 1 result in crashes/frozen notebook
            dataset = CustomLoadPreprocess(args, rgb_folder)
            loader = DataLoader(dataset, batch_size=batch_size, drop_last=False, shuffle=False, num_workers=workers)

            print("Intrin Num:{}".format(i))

            # Only append a fully completed batch, allows us to safely resume a run thats been stopped
            full_batch = []
            with torch.no_grad():
                for batch in loader:                    
                    imgs = batch["img"].to(device)        
                    file_ids = batch["img_name"]

                    norm_out_list, _, _ = model(imgs)   
                    norm_out = norm_out_list[-1]
                    norm_out = norm_out[:, :3, :, :]

                    norm_out = torch.nn.functional.interpolate(
                         norm_out,
                         size=target_size,
                         mode="bicubic",
                         align_corners=False
                    )

                    norm_out = torch.nn.functional.normalize(norm_out, dim=1)
                    norm_out = (norm_out + 1) * 127.5
                    norm_out = norm_out.clamp(0, 255).byte().cpu().numpy()
                   
                    for j, file_id in enumerate(file_ids):   
                        full_batch.append(ProcessedImage(uuid=uuid, file_id=file_id, image=norm_out[j].tobytes()))
            
            for entry in full_batch:
                data.append({
                    "split": set.name,
                    "uuid": entry.uuid,
                    "frame_id": entry.file_id,                    
                    "normal": entry.image
                })

            # Do occasional save if requested
            if args.save_iter != -1 and ((i-1) % args.save_iter) == 0:
                save(out_path, pd.concat([df, pd.DataFrame(data)], ignore_index=True))
        
        # Save on subset completion
        save(out_path, pd.concat([df, pd.DataFrame(data)], ignore_index=True))  
                
def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate normal map priors')

    parser.add_argument('--in_folder', type=str, required=True, help='Input folder containing images')
    parser.add_argument('--out_folder', type=str, required=True, help='Output folder to save normal priors')
    parser.add_argument('--save_iter', type=int, default=-1, help='How often to make an intermediate save')

    parser.add_argument('--checkpoint',type=str, default=os.path.join(REPO_PATH, "checkpoints", "scannet.pt")
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
