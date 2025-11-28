import pandas as pd
import numpy as np
import cv2
import os
import argparse
import os
import glob

def save(path, intrin_data, rgb_data, pose_data):
    rgb_out_path = os.path.join(args.out_folder, "srn_cars_rgbs.parquet")  
    pose_out_path = os.path.join(args.out_folder, "srn_cars_poses.parquet")  
    intrin_out_path = os.path.join(args.out_folder, "srn_cars_intrins.parquet")  

    intrin_df = pd.DataFrame(intrin_data)
    intrin_df.to_parquet(intrin_out_path)
    
    pose_df = pd.DataFrame(pose_data)
    pose_df.to_parquet(pose_out_path)
    
    rgb_df = pd.DataFrame(rgb_data)
    rgb_df.to_parquet(rgb_out_path)

def main(args):
    in_path = os.path.join(args.in_folder, "srn_cars")    
    print(in_path)

    sets = ["test", "train", "val"]

    rgb_data = []
    pose_data = []
    intrin_data = []

    for set in sets:
        subset = "cars_{}".format(set)
        search_path = os.path.join(in_path, subset)

        # Allows us to resume from previous progress
        intrins = sorted(glob.glob(os.path.join(search_path, "*", "intrinsics.txt")))
        
        print("Split: {}".format(set))

        for i, intrin in enumerate(intrins):
            folder_path = os.path.dirname(intrin)   
            uuid = os.path.basename(folder_path)        

            rgbs = glob.glob(os.path.join(folder_path, "rgb", "*.png"))
            poses = glob.glob(os.path.join(folder_path, "pose", "*.txt"))

            print("Intrin Num:{}, RGBs found: {}, Poses found: {}".format(i, len(rgbs), len(poses)))
            
            with open(intrin, 'r') as file:
                data = file.read()

                intrin_data.append({
                    "split": set,
                    "uuid": uuid,
                    "intrinsics": data            
                }) 

            for rgb_path in rgbs:
                file_id = int(os.path.basename(rgb_path).split('.')[0]) 
                bgr = cv2.imread(rgb_path)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                rgb = rgb.transpose(2, 0, 1)

                # C H W format
                rgb_data.append({
                    "split": set,
                    "uuid": uuid,
                    "frame_id": file_id,                    
                    "depth": rgb.tobytes()
                })

            for pose_path in poses:
                with open(pose_path, 'r') as file:
                    file_id = int(os.path.basename(pose_path).split('.')[0]) 
                    data = file.read()    

                    pose_data.append({
                        "split": set,
                        "uuid": uuid,
                        "frame_id": file_id,                    
                        "pose": data
                    })  

            # Do occasional save if requested
            if args.save_iter != -1 and ((i-1) % args.save_iter) == 0:
                save(args.out_folder, intrin_data, rgb_data, pose_data)
        
        # Save on subset completion
        save(args.out_folder, intrin_data, rgb_data, pose_data)                

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