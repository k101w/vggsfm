

import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import os
from tqdm import tqdm
import json
import pdb

# root_dir = '../eth3d_slam_training'
# with torch.no_grad():
#     for dirpath, dirnames, filenames in os.walk(root_dir):
      
#         for scene_name in tqdm(dirnames):
#             cmd = 'CUDA_VISIBLE_DEVICES=7 python demo.py'
#             print(scene_name)
#             rgb_folder = os.path.join(dirpath, scene_name, 'rgb')
#             cmd+=f' SCENE_DIR={rgb_folder}'
#             cmd+=f' scene_name={scene_name}'
#             print(cmd)
#             os.system(cmd)
#         break



# Open the JSON file

root_dir = '../indoor'
with open(os.path.join(root_dir,'output.json'), 'r') as f:
    # Load the JSON data into a Python dictionary
    data = json.load(f)
sc = []
for i in range(len(data)):
    sc.append(data[i][0]['scene'])
print(sc)

with torch.no_grad():
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for scene_name in tqdm(dirnames):
            if scene_name in sc:
                print(f'done scene {scene_name}')
                continue
            
            cmd = 'python demo.py'
            print(scene_name)
            rgb_folder = os.path.join(dirpath, scene_name)
            cmd+=f' SCENE_DIR={rgb_folder}'
            cmd+=f' scene_name={scene_name}'
            print(cmd)
            os.system(cmd)
            exit()
        break
