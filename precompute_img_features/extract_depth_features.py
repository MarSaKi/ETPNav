#!/usr/bin/env python3

''' Script to precompute image features using a Pytorch ResNet CNN, using 36 discretized views
    at each viewpoint in 30 degree increments, and the provided camera WIDTH, HEIGHT 
    and VFOV parameters. '''

import os
import sys

import argparse
import numpy as np
import json
import math
import h5py
import copy
from PIL import Image
import time
from progressbar import ProgressBar

import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass

import clip
from resnet_encoder import VlnResnetDepthEncoder
from gym.spaces import Box

VIEWPOINT_SIZE = 36 # Number of discretized views from one viewpoint

def load_viewpoint_ids(connectivity_dir):
    viewpoint_ids = []
    with open(os.path.join(connectivity_dir, 'scans.txt')) as f:
        scans = [x.strip() for x in f]      # load all scans
    for scan in scans:
        with open(os.path.join(connectivity_dir, '%s_connectivity.json'%scan)) as f:
            data = json.load(f)
            viewpoint_ids.extend([(scan, x['image_id']) for x in data if x['included']])
    print('Loaded %d viewpoints' % len(viewpoint_ids))
    return viewpoint_ids

def build_feature_extractor(model_name, checkpoint_file):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model, preprocess = clip.load(model_name, device='cuda')
    model = VlnResnetDepthEncoder(
        {"depth": Box(low=0.0, high=1.0, shape=(256,256,1), dtype='float32')},
        output_size=256,
        checkpoint=checkpoint_file,
        backbone=model_name,
        spatial_output=False,
    ).to(device)
    model.eval()
    return model, device

def process_features(proc_id, out_queue, scanvp_list, args):
    print('start proc_id: %d' % proc_id)

    # Set up PyTorch CNN model
    torch.set_grad_enabled(False)
    model, device = build_feature_extractor(args.model_name, args.checkpoint_file)

    with h5py.File(args.img_db, 'r') as f:
        for scan_id, viewpoint_id in scanvp_list:
            data = f[f'{scan_id}_{viewpoint_id}'][...].astype('float32')
            # print(data.shape, data.min(), data.max())
            images = torch.from_numpy(data).to(device)
            fts = model({"depth": images})
            fts = fts.data.cpu().numpy()
            logits = []
            out_queue.put((scan_id, viewpoint_id, fts, logits))

    out_queue.put(None)

def build_feature_file(args):
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    scanvp_list = load_viewpoint_ids(args.connectivity_dir)

    num_workers = min(args.num_workers, len(scanvp_list))
    num_data_per_worker = len(scanvp_list) // num_workers

    out_queue = mp.Queue()
    processes = []
    for proc_id in range(num_workers):
        sidx = proc_id * num_data_per_worker
        eidx = None if proc_id == num_workers - 1 else sidx + num_data_per_worker

        process = mp.Process(
            target=process_features,
            args=(proc_id, out_queue, scanvp_list[sidx: eidx], args)
        )
        process.start()
        processes.append(process)
    
    num_finished_workers = 0
    num_finished_vps = 0

    progress_bar = ProgressBar(maxval=len(scanvp_list))
    progress_bar.start()

    with h5py.File(args.output_file, 'w') as outf:
        while num_finished_workers < num_workers:
            res = out_queue.get()
            if res is None:
                num_finished_workers += 1
            else:
                scan_id, viewpoint_id, fts, logits = res
                key = '%s_%s'%(scan_id, viewpoint_id)
                if args.out_image_logits:
                    data = np.hstack([fts, logits])
                else:
                    data = fts
                outf.create_dataset(key, data.shape, dtype='float32', compression='gzip')
                outf[key][...] = data
                outf[key].attrs['scanId'] = scan_id
                outf[key].attrs['viewpointId'] = viewpoint_id

                num_finished_vps += 1
                progress_bar.update(num_finished_vps)

    progress_bar.finish()
    for process in processes:
        process.join()
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='resnet50')
    parser.add_argument('--checkpoint_file', default='data/ddppo-models/gibson-2plus-resnet50.pth')
    parser.add_argument('--connectivity_dir', default='precompute_img_features/connectivity')
    parser.add_argument('--img_db', default='pretrain_src/img_features/habitat_256x256_vfov60_depth.hdf5')
    parser.add_argument('--out_image_logits', action='store_true', default=False)
    parser.add_argument('--output_file', default='pretrain_src/img_features/ddppo_resnet50_depth_features.hdf5')
    parser.add_argument('--batch_size', default=36, type=int)
    parser.add_argument('--num_workers', type=int, default=1)
    args = parser.parse_args()

    build_feature_file(args)