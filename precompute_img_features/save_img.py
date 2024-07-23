import os
import sys
import math
import json
from grpc import Compression
from tqdm import tqdm
import numpy as np
import h5py
from progressbar import ProgressBar
import torch.multiprocessing as mp
import argparse
import cv2
# change to your Matterport3DSimulator path, build in cpu mode
# https://github.com/peteanderson80/Matterport3DSimulator
sys.path.insert(0, '/home/dongan/workspace/lib/Matterport3DSimulator/build')
import MatterSim
from habitat_utils import HabitatUtils
from scipy.spatial.transform import Rotation as R

VIEWPOINT_SIZE = 36
WIDTH = 256
HEIGHT = 256
VFOV = 60

def build_simulator(connectivity_dir):
    sim = MatterSim.Simulator()
    sim.setNavGraphPath(connectivity_dir)
    sim.setCameraResolution(WIDTH, HEIGHT)
    sim.setCameraVFOV(math.radians(VFOV))
    sim.setDiscretizedViewingAngles(True)
    sim.setRenderingEnabled(False)
    sim.setDepthEnabled(False)
    sim.setPreloadingEnabled(False)
    sim.setBatchSize(1)
    sim.initialize()
    return sim

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

def get_img(proc_id, out_queue, scanvp_list, args):
    print('start proc_id: %d' % proc_id)

    # Set up the simulator
    sim = build_simulator(args.connectivity_dir)
    
    pre_scan = None
    habitat_sim = None
    for scan_id, viewpoint_id in scanvp_list:
        if scan_id != pre_scan:
            if habitat_sim != None:
                habitat_sim.sim.close()
            habitat_sim = HabitatUtils(f'data/scene_datasets/mp3d/{scan_id}/{scan_id}.glb', 
                                       int(0), 60, HEIGHT, WIDTH)
            pre_scan = scan_id

        images = []
        for ix in range(VIEWPOINT_SIZE):
            if ix == 0:
                sim.newEpisode([scan_id], [viewpoint_id], [0], [math.radians(-30)])
            elif ix % 12 == 0:
                sim.makeAction([0], [1.0], [1.0])
            else:
                sim.makeAction([0], [1.0], [0])
            state = sim.getState()[0]
            assert state.viewIndex == ix

            x, y, z, h, e = state.location.x, state.location.y, state.location.z, state.heading, state.elevation
            habitat_position = [x, z-1.25, -y]
            mp3d_h = np.array([0, 2*math.pi-h, 0]) # counter-clock heading
            mp3d_e = np.array([e, 0, 0])
            rotvec_h = R.from_rotvec(mp3d_h)
            rotvec_e = R.from_rotvec(mp3d_e)
            habitat_rotation = (rotvec_h * rotvec_e).as_quat()
            habitat_sim.sim.set_agent_state(habitat_position, habitat_rotation)

            if args.img_type == 'rgb':
                image = habitat_sim.render('rgb')[:, :, ::-1]
            elif args.img_type == 'depth':
                image = habitat_sim.render('depth')
            images.append(image)
        images = np.stack(images, axis=0)
        out_queue.put((scan_id, viewpoint_id, images))

    out_queue.put(None)

def build_img_file(args):

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
            target=get_img,
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
                scan_id, viewpoint_id, images = res
                key = '%s_%s'%(scan_id, viewpoint_id)
                if args.img_type == 'rgb':
                    outf.create_dataset(key, data=images, dtype='uint8', compression='gzip')
                elif args.img_type == 'depth':
                    outf.create_dataset(key, data=images, dtype='float32', compression='gzip')

                num_finished_vps += 1
                progress_bar.update(num_finished_vps)

    progress_bar.finish()
    for process in processes:
        process.join()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--connectivity_dir', default='precompute_img_features/connectivity')
    parser.add_argument('--output_file', default=None)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--img_type', type=str, default='rgb', choices=['rgb', 'depth'])
    args = parser.parse_args()
    os.system('mkdir -p pretrain_src/img_features')
    if args.img_type == 'rgb':
        args.output_file = f'pretrain_src/img_features/habitat_{HEIGHT}x{WIDTH}_vfov{VFOV}_bgr.hdf5'
    elif args.img_type == 'depth':
        args.output_file = f'pretrain_src/img_features/habitat_{HEIGHT}x{WIDTH}_vfov{VFOV}_depth.hdf5'
    else:
        raise NotImplementedError

    build_img_file(args)