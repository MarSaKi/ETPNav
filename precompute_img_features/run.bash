export CUDA_VISIBLE_DEVICES=0
# https://github.com/peteanderson80/Matterport3DSimulator
# change to your Matterport3DSimulator path, Off-screen CPU rendering using OSMesa
# cd /root/mount/Matterport3DSimulator
# mkdir build && cd build
# cmake -DOSMESA_RENDERING=ON ..
# make
export PYTHONPATH=Matterport3DSimulator/build:$PYTHONPATH

python precompute_img_features/save_img.py --img_type rgb
python precompute_img_features/save_img.py --img_type depth
python precompute_img_features/extract_rgb_features.py
python precompute_img_features/extract_depth_features.py