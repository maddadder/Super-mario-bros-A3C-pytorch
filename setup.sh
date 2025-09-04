#get cuda:
#https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/

glxinfo | grep -i "OpenGL renderer"
#if not nvidia then run:
sudo prime-select nvidia

#if ubuntu 24.04
conda remove --name cenv --all
conda create -n cenv python=3.8
conda activate cenv
pip install -r requirements.txt
python train.py
#else
sudo apt update
sudo apt install build-essential
python3 -m venv ./env
source env/bin/activate
pip install -r requirements.txt
python train.py

