#!/usr/bin/env bash

# launch from root of the project

# assuming conda and cuda 10.2 installed
# script tested on AWS AMI ami-0abcbc65f89fb220e (Deep Learning AMI (Ubuntu 18.04) Version 30.0)

conda update -y -n base -c defaults conda
conda create -n nir-graph
source ~/anaconda3/etc/profile.d/conda.sh
conda activate nir-graph

conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch

pip install -r requirements.txt

pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.11.0+cu102.html