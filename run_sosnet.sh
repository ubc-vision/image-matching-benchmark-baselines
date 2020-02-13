#!/bin/bash

# sosnet
echo "Extracting SOSnet"

python extract_descriptors_sosnet.py --dataset_path=../benchmark-patches-8k --method_name=sift8k_8000_sosnet
python extract_descriptors_sosnet.py --dataset_path=../benchmark-patches-8k-upright-no-dups --method_name=sift8k_8000_sosnet-upright-no-dups

python extract_descriptors_sosnet.py --dataset_path=../benchmark-patches-default --method_name=siftdef_2048_sosnet
python extract_descriptors_sosnet.py --dataset_path=../benchmark-patches-default-upright-no-dups --method_name=siftdef_2048_sosnet-upright-no-dups
