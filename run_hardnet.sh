#!/bin/bash

# hardnet
echo "Extracting Hardnet"

python extract_descriptors_hardnet.py --dataset_path=../benchmark-patches-8k --method_name=sift8k_8000_hardnet
python extract_descriptors_hardnet.py --dataset_path=../benchmark-patches-8k-upright-no-dups --method_name=sift8k_8000_hardnet-upright-no-dups

#python extract_descriptors_hardnet.py --dataset_path=../benchmark-patches-default --method_name=siftdef_2048_hardnet
#python extract_descriptors_hardnet.py --dataset_path=../benchmark-patches-default-upright-no-dups --method_name=siftdef_2048_hardnet-upright-no-dups
