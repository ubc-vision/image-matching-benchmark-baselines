#!/bin/bash

# geodesc
echo "Extracting Geodesc"

python extract_descriptors_geodesc.py --dataset_path=../benchmark-patches-8k --method_name=sift8k_8000_geodesc
python extract_descriptors_geodesc.py --dataset_path=../benchmark-patches-8k-upright-no-dups --method_name=sift8k_8000_geodesc-upright-no-dups

#python extract_descriptors_geodesc.py --dataset_path=../benchmark-patches-default --method_name=siftdef_2048_geodesc
#python extract_descriptors_geodesc.py --dataset_path=../benchmark-patches-default-upright-no-dups --method_name=siftdef_2048_geodesc-upright-no-dups
