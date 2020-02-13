#!/bin/bash

# logpolar
echo "Extracting Log-polar descriptors"

python extract_descriptors_logpolar.py --dataset_path=../benchmark-patches-8k --config_file=third_party/log_polar_descriptors/configs/init_one_example_ptn_96.yml --method_name=sift8k_8000_logpolar96
python extract_descriptors_logpolar.py --dataset_path=../benchmark-patches-8k-upright-no-dups --config_file=third_party/log_polar_descriptors/configs/init_one_example_ptn_96.yml --method_name=sift8k_8000_logpolar96-upright-no-dups

#python extract_descriptors_logpolar.py --dataset_path=../benchmark-patches-default --config_file=third_party/log_polar_descriptors/configs/init_one_example_ptn_96.yml --method_name=siftdef_2048_logpolar96
#python extract_descriptors_logpolar.py --dataset_path=../benchmark-patches-default-upright-no-dups --config_file=third_party/log_polar_descriptors/configs/init_one_example_ptn_96.yml --method_name=siftdef_2048_logpolar96-upright-no-dups
