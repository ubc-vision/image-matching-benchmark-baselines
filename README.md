# Summary

This repository contains utilities to extract local features for the [Image Matching Benchmark](https://github.com/ubc-vision/image-matching-benchmark) and its associated challenge. For details please refer to the [website](https://image-matching-challenge.github.io).

## Data

Data can be downloaded [here](https://www.cs.ubc.ca/~kmyi/imw2020/data.html): you may want to download the images for validation and testing. Most of the scripts assume that the images are in `../imw-2020`, as follows:

```
$ ~/image-matching-benchmark-baselines $ ls ../imw-2020/
british_museum           lincoln_memorial_statue  milan_cathedral  piazza_san_marco  sacre_coeur      st_pauls_cathedral  united_states_capitol
florence_cathedral_side  london_bridge            mount_rushmore   reichstag         sagrada_familia  st_peters_square

$ ~/image-matching-benchmark-baselines $ ls ../imw-2020/british_museum/
00350405_2611802704.jpg  26237164_4796395587.jpg  45839934_4117745134.jpg  [...]
```

You may need to format the validation set in this way.

## Installation

Initialize the submodules by running the following:
```
git submodule update --init
```

We provide support for the following methods:

* [Hardnet](https://github.com/DagnyT/hardnet.git)
* [HardnetAmos](https://github.com/pultarmi/HardNet_MultiDataset)
* [GeoDesc](https://github.com/lzx551402/geodesc.git)
* [SOSNet](https://github.com/yuruntian/SOSNet.git)
* [L2Net](https://github.com/yuruntian/L2-Net)
* [Log-polar descriptor](https://github.com/DagnyT/hardnet_ptn.git)
* [Superpoint](https://github.com/MagicLeapResearch/SuperPointPretrainedNetwork)
* [D2-Net](https://github.com/mihaidusmanu/d2-net)
* [DELF](https://github.com/tensorflow/models/blob/master/research/delf/INSTALL_INSTRUCTIONS.md)
* [Contextdesc](https://github.com/lzx551402/contextdesc)
* [LFNet](https://github.com/vcg-uvic/lf-net-release)
* [R2D2](https://github.com/naver/r2d2)

We have pre-packaged conda environments: see below for details. You can install miniconda following [these instructions](https://docs.conda.io/en/latest/miniconda.html) (we have had problems with the latest version -- consider an [older one](https://repo.continuum.io/miniconda/Miniconda3-4.5.12-Linux-x86_64.sh)). You can install an environment with:
```
conda env create -f system/<environment>.yml
```

And switch between them with:
```
conda deactivate
conda activate <environment>
```

## Patch-based descriptors

### Pre-extracting patches for patch-based descriptors

Many learned descriptors require pre-generated patches. This functionality is useful by itself, so we moved it to a [separate package](https://pypi.org/project/extract-patches/). You can install it with `pip install extract_patches`: please note that this requires python 3.6, as the package is generated via nbdev). You may do do this with the `system/r2d2-python3.6.yml` environment (which also requires 3.6 due to formatted string literals) or create a different environment.

To extract patches with the default configuration to `../benchmark-patches-8k`, run:
```
python detect_sift_keypoints_and_extract_patches.py
```

This will create the following HDF5 files:
```
$ stat -c "%s %n" ../benchmark-patches-8k/british_museum/*
6414352 ../benchmark-patches-8k/british_museum/angles.h5
12789024 ../benchmark-patches-8k/british_museum/keypoints.h5
2447913728 ../benchmark-patches-8k/british_museum/patches.h5
6414352 ../benchmark-patches-8k/british_museum/scales.h5
6414352 ../benchmark-patches-8k/british_museum/scores.h5
```

You can also extract patches with a fixed orientation with the flag `--force_upright=no-dups-more-points`: this option will filter out duplicate orientations and add more points until it reaches the keypoint budget (if possible).
```
python detect_sift_keypoints_and_extract_patches.py --force_upright=no-dups-more-points --folder_outp=../benchmark-patches-8k-upright-no-dups
```

These settings generate about (up to) 8000 features per image, which requires lowering the SIFT detection threshold. If you want fewer features (~2k), you may want to use the default detection threshold, as the results are typically slightly better:
```
python detect_sift_keypoints_and_extract_patches.py --n_keypoints 2048 --folder_outp=../benchmark-patches-default --lower_sift_threshold=False
python detect_sift_keypoints_and_extract_patches.py --n_keypoints 2048 --force_upright=no-dups-more-points --folder_outp=../benchmark-patches-default-upright-no-dups --lower_sift_threshold=False
```

After this you can extract features with `run_<method>.sh`, or following the instructions below. The shell scripts use reasonable defaults: please refer to each individual wrapper for further settings (upright patches, different NMS, etc).

### Extracting descriptors from pre-generated patches 

For HardNet (environment `hardnet`):
```
python extract_descriptors_hardnet.py
```

For SOSNet (environment `hardnet`):
```
python extract_descriptors_sosnet.py
```

For L2Net (environment `hardnet`):
```
python extract_descriptors_l2net.py
```

The Log-Polar Descriptor (environment `hardnet`) requires access to the original images. For the log-polar models, use:
```
python extract_descriptors_logpolar.py --config_file=third_party/log_polar_descriptors/configs/init_one_example_ptn_96.yml --method_name=sift8k_8000_logpolar96
```

and for the cartesian models, use:
```
python extract_descriptors_logpolar.py --config_file=third_party/log_polar_descriptors/configs/init_one_example_stn_16.yml --method_name=sift8k_8000_cartesian16
```

For Geodesc (environment `geodesc`):
```
wget http://home.cse.ust.hk/~zluoag/data/geodesc.pb -O third_party/geodesc/model/geodesc.pb
python extract_descriptors_geodesc.py
```

Check the files for more options.

## End-to-end methods

### Superpoint

Use environment `hardnet`. Keypoints are sorted by score and only the top `num_kp` are kept. You can extract features with default parameters with the following:
```
python third_party/superpoint_forked/superpoint.py --cuda --num_kp=2048 --method_name=superpoint_default_2048
```

You can also lower the detection threshold to extract more features, and resize the images to a fixed size (on the largest dimension), e.g.:
```
python third_party/superpoint_forked/superpoint.py --cuda --num_kp=8000 --conf_thresh=0.0001 --nms_dist=2 --resize_image_to=1024 --num_kp=8000 --method_name=superpoint_8k_resize1024_nms2
```

### D2-Net

Use environment `hardnet`. Following D2-Net's settings, you can generate text lists of the images with:
```
python generate_image_lists.py
```
Download the weights (use this set, as the default has some overlap with out test subset):
```bash
mkdir third_party/d2net/models
wget https://dsmn.ml/files/d2-net/d2_tf_no_phototourism.pth -O third_party/d2net/models/d2_tf_no_phototourism.pth
```
You can then extract single-scale D2-Net features with:
```
python extract_d2net.py --num_kp=8000 --method_name=d2net-default_8000
```
and multi-scale D2-Net features (add the `--cpu` flag if your GPU runs out of memory) with:
```
python extract_d2net.py --num_kp=8000 --multiscale --method_name=d2net-multiscale_8000
```
(If the multi-scale variant crashes, please check [this](https://github.com/mihaidusmanu/d2-net/issues/22).)


### ContextDesc

Use environment `hardnet` and download the model weights:
```
mkdir third_party/contextdesc/pretrained
wget https://research.altizure.com/data/contextdesc_models/contextdesc_pp.tar -O third_party/contextdesc/pretrained/contextdesc_pp.tar
wget https://research.altizure.com/data/contextdesc_models/retrieval_model.tar -O third_party/contextdesc/pretrained/retrieval_model.tar
wget https://research.altizure.com/data/contextdesc_models/contextdesc_pp_upright.tar -O third_party/contextdesc/pretrained/contextdesc_pp_upright.tar
tar -C third_party/contextdesc/pretrained/ -xf third_party/contextdesc/pretrained/contextdesc_pp.tar
tar -C third_party/contextdesc/pretrained/ -xf third_party/contextdesc/pretrained/contextdesc_pp_upright.tar
tar -C third_party/contextdesc/pretrained/ -xf third_party/contextdesc/pretrained/retrieval_model.tar
rm third_party/contextdesc/pretrained/contextdesc_pp.tar
rm third_party/contextdesc/pretrained/contextdesc_pp_upright.tar
rm third_party/contextdesc/pretrained/retrieval_model.tar
```
Generate the `.yaml` file for ContextDesc:
```
python generate_yaml.py --num_keypoints=8000
```
Extract ContextDesc: 
```
python third_party/contextdesc/evaluations.py --config yaml/imw-2020.yaml
```
You may delete the `tmp` folder after extracting the features:
```
rm -rf ../benchmark-features/tmp_contextdesc
```


### DELF

You can install DELF from the tensorflow models repository, following [these instructions](https://github.com/tensorflow/models/blob/master/research/delf/INSTALL_INSTRUCTIONS.md).

You have to download the model:
```
mkdir third_party/tensorflow_models/research/delf/delf/python/examples/parameters/
wget http://storage.googleapis.com/delf/delf_gld_20190411.tar.gz -O third_party/tensorflow_models/research/delf/delf/python/examples/parameters/delf_gld_20190411.tar.gz
tar -C third_party/tensorflow_models/research/delf/delf/python/examples/parameters/ -xvf third_party/tensorflow_models/research/delf/delf/python/examples/parameters/delf_gld_20190411.tar.gz
```
and add the folder `third_party/tensorflow_models/research` to $PYTHONPATH. See `run_delf.py` for usage.


### LF-Net

Use environment `lfnet` and download the model weights:
```
mkdir third_party/lfnet/release
wget https://cs.ubc.ca/research/kmyi_data/files/2018/ono2018lfnet/lfnet-norotaug.tar.gz -O third_party/lfnet/release/lfnet-norotaug.tar.gz
tar -C third_party/lfnet/release/ -xf third_party/lfnet/release/lfnet-norotaug.tar.gz
```
Use environment 'lfnet'. Refer to extract_lfnet.py for more options. Extract LF-Net with default 2K keypoints and without resize image:
```
python extract_lfnet.py --out_dir=../benchmark-features/lfnet
```


### R2D2

Use the environment `r2d2-python-3.6` (requires 3.6 for f-strings). For options, please see the script. The authors provide three pre-trained models which can be used with:

```
python extract_r2d2.py --model=third_party/r2d2/models/r2d2_WAF_N16.pt --num_keypoints=8000 --save_path=../benchmark-features/r2d2-waf-n16-8k
python extract_r2d2.py --model=third_party/r2d2/models/r2d2_WASF_N16.pt --num_keypoints=8000 --save_path=../benchmark-features/r2d2-wasf-n16-8k
python extract_r2d2.py --model=third_party/r2d2/models/r2d2_WASF_N8_big.pt --num_keypoints=8000 --save_path=../benchmark-features/r2d2-wasf-n8-big-8k
```

## VLFeat features (via Matlab)

Matlab-based features are in a [separate repository](https://github.com/ducha-aiki/sfm-benchmark-matlab-features). You can run:
```bash
./run_vlfeat_alone.sh
./run_vlfeat_with_affnet_and_hardnet.sh
```
