import torch
import numpy as np
import argparse
import h5py
from tqdm import tqdm
import os
import sys
import shutil
import json

from utils import cv2_greyscale, cv2_scale, np_reshape, str2bool, save_h5
import tensorflow as tf
import torchvision.transforms as transforms

sys.path.append(os.path.join('third_party', 'geodesc'))
from third_party.geodesc.utils.tf import load_frozen_model


def get_transforms():

    transform = transforms.Compose([
        transforms.Lambda(cv2_greyscale), transforms.Lambda(cv2_scale),
        transforms.Lambda(np_reshape)
    ])

    return transform


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        default=os.path.join('..', 'benchmark-patches-8k'),
        type=str,
        help='Path to the pre-generated patches')
    parser.add_argument(
        "--save_path",
        default=os.path.join('..', 'benchmark-features'),
        type=str,
        help='Path to store the features')
    parser.add_argument(
        "--method_name", default='sift8k_8000_geodesc', type=str)
    parser.add_argument(
        "--weights_path",
        default=os.path.join('third_party', 'geodesc', 'model', 'geodesc.pb'),
        type=str,
        help='Path to the model weights')
    parser.add_argument(
        "--subset",
        default='both',
        type=str,
        help='Options: "val", "test", "both", "spc-fix", "lms-fix"')
    parser.add_argument(
        "--clahe-mode",
        default='None',
        type=str,
        help='can be None, detector, descriptor, both')

    args = parser.parse_args()

    if args.subset not in ['val', 'test', 'both', 'spc-fix', 'lms-fix']:
        raise ValueError('Unknown value for --subset')
    seqs = []
    if args.subset == 'spc-fix':
        seqs += ['st_pauls_cathedral']
    elif args.subset == 'lms-fix':
        seqs += ['lincoln_memorial_statue']
    else:
        if args.subset in ['val', 'both']:
            with open(os.path.join('data', 'val.json')) as f:
                seqs += json.load(f)
        if args.subset in ['test', 'both']:
            with open(os.path.join('data', 'test.json')) as f:
                seqs += json.load(f)
    print('Processing the following scenes: {}'.format(seqs))

    suffix = ""
    if args.clahe_mode.lower() == 'detector':
        suffix = "_clahe_det"
    elif args.clahe_mode.lower() == 'descriptor':
        suffix = "_clahe_desc"
    elif args.clahe_mode.lower() == 'both':
        suffix = "_clahe_det_desc"
    elif args.clahe_mode.lower() == 'none':
        pass
    else:
        raise ValueError("unknown CLAHE mode. Try detector, descriptor or both")
    args.method_name += suffix

    print('Saving descriptors to folder: {}'.format(args.method_name))

    transforms = get_transforms()

    graph = load_frozen_model(args.weights_path, print_nodes=False)

    with tf.Session(graph=graph) as sess:
        for idx, seq_name in enumerate(seqs):
            print('Processing "{}"'.format(seq_name))

            seq_descriptors = {}
            patches_h5py_file = os.path.join(args.dataset_path, seq_name,
                                         'patches{}.h5'.format(suffix))

            with h5py.File(patches_h5py_file, 'r') as patches_h5py:
                for key, patches in tqdm(patches_h5py.items()):
                    patches = patches.value
                    bs = 128
                    descriptors = []

                    for i in range(0, len(patches), bs):
                        seq_data = patches[i:i + bs, :, :, :]
                        seq_data = np.array(
                            [transforms(patch)
                             for patch in seq_data]).squeeze(axis=3)
                        # compute output
                        processed_seq = np.zeros(
                            (len(seq_data), 32, 32), np.float32)

                        for j in range(len(seq_data)):
                            processed_seq[j] = (seq_data[j] - np.mean(
                                seq_data[j])) / (np.std(seq_data[j]) + 1e-8)

                        processed_seq = np.expand_dims(processed_seq, axis=-1)

                        descs = sess.run("squeeze_1:0",
                                         feed_dict={"input:0": processed_seq})
                        if descs.ndim == 1:
                            descs = descs[None, ...]
                        descriptors.extend(descs)

                    descriptors = np.array(descriptors)
                    seq_descriptors[key] = descriptors.astype(np.float32)

            print('Processed {} images: {} descriptors/image'.format(
                len(seq_descriptors),
                np.array([s.shape[0]
                          for s in seq_descriptors.values()]).mean()))

            cur_path = os.path.join(args.save_path, args.method_name, seq_name)
            if not os.path.exists(cur_path):
                os.makedirs(cur_path)
            save_h5(seq_descriptors, os.path.join(cur_path, 'descriptors.h5'))
            sub_files_in = ['keypoints{}.h5'.format(suffix), 'scales{}.h5'.format(suffix), 'angles{}.h5'.format(suffix), 'scores{}.h5'.format(suffix)]
            sub_files_out = ['keypoints.h5', 'scales.h5', 'angles.h5', 'scores.h5']
            for sub_file_in, sub_file_out in zip(sub_files_in, sub_files_out):
                shutil.copyfile(
                    os.path.join(args.dataset_path, seq_name, sub_file_in),
                    os.path.join(cur_path, sub_file_out))

            print('Done sequence: {}'.format(seq_name))
