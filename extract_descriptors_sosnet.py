import argparse
import torch
import numpy as np
import h5py
from tqdm import tqdm
import os
import sys
import shutil
import json

from utils import cv2_scale, cv2_greyscale, np_reshape, str2bool, save_h5
from third_party.SOSNet.codes.sosnet_model import SOSNet32x32
import torchvision.transforms as transforms


def get_transforms():

    transform = transforms.Compose([
        transforms.Lambda(cv2_greyscale), transforms.Lambda(cv2_scale),
        transforms.Lambda(np_reshape), transforms.ToTensor()
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
        "--method_name", default='sift8k_8000_sosnet', type=str)
    parser.add_argument(
        "--weights_path",
        default=os.path.join('third_party', 'SOSNet', 'sosnet-weights',
                             'sosnet-32x32-liberty.pth'),
        type=str,
        help='Path to the model weights')
    parser.add_argument(
        "--subset",
        default='both',
        type=str,
        help='Options: "val", "test", "both", "spc-fix"')
    parser.add_argument(
        "--clahe-mode",
        default='None',
        type=str,
        help='can be None, detector, descriptor, both')

    args = parser.parse_args()

    if args.subset not in ['val', 'test', 'both', 'spc-fix']:
        raise ValueError('Unknown value for --subset')
    seqs = []
    if args.subset == 'spc-fix':
        seqs += ['st_pauls_cathedral']
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

    model = SOSNet32x32()
    model.load_state_dict(torch.load(args.weights_path))
    print('Loaded weights: {}'.format(args.weights_path))

    model.cuda()
    model.eval()

    for idx, seq_name in enumerate(seqs):
        print('Processing "{}"'.format(seq_name))

        seq_descriptors = {}
        patches_h5py_file = os.path.join(args.dataset_path, seq_name,
                                         'patches{}.h5'.format(suffix))

        with h5py.File(patches_h5py_file, 'r') as patches_h5py:
            for key, patches in tqdm(patches_h5py.items()):
                patches = patches.value
                bs = 128
                descriptors = np.zeros((len(patches), 128))

                for i in range(0, len(patches), bs):
                    data_a = patches[i:i + bs, :, :, :]
                    data_a = torch.stack(
                        [transforms(patch) for patch in data_a]).cuda()
                    # compute output
                    with torch.no_grad():
                        out_a = model(data_a)
                        descriptors[i:i + bs] = out_a.cpu().detach().numpy()

                seq_descriptors[key] = descriptors.astype(np.float32)

        print('Processed {} images: {} descriptors/image'.format(
            len(seq_descriptors),
            np.array([s.shape[0] for s in seq_descriptors.values()]).mean()))

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
