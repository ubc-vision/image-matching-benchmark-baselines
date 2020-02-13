import os
import sys
from PIL import Image
import numpy as np
import torch
import argparse
from glob import glob
import h5py
import json

sys.path.append(os.path.join('third_party', 'r2d2'))
from third_party.r2d2.tools import common
from third_party.r2d2.tools.dataloader import norm_RGB
from third_party.r2d2.nets.patchnet import *
from third_party.r2d2.extract import load_network, NonMaxSuppression, extract_multiscale

# Adapted from third_party/r2d2/extract.py
if __name__ == '__main__':
    parser = argparse.ArgumentParser("Extract R2D2 features for IMW2020")

    parser.add_argument("--model", type=str, required=True, help='Model path')
    parser.add_argument(
        "--num_keypoints", type=int, default=5000, help='Number of keypoints')
    parser.add_argument("--scale-f", type=float, default=2**0.25)
    parser.add_argument("--min-size", type=int, default=256)
    parser.add_argument("--max-size", type=int, default=1024)
    parser.add_argument("--min-scale", type=float, default=0)
    parser.add_argument("--max-scale", type=float, default=1)
    parser.add_argument("--reliability-thr", type=float, default=0.7)
    parser.add_argument("--repeatability-thr", type=float, default=0.7)
    parser.add_argument(
        "--gpu", type=int, nargs='+', default=[0], help='Use -1 for CPU')
    parser.add_argument(
        "--data_path", type=str, default=os.path.join('..', 'imw-2020'))
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help='Path to store the features')
    parser.add_argument(
        "--subset",
        default='both',
        type=str,
        help='Options: "val", "test", "both"')

    args = parser.parse_args()

    seqs = []
    if args.subset not in ['val', 'test', 'both']:
        raise ValueError('Unknown value for --subset')
    if args.subset in ['val', 'both']:
        with open(os.path.join('data', 'val.json')) as f:
            seqs += json.load(f)
    if args.subset in ['test', 'both']:
        with open(os.path.join('data', 'test.json')) as f:
            seqs += json.load(f)
    print('Processing the following scenes: {}'.format(seqs))

    iscuda = common.torch_set_gpu(args.gpu)

    net = load_network(args.model)
    if iscuda:
        net = net.cuda()

    detector = NonMaxSuppression(
        rel_thr=args.reliability_thr, rep_thr=args.repeatability_thr)

    for seq in seqs:
        print('Processing scene "{}"'.format(seq))
        if not os.path.isdir('{}/{}'.format(args.save_path, seq)):
            os.makedirs('{}/{}'.format(args.save_path, seq))

        images = glob('{}/{}/*.jpg'.format(args.data_path, seq))

        num_kp = []
        with h5py.File('{}/{}/keypoints.h5'.format(args.save_path, seq), 'w') as f_kp, \
             h5py.File('{}/{}/descriptors.h5'.format(args.save_path, seq), 'w') as f_desc, \
             h5py.File('{}/{}/scores.h5'.format(args.save_path, seq), 'w') as f_score, \
             h5py.File('{}/{}/scales.h5'.format(args.save_path, seq), 'w') as f_scale:
            for fn in images:
                key = os.path.splitext(os.path.basename(fn))[0]
                img = Image.open(fn).convert('RGB')
                img = norm_RGB(img)[None]
                if iscuda:
                    img = img.cuda()

                xys, desc, scores = extract_multiscale(
                    net,
                    img,
                    detector,
                    scale_f=args.scale_f,
                    min_scale=args.min_scale,
                    max_scale=args.max_scale,
                    min_size=args.min_size,
                    max_size=args.max_size,
                    verbose=False)

                kp = xys.cpu().numpy()[:, :2]
                scales = xys.cpu().numpy()[:, 2]
                desc = desc.cpu().numpy()
                scores = scores.cpu().numpy()
                idxs = scores.argsort()[-args.num_keypoints:]

                f_kp[key] = kp[idxs]
                f_desc[key] = desc[idxs]
                f_score[key] = scores[idxs]
                f_scale[key] = scales[idxs]
                num_kp.append(len(f_kp[key]))

                print('Image "{}/{}" -> {} features'.format(
                    seq, key, num_kp[-1]))

        print('Finished processing scene "{}" -> {} features/image'.format(
            seq, np.array(num_kp).mean()))
