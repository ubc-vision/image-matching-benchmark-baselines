# Forked from https://github.com/mihaidusmanu/d2-net:extract_features.py

import argparse
import numpy as np
import imageio
import torch
import json
from tqdm import tqdm
from time import time
import os
import sys

import scipy
import scipy.io
import scipy.misc

sys.path.append(os.path.join('third_party', 'd2net'))
from lib.model_test import D2Net
from lib.utils import preprocess_image
from lib.pyramid import process_multiscale
from utils import save_h5


# Argument parsing
parser = argparse.ArgumentParser(description='Feature extraction script')

parser.add_argument(
    "--save_path",
    default=os.path.join('..', 'benchmark-features'),
    type=str,
    help='Path to store the features')

parser.add_argument(
    "--method_name", default='d2-net-singlescale_8000', type=str)

parser.add_argument(
    '--preprocessing', type=str, default='caffe',
    help='image preprocessing (caffe or torch)'
)
parser.add_argument(
    '--model_file', type=str,
    default=os.path.join(
        'third_party', 'd2net', 'models', 'd2_tf_no_phototourism.pth'),
    help='path to the full model'
)

parser.add_argument(
    '--max_edge', type=int, default=1600,
    help='maximum image size at network input'
)
parser.add_argument(
    '--max_sum_edges', type=int, default=2800,
    help='maximum sum of image sizes at network input'
)

parser.add_argument(
    '--multiscale', dest='multiscale', action='store_true',
    help='extract multiscale features'
)
parser.set_defaults(multiscale=False)

parser.add_argument(
    '--no-relu', dest='use_relu', action='store_false',
    help='remove ReLU after the dense feature extraction module'
)
parser.set_defaults(use_relu=True)

parser.add_argument(
    '--cpu', dest='cpu', action='store_true',
    help='Use CPU instead of GPU'
)

parser.add_argument(
    '--num_kp',
    type=int,
    default=0,
    help='Number of keypoints to save (0 to keep all)')

parser.set_defaults(cpu=False)

parser.add_argument(
    "--subset",
    default='both',
    type=str,
    help='Options: "val", "test", "both", "spc-fix"')

args, unparsed = parser.parse_known_args()

# Parse dataset
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

# CUDA
if args.cpu:
    print('Using CPU')
    use_cuda = False
    device = None
else:
    print('Using GPU (if available)')
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

print(args)

# Creating CNN model
model = D2Net(
    model_file=args.model_file,
    use_relu=args.use_relu,
    use_cuda=use_cuda
)

for seq in seqs:
    print('Processing "{}"'.format(seq))
    seq_keypoints = {}
    seq_scores = {}
    seq_descriptors = {}
    seq_scales = {}

    # Open the text file
    with open(os.path.join('txt', 'list-{}.txt'.format(seq)), 'r') as f:
        lines = f.readlines()

    # Process each image
    for line in tqdm(lines, total=len(lines)):
        path = line.strip()
        key = os.path.basename(os.path.splitext(path)[0])
        
        image = imageio.imread(path)
        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]
            image = np.repeat(image, 3, -1)

        # TODO: switch to PIL.Image due to deprecation of scipy.misc.imresize.
        resized_image = image
        if max(resized_image.shape) > args.max_edge:
            resized_image = scipy.misc.imresize(
                resized_image,
                args.max_edge / max(resized_image.shape)
            ).astype('float')
        if sum(resized_image.shape[: 2]) > args.max_sum_edges:
            resized_image = scipy.misc.imresize(
                resized_image,
                args.max_sum_edges / sum(resized_image.shape[: 2])
            ).astype('float')

        fact_i = image.shape[0] / resized_image.shape[0]
        fact_j = image.shape[1] / resized_image.shape[1]

        t_start = time()
        input_image = preprocess_image(
            resized_image,
            preprocessing=args.preprocessing
        )
        with torch.no_grad():
            if args.multiscale:
                keypoints, scores, descriptors = process_multiscale(
                    torch.tensor(
                        input_image[np.newaxis, :, :, :].astype(np.float32),
                        device=device
                    ),
                    model
                )
            else:
                keypoints, scores, descriptors = process_multiscale(
                    torch.tensor(
                        input_image[np.newaxis, :, :, :].astype(np.float32),
                        device=device
                    ),
                    model,
                    scales=[1]
                )
        t_end = time()

        # Input image coordinates
        keypoints[:, 0] *= fact_i
        keypoints[:, 1] *= fact_j

        # Sort the scores and subsample
        indices = np.argsort(scores)[::-1]
        if args.num_kp > 0:
            top_k = indices[:args.num_kp]
        else:
            top_k = indices

        # Flip coordinates: network provides [y, x]
        seq_keypoints[key] = np.concatenate(
                [keypoints[top_k, 1][..., None],
                keypoints[top_k, 0][..., None]],
                axis=1)
        seq_scales[key] = keypoints[top_k, 2]
        seq_scores[key] = scores[top_k]
        seq_descriptors[key] = descriptors[top_k, :]

        # print('Processed "{}" in {:.02f} sec. Found {} features'.format(
        #     key, t_end - t_start, keypoints.shape[0]))

    print('Average number of keypoints per image: {:.02f}'.format(
        np.mean([v.shape[0] for v in seq_keypoints.values()])))

    cur_path = os.path.join(args.save_path, args.method_name, seq)
    if not os.path.exists(cur_path):
        os.makedirs(cur_path)
    save_h5(seq_descriptors, os.path.join(cur_path, 'descriptors.h5'))
    save_h5(seq_keypoints, os.path.join(cur_path, 'keypoints.h5'))
    save_h5(seq_scores, os.path.join(cur_path, 'scores.h5'))
    save_h5(seq_scales, os.path.join(cur_path, 'scales.h5'))

print('Done')
