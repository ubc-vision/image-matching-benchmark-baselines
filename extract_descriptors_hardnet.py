import torch
import numpy as np
import argparse
import h5py
from tqdm import tqdm
import os
import sys
import shutil
import json

import torchvision.transforms as transforms
from utils import cv2_greyscale, str2bool, save_h5


def get_transforms(color):

    MEAN_IMAGE = 0.443728476019
    STD_IMAGE = 0.20197947209

    transform = transforms.Compose([
        transforms.Lambda(cv2_greyscale), transforms.Lambda(cv2_scale),
        transforms.Lambda(np_reshape), transforms.ToTensor(),
        transforms.Normalize((MEAN_IMAGE, ), (STD_IMAGE, ))
    ])

    return transform


def remove_option(parser, arg):
    for action in parser._actions:
        if (vars(action)['option_strings']
            and vars(action)['option_strings'][0] == arg) \
                or vars(action)['dest'] == arg:
            parser._remove_action(action)

    for action in parser._action_groups:
        vars_action = vars(action)
        var_group_actions = vars_action['_group_actions']
        for x in var_group_actions:
            if x.dest == arg:
                var_group_actions.remove(x)
                return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        default=os.path.join('..', 'benchmark-patches-8k'),
        type=str,
        help='Path to the pre-generated patches')
    parser.add_argument(
        "--mrSize", default=12.0, type=float,
        help=' patch size in image is mrSize * pt.size. Default mrSize is 12' )
    parser.add_argument(
        "--save_path",
        default=os.path.join('..', 'benchmark-features'),
        type=str,
        help='Path to store the features')
    parser.add_argument(
        "--method_name", default='sift8k_8000_hardnet', type=str)
    parser.add_argument(
        "--weights_path",
        default=os.path.join('third_party', 'hardnet', 'pretrained',
                             'train_liberty_with_aug',
                             'checkpoint_liberty_with_aug.pth'),
        type=str,
        help='Path to the model weights')
    parser.add_argument(
        "--data_seq_json",
        default='data/val.json',
        type=str,

        help='Path to json with seq names')
    parser.add_argument(
        "--clahe-mode",
        default='None',
        type=str,
        help='can be None, detector, descriptor, both')

    args = parser.parse_args()

    seqs = []
    with open(args.data_seq_json, 'r') as f:
        seqs += json.load(f)
    print('Processing the following scenes: {}'.format(seqs))


    # Hacky work-around: reset argv for the HardNet argparse
    sys.path.append(os.path.join('third_party', 'hardnet', 'code'))
    sys.argv = [sys.argv[0]]
    from third_party.hardnet.code.HardNet import HardNet
    from third_party.hardnet.code.Utils import cv2_scale, np_reshape
    import torch
    try:
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')
    except:
        device = torch.device('cpu')

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

    if abs(args.mrSize - 12.) > 0.1:
        suffix+= '_mrSize{:.1f}'.format(args.mrSize)

    args.method_name += suffix
    print('Saving descriptors to folder: {}'.format(args.method_name))

    transforms = get_transforms(False)

    model = HardNet()
    try:
        model.load_state_dict(torch.load(args.weights_path,map_location=device)['state_dict'])
    except:
        model.load_state_dict(torch.load(args.weights_path,map_location=device))
    print('Loaded weights: {}'.format(args.weights_path))

    model = model.to(device)
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
                        [transforms(patch) for patch in data_a]).to(device)
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
