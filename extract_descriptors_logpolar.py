import torch
import numpy as np
import argparse
import h5py
from tqdm import tqdm
import os
import sys
import cv2
from utils import str2bool, save_h5
import shutil
import json


def l_clahe(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sequences_folder",
        default=os.path.join('..', 'imw-2020'),
        help="path to config file",
        type=str)
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
        "--method_name", default='sift8k_8000_logpolar96', type=str)
    parser.add_argument(
        "--config_file",
        default='third_party/log_polar_descriptors/configs/init_one_example_ptn_96.yml',
        help="path to config file",
        type=str)
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
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER)

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

    # Hacky work-around: reset argv for the HardNet argparse
    sys.path.append(os.path.join('third_party', 'log_polar_descriptors'))
    sys.argv = [sys.argv[0]]
    from third_party.log_polar_descriptors.configs.defaults import _C as cfg
    from third_party.log_polar_descriptors.modules.hardnet.models import HardNet
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
        raise ValueError(
            "unknown CLAHE mode. Try detector, descriptor or both")

    args.method_name += suffix

    print('Saving descriptors to folder: {}'.format(args.method_name))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_gpus = int(
        os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    model = HardNet(
        transform=cfg.TEST.TRANSFORMER,
        coords=cfg.TEST.COORDS,
        patch_size=cfg.TEST.IMAGE_SIZE,
        scale=cfg.TEST.SCALE,
        is_desc256=cfg.TEST.IS_DESC_256,
        orientCorrect=cfg.TEST.ORIENT_CORRECTION)

    model.load_state_dict(
        torch.load(
            os.path.join('third_party', 'log_polar_descriptors',
                         cfg.TEST.MODEL_WEIGHTS))['state_dict'])
    model.eval()
    model.to(device)

    for idx, seq_name in enumerate(seqs):

        print('Processing "{}"'.format(seq_name))

        keypoints = h5py.File(
            os.path.join(args.dataset_path, seq_name,
                         'keypoints{}.h5'.format(suffix)), 'r')
        scales = h5py.File(
            os.path.join(args.dataset_path, seq_name,
                         'scales{}.h5'.format(suffix)), 'r')
        angles = h5py.File(
            os.path.join(args.dataset_path, seq_name,
                         'angles{}.h5'.format(suffix)), 'r')

        seq_descriptors = {}
        for key, keypoints in tqdm(keypoints.items()):
            img = cv2.imread(
                os.path.join(args.sequences_folder, seq_name, key + '.jpg'))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if args.clahe_mode.lower() in ['descriptor', 'both']:
                img = l_clahe(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            # pad image and fix keypoints
            if img.shape[0] > cfg.TEST.PAD_TO or img.shape[1] > cfg.TEST.PAD_TO:
                raise RuntimeError(
                    "Image {} exceeds acceptable size".format(img.shape))

            fillHeight = cfg.TEST.PAD_TO - img.shape[0]
            fillWidth = cfg.TEST.PAD_TO - img.shape[1]

            padLeft = int(np.round(fillWidth / 2))
            padRight = int(fillWidth - padLeft)
            padUp = int(np.round(fillHeight / 2))
            padDown = int(fillHeight - padUp)

            img = np.pad(img,
                         pad_width=((padUp, padDown), (padLeft, padRight)),
                         mode='reflect')

            # Iterate over keypoints
            keypoint_locations = []
            for kpIDX, kp_loc in enumerate(keypoints):
                normKp_a = 2 * np.array([[(kp_loc[0] + padLeft) /
                                          (cfg.TEST.PAD_TO),
                                          (kp_loc[1] + padUp) /
                                          (cfg.TEST.PAD_TO)]]) - 1
                keypoint_locations.append(normKp_a)

            all_desc = []
            bs = 500
            for i in range(0, len(keypoint_locations), bs):
                oris = np.array([
                    np.deg2rad(orient) for orient in angles[key][:][i:i + bs]
                ])

                theta = [
                    torch.from_numpy(np.array(
                        keypoint_locations)[i:i + bs]).float().squeeze(),
                    torch.from_numpy(scales[key][:][i:i + bs]).float().squeeze(),
                    torch.from_numpy(oris).float()
                ]

                imgs = torch.from_numpy(img).unsqueeze(0).to(device)
                img_keypoints = [
                    theta[0].to(device), theta[1].to(device),
                    theta[2].to(device)
                ]

                # Deal with batches size 1
                if len(oris) == 1:
                    img_keypoints[0] = img_keypoints[0].unsqueeze(0)
                    img_keypoints[1] = img_keypoints[1].unsqueeze(0)
                    img_keypoints[2] = img_keypoints[2].unsqueeze(0)

                descriptors, patches = model({
                    key: imgs
                }, img_keypoints, [key] * len(img_keypoints[0]))
                all_desc.append(descriptors.data.cpu().numpy())
            seq_descriptors[key] = np.vstack(np.array(all_desc)).astype(
                np.float32)

        cur_path = os.path.join(args.save_path, args.method_name, seq_name)
        if not os.path.exists(cur_path):
            os.makedirs(cur_path)
        save_h5(seq_descriptors, os.path.join(cur_path, 'descriptors.h5'))
        sub_files_in = [
            'keypoints{}.h5'.format(suffix), 'scales{}.h5'.format(suffix),
            'angles{}.h5'.format(suffix), 'scores{}.h5'.format(suffix)
        ]
        sub_files_out = ['keypoints.h5', 'scales.h5', 'angles.h5', 'scores.h5']

        for sub_file_in, sub_file_out in zip(sub_files_in, sub_files_out):
            shutil.copyfile(
                os.path.join(args.dataset_path, seq_name, sub_file_in),
                os.path.join(cur_path, sub_file_out))

        print('Done sequence: {}'.format(seq_name))
