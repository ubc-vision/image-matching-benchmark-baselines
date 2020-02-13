import numpy as np
import os
import cv2
from tqdm import tqdm
import argparse
import json

from extract_patches.core import extract_patches
from utils import save_h5


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False


def l_clahe(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)


def get_SIFT_keypoints(sift, img, lower_detection_th=False):

    # convert to gray-scale and compute SIFT keypoints
    keypoints = sift.detect(img, None)

    response = np.array([kp.response for kp in keypoints])
    respSort = np.argsort(response)[::-1]

    pt = np.array([kp.pt for kp in keypoints])[respSort]
    size = np.array([kp.size for kp in keypoints])[respSort]
    angle = np.array([kp.angle for kp in keypoints])[respSort]
    response = np.array([kp.response for kp in keypoints])[respSort]

    return pt, size, angle, response


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--scenes_folder",
        default=os.path.join('..', 'imw-2020'),
        help="path to config file",
        type=str)
    parser.add_argument(
        "--folder_outp",
        default=os.path.join('..', 'benchmark-patches-8k'),
        type=str)
    parser.add_argument(
        "--mrSize",
        default=12.0,
        type=float,
        help=' patch size in image is mrSize * pt.size. Default mrSize is 12')
    parser.add_argument(
        "--lower_sift_threshold",
        default='True',
        type=str2bool,
        help='Lower detection threshold (useful to extract 8k features)')
    parser.add_argument(
        "--clahe-mode",
        default='None',
        type=str,
        help='can be None, detector, descriptor, both')
    parser.add_argument(
        "--subset",
        default='both',
        type=str,
        help='Options: "val", "test", "both", "spc-fix"')
    parser.add_argument(
        "--force_upright",
        default='off',
        type=str,
        help='Options: "off", "no-dups", "no-dups-more-points"')
    parser.add_argument("--n_keypoints", default=8000, type=int)

    args = parser.parse_args()

    if args.subset not in ['val', 'test', 'both', 'spc-fix']:
        raise ValueError('Unknown value for --subset')

    if args.lower_sift_threshold:
        print('Instantiating SIFT detector with a lower detection threshold')
        sift = cv2.xfeatures2d.SIFT_create(
            contrastThreshold=-10000, edgeThreshold=-10000)
    else:
        print('Instantiating SIFT detector with default values')
        sift = cv2.xfeatures2d.SIFT_create()

    if not os.path.isdir(args.folder_outp):
        os.makedirs(args.folder_outp)

    scenes = []
    if args.subset == 'spc-fix':
        scenes += ['st_pauls_cathedral']
    else:
        if args.subset in ['val', 'both']:
            with open(os.path.join('data', 'val.json')) as f:
                scenes += json.load(f)
        if args.subset in ['test', 'both']:
            with open(os.path.join('data', 'test.json')) as f:
                scenes += json.load(f)
    print('Processing the following scenes: {}'.format(scenes))

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

    assert(args.mrSize > 0)
    if abs(args.mrSize - 12.) > 0.1:
        suffix += '_mrSize{:.1f}'.format(args.mrSize)

    for scene in scenes:
        print('Processing "{}"'.format(scene))
        scene_patches, scene_kp, scene_loc, scene_scale, \
                sec_ori, sec_resp = {}, {}, {}, {}, {}, {}

        scene_path = os.path.join(args.scenes_folder,
                                     scene)
        num_patches = []
        img_list = [x for x in os.listdir(scene_path) if x.endswith('.jpg')]
        for im_path in tqdm(img_list):
            img_name = im_path.replace('.jpg', '')
            im = cv2.cvtColor(
                cv2.imread(os.path.join(scene_path, im_path)),
                cv2.COLOR_BGR2RGB)
            if args.clahe_mode.lower() in ['detector', 'both']:
                img_gray = cv2.cvtColor(l_clahe(im), cv2.COLOR_RGB2GRAY)
            else:
                img_gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

            keypoints, scales, angles, responses = get_SIFT_keypoints(sift,
                                                                      img_gray)

            if args.force_upright == 'off':
                # Nothing to do
                kpts = [
                    cv2.KeyPoint(
                        x=point[0],
                        y=point[1],
                        _size=scales[i],
                        _angle=angles[i]) for i, point in enumerate(keypoints)
                ]
            elif args.force_upright == 'no-dups':
                # Set orientation to zero, remove duplicates later
                # This is a subset of the previous set
                kpts = [
                    cv2.KeyPoint(
                        x=keypoints[i][0],
                        y=keypoints[i][1],
                        _size=scales[i],
                        _angle=0) for i, point in enumerate(keypoints)
                ]
            elif args.force_upright == 'no-dups-more-points':
                # Copy without duplicates, set orientation to zero
                # The cropped list may contain new points
                kpts = [
                    cv2.KeyPoint(
                        x=keypoints[i][0],
                        y=keypoints[i][1],
                        _size=scales[i],
                        _angle=0) for i, point in enumerate(keypoints)
                    if point not in keypoints[:i]
                ]
            else:
                raise ValueError('Unknown --force_upright setting')

            # apply CLAHE
            im = cv2.cvtColor(
                cv2.imread(os.path.join(scene_path, im_path)),
                cv2.COLOR_BGR2RGB)
            if args.clahe_mode.lower() in ['descriptor', 'both']:
                im = l_clahe(im)

            # Extract patches
            patches = extract_patches(
                kpts, im, 32, args.mrSize)
            keypoints = np.array([(x.pt[0], x.pt[1]) for x in kpts ]).reshape(-1, 2)
            scales = np.array([args.mrSize * x.size for x in kpts ]).reshape(-1, 1)
            angles = np.array([x.angle for x in kpts ]).reshape(-1, 1)
            responses = np.array([x.response for x in kpts ]).reshape(-1, 1)

            # Crop
            patches = np.array(patches)[:args.n_keypoints].astype(np.uint8)
            keypoints = keypoints[:args.n_keypoints]
            scales = scales[:args.n_keypoints]
            angles = angles[:args.n_keypoints]
            responses = responses[:args.n_keypoints]

            # Remove duplicates after cropping
            if args.force_upright == 'no-dups':
                _, unique = np.unique(keypoints, axis=0, return_index=True)
                patches = patches[unique]
                keypoints = keypoints[unique]
                scales = scales[unique]
                angles = angles[unique]
                responses = responses[unique]

            # Patches are already uint8
            num_patches.append(patches.shape[0])

            scene_patches[img_name] = patches
            scene_kp[img_name] = keypoints
            scene_scale[img_name] = scales
            sec_ori[img_name] = angles
            sec_resp[img_name] = responses

        print('Processed {} images: {} patches/image'.format(
            len(num_patches), np.array(num_patches).mean()))

        cur_path = os.path.join(args.folder_outp, scene)
        # if args.force_upright == 'no-dups':
        #     cur_path += '_upright_v1'
        # elif args.force_upright == 'no-dups-more-points':
        #     cur_path += '_upright_v2'
        if not os.path.isdir(cur_path):
            os.makedirs(cur_path)

        save_h5(scene_patches,
                os.path.join(cur_path, 'patches{}.h5'.format(suffix)))
        save_h5(scene_kp, os.path.join(cur_path,
                                     'keypoints{}.h5'.format(suffix)))
        save_h5(scene_scale, os.path.join(cur_path,
                                        'scales{}.h5'.format(suffix)))
        save_h5(sec_ori, os.path.join(cur_path, 'angles{}.h5'.format(suffix)))
        save_h5(sec_resp, os.path.join(cur_path, 'scores{}.h5'.format(suffix)))

    print('Done!')
