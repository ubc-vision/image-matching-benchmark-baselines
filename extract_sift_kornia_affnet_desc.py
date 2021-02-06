import  os
import sys
from PIL import Image
import numpy as np
import torch.nn as nn
import torch
import argparse
from glob import glob
import h5py
import json
from torch import tensor
sys.path.append(os.path.join('third_party', 'r2d2'))
import kornia as K
import kornia.feature as KF
from kornia_moons.feature import *
import cv2
def get_local_descriptors(img, cv2_sift_kpts, kornia_descriptor, aff):
  #We will not train anything, so let's save time and memory by no_grad()
  with torch.no_grad():
    timg = K.color.rgb_to_grayscale(K.image_to_tensor(img, False))/255.
    timg = timg.cuda()
    lafs = laf_from_opencv_SIFT_kpts(cv2_sift_kpts).cuda()
    angles = KF.laf.get_laf_orientation(lafs)
    # We will estimate affine shape of the feature and re-orient the keypoints with the OriNet
    lafs_new = aff(lafs,timg)
    patches = KF.extract_patches_from_pyramid(timg,lafs_new, 32)
    B, N, CH, H, W = patches.size()
    # Descriptor accepts standard tensor [B, CH, H, W], while patches are [B, N, CH, H, W] shape
    # So we need to reshape a bit :) 
    descs = kornia_descriptor(patches.view(B * N, CH, H, W)).view(B * N, -1)
  return descs.detach().cpu().numpy()

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Extract Kornia handcrafted features for IMW2020")

    parser.add_argument(
        "--num_keypoints", type=int, default=8000, help='Number of keypoints')
    parser.add_argument("--mrsize", type=float, default=6.0)
    parser.add_argument("--patchsize", type=float, default=32)
    parser.add_argument("--upright", type=bool, default=True)
    parser.add_argument("--affine", type=bool, default=True)
    parser.add_argument("--descriptor", type=str, default='HardNet', help='hardnet, sift, rootsift, tfeat, sosnet')
    parser.add_argument(
        "--data_path", type=str, default=os.path.join('..', 'data'))
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
    PS = args.patchsize
    device = torch.device('cpu')
    try:
        if torch.cuda.is_available():
            device = torch.device('cuda')
    except:
        print ('CPU mode')
    sift = cv2.SIFT_create(
            contrastThreshold=-10000, edgeThreshold=-10000)
    if args.descriptor.lower() == 'sift':
        descriptor = KF.SIFTDescriptor(PS, rootsift=False)
    elif args.descriptor.lower() == 'rootsift':
        descriptor = KF.SIFTDescriptor(PS, rootsift=True)
    elif args.descriptor.lower() == 'hardnet':
        PS = 32
        descriptor = KF.HardNet(True)
    elif args.descriptor.lower() == 'sosnet':
        PS = 32
        descriptor = KF.SOSNet(True)
    elif args.descriptor.lower() == 'tfeat':
        PS = 32
        descriptor = KF.TFeat(True)
    else:
        raise ValueError('Unknown descriptor')
    descriptor = descriptor.to(device)
    print (device)
    descriptor.eval()
    aff_est = KF.LAFAffNetShapeEstimator(True).to(device)
    aff_est.eval()
    from tqdm import tqdm
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
    NUM_KP = args.num_keypoints
    for seq in seqs:
        print('Processing scene "{}"'.format(seq))
        if not os.path.isdir('{}/{}'.format(args.save_path, seq)):
            os.makedirs('{}/{}'.format(args.save_path, seq))
        images = glob('{}/{}/set_100/images/*.jpg'.format(args.data_path, seq))
        num_kp = []
        with h5py.File('{}/{}/keypoints.h5'.format(args.save_path, seq), 'w') as f_kp, \
             h5py.File('{}/{}/descriptors.h5'.format(args.save_path, seq), 'w') as f_desc, \
             h5py.File('{}/{}/scores.h5'.format(args.save_path, seq), 'w') as f_score, \
             h5py.File('{}/{}/angles.h5'.format(args.save_path, seq), 'w') as f_ang, \
             h5py.File('{}/{}/scales.h5'.format(args.save_path, seq), 'w') as f_scale:
            for fn in tqdm(images):
                key = os.path.splitext(os.path.basename(fn))[0]
                im = cv2.cvtColor(cv2.imread(fn), cv2.COLOR_BGR2RGB)
                pts, size, angle, response = get_SIFT_keypoints(sift, im)
                if args.upright:
                    kpts = [
                        cv2.KeyPoint(
                            x=pt[0],
                            y=pt[1],
                            _size=size[i],
                            _angle=0, _response=response[i])  for i, pt in enumerate(pts) if (pt not in pts[:i]) ]
                    kpts = kpts[:NUM_KP]
                else:
                    kpts = [
                        cv2.KeyPoint(
                            x=pt[0],
                            y=pt[1],
                            _size=size[i],
                            _angle=angle[i], _response=response[i])  for i, pt in enumerate(pts)  ]
                    kpts = kpts[:NUM_KP]
                with torch.no_grad():
                   descs = get_local_descriptors(im, kpts, descriptor, aff_est) 
                keypoints = np.array([(x.pt[0], x.pt[1]) for x in kpts ]).reshape(-1, 2)
                scales = np.array([12.0* x.size for x in kpts ]).reshape(-1, 1)
                angles = np.array([x.angle for x in kpts ]).reshape(-1, 1)
                responses = np.array([x.response for x in kpts ]).reshape(-1, 1)
                f_kp[key] = keypoints
                f_desc[key] = descs
                f_score[key] = responses
                f_ang[key] = angles
                f_scale[key] = scales
                num_kp.append(len(keypoints))
        print('Finished processing scene "{}" -> {} features/image'.format(
            seq, np.array(num_kp).mean()))
