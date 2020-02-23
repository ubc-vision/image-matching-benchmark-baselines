# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2018
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Daniel DeTone (ddetone)
#                       Tomasz Malisiewicz (tmalisiewicz)
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

# Code forked by E. Trulls from:
# https://github.com/MagicLeapResearch/SuperPointPretrainedNetwork
# with minimal changes. Includes options to resize images to a fixed size.

import argparse
import glob
import numpy as np
import os
from time import time
import json

import cv2
import torch

import h5py
from IPython import embed
from imageio import imread, imwrite

# Stub to warn about opencv version.
if int(cv2.__version__[0]) < 3:  # pragma: no cover
    print('Warning: OpenCV 3 is not installed')

# Jet colormap for visualization.
myjet = np.array(
    [[0., 0., 0.5], [0., 0., 0.99910873], [0., 0.37843137, 1.],
     [0., 0.83333333, 1.], [0.30044276, 1., 0.66729918],
     [0.66729918, 1., 0.30044276], [1., 0.90123457, 0.], [1., 0.48002905, 0.],
     [0.99910873, 0.07334786, 0.], [0.5, 0., 0.]])


def save_h5(dict_to_save, filename):
    """Saves dictionary to hdf5 file"""

    with h5py.File(filename, 'w') as f:
        for key in dict_to_save:
            f.create_dataset(key, data=dict_to_save[key])


class SuperPointNet(torch.nn.Module):
    """ Pytorch definition of SuperPoint Network. """

    def __init__(self):
        super(SuperPointNet, self).__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
        # Shared Encoder.
        self.conv1a = torch.nn.Conv2d(
            1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = torch.nn.Conv2d(
            c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = torch.nn.Conv2d(
            c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = torch.nn.Conv2d(
            c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = torch.nn.Conv2d(
            c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = torch.nn.Conv2d(
            c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = torch.nn.Conv2d(
            c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = torch.nn.Conv2d(
            c4, c4, kernel_size=3, stride=1, padding=1)
        # Detector Head.
        self.convPa = torch.nn.Conv2d(
            c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = torch.nn.Conv2d(
            c5, 65, kernel_size=1, stride=1, padding=0)
        # Descriptor Head.
        self.convDa = torch.nn.Conv2d(
            c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = torch.nn.Conv2d(
            c5, d1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """ Forward pass that jointly computes unprocessed point and descriptor
    tensors.
    Input
      x: Image pytorch tensor shaped N x 1 x H x W.
    Output
      semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
      desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
    """
        # Shared Encoder.
        x = self.relu(self.conv1a(x))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        # Detector Head.
        cPa = self.relu(self.convPa(x))
        semi = self.convPb(cPa)
        # Descriptor Head.
        cDa = self.relu(self.convDa(x))
        desc = self.convDb(cDa)
        dn = torch.norm(desc, p=2, dim=1)  # Compute the norm.
        desc = desc.div(torch.unsqueeze(dn, 1))  # Divide by norm to normalize.
        return semi, desc


class SuperPointFrontend(object):
    """ Wrapper around pytorch net to help with pre and post image processing. """

    def __init__(self,
                 weights_path,
                 nms_dist,
                 conf_thresh,
                 nn_thresh,
                 cuda=False):
        self.name = 'SuperPoint'
        self.cuda = cuda
        self.nms_dist = nms_dist
        self.conf_thresh = conf_thresh
        self.nn_thresh = nn_thresh  # L2 descriptor distance for good match.
        self.cell = 8  # Size of each output cell. Keep this fixed.
        self.border_remove = 4  # Remove points this close to the border.

        # Load the network in inference mode.
        self.net = SuperPointNet()
        if cuda:
            # Train on GPU, deploy on GPU.
            self.net.load_state_dict(torch.load(weights_path))
            self.net = self.net.cuda()
        else:
            # Train on GPU, deploy on CPU.
            self.net.load_state_dict(
                torch.load(
                    weights_path, map_location=lambda storage, loc: storage))
        self.net.eval()

    def nms_fast(self, in_corners, H, W, dist_thresh):
        """
    Run a faster approximate Non-Max-Suppression on numpy corners shaped:
      3xN [x_i,y_i,conf_i]^T

    Algo summary: Create a grid sized HxW. Assign each corner location a 1, rest
    are zeros. Iterate through all the 1's and convert them either to -1 or 0.
    Suppress points by setting nearby values to 0.

    Grid Value Legend:
    -1 : Kept.
     0 : Empty or suppressed.
     1 : To be processed (converted to either kept or supressed).

    NOTE: The NMS first rounds points to integers, so NMS distance might not
    be exactly dist_thresh. It also assumes points are within image boundaries.

    Inputs
      in_corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
      H - Image height.
      W - Image width.
      dist_thresh - Distance to suppress, measured as an infinty norm distance.
    Returns
      nmsed_corners - 3xN numpy matrix with surviving corners.
      nmsed_inds - N length numpy vector with surviving corner indices.
    """
        grid = np.zeros((H, W)).astype(int)  # Track NMS data.
        inds = np.zeros((H, W)).astype(int)  # Store indices of points.
        # Sort by confidence and round to nearest int.
        inds1 = np.argsort(-in_corners[2, :])
        corners = in_corners[:, inds1]
        rcorners = corners[:2, :].round().astype(int)  # Rounded corners.
        # Check for edge case of 0 or 1 corners.
        if rcorners.shape[1] == 0:
            return np.zeros((3, 0)).astype(int), np.zeros(0).astype(int)
        if rcorners.shape[1] == 1:
            out = np.vstack((rcorners, in_corners[2])).reshape(3, 1)
            return out, np.zeros((1)).astype(int)
        # Initialize the grid.
        for i, rc in enumerate(rcorners.T):
            grid[rcorners[1, i], rcorners[0, i]] = 1
            inds[rcorners[1, i], rcorners[0, i]] = i
        # Pad the border of the grid, so that we can NMS points near the border.
        pad = dist_thresh
        grid = np.pad(grid, ((pad, pad), (pad, pad)), mode='constant')
        # Iterate through points, highest to lowest conf, suppress neighborhood.
        count = 0
        for i, rc in enumerate(rcorners.T):
            # Account for top and left padding.
            pt = (rc[0] + pad, rc[1] + pad)
            if grid[pt[1], pt[0]] == 1:  # If not yet suppressed.
                grid[pt[1] - pad:pt[1] + pad + 1, pt[0] - pad:pt[0] + pad +
                     1] = 0
                grid[pt[1], pt[0]] = -1
                count += 1
        # Get all surviving -1's and return sorted array of remaining corners.
        keepy, keepx = np.where(grid == -1)
        keepy, keepx = keepy - pad, keepx - pad
        inds_keep = inds[keepy, keepx]
        out = corners[:, inds_keep]
        values = out[-1, :]
        inds2 = np.argsort(-values)
        out = out[:, inds2]
        out_inds = inds1[inds_keep[inds2]]
        return out, out_inds

    def run(self, img):
        """ Process a numpy image to extract points and descriptors.
    Input
      img - HxW numpy float32 input image in range [0,1].
    Output
      corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
      desc - 256xN numpy array of corresponding unit normalized descriptors.
      heatmap - HxW numpy heatmap in range [0,1] of point confidences.
      """
        assert img.ndim == 2, 'Image must be grayscale.'
        assert img.dtype == np.float32, 'Image must be float32.'
        H, W = img.shape[0], img.shape[1]
        inp = img.copy()
        inp = (inp.reshape(1, H, W))
        inp = torch.from_numpy(inp)
        inp = torch.autograd.Variable(inp).view(1, 1, H, W)
        if self.cuda:
            inp = inp.cuda()
        # Forward pass of network.
        outs = self.net.forward(inp)
        semi, coarse_desc = outs[0], outs[1]
        # Convert pytorch -> numpy.
        semi = semi.data.cpu().numpy().squeeze()
        # --- Process points.
        dense = np.exp(semi)  # Softmax.
        dense = dense / (np.sum(dense, axis=0) + .00001)  # Should sum to 1.
        # Remove dustbin.
        nodust = dense[:-1, :, :]
        # Reshape to get full resolution heatmap.
        Hc = int(H / self.cell)
        Wc = int(W / self.cell)
        nodust = nodust.transpose(1, 2, 0)
        heatmap = np.reshape(nodust, [Hc, Wc, self.cell, self.cell])
        heatmap = np.transpose(heatmap, [0, 2, 1, 3])
        heatmap = np.reshape(heatmap, [Hc * self.cell, Wc * self.cell])
        xs, ys = np.where(heatmap >= self.conf_thresh)  # Confidence threshold.
        if len(xs) == 0:
            return np.zeros((3, 0)), None, None
        pts = np.zeros((3, len(xs)))  # Populate point data sized 3xN.
        pts[0, :] = ys
        pts[1, :] = xs
        pts[2, :] = heatmap[xs, ys]
        pts, _ = self.nms_fast(
            pts, H, W, dist_thresh=self.nms_dist)  # Apply NMS.
        inds = np.argsort(pts[2, :])
        pts = pts[:, inds[::-1]]  # Sort by confidence.
        # Remove points along border.
        bord = self.border_remove
        toremoveW = np.logical_or(pts[0, :] < bord, pts[0, :] >= (W - bord))
        toremoveH = np.logical_or(pts[1, :] < bord, pts[1, :] >= (H - bord))
        toremove = np.logical_or(toremoveW, toremoveH)
        pts = pts[:, ~toremove]
        # --- Process descriptor.
        D = coarse_desc.shape[1]
        if pts.shape[1] == 0:
            desc = np.zeros((D, 0))
        else:
            # Interpolate into descriptor map using 2D point locations.
            samp_pts = torch.from_numpy(pts[:2, :].copy())
            samp_pts[0, :] = (samp_pts[0, :] / (float(W) / 2.)) - 1.
            samp_pts[1, :] = (samp_pts[1, :] / (float(H) / 2.)) - 1.
            samp_pts = samp_pts.transpose(0, 1).contiguous()
            samp_pts = samp_pts.view(1, 1, -1, 2)
            samp_pts = samp_pts.float()
            if self.cuda:
                samp_pts = samp_pts.cuda()
            desc = torch.nn.functional.grid_sample(coarse_desc, samp_pts)
            desc = desc.data.cpu().numpy().reshape(D, -1)
            desc /= np.linalg.norm(desc, axis=0)[np.newaxis, :]
        return pts, desc, heatmap


class VideoStreamer(object):
    """ Class to help process image streams. Three types of possible inputs:"
    1.) USB Webcam.
    2.) A directory of images (files in directory matching 'img_glob').
    3.) A video file, such as an .mp4 or .avi file.
    """

    def __init__(self, basedir, camid, height, width, skip, img_glob,
                 resize_to):
        self.cap = []
        self.camera = False
        self.video_file = False
        self.listing = []
        self.sizer = [height, width]
        self.resize_to = resize_to
        self.i = 0
        self.skip = skip
        self.maxlen = 1000000

        # If the "basedir" string is the word camera, then use a webcam.
        if basedir == "camera/" or basedir == "camera":
            print('==> Processing Webcam Input.')
            self.cap = cv2.VideoCapture(camid)
            self.listing = range(0, self.maxlen)
            self.camera = True
        else:
            # Try to open as a video.
            self.cap = cv2.VideoCapture(basedir)
            lastbit = basedir[-4:len(basedir)]
            if (type(self.cap) == list or
                    not self.cap.isOpened()) and (lastbit == '.mp4'):
                raise IOError('Cannot open movie file')
            elif type(self.cap) != list and self.cap.isOpened() and (
                    lastbit != '.txt'):
                print('==> Processing Video Input.')
                num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.listing = range(0, num_frames)
                self.listing = self.listing[::self.skip]
                self.camera = True
                self.video_file = True
                self.maxlen = len(self.listing)
            else:
                print('==> Processing Image Directory Input.')
                search = os.path.join(basedir, img_glob)
                self.listing = glob.glob(search)
                self.listing.sort()
                self.listing = self.listing[::self.skip]
                self.maxlen = len(self.listing)
                if self.maxlen == 0:
                    raise IOError(
                        'No images were found (maybe bad \'--img_glob\' parameter?)'
                    )

    def read_image(self, impath, img_size):
        """ Read image as grayscale and resize to img_size.
    Inputs
      impath: Path to input image.
      img_size: (W, H) tuple specifying resize size.
    Returns
      grayim: float32 numpy array sized H x W with values in range [0, 1].
    """
        grayim = cv2.imread(impath, 0)
        if grayim is None:
            raise Exception('Error reading image %s' % impath)

        # Image is resized with opencv
        interp = cv2.INTER_CUBIC
        if img_size == 0:
            factor = [1, 1]
        elif isinstance(img_size, (list, tuple)):
            factor = (grayim.shape[1] / img_size[1],
                      grayim.shape[0] / img_size[0])
            grayim = cv2.resize(
                grayim, (img_size[1], img_size[0]), interpolation=interp)
        else:
            [h, w] = grayim.shape
            if h > w:
                new_w, new_h = int(w * img_size / h), int(img_size)
            else:
                new_w, new_h = int(img_size), int(h * img_size / w)
            grayim = cv2.resize(grayim, (new_w, new_h), interpolation=interp)
            factor = (w / new_w, h / new_h)
        grayim = (grayim.astype('float32') / 255.)
        return grayim, factor

    def next_frame(self):
        """ Return the next frame, and increment internal counter.
    Returns
       image: Next H x W image.
       status: True or False depending whether image was loaded.
    """
        if self.i == self.maxlen:
            return (None, False, False)
        if self.camera:
            raise RuntimeError("Deprecated")
            ret, input_image = self.cap.read()
            if ret is False:
                print(
                    'VideoStreamer: Cannot get image from camera (maybe bad --camid?)'
                )
                return (None, False, False)
            if self.video_file:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.listing[self.i])
            input_image = cv2.resize(
                input_image, (self.sizer[1], self.sizer[0]),
                interpolation=cv2.INTER_AREA)
            input_image = cv2.resize(
                input_image, (self.sizer[1], self.sizer[0]),
                interpolation=cv2.inter_area)
            input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
            input_image = input_image.astype('float') / 255.0
        else:
            image_file = self.listing[self.i]
            # input_image = self.read_image(image_file, self.sizer)
            input_image, factor = self.read_image(image_file, self.resize_to)
        # Increment internal counter.
        self.i = self.i + 1
        input_image = input_image.astype('float32')
        return (input_image, True, factor)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sequences_folder",
        default=os.path.join('..', 'imw-2020'),
        help="path to config file",
        type=str)
    parser.add_argument(
        '--img_glob',
        type=str,
        default='*.jpg',
        help='Glob match if directory of images is specified '
        '(default: \'*.jpg\').')
    parser.add_argument(
        '--nms_dist',
        type=int,
        default=4,
        help='Non Maximum Suppression (NMS) distance (default: 4).')
    parser.add_argument(
        '--conf_thresh',
        type=float,
        default=0.015,
        help='Detector confidence threshold (default: 0.015).')
    parser.add_argument(
        '--resize_image_to',
        type=float,
        default=0,
        help='Resize the largest image dimension to this value (default: 0, '
        'does nothing).')
    parser.add_argument(
        '--cuda',
        action='store_true',
        help='Use cuda GPU to speed up network processing speed (default: False)'
    )
    parser.add_argument(
        "--save_path",
        default=os.path.join('..', 'benchmark-features'),
        type=str,
        help='Path to store the features')
    parser.add_argument(
        "--method_name", default='superpoint_default', type=str)
    parser.add_argument(
        '--num_kp',
        type=int,
        default=0,
        help='Number of keypoints to save (0 to keep all)')
    parser.add_argument(
        "--subset",
        default='both',
        type=str,
        help='Options: "val", "test", "both", "spc-fix"')

    opt, unparsed = parser.parse_known_args()
    print(opt)

    if opt.subset not in ['val', 'test', 'both', 'spc-fix']:
        raise ValueError('Unknown value for --subset')
    seqs = []
    if opt.subset == 'spc-fix':
        seqs += ['st_pauls_cathedral']
    else:
        if opt.subset in ['val', 'both']:
            with open(os.path.join('data', 'val.json')) as f:
                seqs += json.load(f)
        if opt.subset in ['test', 'both']:
            with open(os.path.join('data', 'test.json')) as f:
                seqs += json.load(f)
    print('Processing the following scenes: {}'.format(seqs))

    print('Saving descriptors to folder: {}'.format(opt.method_name))

    print('==> Loading pre-trained network.')
    # This class runs the SuperPoint network and processes its outputs.
    fe = SuperPointFrontend(
        weights_path=os.path.join('third_party', 'superpoint_ml_repo',
                                  'superpoint_v1.pth'),
        nms_dist=opt.nms_dist,
        conf_thresh=opt.conf_thresh,
        nn_thresh=0.7,
        cuda=opt.cuda)
    print('==> Successfully loaded pre-trained network.')

    # Font parameters for visualizaton.
    font = cv2.FONT_HERSHEY_DUPLEX
    font_clr = (255, 255, 255)
    font_pt = (4, 12)
    font_sc = 0.4

    for seq in seqs:
        print('Processing "{}"'.format(seq))
        count = 0
        vs = VideoStreamer(
            os.path.join(opt.sequences_folder, seq), None, None, None, 1,
            opt.img_glob, opt.resize_image_to)
        start = time()

        start1 = time()
        seq_keypoints = {}
        seq_descriptors = {}
        seq_scores = {}
        while True:
            # Get a new image.
            img, status, factor = vs.next_frame()
            if status is False:
                print('Failed or over!')
                break

            key = vs.listing[count].split("/")[-1].split('.')[0]

            # Compute superpoint
            # Points are ordered: [x, y]
            pts, desc, heatmap = fe.run(img)
            print('Processing image "{}": found {} points'.format(
                key, pts.shape[-1]))

            # Undo the scaling
            pts = np.array([[(p[0] + .5) * factor[0] - .5,
                             (p[1] + .5) * factor[1] - .5, p[2]]
                            for p in pts.T])

            # Scores should be sorted
            assert(all(pts[:-1, 2] - pts[1:, 2] >= 0))

            # Crop
            if opt.num_kp > 0:
                pts = pts[:opt.num_kp, :]
                desc = desc.T[:opt.num_kp, :]

            seq_keypoints[key] = pts[:, :2]
            seq_descriptors[key] = desc
            seq_scores[key] = pts[:, 2]

            # draw image
            # im = imread(vs.listing[count])
            # for _p in pts:
            #     cv2.circle(im, (int(_p[0]), int(_p[1])), 2, [255, 255, 0])
            # imwrite('{}/{}/{}-debug.jpg'.format(opt.write_dir, seq, key),
            #         im)

            count += 1

        end1 = time()
        print('Processed "{}" ({} images) in {:0.02f} sec.'.format(
            seq, count, end1 - start1))
        print('Average number of keypoints: {}'.format(
            np.mean([v.shape[0] for v in seq_keypoints.values()])))

        cur_path = os.path.join(opt.save_path, opt.method_name, seq)
        if not os.path.exists(cur_path):
            os.makedirs(cur_path)
        save_h5(seq_descriptors, os.path.join(cur_path, 'descriptors.h5'))
        save_h5(seq_keypoints, os.path.join(cur_path, 'keypoints.h5'))
        save_h5(seq_scores, os.path.join(cur_path, 'scores.h5'))

    print('Done!')
