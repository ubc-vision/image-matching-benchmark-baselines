from __future__ import print_function
import os
import sys
from PIL import Image
import numpy as np
import argparse
import glob
import h5py
import json
import tensorflow as tf
import importlib
import time
import cv2
from tqdm import tqdm
import pickle

sys.path.insert(0,os.path.join('third_party','lfnet'))

from mydatasets import *

from det_tools import *
from eval_tools import draw_keypoints
from common.tf_train_utils import get_optimizer
from common.argparse_utils import *
from imageio import imread, imsave
from inference import *
from run_lfnet import build_networks

MODEL_PATH = './third_party/lfnet/models'
if MODEL_PATH not in sys.path:
    sys.path.append(MODEL_PATH)

# Adapted from third_party/r2d2/extract.py
if __name__ == '__main__':
    parser = get_parser()

    general_arg = add_argument_group('General', parser)
    general_arg.add_argument('--num_threads', type=int, default=8,
                            help='the number of threads (for dataset)')
    general_arg.add_argument(
            "--subset",
            default='both',
            type=str,
            help='Options: "val", "test", "both"')

    io_arg = add_argument_group('In/Out', parser)
    io_arg.add_argument('--in_dir', type=str, default=os.path.join('..', 'imw-2020'),
                            help='input image directory')
    # io_arg.add_argument('--in_dir', type=str, default='./release/outdoor_examples/images/sacre_coeur/dense/images',
    #                         help='input image directory')
    io_arg.add_argument('--out_dir', type=str, required=True,
                            help='where to save keypoints')

    model_arg = add_argument_group('Model', parser)
    model_arg.add_argument('--model', type=str, default='./third_party/lfnet/release/lfnet-norotaug/',
                            help='model file or directory')
    model_arg.add_argument('--top_k', type=int, default=2000,
                            help='number of keypoints')
    model_arg.add_argument('--max_longer_edge', type=int, default=0,
                            help='resize image (do nothing if max_longer_edge <= 0)')

    tmp_config, unparsed = get_config(parser)

    if len(unparsed) > 0:
        raise ValueError('Miss finding argument: unparsed={}\n'.format(unparsed))

    # restore other hyperparams to build model
    if os.path.isdir(tmp_config.model):
        config_path = os.path.join(tmp_config.model, 'config.pkl')
    else:
        config_path = os.path.join(os.path.dirname(tmp_config.model), 'config.pkl')
    try:
        with open(config_path, 'rb') as f:
            config = pickle.load(f)
    except:
        raise ValueError('Fail to open {}'.format(config_path))

    for attr, dst_val in sorted(vars(tmp_config).items()):
        if hasattr(config, attr):
            src_val = getattr(config, attr)
            if src_val != dst_val:
                setattr(config, attr, dst_val)
        else:
            setattr(config, attr, dst_val)

    seqs = []
    if config.subset not in ['val', 'test', 'both']:
        raise ValueError('Unknown value for --subset')
    if config.subset in ['val', 'both']:
        with open(os.path.join('data', 'val.json')) as f:
            seqs += json.load(f)
    if config.subset in ['test', 'both']:
        with open(os.path.join('data', 'test.json')) as f:
            seqs += json.load(f)


    # Build Networks
    tf.reset_default_graph()

    photo_ph = tf.placeholder(tf.float32, [1, None, None, 1]) # input grayscale image, normalized by 0~1
    is_training = tf.constant(False) # Always False in testing

    ops = build_networks(config, photo_ph, is_training)

    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth = True 
    sess = tf.Session(config=tfconfig)
    sess.run(tf.global_variables_initializer())

    # load model
    saver = tf.train.Saver()
    print('Load trained models...')

    if os.path.isdir(config.model):
        checkpoint = tf.train.latest_checkpoint(config.model)
        model_dir = config.model
    else:
        checkpoint = config.model
        model_dir = os.path.dirname(config.model)


    if checkpoint is not None:
        print('Checkpoint', os.path.basename(checkpoint))
        print("[{}] Resuming...".format(time.asctime()))
        saver.restore(sess, checkpoint)
    else:
        raise ValueError('Cannot load model from {}'.format(model_dir))    
    print('Done.')


    print('Processing the following scenes: {}'.format(seqs))
    for seq in seqs:

        print('Processing scene "{}"'.format(seq))

        if not os.path.isdir('{}/{}'.format(config.out_dir, seq)):
            os.makedirs('{}/{}'.format(config.out_dir, seq))

        images = glob.glob('{}/{}/*.jpg'.format(config.in_dir, seq))

        num_kp = []
        with h5py.File('{}/{}/keypoints.h5'.format(config.out_dir, seq), 'w') as f_kp, \
             h5py.File('{}/{}/descriptors.h5'.format(config.out_dir, seq), 'w') as f_desc:
            for fn in images:
                key = os.path.splitext(os.path.basename(fn))[0]
                print(key)
                photo = imread(fn)
                height, width = photo.shape[:2]
                longer_edge = max(height, width)
                if config.max_longer_edge > 0 and longer_edge > config.max_longer_edge:
                    if height > width:
                        new_height = config.max_longer_edge
                        new_width = int(width * config.max_longer_edge / height)
                    else:
                        new_height = int(height * config.max_longer_edge / width)
                        new_width = config.max_longer_edge
                    photo = cv2.resize(photo, (new_width, new_height))
                    height, width = photo.shape[:2]
                rgb = photo.copy()
                if photo.ndim == 3 and photo.shape[-1] == 3:
                    photo = cv2.cvtColor(photo, cv2.COLOR_RGB2GRAY)
                photo = photo[None,...,None].astype(np.float32) / 255.0 # normalize 0-1
                assert photo.ndim == 4 # [1,H,W,1]

                feed_dict = {
                    photo_ph: photo,
                }

                fetch_dict = {
                    'kpts': ops['kpts'],
                    'feats': ops['feats'],
                    'kpts_scale': ops['kpts_scale'],
                    'kpts_ori': ops['kpts_ori'],
                    'scale_maps': ops['scale_maps'],
                    'degree_maps': ops['degree_maps'],
                }

                outs = sess.run(fetch_dict, feed_dict=feed_dict)     


                f_kp[key] = outs['kpts']
                f_desc[key] = outs['feats']
                num_kp.append(len(f_kp[key]))

                print('Image "{}/{}" -> {} features'.format(
                    seq, key, num_kp[-1]))

        print('Finished processing scene "{}" -> {} features/image'.format(
            seq, np.array(num_kp).mean()))
