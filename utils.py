import cv2
import numpy as np
import h5py

cv2_greyscale = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)

cv2_scale = lambda x: cv2.resize(x, dsize=(32, 32),
                                 interpolation=cv2.INTER_LINEAR)
# reshape image
np_reshape = lambda x: np.reshape(x, (32, 32, 1))


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False


def save_h5(dict_to_save, filename):
    """Saves dictionary to hdf5 file"""

    with h5py.File(filename, 'w') as f:
        for key in dict_to_save:
            f.create_dataset(key, data=dict_to_save[key])
