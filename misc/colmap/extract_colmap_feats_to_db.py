import os
import argparse

def extract_features(dsp, upright_data_path, save_path):

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for folder in os.listdir(upright_data_path):

        images_path = os.path.join(upright_data_path, folder, 'set_100', 'images')
        database_path = os.path.join(save_path, folder+'.db')
        print('colmap database_creator --database_path {}'.format(database_path))

        os.system('colmap database_creator --database_path {}'.format(database_path))

        if dsp:
            os.system(
                'CUDA_VISIBLE_DEVICES="0" colmap feature_extractor --database_path {} --image_path {} --SiftExtraction.gpu_index 0'
                ' --SiftExtraction.estimate_affine_shape 1 --SiftExtraction.edge_threshold 10000 --SiftExtraction.peak_threshold 0.00000001 --SiftExtraction.domain_size_pooling 1'.format(database_path,
                                                                                                   images_path))
        else:
            os.system('CUDA_VISIBLE_DEVICES="0" colmap feature_extractor --database_path {} --image_path {}'
                      ' --SiftExtraction.estimate_affine_shape 1 --SiftExtraction.gpu_index 0  --SiftExtraction.edge_threshold 10000 --SiftExtraction.peak_threshold 0.00000001'.format(database_path, images_path))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--upright_data_path",
        default=os.path.join('../../..', 'data_upright'),
        type=str,
        help='Path to the dataset')

    parser.add_argument(
        "--save_path",
        default=os.path.join('../../..', 'colmap_descriptors'),
        type=str,
        help='Path to store the features')

    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    extract_features(False, args.upright_data_path, args.save_path)
    extract_features(True, args.upright_data_path, args.save_path.replace('colmap_descriptors','colmap_descriptors_dsp'))