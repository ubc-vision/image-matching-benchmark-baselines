
import argparse
import yaml
import os
if not os.path.isdir('yaml'):
    os.makedirs('yaml')

parser = argparse.ArgumentParser(description='Geenerate yaml for contextdesc script')

parser.add_argument(
  '--data_root',
  default = '../imw-2020',
  type = str,
  help = 'path to dataset folder')

parser.add_argument(
  '--dump_root',
  default = '../benchmark-features',
  type = str,
  help = 'path to dump folder')

parser.add_argument(
  '--num_keypoints',
  default = 8000,
  type = int,
  help = 'number of keypoints to extract'
  )

args, unparsed = parser.parse_known_args()

dict_file = {}
dict_file['data_name']='imw2019'
dict_file['data_split'] =''
dict_file['data_root'] = args.data_root

dict_file['truncate'] = [0, None]

dict_file['pretrained'] = {}
dict_file['pretrained']['loc_model'] = 'third_party/contextdesc/pretrained/contextdesc++'
dict_file['pretrained']['reg_model'] = 'third_party/contextdesc/pretrained/retrieval_model'

dict_file['reg_feat'] ={}
dict_file['reg_feat']['infer'] = True
dict_file['reg_feat']['overwrite']= False
dict_file['reg_feat']['max_dim']= 1024

dict_file['loc_feat'] = {}
dict_file['loc_feat']['infer']= True
dict_file['loc_feat']['overwrite']= False
dict_file['loc_feat']['n_feature']= args.num_keypoints
dict_file['loc_feat']['batch_size']= 512
dict_file['loc_feat']['dense_desc']= False
dict_file['loc_feat']['peak_thld']= 0
dict_file['loc_feat']['max_dim']= 1280
dict_file['loc_feat']['upright']= False
dict_file['loc_feat']['scale_diff']= True

dict_file['aug_feat'] = {}
dict_file['aug_feat']['infer']= True
dict_file['aug_feat']['overwrite']= False
dict_file['aug_feat']['quantz']= True

dict_file['post_format'] = {}
dict_file['post_format']['enable'] = True
dict_file['post_format']['suffix'] = ''

dict_file['dump_root'] = os.path.join(args.dump_root,'tmp_contextdesc')
dict_file['submission_root'] = os.path.join(args.dump_root,'contextdesc_{}'.format(dict_file['loc_feat']['n_feature']))

with open(r'yaml/imw-2020.yaml', 'w') as file:
    documents = yaml.dump(dict_file, file)