import os
import json

seqs = []
with open(os.path.join('data', 'val.json')) as f:
    seqs += json.load(f)
with open(os.path.join('data', 'test.json')) as f:
    seqs += json.load(f)

for seq in seqs:
    # cvpr'19 model
    os.system('python extract_delf.py --config_path=third_party/delf-config/delf_config_example.pbtxt --list_images_path=txt/list-{}.txt --output_dir=../benchmark-features/delf-gld-default/{}'.format(seq, seq))
