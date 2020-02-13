import os
from glob import glob

src = os.path.join('..', 'imw-2020')

seqs = [os.path.basename(p) for p in glob(os.path.join(src, '*'))]
print(seqs)

if not os.path.isdir('txt'):
    os.makedirs('txt')

for seq in seqs:
    ims = glob(os.path.join(src, seq, '*.jpg'))
    with open(os.path.join('txt', 'list-{}.txt'.format(seq)), 'w') as f:
        for im in ims:
            f.write('{}\n'.format(im))
