import argparse
import os
import os.path
import cPickle as pkl

import numpy as np

from scipy import misc


def aggregate(directory, **kwargs):
    for file in os.listdir(directory):
        if file.endswith(".jpg"):
            full_path = os.path.join(directory, file)
            image = misc.imread(full_path)
            label, id, _ = file.split('.')
            label = np.array([label == 'dog', label == 'cat'], dtype='int64')
            with open(full_path + '.pkl', 'w') as fout:
                pkl.dump((image, label), fout)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory',
                        default='/home/dima/Downloads/datasets/cats_vs_dogs/train/')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    aggregate(**args.__dict__)