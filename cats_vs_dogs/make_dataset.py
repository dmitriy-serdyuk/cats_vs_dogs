import argparse
import os
import os.path
import cPickle as pkl

import numpy as np

from scipy import misc


def aggregate(directory, **kwargs):
    dirs = os.listdir(directory)
    total = len(dirs)
    for i, file in enumerate(dirs):
        if file.endswith(".jpg"):
            full_path = os.path.join(directory, file)
            image = misc.imread(full_path)
            label, id, _ = file.split('.')
            label = int(label == 'dog')
            with open(full_path + '.pkl', 'w') as fout:
                pkl.dump((image, label), fout)
        if i % 100 == 0:
            print '.. aggregated', i, '/', total


def make_datasets(train_share, valid_share, directory, seed, **kwargs):
    assert train_share + valid_share < 1.
    all_files = []
    for file in os.listdir(directory):
        if file.endswith(".pkl"):
            all_files += file
    n_train = int(len(all_files) * train_share)
    n_valid = int(len(all_files) * valid_share)

    rng = np.random.RandomState(seed)
    rng.shuffle(all_files)
    train = all_files[:n_train]
    valid = all_files[n_train:(n_train + n_valid)]
    test = all_files[(n_train + n_valid):]
    save_path = os.path.join(directory, '../datasets.pkl')
    with open(save_path, 'w') as fout:
        pkl.dump((train, valid, test), fout)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory',
                        default='/home/dima/Downloads/datasets/cats_vs_dogs/train/',
                        help='Directory with data (extracted kaggle dataset)')
    parser.add_argument('--train_share', type=float,
                        default=0.6,
                        help='Train dataset share')
    parser.add_argument('--valid_share', type=float,
                        default=0.2,
                        help='Validation dataset share')
    parser.add_argument('--seed', type=int,
                        default=1,
                        help='Random number generator seed')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    aggregate(**args.__dict__)
    make_datasets(**args.__dict__)