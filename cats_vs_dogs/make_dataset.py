__authors__ = "Vincent Dumoulin, Dmitry Serdyuk"
__maintainer__ = "Dmitry Serdyuk"

import argparse
import os
import os.path
import cPickle as pkl
import tables

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
            all_files += [file]
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


def create_hdf5(rng):
    filters = tables.Filters(complib='blosc', complevel=5)
    h5file = tables.open_file('dummy.h5', mode='w',
                              title='Cats vs Dogs dataset',
                              filters=filters)
    group = h5file.create_group(h5file.root, 'Data', 'Data')
    atom = tables.UInt8Atom()
    X = h5file.create_vlarray(group, 'X', atom=atom, title='Data values',
                              expectedrows=500, filters=filters)
    y = h5file.create_carray(group, 'y', atom=atom, title='Data targets',
                             shape=(500, 1), filters=filters)
    s = h5file.create_carray(group, 's', atom=atom, title='Data shapes',
                             shape=(500, 3), filters=filters)

    shapes = rng.randint(low=10, high=101, size=(500, 2))
    for i, shape in enumerate(shapes):
        size = (shape[0], shape[1], 3)
        image = rng.uniform(low=0, high=1, size=size)
        target = rng.randint(low=0, high=2)

        X.append(image.flatten())
        y[i] = target
        s[i] = np.array(size)
        if i % 100 == 0:
            print i
            h5file.flush()
    h5file.flush()


def parse_args():
    parser = argparse.ArgumentParser('Aggregates cats vs dogs dataset')
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
    parser.add_argument('--no_make_dataset', action='store_true',
                        default=False,
                        help='Skip datasets making')
    parser.add_argument('--no_aggregate', action='store_true',
                        default=False,
                        help='Skip images aggregating ')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if not args.no_aggregate:
        print '.. aggregating...'
        aggregate(**args.__dict__)
    if not args.no_make_dataset:
        print '.. making datasets...'
        make_datasets(**args.__dict__)
    print '.. finished, exiting'