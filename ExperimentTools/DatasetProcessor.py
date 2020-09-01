import random
import math
import tensorflow as tf
import numpy as np

from Settings import MessageTools as mt
from Settings.MessageTools import print_notice, get_notice


def dataset_to_train_and_test(dataset, split_count):
    inp = ""
    while not (str.isdigit(inp) and 1 <= int(inp) <= 3):
        print_notice("[1] Random Selection")
        print_notice("[2] Select Training Set By ID")
        print_notice("[3] Select Testing Set By ID")
        inp = input("Enter a data selection criteria option > ")

        if not (str.isdigit(inp) and 1 <= int(inp) <= 3):
            print_notice("Invalid selection", mt.MessagePrefix.ERROR)

    inp = int(inp)

    if inp == 1:
        random.shuffle(dataset)

        return dataset[0:split_count], dataset[split_count:]
    else:
        print_notice("The following core IDs are available for selection:")
        for ind, ds in enumerate(dataset):
            print("[%s] %s" % (str(ind), ds))

        if inp == 2:
            allowance = len(dataset)
            inp = input(get_notice("Please enter %s ID%s (separate multiple IDs with a comma ',') > "
                                   % (str(allowance), "s" if allowance > 1 else "")))

            ids = inp.split(',')
            ids = [int(x) for x in ids]

            train_set = [dataset[x] for x in ids]
            test_set = [x for x in dataset if x not in train_set]

            return train_set, test_set
        elif inp == 3:
            allowance = len(dataset) - split_count
            inp = input(get_notice("Please enter %s ID%s (separate multiple IDs with a comma ',') > "
                                   % (str(allowance), "s" if allowance > 1 else "")))

            ids = inp.split(',')
            ids = [int(x) for x in ids]

            test_set = [dataset[x] for x in ids]
            train_set = [x for x in dataset if x not in test_set]

            return train_set, test_set


def chunks(dataset, n):
    dataset_chunks = list()

    jump = math.floor(len(dataset) / n)

    for i in range(n):
        ind = i * jump
        dataset_chunks.append(dataset[ind:ind + jump])

    return dataset_chunks


def dataset_to_k_cross_fold(dataset, k):
    random.shuffle(dataset)
    groups = chunks(dataset, k)
    train_files = list()
    test_files = groups

    for ind, group in enumerate(groups):
        train_files.insert(ind, [data for data in dataset if data not in group])

    return train_files, test_files


def prepare_tf_set(set_filenames, voxel_dimensions, epochs, batch_size):
    print_notice("Preparing TFData dataset...")

    train_ds = tf.data.TFRecordDataset(filenames=set_filenames, num_parallel_reads=len(set_filenames))

    example = {
        'aggregate': tf.io.FixedLenFeature([], dtype=tf.string),
        'binder': tf.io.FixedLenFeature([], dtype=tf.string)
    }

    def _parse_voxel_function(example_proto):
        return tf.io.parse_single_example(example_proto, example)

    def _decode_voxel_function(serialised_example):
        aggregate = tf.io.decode_raw(serialised_example['aggregate'], tf.bool)
        binder = tf.io.decode_raw(serialised_example['binder'], tf.bool)

        segments = [aggregate, binder]

        for ind in range(2):
            segments[ind] = tf.cast(segments[ind], dtype=tf.bfloat16)

            segments[ind] = tf.reshape(segments[ind], voxel_dimensions)

            segments[ind] = tf.expand_dims(segments[ind], -1)

        return segments

    def _rescale_voxel_values(features, labels):
        segments = [features, labels]
        # Data must be between [-1, 1] for GAN (voxels are stored between [0, 1])
        for ind in range(2):
            segments[ind] = segments[ind] * 2.0
            segments[ind] = segments[ind] - 1.0

        return segments[0], segments[1]

    # Training dataset(s)
    print("\t\tTraining Sets:")

    for filename in set_filenames:
        print("\t\t\t%s" % filename)

    # Shuffle filenames, not images, as this is done in memory
    train_ds = train_ds.shuffle(buffer_size=len(set_filenames))

    train_ds = train_ds.map(_parse_voxel_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_ds = train_ds.map(_decode_voxel_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_ds = train_ds.map(_rescale_voxel_values, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    train_ds = train_ds.batch(batch_size=batch_size)

    print("\t\tDataset Size: (CALCULATING)\r", end='', flush=True)
    batches = sum(1 for _ in train_ds)

    print("\t\tDataset Size: %s " % str(batches * batch_size))
    print("\t\tDataset Format: %s Epochs of %s batches, %s voxels each" % (str(epochs), str(batches), str(batch_size)))

    train_ds = train_ds.repeat(epochs)
    print("\t\tTotal Datapoints: %s" % (str(epochs * batch_size * batches)))

    train_ds = train_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return train_ds, batches * epochs
