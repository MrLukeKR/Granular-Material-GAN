import random
import math

import tensorflow as tf

from Settings.MessageTools import print_notice


def dataset_to_train_and_test(dataset, split_count):
    random.shuffle(dataset)

    return dataset[0:split_count], dataset[split_count:]


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

    train_ds = train_ds.prefetch(1)

    return train_ds, batches * epochs
