import h5py
import numpy as np
import tensorflow as tf


class VoxelGenerator:
    def __init__(self, file):
        self.file = file

    def __call__(self):
        with h5py.File(self.file, 'r') as hf:
            for vox in hf["voxels"]:
                aggregate = np.array([x == 255 for x in vox], dtype=bool)
                binder = np.array([x == 255 for x in vox], dtype=bool)

                yield aggregate, binder
