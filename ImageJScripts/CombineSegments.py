import os
import numpy as np
from matplotlib import pyplot as plt

from PIL import Image
from os import listdir
from os.path import isdir, join, isfile


# base_directory = "X:/Doctorate/Phase1/data/CT-Scans/03_Segmented/Regions-Of-Interest/Aggregate-CT-Scans/";
base_directory = "D:/OneDrive - The University of Nottingham/Research/Data/CT Scans/03_Segmented/Regions-Of-Interest/Aggregate-CT-Scans/";
results_dirs = [f + '/' for f in listdir(base_directory) if isdir(join(base_directory, f))]

for results_dir in results_dirs:
	aggregate_dir = base_directory + results_dir + "/Aggregate/"
	binder_dir = base_directory + results_dir + "/Binder/"
	save_dir = base_directory + results_dir

	agg_files = [aggregate_dir + f for f in listdir(aggregate_dir) if isfile(join(aggregate_dir, f))]
	bin_files = [binder_dir + f for f in listdir(binder_dir) if isfile(join(binder_dir, f))]

	for ind, res in enumerate(agg_files):
		agg_im = Image.open(res)
		agg_im = np.array(agg_im)

		bin_im = Image.open(bin_files[ind])
		bin_im = np.array(bin_im)
		bin_im = bin_im // 2
		bin_im += agg_im

		# plt.figure()
		# cmap = plt.imshow(bin_im)
		# cmap.set_cmap("binary")
		# plt.show()

		combined = Image.fromarray(np.uint8(bin_im))

		filename = "Segment" + ((len(str(len(agg_files))) - len(str(ind))) * '0') + str(ind) + ".tif"
		print(save_dir + filename)
		combined.save(save_dir + filename)
		