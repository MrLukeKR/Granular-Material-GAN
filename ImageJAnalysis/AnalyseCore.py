import sc.fiji.analyzeSkeleton.AnalyzeSkeleton_;
import os

from ij import IJ as ij
from os import listdir
from os.path import isdir, join

base_directory = "/run/media/***REMOVED***/Experiments/Doctorate/Phase1/experiments/Results/Figures/Aggregate-CT-Scans/";
results_dirs = [f + '/' for f in listdir(base_directory) if isdir(join(base_directory, f))]

def analyseCore(results_dir):
	ij.run("Set Scale...", "distance=1 known=0.096 unit=mm global")
	imp = ij.getImage()

	ij.log("Cropping to ROI...");
	ij.run("Select Bounding Box");
	ij.run("Crop");
	ij.run("Fit Circle to Image", "threshold=253.02");
	ij.run("Select Bounding Box");
	ij.run("Crop");

	ij.log("Isolating Air Voids...");
	ij.run("Invert", "stack");
	ij.run("Mean 3D...", "x=2 y=2 z=2");
	ij.run("Invert", "stack");
	ij.setAutoThreshold(imp, "Default", );
	ij.run("Convert to Mask", "method=Default background=Light calculate");
	ij.run("Invert LUT");

	ij.log("Analysing Particles for Void Volume, Average Diameter and Euler Characteristic...")
	ij.run("Particle Analyser", "enclosed_volume euler thickness min=0.000 max=Infinity surface_resampling=2 surface=Gradient split=0.000 volume_resampling=2");
	ij.saveAs("Results", results_dir + "AnalyseParticles_Results.csv");

	raise Exception()

	ij.log("Converting to Skeleton...");
	ij.run("Skeletonize (2D/3D)");

	ij.log("Analysing Skeleton...");
	ij.run("Analyze Skeleton (2D/3D)", "prune=none show");

	ij.saveAs("Results", results_dir + "AnalyseSkeleton_Results.csv");

	for new_dir in ["/Skeleton/", "/Skeleton/Tagged/", "/Skeleton/Original/"]:
		if not os.path.exists(results_dir + new_dir):
			os.makedirs(results_dir + new_dir)

	ij.selectWindow("CoreSlices_PostProcessed");
	ij.run("Image Sequence... ", "format=TIFF use save=[" + results_dir + "/Skeleton/Original/]");
	ij.selectWindow("Tagged skeleton");
	ij.run("Image Sequence... ", "format=TIFF use save=[" + results_dir + "/Skeleton/Tagged/]");

for results_dir in results_dirs:
	if (results_dir.startswith("Results")):
		ij.log(results_dir)
		base_result_dir = base_directory + results_dir
		fold_dirs = [f + '/' for f in listdir(base_result_dir) if isdir(join(base_result_dir, f))]

		for fold_dir in fold_dirs:
			if(fold_dir.startswith("Fold")):
				ij.log(fold_dir)
				core_dir = base_directory + results_dir + fold_dir + "Outputs/"
				core = [f + '/' for f in listdir(core_dir) if isdir(join(core_dir, f))]
				core_slice_dir = core_dir + core[0] + "CoreSlices_PostProcessed/"

				analysis_dir = core_dir + "/Analysis/"

				if not os.path.exists(analysis_dir):
					os.makedirs(analysis_dir)

				ij.log(core_slice_dir)
				ij.run("Image Sequence...", "open=[" + core_slice_dir + "] sort")
				analyseCore(analysis_dir)
				ij.run("Close All")
