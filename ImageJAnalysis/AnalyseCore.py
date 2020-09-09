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

	ij.log("Cropping to content...")
	ij.run("Select Bounding Box")
	ij.run("Crop")

	ij.log("Cropping to 5cm x 5cm ROI...");
	# center_slice = imp_stack.size() // 2

	roi_dim = int(50 // 0.096) # How many pixels in 50mm 
	half_roi_dim = roi_dim // 2
	cal = imp.getCalibration()

	im_width = imp.getDimensions()[0]
	im_height = imp.getDimensions()[1]

	im_mid_width = im_width // 2
	im_mid_height = im_height // 2

	roi_start_x = int(im_mid_width - half_roi_dim)
	roi_start_y = int(im_mid_height - half_roi_dim)

	ij.makeRectangle(roi_start_x, roi_start_y, roi_dim, roi_dim)
	ij.run("Crop")

	#TODO: Crop in the z-dimension (currently unknown possibility due to small cores)

	ij.log("Calculating Volumes...");
	imp_stack = imp.getImageStack()
	
	voidVoxels = 0
	binderVoxels = 0
	aggregateVoxels = 0
	
	for ind in range(imp_stack.size()):
		ImProc  = imp_stack.getProcessor(ind + 1)
		ListBin = ImProc.getHistogram()
		voidVoxels += int(ListBin[0])
		binderVoxels += int(ListBin[127])
		aggregateVoxels += int(ListBin[255])

	totalVoxels = sum([voidVoxels, binderVoxels, aggregateVoxels])
	voidPercent = float(voidVoxels) / totalVoxels
	binderPercent = float(binderVoxels) / totalVoxels
	aggregatePercent = float(aggregateVoxels) / totalVoxels
	
	histFile = open(results_dir + "Histogram_Results.csv", "w")
	histFile.write("Void_Voxels, Binder_Voxels, Aggregate_Voxels, Total_Voxels\n")
	histFile.write("%s,%s,%s,%s\n" % (str(voidVoxels), str(binderVoxels), str(aggregateVoxels), str(totalVoxels)))
	histFile.write("%s,%s,%s,%s" % (str(voidPercent), str(binderPercent), str(aggregatePercent), "1.0"))
	histFile.close()
	
	ij.log("Smoothing Images...");
	ij.run("Invert", "stack");
	ij.run("Mean 3D...", "x=2 y=2 z=2");
	ij.run("Invert", "stack");	

	ij.log("Isolating Air Voids...");
	ij.setAutoThreshold(imp, "Default", );
	ij.run("Convert to Mask", "method=Default background=Light calculate");
	ij.run("Invert LUT");

	ij.log("Analysing Particles for Void Volume, Average Diameter and Euler Characteristic...")
	ij.run("Particle Analyser", "euler thickness min=0.000 max=Infinity surface_resampling=2 surface=Gradient split=0.000 volume_resampling=2");
	ij.saveAs("Results", results_dir + "AnalyseParticles_Results.csv");
	# ij.selectWindow("CoreSlices_PostProcessed");
	# ij.run("Close")
	# ij.run("Collect Garbage");

	ij.log("Converting to Skeleton...");
	ij.run("Skeletonize (2D/3D)");

	ij.log("Analysing Skeleton for Tortuosity...");
	ij.run("Analyze Skeleton (2D/3D)", "prune=none show");

	ij.saveAs("Results", results_dir + "AnalyseSkeleton_Results.csv");

	for new_dir in ["/Skeleton/", "/Skeleton/Tagged/", "/Skeleton/Original/"]:
		if not os.path.exists(results_dir + new_dir):
			os.makedirs(results_dir + new_dir)

	ij.selectWindow("CoreSlices_PostProcessed");
	ij.run("Image Sequence... ", "format=TIFF use save=[" + results_dir + "/Skeleton/Original/]");
	ij.selectWindow("Tagged skeleton");
	ij.run("Image Sequence... ", "format=TIFF use save=[" + results_dir + "/Skeleton/Tagged/]");
	ij.run("Collect Garbage");
	ij.run("Close All")

for results_dir in results_dirs:
	if (results_dir.startswith("Results")):
		ij.log(results_dir)
		base_result_dir = base_directory + results_dir
		fold_dirs = [f + '/' for f in listdir(base_result_dir) if isdir(join(base_result_dir, f))]

		for fold_dir in fold_dirs:
			if(fold_dir.startswith("Fold")):
				ij.run("Collect Garbage");
				ij.log(fold_dir)
				core_dir = base_directory + results_dir + fold_dir + "Outputs/"
				core = [f + '/' for f in listdir(core_dir) if isdir(join(core_dir, f)) and f.startswith("1")]
				core_slice_dir = core_dir + core[0] + "CoreSlices_PostProcessed/"

				analysis_dir = core_dir + "/Analysis/"

				if not os.path.exists(analysis_dir):
					os.makedirs(analysis_dir)

				ij.log(core_slice_dir)
				ij.run("Image Sequence...", "open=[" + core_slice_dir + "] sort")
				analyseCore(analysis_dir)
