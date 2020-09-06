base_directory = "D:/OneDrive - The University of Nottingham/Research/Results/";
results_dirs = getFileList(base_directory);

for(i=0; i < lengthOf(results_dirs); i++)
{
	if (results_dirs[i].startsWith("Results"))
	{
		print(results_dirs[i]);		
		fold_dirs = getFileList(base_directory + results_dirs[i]);
		
		for(j=0; j < lengthOf(fold_dirs); j++)
		{
			if(fold_dirs[j].startsWith("Fold"))
			{
				print(fold_dirs[j]);
				core_dir = base_directory + results_dirs[i] + fold_dirs[j] + "Outputs/";
				core = getFileList(core_dir);
				core_slice_dir = core_dir + core[0] + "CoreSlices_PostProcessed/";

				print(core_slice_dir);
				run("Image Sequence...", "select=[" + core_slice_dir + "] dir=[" + core_slice_dir + "] sort");
				analyseCore();
				close("*");			
			}
		}
	}
	
}

return;


function analyseCore(){
	print("Cropping to ROI...");
	run("Select Bounding Box");
	run("Crop");
	run("Fit Circle to Image", "threshold=253.02");
	run("Select Bounding Box");
	run("Crop");

	print("Isolating Air Voids...");
	run("Invert", "stack");
	run("Mean 3D...", "x=2 y=2 z=2");
	run("Invert", "stack");
	setAutoThreshold("Default");
	//run("Threshold...");
	run("Convert to Mask", "method=Default background=Light calculate");
	run("Invert LUT");

	print("Removing Small Particles...");
	run("Dilate", "stack");
	run("Erode", "stack");

	print("Converting to Skeleton...");
	run("Skeletonize (2D/3D)");
	
	print("Analysing Skeleton...");
	run("Analyse Skeleton");
}
