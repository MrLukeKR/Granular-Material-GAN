id="15-2975";

threshold_skip_original = 64
threshold_skip = threshold_skip_original
threshold_low=0
threshold_high=threshold_skip

known_airvoid=10.6
known_binder=4.5
current_content=0

nBins = 2;

run("Clear Results");
run("Image Sequence...", "select= dir=X:/Doctorate/Phase1/data/CT-Scans/02_Processed/Aggregate-CT-Scans/" + id +  "/ sort");
original = getImageID();

width = getWidth;
height = getHeight;

makeOval(0, 0, width, height);
Roi.setName("Core");
roiManager("add");
run("Set Scale...", "distance=0 known=0 unit=pixel");

getStatistics(area);

core_volume = area * nSlices;
total_content = width * height * nSlices;

print("Core Volume: " + ((core_volume / total_content) * 100) + "% of image stack");
print("Background Volume: " + (100 - (core_volume / total_content) * 100) + "% of image stack");

timeout_max = 20;	

base_dir = "X:/Doctorate/Phase1/data/CT-Scans/03_Segmented/Cores/Aggregate-CT-Scans/" + id + "/";
void_dir = base_dir + "/Void/";
binder_dir = base_dir + "/Binder/";
File.makeDirectory(base_dir); 
File.makeDirectory(void_dir);
File.makeDirectory(binder_dir);

void_upper_threshold = thresholdSegment(known_airvoid, threshold_low, threshold_high, core_volume);

run("Invert LUT");
run("Image Sequence... ", "select=" + void_dir + " dir=" + void_dir + " format=TIFF");

air_void_image = getImageID();
threshold_skip = threshold_skip_original;
threshold_low = void_upper_threshold + 1;
threshold_high = threshold_low + threshold_skip;

binder_upper_threshold = thresholdSegment(known_binder, threshold_low, threshold_high, core_volume);
run("Invert LUT");
run("Image Sequence... ", "select=" + binder_dir + " dir=" + binder_dir + " format=TIFF");

function thresholdStack(threshold_low, threshold_high, width, height, core_volume)
{
		total_dark = 0;
		total_light = 0;
	
		for (i=1; i<=nSlices; i++) {
			setSlice(i);
			run("Duplicate...", "title=temp");
			makeOval(0, 0, width, height);
			setThreshold(threshold_low, threshold_high);
			run("Convert to Mask");
			invertingLUT = is("Inverting LUT");
			run("Copy");
			close;
     		selectImage(copy);
     		run("Paste");
     		if (i==1 && invertingLUT != is("Inverting LUT"))
         		run("Invert LUT");

         	makeOval(0, 0, width, height);
	
			getHistogram(values, counts, nBins);
			total_dark += counts[0];
			total_light += counts[1];
		}
		
		current_content = (total_light / core_volume) * 100;

		return current_content;
}


function thresholdSegment(known_content, threshold_low, threshold_high, core_volume)
{
	distance_to_actual = known_content;
	min_distance_to_actual = known_content;
	timeout_count = 0;
	
	while ((distance_to_actual != min_distance_to_actual || threshold_skip > 0) && timeout_count < timeout_max){		
		if (timeout_count > 0)
			close();
		
		print("Threshold " + threshold_low + " to " + threshold_high); 
		
		selectImage(original);
		run("Duplicate...", "title=Calculating duplicate");
		copy = getImageID();
	
		selectImage(copy);
		
		current_content = thresholdStack(threshold_low, threshold_high, width, height, core_volume);

		distance_to_actual = abs(known_content - current_content);
		min_distance_to_actual = minOf(distance_to_actual, min_distance_to_actual);

		print("\tContent: " + (current_content) + "%");
		print("\tDistance to Actual: " + distance_to_actual + "%");
		print("\tMinimum Distance to Actual: " + min_distance_to_actual + "%");

		if (threshold_skip == 1)
			threshold_skip = 0;
		if (threshold_skip > 1 && current_content > known_content){
			threshold_high -= threshold_skip;

			threshold_skip = round(threshold_skip / 2);
		} 
		
		threshold_high += threshold_skip;
		timeout_count++;
	}	

	if (min_distance_to_actual != distance_to_actual)
	{
		print("Previous stack was closer to actual -- Reverting...");
		
		close();
		
		selectImage(original);
		run("Duplicate...", "duplicate");
		copy = getImageID();
		selectImage(copy);
		
		threshold_high -= threshold_skip;
		current_content = thresholdStack(threshold_low, threshold_high, width, height, core_volume);

	}

	return threshold_high;
}