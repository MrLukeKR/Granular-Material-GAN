ids = newArray("15-2974", "15-2975", "15-2983", "16-399", "16-401", "16-402",
			   "15-2986", "15-2988", "15-2992", "15-2993", "16-405", "16-408",
			   "15-2998", "15-2999", "15-3007", "16-647", "16-649", "16-651",
			   "15-3011", "15-3013", "15-3017", "15-3018", "16-411", "16-413",
			   "15-3025", "15-3028", "15-3031", "16-414", "16-415", "16-418");

voids = newArray( 9.4, 10.6, 10.6, 12.1, 14.6, 12.8,
				 13.7, 15.4, 15.1, 16  , 13.3, 14.6,
				 17.2, 16.4, 18.3, 17.9, 15.8, 18.3,
				 21.5, 19.6, 22.9, 21.2, 19.6, 19.7,
				 24.1, 24.8, 26  , 22.4, 23.8, 22.9);
				 
binders = newArray(4.5, 4.5, 4.5, 3.8, 3.8, 3.8,
				   4.2, 4.2, 4.2, 4.2, 3.8, 3.8,
				   3.8, 3.8, 3.8, 3.8, 3.8, 3.8,
				   3.3, 3.3, 3.3, 3.3, 3.8, 3.8,
				   2.7, 2.7, 2.7, 3.8, 3.8, 3.8);

dusts = newArray(45.6, 45.6, 45.6, 19, 19, 19,
				 22  , 22  , 22  , 22, 19, 19,
				 19  , 19  , 19  , 20, 20, 20,
				 17  , 17  , 17  , 17, 19, 19,
				 10  , 10  , 10  , 19, 19, 19);

include_dust = true;

for (i = 0; i < ids.length; i++){
	if(include_dust)
	{
		mastic = (dusts[i] / 100) * (100 - voids[i] - binders[i]) + binders[i];
		segmentCore(ids[i], voids[i], mastic);
	}else
		segmentCore(ids[i], voids[i], binders[i]);
}

function segmentCore(id, known_airvoid, known_binder){
	threshold_skip_original = 64;

	current_content=0;
	
	use_roi = true;
	
	nBins = 2;
	
	run("Clear Results");
	run("Image Sequence...", "select= dir=X:/Doctorate/Phase1/data/CT-Scans/02_Processed/Aggregate-CT-Scans/" + id +  "/ sort");
	original = getImageID();
	
	width = getWidth;
	height = getHeight;

	mm_per_pixel = height / 100;
	roi_dim = 60;
	roi_depth = 40;
	
	roi_pixels = roi_dim * mm_per_pixel;
	roi_depth_pixels = roi_depth * mm_per_pixel;

	half_stack = round(nSlices / 2);
	start_stack = half_stack -= round(roi_depth_pixels / 2);
	
	run("Set Scale...", "distance=" + mm_per_pixel + " known=1 unit=pixel");
	
	
	if (!use_roi){
		makeOval(0, 0, width, height);
		Roi.setName("Core");
		roiManager("add");
	}else{
		startX = round(width / 2);
		startY = round(height / 2);
		startX -= roi_pixels / 2;
		startY -= roi_pixels / 2;
		
		makeRectangle(startX, startY, roi_pixels, roi_pixels);
	
		run("Crop");
		//run("Delete Slice Range", "first=1 last=" + (start_stack - 1));
		//run("Delete Slice Range", "first=" + (start_stack + roi_depth_pixels + 1) + " last=" + nSlices);
	}
	
	getStatistics(area);
	
	core_volume = area * nSlices;
	total_content = width * height * nSlices;
	
	print("Core Volume: " + ((core_volume / total_content) * 100) + "% of image stack");
	print("Background Volume: " + (100 - (core_volume / total_content) * 100) + "% of image stack");
	
	timeout_max = 20;
	
	void_upper_threshold = thresholdSegment(known_airvoid, 0, threshold_skip_original, core_volume);
	
	thresh_low = void_upper_threshold + 1;
	thresh_high = thresh_low + threshold_skip_original;
	
	binder_upper_threshold = thresholdSegment(known_binder, thresh_low, thresh_high, core_volume);
			
	selectImage(original);
	run("Duplicate...", "title=Voids duplicate");
	airvoid = getImageID();	
	run("Duplicate...", "title=Binder duplicate");
	binder = getImageID();	
	run("Duplicate...", "title=Aggregate duplicate");
	aggregate = getImageID();	

	base_dir = "X:/Doctorate/Phase1/data/CT-Scans/03_Segmented/Regions-Of-Interest/Aggregate-CT-Scans/" + ids[i] + "/";
	File.makeDirectory(base_dir);
	File.makeDirectory(base_dir + "Voids/");
	File.makeDirectory(base_dir + "Binder/");
	File.makeDirectory(base_dir + "Aggregate/");

	print("Air Voids: Threshold 0 to " + void_upper_threshold); 
	thresholdStack(0, void_upper_threshold, width, height, core_volume, airvoid);
	run("Image Sequence... ", "select=" + base_dir + "Voids/ dir=" + base_dir + "Voids/ format=TIFF");

	print("Binder: Threshold " + (void_upper_threshold + 1) + " to " + binder_upper_threshold); 
	thresholdStack(void_upper_threshold + 1, binder_upper_threshold, width, height, core_volume, binder);
	run("Image Sequence... ", "select=" + base_dir + "Binder/ dir=" + base_dir + "Binder/ format=TIFF");
	
	print("Aggregates: Threshold " + (binder_upper_threshold + 1) + " to 255"); 
	thresholdStack(binder_upper_threshold + 1, 255, width, height, core_volume, aggregate);
	run("Image Sequence... ", "select=" + base_dir + "Aggregate/ dir=" + base_dir + "Aggregate/ format=TIFF");

	close("*");
}

function thresholdStack(threshold_low, threshold_high, width, height, core_volume, image)
{
		total_dark = 0;
		total_light = 0;

		selectImage(image);
		for (i=1; i<=nSlices; i++) 
		{
			setSlice(i);
			run("Duplicate...", "title=temp");
			if(!use_roi)
				makeOval(0, 0, width, height);
			setThreshold(threshold_low, threshold_high);
			run("Convert to Mask");
			invertingLUT = is("Inverting LUT");
			run("Copy");
			close;
     		selectImage(image);
     		run("Paste");
     		if (i==1 && invertingLUT != is("Inverting LUT"))
         		run("Invert LUT");
			if(!use_roi)
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
	threshold_skip = threshold_skip_original;
	min_threshold = threshold_high;
	
	while (threshold_skip > 0 && timeout_count < timeout_max){		
		print("Threshold " + threshold_low + " to " + threshold_high); 
		
		selectImage(original);
		run("Duplicate...", "title=Calculating duplicate");
		copy = getImageID();
	
		selectImage(copy);
		
		current_content = thresholdStack(threshold_low, threshold_high, width, height, core_volume, copy);

		distance_to_actual = abs(known_content - current_content);
		min_distance_to_actual = minOf(distance_to_actual, min_distance_to_actual);
		if (distance_to_actual == min_distance_to_actual)
			min_threshold = threshold_high;

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

		close();
	}
	
	return min_threshold;
}