ids = newArray("15-2974", "15-2975", "15-2983", "16-399", "16-401", "16-402",
			   "15-2986", "15-2988", "15-2992", "15-2993", "16-405", "16-408",
			   "15-2998", "15-2999", "15-3007", "16-647", "16-649", "16-651",
			   "15-3011", "15-3013", "15-3017", "15-3018", "16-411", "16-413",
			   "15-3025", "15-3028", "15-3031", "16-414", "16-415", "16-418");

for (i = 0; i < ids.length; i++)
	applyPreprocessing(ids[i]);

function applyPreprocessing(id){
	
	run("Image Sequence...", "select= dir=X:/Doctorate/Phase1/data/CT-Scans/01_Unprocessed/Aggregate-CT-Scans/" + id +  "/ sort");
	run("8-bit");
	run("Enhance Contrast...", "saturated=0.3 normalize process_all");
	
	blocksize = 127;
	histogram_bins = 256;
	maximum_slope = 3;
	mask = "*None*";
	fast = true;
	process_as_composite = true;
	 
	getDimensions( width, height, channels, slices, frames );
	isComposite = channels > 1;
	parameters =
	  "blocksize=" + blocksize +
	  " histogram=" + histogram_bins +
	  " maximum=" + maximum_slope +
	  " mask=" + mask;
	if ( fast )
	  parameters += " fast_(less_accurate)";
	if ( isComposite && process_as_composite ) {
	  parameters += " process_as_composite";
	  channels = 1;
	}
	   
	for ( f=1; f<=frames; f++ ) {
	  Stack.setFrame( f );
	  for ( s=1; s<=slices; s++ ) {
	    Stack.setSlice( s );
	    for ( c=1; c<=channels; c++ ) {
	      Stack.setChannel( c );
	      run( "Enhance Local Contrast (CLAHE)", parameters );
	    }
	  }
	}
	
	run("Gaussian Blur 3D...", "x=2 y=2 z=2");
	run("Median 3D...", "x=2 y=2 z=2");
	run("Image Sequence... ", "select=X:/Doctorate/Phase1/data/CT-Scans/02_Processed/Aggregate-CT-Scans/" + id + "/ dir=X:/Doctorate/Phase1/data/CT-Scans/02_Processed/Aggregate-CT-Scans/" + id + "/ format=TIFF");
	close();
}