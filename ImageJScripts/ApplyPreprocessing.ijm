id="15-2974";
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

run("Median 3D...", "x=1 y=1 z=1");
run("Image Sequence... ", "select=X:/Doctorate/Phase1/data/CT-Scans/02_Processed/Aggregate-CT-Scans/" + id + "/ dir=X:/Doctorate/Phase1/data/CT-Scans/02_Processed/Aggregate-CT-Scans/" + id + "/ format=TIFF");
close();
