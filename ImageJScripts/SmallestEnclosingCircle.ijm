/* This macro creates a circular selection that is the smallest circle
   enclosing the current selection.
   Version: 2009-06-12 Michael Schmid

   Restrictions:
   - Does not work with composite selections
   - Due to rounding errors, some selection points may be slightly outside the circle
*/


//global variables used for passing or as return values
var fourIndices = newArray(4);
var xcenter, ycenter, radius;

macro 'Smallest Enclosing Circle' {
  if (selectionType<0) exit("Error: Roi Required");
  if (selectionType==9) exit("This macro does not work with composite selections");
  run("Line Width...", "line=1");
  getSelectionCoordinates(xCoordinates, yCoordinates);
  smallestEnclosingCircle(xCoordinates, yCoordinates);
  diameter = round(2*radius);
  makeOval(round(xcenter-radius), round(ycenter-radius), diameter, diameter);
}

/* Finds the smallest circle enclosing a set of points */
/* Input: arrays of x and y coordinates of the points */
/* Returns global variables xcenter, ycenter, radius */
function smallestEnclosingCircle(x,y) {
  n = x.length;
  if (n==1)
    return newArray(x[0], y[0], 0);
  else if (n==2)
    return circle2(x[0], y[0], x[1], y[1]);
  else if (n==3)
    return circle3(x[0], y[0], x[1], y[1], x[2], y[2]);
  //As starting point, find indices of min & max x & y
  xmin = 999999999; ymin=999999999; xmax=-1; ymax=-1;
  for (i=0; i<n; i++) {
    if (x[i]<xmin) {xmin=x[i]; fourIndices[0]=i;}
    if (x[i]>xmax) {xmax=x[i]; fourIndices[1]=i;}
    if (y[i]<ymin) {ymin=y[i]; fourIndices[2]=i;}
    if (y[i]>ymax) {ymax=y[i]; fourIndices[3]=i;}
  }
  do {
    badIndex = circle4(x, y);  //get circle through points listed in fourIndices
    newIndex = -1;
    largestRadius = -1;
    for (i=0; i<n; i++) {      //get point most distant from center of circle
      r = vecLength(xcenter-x[i], ycenter-y[i]);
      if (r > largestRadius) {
        largestRadius = r;
        newIndex = i;
      }
    }
    //print(largestRadius);
    retry = (largestRadius > radius*1.0000000000001);
    fourIndices[badIndex] = newIndex; //add most distant point
  } while (retry);
}

//circle spanned by diameter between two points.
function circle2(xa,ya,xb,yb) {
  xcenter = 0.5*(xa+xb);
  ycenter = 0.5*(ya+yb);
  radius = 0.5*vecLength(xa-xb, ya-yb);
  return;
}
//smallest circle enclosing 3 points.
function circle3(xa,ya,xb,yb,xc,yc) {
  xab = xb-xa; yab = yb-ya; c = vecLength(xab, yab);
  xac = xc-xa; yac = yc-ya; b = vecLength(xac, yac);
  xbc = xc-xb; ybc = yc-yb; a = vecLength(xbc, ybc);
  if (b==0 || c==0 || a*a>=b*b+c*c) return circle2(xb,yb,xc,yc);
  if (b*b>=a*a+c*c) return circle2(xa,ya,xc,yc);
  if (c*c>=a*a+b*b) return circle2(xa,ya,xb,yb);
  d = 2*(xab*yac - yab*xac);
  xcenter = xa + (yac*c*c-yab*b*b)/d;
  ycenter = ya + (xab*b*b-xac*c*c)/d;
  radius = vecLength(xa-xcenter, ya-ycenter);
  return;
}
//Get enclosing circle for 4 points of the x, y array and return which
//of the 4 points we may eliminate
//Point indices of the 4 points are in global array fourIndices
function circle4(x, y) {
  rxy = newArray(12); //0...3 is r, 4...7 is x, 8..11 is y
  circle3(x[fourIndices[1]], y[fourIndices[1]], x[fourIndices[2]], y[fourIndices[2]], x[fourIndices[3]], y[fourIndices[3]]);
  rxy[0] = radius; rxy[4] = xcenter; rxy[8] = ycenter;
  circle3(x[fourIndices[0]], y[fourIndices[0]], x[fourIndices[2]], y[fourIndices[2]], x[fourIndices[3]], y[fourIndices[3]]);
  rxy[1] = radius; rxy[5] = xcenter; rxy[9] = ycenter;
  circle3(x[fourIndices[0]], y[fourIndices[0]], x[fourIndices[1]], y[fourIndices[1]], x[fourIndices[3]], y[fourIndices[3]]);
  rxy[2] = radius; rxy[6] = xcenter; rxy[10] = ycenter;
  circle3(x[fourIndices[0]], y[fourIndices[0]], x[fourIndices[1]], y[fourIndices[1]], x[fourIndices[2]], y[fourIndices[2]]);
  rxy[3] = radius; rxy[7] = xcenter; rxy[11] = ycenter;
  radius = 0;
  for (i=0; i<4; i++)
    if (rxy[i]>radius) {
      badIndex = i;
      radius = rxy[badIndex];
    }
  xcenter = rxy[badIndex + 4]; ycenter = rxy[badIndex + 8];
  return badIndex;
}

function vecLength(dx, dy) {
  return sqrt(dx*dx+dy*dy);
}
