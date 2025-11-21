## Testing Models for Geometry Prior Subset.

*These are tested on CO3D images, here, a hydrant.*
This should work with any image input. 

#### Sobel Operator
Calculates the image gradient magnitude at each pixel. 
Uses two 3x3 conv. kernels to eventually get a `final edge strength'.

**Result:** RGB edge map. 

#### Canny Edge Detector
Edge detection algorithm. Finding the intensity gradients (like Sobel) is one part. Also, uses Non Maximum Suppression to thin edges. 

**Result:** RGB edge map. 


#### MiDaS (Monocular Depth Estimation Model)
Pre-trained model, trained on a large set of depth datasets. 

**Result:** Relative depth map. 

#### MiDaS + Edge Detection 
Using Sobel (or other operators) on the single channel output from MiDaS. Ideally, detecting changes in distance rather than intensity. 

**Result:** Geometric edge map. 
- Potentially could be useful for identifying occlusion boundaries. 

#### Mask RCNN, Detectron
Instance segmentation models. 

**Result:** Segmentations masks, visibility/occlusion prior. 

