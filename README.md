# Multiview Stereo Reconstruction
Note: we restrict our focus on two view stereo reconstruction.
## STEP 1: Camera Calibration
 - intrinsics parameters: projection matrices P and P' for left and right camera
 - extrinsics parameters: rotataton R and translation T of one camera with respect to the other camera
## STEP 2: Correspondence Matching
 - sparse correspondence (sift key points -> estimate fundamental matrix F)
 - dense correspondence (region-based 1D searching)
## STEP 3: 3D Reconstruction
 - disparity -> depth map -> 3D dense point cloud
