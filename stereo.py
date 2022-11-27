#!/usr/bin/env python
# --------------------------------------------------------------------
# Simple sum of squared differences (SSD) stereo-matching using Numpy
# --------------------------------------------------------------------

# Copyright (c) 2016 David Christian
# Licensed under the MIT License
import numpy as np
from PIL import Image

def stereo_match(left_img, right_img, kernel, max_offset):
    # Load in both images, assumed to be RGBA 8bit per channel images
    left_img = Image.open(left_img).convert('L')
    left = np.asarray(left_img)
    right_img = Image.open(right_img).convert('L')
    right = np.asarray(right_img)    
    w, h = left_img.size  # assume that both images are same size   
    
    # Depth (or disparity) map
    depth = np.zeros((w, h), np.uint8)
    depth.shape = h, w
       
    kernel_half = int(kernel / 2)
    offset_adjust = 255 / max_offset  # this is used to map depth map output to 0-255 range
      
    for y in range(kernel_half, h - kernel_half):              
        print("\rProcessing.. %d%% complete"%(y / (h - kernel_half) * 100), end="", flush=True)        
        
        for x in range(kernel_half, w - kernel_half):
            best_offset = 0
            prev_ssd = 65534
            
            for offset in range(max_offset):               
                ssd = 0
                ssd_temp = 0                            
                
                # v and u are the x,y of our local window search, used to ensure a good match
                # because the squared differences of two pixels alone is not enough ot go on
                for v in range(-kernel_half, kernel_half):
                    for u in range(-kernel_half, kernel_half):
                        # iteratively sum the sum of squared differences value for this block
                        # left[] and right[] are arrays of uint8, so converting them to int saves
                        # potential overflow
                        ssd_temp = int(left[y+v, x+u]) - int(right[y+v, (x+u) - offset])  
                        ssd += ssd_temp * ssd_temp              
                
                # if this value is smaller than the previous ssd at this block
                # then it's theoretically a closer match. Store this value against
                # this block..
                if ssd < prev_ssd:
                    prev_ssd = ssd
                    best_offset = offset
                            
            # set depth output for this x,y location to the best match
            depth[y, x] = best_offset * offset_adjust
                                
    # Convert to PIL and save it
    Image.fromarray(depth).save('depth.png')

    

if __name__ == '__main__':
    stereo_match("im2.png", "im1.png", 6, 30)   #6x6 local search kernel, 30 pixel search range

    # this is because I thought I had to create the depth map from the disparity map
    #img = Image.open("depth2.png").convert('L')
    #depth = np.asarray(img)
    #w, h = img.size
    #depth_map = np.zeros(depth.shape, np.uint8)
    # Focal Length - Pixels | Baseline -cm | Depth_map - cm
    #for x in range(h):
     #   for y in range(w):
    #        if depth[x,y] !=0:
   #             depth_map[x,y]=(1733.74 * 536.62) /depth[x,y];
    #Image.fromarray(depth_map).convert("L").save('depth3.png')
