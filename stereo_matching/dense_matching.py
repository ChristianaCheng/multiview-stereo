import numpy as np
import cv2
import time
def compute_similarity(patch,window,method):
  # can handle batch or non batch computation
    if method == "SSD":
      sqrt_diff = np.square(patch-window).astype(int)
      if patch.ndim == 2:
        similarity_score = np.sum(sqrt_diff)
      elif patch.ndim == 3:
        similarity_score = list(map(np.sum,sqrt_diff))
    elif method == "NCC":
      pass
    return similarity_score

def resize_images(I_left,I_right,max_disp,resize_factor):
  # resize images reduces time
  # use resize_factor = 0.1 for testing
  I_left = cv2.imread(I_left)[:,:,0]
  I_right = cv2.imread(I_right)[:,:,0]

  height = int((I_left[:,0]).size*resize_factor)
  width = int((I_left[0].size)*resize_factor)
  dsize = (width, height)
  # resize image
  I_left = cv2.resize(I_left, dsize)
  I_right = cv2.resize(I_right, dsize)
  max_disp = int(resize_factor*max_disp)

  return I_left,I_right,max_disp

#-------------------------------------------------------------------------

def stereo_match(I_left, I_right, kernel, max_offset,method,batch=True,depth_map=False):

  left = I_left.astype(np.uint8)
  right= I_right.astype(np.uint8)
  h,w = left.shape  # assume that both images are same size   
  print(w,h)
  # Depth (or disparity) map
  depth = np.zeros((w, h), np.uint8)
  depth.shape = h,w
      
  kernel_half = int(kernel / 2)
  offset_adjust = 255 / max_offset  # this is used to map depth map output to 0-255 range

    
  for y in range(kernel_half, h - kernel_half):              
      print("\rProcessing.. %d%% complete"%(y / (h - kernel_half) * 100), end="", flush=True)        
      
      for x in range(kernel_half, w - kernel_half):

          if not batch:
            scores = []
            for offset in range(max_offset):     
                patch = left[y-kernel_half:y+kernel_half+1, x-kernel_half:x+kernel_half+1].astype(int)
                if x-kernel_half-offset >= 0:
                  window = right[y-kernel_half:y+kernel_half+1, x-kernel_half-offset:x+kernel_half+1-offset].astype(int)
                  similarity_score = compute_similarity(patch,window,method)
                  scores.append(similarity_score)
             
          # batch computation
          # similarity scores are computed parellely for each scanline
          else:
                windows = np.array([   
      right[y-kernel_half:y+kernel_half+1, x-kernel_half-offset:x+kernel_half+1-offset]\
      for offset in range(max_offset) if x-kernel_half-offset >= 0
              ]).astype(int)

                length = windows.shape[0]

                patch = left[y-kernel_half:y+kernel_half+1, x-kernel_half:x+kernel_half+1].astype(int)
                
                patches = np.array([   
                    patch for i in range(length)
                          ]).astype(int)
                scores = compute_similarity(patches,windows,method)

          if method in ['SSD']:
                    best_offset =  np.argmin(scores) 
          elif method in ["NCC"]:
                    best_offset =  np.argmax(scores) 
          # set depth output for this x,y location to the best match
          depth[y, x] = best_offset * offset_adjust
  if depth_map:
   pass
   #return depth_map         

  # cv2_imshow(depth)   
  cv2.imwrite('depth_3_SSD+batch.png',depth)                        

if __name__ == '__main__':
    start = time.time()
    I_left,I_right,max_disp = resize_images("im0.png","im1.png",290,0.1)
    stereo_match(I_left, I_right, 3, max_disp,"SSD")  
    print("\nTook:",time.time()-start)