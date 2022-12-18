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
        patch = patch.flatten().astype(int)
        window = window.flatten().astype(int)
        mean_left = np.mean(patch)
        mean_right = np.mean(window)
        std_left = np.std(patch)
        std_right = np.std(window)
        patch = (patch-mean_left)/(std_left+1e-13)
        window = (window-mean_right)/(std_right+1e-13)
        similarity_score = np.dot(patch,window)
    elif method == "SAD":
      abs_diff = np.absolute(patch-window).astype(int)
      if patch.ndim == 2:
        similarity_score = np.sum(abs_diff)
      elif patch.ndim == 3:
        similarity_score = list(map(np.sum,abs_diff))
    return similarity_score

def resize_images(I_left,I_right,max_disp,resize_factor):
  # resize images reduces time
  # use resize_factor = 0.1 for testing
  I_left = cv2.imread(I_left)
  #[:,:,0]
  # convert color image to graysclae
  I_left = cv2.cvtColor(I_left, cv2.COLOR_BGR2GRAY)
  I_right = cv2.imread(I_right)
  #[:,:,0]
  # convert color image to graysclae
  I_right = cv2.cvtColor(I_right, cv2.COLOR_BGR2GRAY)

  height = int((I_left[:,0]).size*resize_factor)
  width = int((I_left[0].size)*resize_factor)
  dsize = (width, height)
  # resize image
  I_left = cv2.resize(I_left, dsize)
  I_right = cv2.resize(I_right, dsize)
  max_disp = int(resize_factor*max_disp)

  return I_left,I_right,max_disp

def compute_depth(depth,f,b,h,w):
    
    depth_map = np.zeros(depth.shape, np.uint8)
    
    for x in range(h):

        for y in range(w):

            if depth[x,y] !=0:
                    depth_map[x,y]=(f * b) /depth[x,y]
            if depth[x,y] ==0:
                depth_map[x,y]=0

    cv2.imwrite('depth_map_3_SSD+batch.png',depth_map)     
#-------------------------------------------------------------------------

def stereo_match(I_left, I_right,kernel,max_offset,method,batch=False,outfile=None):

  left = I_left.astype(np.uint8)
  right= I_right.astype(np.uint8)
  h,w = left.shape  # assume that both images are same size   
  print(w,h)
  disp = np.zeros((w, h), np.uint8)
  disp.shape = h,w
      
  kernel_half = int(kernel / 2)
  offset_adjust = 255 / max_offset  # this is used to map depth/disp map output to 0-255 range

    
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

          if method in ["SSD","SAD"]:
                    best_offset =  np.argmin(scores) 
          elif method in ["NCC"]:
                    best_offset =  np.argmax(scores) 
          # set disp output for this x,y location to the best match
          disp[y, x] = best_offset * offset_adjust        

  if outfile != None:
    print('Saving disparity map...')
    out1 = f"{outfile}_{method}_{kernel}.png"
    disp = cv2.normalize(disp, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX)
    cv2.imwrite(out1,disp)
  # if depthmap:
  #   print('Saving depth map...')
  #   #disp = cv2.normalize(disp, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX)
  #   #disp = disp.astype(np.uint8)
  #   # depth_map = b*f/(disp+1e-13)
  #   # depth_map = depth_map.astype(np.uint8)
  #   # depth_map = cv2.medianBlur(depth_map,3)  
  #   # depth_map = np.zeros(disp.shape, np.uint8)
  #   # for x in range(h):
  #   #     for y in range(w):
  #   #         if disp[x,y] !=0:
  #   #             depth_map[x,y]=(f * b) /disp[x,y]
  #   #         if disp[x,y] ==0:
  #   #             depth_map[x,y]=0
  #   # out2 = f"{outfile}_{method}_{kernel}_depthmap.png"
  #   # cv2.imwrite(out2,depth_map)               

if __name__ == '__main__':
    start = time.time()
    I_left,I_right,max_disp = resize_images("im0.png","im1.png",290,0.5)
    for size in [5,7,9]:
      for method in ["SSD","SAD"]:
        print(method,size)
        stereo_match(I_left, I_right, size, max_disp, method,batch=True,outfile="artroom1") 
        print("\nTook:",time.time()-start)

    
    I_left,I_right,max_disp = resize_images("im0.png","im1.png",290,0.25)
    for size in [5,7,9]:
      print("NCC",size)
      stereo_match(I_left, I_right, size, max_disp,"NCC",outfile="artroom1") 
      print("\nTook:",time.time()-start)

    I_left,I_right,max_disp = resize_images("chess0.png","chess1.png",290,0.5)
    
    for size in [3,5,7,9]:
      for method in ["SSD","SAD"]:
        print(method,size)
        stereo_match(I_left, I_right, size, max_disp,method,batch=True,outfile="chess1") 
        print("\nTook:",time.time()-start)

    I_left,I_right,max_disp = resize_images("chess0.png","chess1.png",290,0.25)
    for size in [3,5,7,9]:
      print("NCC",size)
      stereo_match(I_left, I_right, size, max_disp,"NCC",outfile="chess1") 
      print("\nTook:",time.time()-start)

