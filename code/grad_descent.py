# import haze
# from haze import files
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import errno
import sys
cwd = os.getcwd()
depth_files = os.listdir(cwd+"/depthmap")
print ('files in the folder "/depthmap" are ',depth_files)
if (len(sys.argv)==2):
    iter = sys.argv[1]
else:
    iter = 30

for i in depth_files:
    img_file_name = i.replace("depthmap_","")
    _I = cv2.imread(cwd+"/haze_img/" + img_file_name )
    I = cv2.cvtColor(_I, cv2.COLOR_BGR2RGB)
    H,S,V = cv2.split(cv2.cvtColor(I, cv2.COLOR_BGR2HSV))
    S = S/255.0
    V = V/255.0
    img_h = V.shape[0]
    img_w = V.shape[1]
    zero_array = np.zeros(V.shape)
    theta_0, theta_1, theta_2 = zero_array, zero_array+1 ,zero_array-1
    sum, wsum, vsum, ssum = 0, 0, 0, 0
    sigma = 0
    for j in range(0,iter):
        temp = j - theta_0-np.multiply(theta_1,V)-np.multiply(theta_2,S)
        wsum += np.sum(temp)
        vsum += np.sum(np.multiply(V,temp))
        ssum += np.sum(np.multiply(S,temp))
        sum += np.sum(np.multiply(temp,temp))
        sigma = sum/(img_h*img_w)
        theta_0 += wsum
        theta_1 += vsum
        theta_2 += ssum
    t_0 = theta_0[0,0]
    t_1 = theta_1[0,0]
    t_2 = theta_2[0,0]
    print(t_0,t_1,t_2)
