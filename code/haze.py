# Import all the necessary packages to your arsenal
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import errno
import sys
# print ('Number of arguments:', len(sys.argv), 'arguments.')
# print ('Argument List:', str(sys.argv))
cwd = os.getcwd()
if (len(sys.argv)==1):
    files = os.listdir(cwd+"/clear_img")
    print ('files in the folder "clear_img" are ',files)
    flag = 1
else:
    files = os.listdir(sys.argv[1])
    print ('files in the folder "'+sys.argv[1]+'" are ',files)
    flag = 0
if (os.path.isdir(cwd+"/haze_img")):
    pass
else:
    try:
        os.makedirs(cwd+"/haze_img")
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

for i in files:
    if i.lower().endswith(('.png', '.jpg', '.jpeg')):
        if (flag == 1):
            J = cv2.imread(cwd+"/clear_img/"+str(i))
        else:
            J = cv2.imread(cwd+"/"+sys.argv[1]+"/"+str(i))
        J = cv2.cvtColor(J, cv2.COLOR_BGR2RGB)
        R = J[:,:,2]/255.0
        G = J[:,:,1]/255.0
        B = J[:,:,0]/255.0
        mu, sigma = 0.93, 0.013    # appear to be giving values in the range of 0.86.. to 0.99..
        d = np.random.uniform(0,1,R.shape).astype(np.float32)
        beta = 1.0
        t = np.empty(d.shape)
        t = np.exp(-beta*d)
        A = np.random.normal(mu, sigma, J.shape).astype(np.float32)
        # print(np.max(k),np.min(k))
        I_R = np.multiply(R,t)-np.multiply(A[:,:,2],t)+A[:,:,2]
        I_G = np.multiply(G,t)-np.multiply(A[:,:,1],t)+A[:,:,1]
        I_B = np.multiply(B,t)-np.multiply(A[:,:,0],t)+A[:,:,0]
        I_R = I_R - np.min(I_R)
        I_R = 255.0*I_R/np.max(I_R)
        I_G = I_G - np.min(I_G)
        I_G = 255.0*I_G/np.max(I_G)
        I_B = I_B - np.min(I_B)
        I_B = 255.0*I_B/np.max(I_B)
        # print(np.max(I_R),np.max(I_G),np.max(I_B))
        O = cv2.merge(np.float32(([I_R,I_G,I_B])))
        O = cv2.cvtColor(O, cv2.COLOR_BGR2RGB)
        # Plot the generated Haze map
        plt.imshow(O)
        plt.title('Hazed Image')
        # plt.show()
        cv2.imwrite(cwd+'/haze_img/' + i,O)
        # exit()
