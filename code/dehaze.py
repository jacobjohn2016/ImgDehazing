# Import all the necessary packages to your arsenal
import numpy as np
import cv2
import matplotlib.pyplot as plt

def localmin(D, r=15):
    R = int(r/2)
    imax = D.shape[0]
    jmax = D.shape[1]
    LM = np.zeros([imax,jmax])
    for i in np.arange(D.shape[0]):
        for j in np.arange(D.shape[1]):
            iL = np.max([i-R,0])
            iR = np.min([i+R, imax])
            jT = np.max([j-R,0])
            jB = np.min([j+R, jmax])
            # print(D[iL:iR+1,jT:jB+1].shape)
            LM[i,j] = np.min(D[iL:iR+1,jT:jB+1])
    return LM

filename = '4.png'
# Read the Image
_I = cv2.imread('../data/hazy/' + filename )
# opencv reads any image in Blue-Green-Red(BGR) format,
# so change it to RGB format, which is popular.
I = cv2.cvtColor(_I, cv2.COLOR_BGR2RGB)
# Split Image to Hue-Saturation-Value(HSV) format.
H,S,V = cv2.split(cv2.cvtColor(_I, cv2.COLOR_BGR2HSV) )

# Calculating Depth Map using the linear model fit by ZHU et al.
# Refer Eq(8) in mentioned research paper (README.md file) page 3535.
theta_0 = 0.121779
theta_1 = 0.959710
theta_2 = -0.780245
sigma   = 0.041337
epsilon = np.random.normal(0, sigma, H.shape )
D = theta_0 + theta_1*V + theta_2*S + epsilon

LMD = localmin(D, 15)
# LMD = D
LMD = LMD - np.min(LMD)
LMD = 255*LMD/np.max(LMD)

# Plot the generated raw depth map
plt.imshow(LMD, cmap='inferno')
plt.title('Raw Depth Map')
plt.xticks([]); plt.yticks([])
plt.show()

# save the depthmap.
# Note: It will be saved as gray image.
cv2.imwrite('../data/dmap/' + filename, LMD)
