# Import all the necessary packages to your arsenal
import numpy as np
import cv2
import matplotlib.pyplot as plt


def guide(I,P,r,e):

    h,w=np.shape(I)
    window = np.ones((r,r))/(r*r)
    
    meanI = sig.convolve2d(I, window,mode='same')
    meanP = sig.convolve2d(P, window,mode='same')
    
    corrI = sig.convolve2d(I*I, window,mode='same')
    corrIP = sig.convolve2d(I*P, window,mode='same')
    
    
    varI = corrI - meanI*meanI
    covIP = corrIP - meanI*meanP
    a = covIP/(varI+e)
    b = meanP - a*meanI
    
    meana = sig.convolve2d(a, window,mode='same')
    meanb = sig.convolve2d(b, window,mode='same')
    
    q = meana*I+meanb

    return q

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

r = 8; # try r=2, 4, or 8
eps = 0.2 * 0.2; # try eps=0.1^2, 0.2^2, 0.4^2
eps *= 255 * 255;   # Because the intensity range of our images is [0, 255]
# LMD1=guide(D,LMD,r,eps)
# import guidedfilter
# from guidedfilter import guidedfilter as gF
# LMD2=gF(D,LMD,r,eps)

# Plot the generated raw depth map
plt.imshow(LMD, cmap='inferno')
plt.title('Raw Depth Map')
plt.xticks([]); plt.yticks([])
plt.show()

# save the depthmap.
# Note: It will be saved as gray image.
cv2.imwrite('../data/dmap/' + filename, LMD)
