import cv2 as cv
import numpy as np
from math import exp
import time
import glob

####  defining haze degree factor function  ###

def haze_degree_factor (img):
    
    img1 = cv.resize(img, (320,240),interpolation=cv.INTER_CUBIC)  # resizing the image
    image = cv.cvtColor(img1, cv.COLOR_BGR2RGB)   # converting to RGB color model
    r,g,b = cv.split(image)
    width, height = r.shape   # obtaining shape parameters of image
    pro = width * height  # defining size of image

    if not width or not height:
        return 0
    
    d_i = np.zeros(img1.shape[:2], dtype='uint8')
    b_i = np.zeros(img1.shape[:2], dtype='uint8')
    
    # defining constants needed
    lam = 1/3   # lambda
    mu = 5.1
    gamma = 2.9
    sigma = 0.2461
    
    for x in range(width):
        for y in range(height):
            d_i[x,y] = min(r[x,y],g[x,y],b[x,y])  # minimum value of 3 channels
            b_i[x,y] = max(r[x,y],g[x,y],b[x,y])  # maximum value of 3 channels     
    
    D = d_i.sum()/pro   # dark value
    B = b_i.sum()/pro
    C = D - B   # contrast value

    max_b_i = b_i.max()
    A0 = (lam*max_b_i) + ((1-lam)*B)    # estimating atmospheric light value
    
    x1 = 1 - (D/A0)
    x2 = (C/A0)
    exponent = sigma - ((1/2)*((mu*x1)+(gamma*x2)))

    w = exp(exponent)
    return w

###  approximate ranges of w with fog intensity levels  ###
# if w <= 0.6:
#     str = 'sunny or fog free image'
# elif w>0.6 and w<=0.7:
#     str = 'low fog image'
# elif w>0.7 and w<=0.8:
#     str = 'moderate fog image'
# else:
#     str = 'high fog image'

### testing for folder of images  ###
Total_Images = 0
count = 0
sum = 0

for image in glob.glob("c:/Users/KONDAPALLI RAJESH/.vscode/Visual studio code files/winter Internship/non_foggy/*.png"):
    img = cv.imread(image)
    Total_Images += 1
    
    start1 = time.time()
    w = haze_degree_factor(img)
    
    if w > 0.65:
        count += 1
    
    end1 = time.time()
    sum = sum + end1-start1

print("time taken for single image (in milli seconds): " + str(sum*1000/Total_Images))
print('Total Images in Folder: ' + str(Total_Images))
print('Foggy Images in Folder: ' + str(count))
print('sunny Images in Folder: ' + str(Total_Images - count))


###  testing for single image  ###
img = cv.imread('c:/Users/KONDAPALLI RAJESH/.vscode/Visual studio code files/winter Internship/foggy/foggy_night.png')
img1 = cv.resize(img, (320,240),interpolation=cv.INTER_CUBIC)
cv.imshow('image',img1)

start1 = time.time()
print(haze_degree_factor(img))
end1 = time.time()

print("time taken for the image (in milli seconds): " + str((end1 - start1)*1000))

cv.waitKey(0)
cv.destroyAllWindows()