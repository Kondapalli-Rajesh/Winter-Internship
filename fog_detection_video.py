import cv2 as cv
import numpy as np
from math import exp
import time

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

capture = cv.VideoCapture('c:/Users/KONDAPALLI RAJESH/.vscode/Visual studio code files/winter Internship/videos/testing_videos/Testing_2.mp4')
# obtaining the total number of frames and fps (frames per second) in a video
frames = capture.get(cv.CAP_PROP_FRAME_COUNT)
fps = capture.get(cv.CAP_PROP_FPS)
print(f"Total number of frmaes = " + str(frames) + " and fps = " + str(fps))

count = 1
start = 0
ending_frame = 0
starting_frame = 0
sum = 0

while True:
    ret, frame = capture.read()
    capture.set(cv.CAP_PROP_POS_FRAMES, count)
    
    if ret == False or count >= frames:
        break
    
    start1 = time.time()
    
    w = haze_degree_factor(frame)
    if  w > 0.65 and start == 0:
        starting_frame = count
        start = 1
        print("fog started at frame = " + str(starting_frame))
    
    elif w <= 0.65 and start == 1:
        ending_frame = count
        start = 0 
        print("fog ended at frame = " + str(ending_frame))

    end1 = time.time()
    
    sum = sum + end1-start1
    count = count + 1
    cv.waitKey(20)

capture.release()
cv.destroyAllWindows()

if start == 0 and starting_frame == 0:
    print('There is no fog')
elif start == 1 and ending_frame == 0:
        print("fog ended at frame = " + str(frames))

print("time taken for single frame: " + str(sum*1000/frames))