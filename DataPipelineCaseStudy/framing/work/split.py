import cv2
import numpy as np
import os
import sys

input_file = sys.argv[1]
output_folder = sys.argv[2]

# Playing video from file:
cap = cv2.VideoCapture(input_file)

try:
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
except OSError:
    print ('Error: Creating directory of data')

currentFrame = 0
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret: break
    # Saves image of the current frame in jpg file
    #./data-20-200/
    name = './' + output_folder + '/frame' + str(currentFrame) + '.jpg'
    #print ('Creating ' + name)
    cv2.imwrite(name, frame)
    #print(ret)
    # To stop duplicate images
    currentFrame += 1

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
