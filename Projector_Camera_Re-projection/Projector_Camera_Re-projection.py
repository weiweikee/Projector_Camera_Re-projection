import cv2
import numpy as np
import urllib.request
from goprocam import GoProCamera
from goprocam import constants

#### GoPro Settings ####
gpCam = GoProCamera.GoPro()
gpCam.overview()
########################

## Create Projector Window 
cv2.namedWindow("Projector", cv2.WND_PROP_FULLSCREEN)
cv2.moveWindow("Projector", 3500, 0)
cv2.setWindowProperty("Projector", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

initialPhoto = False
camCalib = False
camCalibPic = False
findCorners = False
loop = 0

## Crop Variables
realW = None
realH = None
realX = None
realY = None

## Get Calibration Data
def load_calibration_data():
    print ("reading calibration file")
    calibrationDataFile = np.load("calibration.npz")
    print (calibrationDataFile.files)
    distCoeff = calibrationDataFile['distCoeff']
    intrinsic_matrix = calibrationDataFile['intrinsic_matrix']
    return distCoeff, intrinsic_matrix

while True:
    proj_img = cv2.imread("CircleGrid_3416x1920.jpg", 0)
    dist, K = load_calibration_data()
    
    ## Take photo without Projection to send to CNN for defect identification
    if initialPhoto == False:
        unProjectedImg = gpCam.take_photo(1)
        initialPhoto = True
        h, w = unProjectedImg.shape[:2]
    	newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w,h), 1, (w,h))
    	dst = cv2.undistort(unProjectedImg, K, dist, None, newcameramtx)
    	x, y, w, h = roi
    	unProjectedImg = dst[y:y+h, x:x+w]
        cv2.imwrite('Initial_Image.png', unProjectedImg)

    ## Show Projector Image if the corners are not found
    if findCorners == False:
        cv2.imshow('Projector', proj_img) 
    

    if camCalib == False and loop > 1:     
        ## Take the picture of dotted grid
        if camCalibPic == False:
            print("Taking Picture")
            imgURL = gpCam.take_photo(1)
            camCalibPic = True

        ## Perform Bounding Box and Match the dotted grid with real image
        else:
            if findCorners == False:
                url = urllib.request.urlopen(imgURL)
                photo = np.array(bytearray(url.read()), dtype=np.uint8)
                img = cv2.imdecode(photo, -1)
                
                ###### undistort ######
                h, w = img.shape[:2]
                newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w,h), 1, (w,h))
                dst = cv2.undistort(img, K, dist, None, newcameramtx)
                x, y, w, h = roi
                img = dst[y:y+h, x:x+w]
                h, w = img.shape[:2]
                ## Crop the distorted part out 
                img = img[300:h-300, 300:w-300]
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                ###### Find estimated Shape ######
                ret, thresh = cv2.threshold(gray, 135, 255,0) ## (gray, 135, 255, 0) for brighter room  ## (gray, 155, 255, 0) for darker room
                ## find all contours
                newImg, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for i in range(len(contours)):
                    cnt = contours[i]
                    # Find bounding box for all contours
                    x,y,w,h = cv2.boundingRect(cnt)
                    # Get bounding box bigger than 200 by 200 and set the crop variables
                    if w > 200 and h > 200:
                        realW = w
                        realX = x
                        realY = y
                        realH = h

                # Draw Rectangle on the bounding box
                if realW != None and realH != None:
                    cv2.rectangle(gray, (realX,realY), (realX+realW, realY+realH), (0,255,0), 2)

                findCorners = True
                # see the bounding box image
                cv2.imshow('Image', gray)
                # see the threshold image
                cv2.imshow('Gray', thresh)

    # Loop is required because projection is slower than the picture taking so 1 loop is a buffer
    loop += 1

    ## if corners
    if findCorners:       
    	# Grab the processed image and reproject
        proj_img = cv2.imread('ProcessingImageWoBG.jpg', 0)
        proj_img = np.array(proj_img, dtype=np.float)
        proj_img /= 255.0
        a_channel = np.ones(proj_img.shape, dtype=np.float)/4.0
        proj_img = proj_img*a_channel

        # reproject the projector image from capture image 
        proj_img = proj_img[realY:realY+realH, realX:realX+realW]
        proj_w, proj_h = proj_img.shape[:2]
        cv2.imshow('Projector', proj_img) 

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
