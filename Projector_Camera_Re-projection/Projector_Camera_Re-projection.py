import cv2
import numpy as np
import urllib.request
from goprocam import GoProCamera
from goprocam import constants

#### GoPro Settings ####
gpCam = GoProCamera.GoPro()
# gpCam.overview()
# gpCam.gpControlSet(constants.Stream.BIT_RATE, constants.Stream.BitRate.B2_4Mbps)
#gpCam.gpControlSet(constants.Stream.WINDOW_SIZE, constants.Stream.WindowSize.W480)
# cap = cv2.VideoCapture("udp://127.0.0.1:10000")
########################

cv2.namedWindow("Projector", cv2.WND_PROP_FULLSCREEN)
cv2.moveWindow("Projector", 3500, 0)
cv2.setWindowProperty("Projector", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

camCalib = False
camCalibPic = False
loop = 0
realW = None
realH = None
realX = None
realY = None
findCorners = False

def load_calibration_data():
    print ("reading calibration file")
    calibrationDataFile = np.load("calibration.npz")
    print (calibrationDataFile.files)
    distCoeff = calibrationDataFile['distCoeff']
    intrinsic_matrix = calibrationDataFile['intrinsic_matrix']
    return distCoeff, intrinsic_matrix

while True:
    proj_img = cv2.imread("CircleGrid_3416x1920.jpg", 0)

    if findCorners == False:
        cv2.imshow('Projector', proj_img) 
        
    if camCalib == False and loop > 1:
        dist, K = load_calibration_data()
        print(dist)
        print(K)
        if camCalibPic == False:
            print("Taking Picture")
            imgURL = gpCam.take_photo(1)
            camCalibPic = True
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
                img = img[300:h-300, 300:w-300]

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                ###### Find estimated Shape ######
                ret, thresh = cv2.threshold(gray, 135, 255,0) #135, 255 for brighter room #155, 255 for darker room
                newImg, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                print(len(contours))
                for i in range(len(contours)):
                    cnt = contours[i]
                    x,y,w,h = cv2.boundingRect(cnt)
                    # find only size rectangle bigger than 200 by 200
                    if w > 200 and h > 200:
                        realW = w
                        realX = x
                        realY = y
                        realH = h

                if realW != None and realH != None:
                    cv2.rectangle(gray, (realX,realY), (realX+realW, realY+realH), (0,255,0), 2)

                findCorners = True
                # see the bounding box image
                cv2.imshow('Image', gray)
                # see the threshold image
                cv2.imshow('Gray', thresh)

    loop += 1

    if findCorners:       
        proj_img = img
        # reproject the projector image from capture image
        proj_img = proj_img[realY:realY+realH, realX:realX+realW]
        proj_w, proj_h = proj_img.shape[:2]
        cv2.imshow('Projector', proj_img) 

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
