# Projector_Camera_Re-projection from Normal
Re-projecting Projector Image from Camera Image
**Python 3.6**
**OpenCv 3.4.0**
**GoProCam**


## Process
This project used a GoPro Camera and a Pico Projector to calibrate the camera and projector. <p>
1. **First**, the camera is calibrated using checkerboard pattern to reduce the GoPro distortion <p>
2. Then, the camera is used to capture projected image pattern (in this case dot pattern) on an normal 2D plane and uses OpenCv contour and bounding box to understand where the projector is projecting in the real world. The bounding box estimation is created through cv2.boundingRect(cnt) function. <p>
3. The captured image from the camera is cropped using the x, y, width, height given the bounding box. <p>
4. This processed image is reprojected onto the normal 2D plane.

## Libraries Used
**OpenCv 3.4.0:** https://opencv.org/ <p>
**GoProCam:** https://github.com/KonradIT/gopro-py-api


## To Do
Make it work on non-normal plane and 3D surfaces.
