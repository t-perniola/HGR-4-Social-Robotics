#import vision_definitions
import cv2
import time
from naoqi import ALProxy
#from PIL import Image, ImageDraw

ROBO_IP = "127.0.0.1"
PORT = 9559

# simuliamo fotocamera robot virtuale, mettendo la nostra

cameraProxy = ALProxy("ALVideoDevice", ROBO_IP, PORT)
cap = cv2.VideoCapture(0) 

if cap.isOpened(): # try to get the first frame rval
    frame = cap.read() 
else: 
    rval = False 

while cap.isOpened(): 
    rval, frame = cap.read() 
    frame = cv2.resize(frame, (640, 480)) 
    cv2.imshow("preview", frame)
    key = cv2.waitKey(1)  
    time.sleep(0.15) 
    b,g,r = cv2.split(frame) # get b,g,r 
    rgb_img = cv2.merge([r,g,b]) # switch it to rgb 
    set_cam = cameraProxy.putImage(0, 640, 480, rgb_img.tobytes()) # serve per simulare fotocamera robot vituale

    if key == 27:
      break 

cap.release()
