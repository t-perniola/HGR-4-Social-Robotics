# coding=utf-8
import cv2
import numpy as np
from naoqi import ALProxy


PEPPER_IP = "192.168.71.245"
PORT = 9559

camProxy = ALProxy("ALVideoDevice", PEPPER_IP, PORT)
## get NAOqi module ALVideoDevice with proxy

tracker = ALProxy("ALTracker", PEPPER_IP, PORT)


# subscribe top camera
AL_kTopCamera = 0
AL_kQVGA = 1            # risoluzione: 320x240
AL_kBGRColorSpace = 13
captureDevice = camProxy.subscribeCamera(
    "f", AL_kTopCamera, AL_kQVGA, AL_kBGRColorSpace, 15)

# create image
width = 320
height = 240
image = np.zeros((height, width, 3), np.uint8)

while True:

    # get image
    result = camProxy.getImageRemote(captureDevice)
    camProxy.releaseImage(captureDevice)

    if result == None:
        print('cannot capture.')
    elif result[6] == None:
        print('no image data string.')
    else:

        # translate value to mat
        values = map(ord, list(result[6]))
        i = 0
        for y in range(0, height):
            for x in range(0, width):
                image.itemset((y, x, 0), values[i + 0])
                image.itemset((y, x, 1), values[i + 1])
                image.itemset((y, x, 2), values[i + 2])
                i += 3

        # show image
        cv2.imshow("pepper-top-camera-320x240", image)

    # exit by [ESC]
    if cv2.waitKey(33) == 27:
        break

camProxy.unsubscribe(captureDevice)
