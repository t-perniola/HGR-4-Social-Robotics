# coding=utf-8
import cv2
import socket
import pickle
import struct 
from naoqi import ALProxy
from PIL import Image

ROBO_IP = "127.0.0.1"
PORT = 9559

cap = cv2.VideoCapture(0)
clientsocket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
clientsocket.connect(('localhost', 8089))

cameraProxy = ALProxy("ALVideoDevice", ROBO_IP, PORT)
subscriber = cameraProxy.subscribeCamera("demo", 0, 2, 13, 5) # params: mod_name, cam_idx, resolution_idx (2: 640x480), colors_idx, fps
    
img = cameraProxy.getImageRemote(subscriber)
cameraProxy.releaseImage(subscriber)

while cap.isOpened():
    ret, frame = cap.read()
    #cv2.imshow("frame", frame)
    data = pickle.dumps(frame) 
    clientsocket.sendall(struct.pack("L", len(data))+data)

    # Break gracefully
    if cv2.waitKey(10) & 0xFF == ord('q'): #se premiamo q -> quit
        break

cap.release()
cv2.destroyAllWindows()

clientsocket.close()