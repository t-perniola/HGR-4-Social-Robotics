# coding=utf-8
import cv2
import socket
import pickle
import numpy as np
import struct ## new

HOST=''
PORT=8089

s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
print('Socket created')

s.bind((HOST,PORT))
print('Socket bind complete')
s.listen(10)
print('Socket now listening')

conn,addr=s.accept()

cap = cv2.VideoCapture(0)

while cap.isOpened(): #finchè webcam attiva...

    # Read feed
    ret, frame = cap.read() #prendiamo il singolo frame           

    # Show to screen
    cv2.imshow('OpenCV Feed', frame)

    # Break gracefully
    if cv2.waitKey(10) & 0xFF == ord('q'): #se premiamo q -> quit
        break

cap.release()
cv2.destroyAllWindows()