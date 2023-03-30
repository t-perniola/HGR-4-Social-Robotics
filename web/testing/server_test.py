# coding=utf-8
import pickle
import socket
import struct
import mediapipe as mp
import numpy as np
import cv2
import utilities as ut
from flask import Flask, render_template, Response


HOST = ''
PORT = 8089

server_test = Flask(__name__)

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print('Socket created')

s.bind((HOST, PORT))
print('Socket bind complete')
s.listen(10)
print('Socket now listening')

conn, addr = s.accept()

data = b'' ### CHANGED
payload_size = struct.calcsize("L") ### CHANGED

sequence = []
predictions = []
gesto = ""

while True:    

    # Retrieve message size
    while len(data) < payload_size:
        data += conn.recv(4096)

    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack("L", packed_msg_size)[0] 

    # Retrieve all data based on message size
    while len(data) < msg_size:
        data += conn.recv(4096)

    frame_data = data[:msg_size]
    data = data[msg_size:]

    # Extract frame
    frame = pickle.loads(frame_data, encoding = 'latin1')

    with ut.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5) as hands:

        image, results = ut.mediapipe_detection(frame, hands) 
        first_hand_keypoints = np.zeros(21*3)
        second_hand_keypoints = np.zeros(21*3)

        if results.multi_hand_landmarks:
            for num, hand_landmarks in enumerate(results.multi_hand_landmarks):

                ut.mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    ut.mp_hands.HAND_CONNECTIONS,
                    ut.mp_drawing.DrawingSpec(
                        color=(150, 50, 10), thickness=2, circle_radius=2),
                    ut.mp_drawing.DrawingSpec(
                        color=(80,255,20), thickness=2, circle_radius=2)
                )
                    
                if num == 0:   
                    first_hand_keypoints = ut.extract_keypoints_hands(results, hand_landmarks)
                    #print("\n1st hand kp:", first_hand_keypoints_test)
                if num == 1:
                    second_hand_keypoints = ut.extract_keypoints_hands(results, hand_landmarks)
                    #print("\n2nd hand kp:", second_hand_keypoints_test)

            keypoints = np.concatenate([first_hand_keypoints, second_hand_keypoints])  

        else: 
            keypoints = np.zeros(21*6)         
                        
        # 2. Prediction logic
        sequence.append(keypoints)
        sequence = sequence[-30:]

        #print(len(sequence))
        #print(len(predictions))
        
        if len(sequence) == 30:            

            res = ut.grid_search.predict_proba([keypoints])
            res = res[0] # è contenuto in un array | TODO togliere
            print("HAGRID", ut.labels_hagrid[np.argmax(res)])
            predictions.append(np.argmax(res))
            
        #3. Viz logic
            if np.unique(predictions[-10:])[0]==np.argmax(res): 
                if res[np.argmax(res)] > 0.5:                     
                    gesto = ut.labels_hagrid[np.argmax(res)]                            
            
            # Viz probabilities
            image = ut.prob_viz(res, ut.labels_hagrid, image, ut.colors_hagrid)        

    # Display
    cv2.imshow('frame', image)
    cv2.waitKey(1)

#s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#s.bind((HOST, PORT))
#s.listen(1)
#conn, addr = s.accept()    
#print("Connected by", addr)
#while True:
    #data = conn.recv(1024)
    #newData = str(data) + "_new"
    #if not data:
    #    break
    #conn.sendall(newData)



