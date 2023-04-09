# coding=utf-8
import socket
import pickle
import struct
import utilities as ut
import numpy as np
#import pepper_server as server_ut
from flask import Flask, render_template

# Declarations
test = Flask(__name__)

HOST = ''
PORT = 8089

# definisco funzione di inizializzazione socket
def init():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print('Socket created')

    s.bind((HOST, PORT))
    print('Socket bind complete')
    s.listen(10)
    print('Socket now listening')

    conn, addr = s.accept()  
    print('Got connection from', addr)

    conn.send('Thank you for connecting'.encode())

    return conn

# avvio socket
conn = init()

# Define ROUTES
@test.route("/")
def pepper():       
    return render_template("pepper.html")

@test.route("/<name>") # tramite richiesta GET, ottenuta dal bottone premuto ...
def gesture(name):  # ... ottengo il gesto corrispondente scelto
    conn.send(name.encode())
    if name == "recog":
        get_video_stream()
        return render_template("web_control.html", gesture = name)
    else:  return render_template("web_control.html", gesture = name)

def get_video_stream():    
    data = b'' 
    payload_size = struct.calcsize("L") 

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

                res = ut.model.predict(np.expand_dims(sequence, axis=0))[0]                
                print("MYDS:", ut.labels[np.argmax(res)])
                predictions.append(np.argmax(res))
                
            #3. Viz logic
                if np.unique(predictions[-10:])[0]==np.argmax(res): 
                    if res[np.argmax(res)] > 0.5:                     
                        gesto = ut.labels[np.argmax(res)]                            
                
                # Viz probabilities
                image = ut.prob_viz(res, ut.labels, image, ut.colors) 


if __name__ == "__main__":
    test.run()


