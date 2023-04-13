# coding=utf-8
import socket
import cv2
import struct
import pickle
import threading
from naoqi import ALProxy

ROBO_IP = "127.0.0.1"
PORT = 9559

def send_stream():
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        cv2.imshow("frame", frame)
        data = pickle.dumps(frame) 
        clientsocket.sendall(struct.pack("L", len(data))+data)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'): #se premiamo q -> quit
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
t1 = threading.Thread(target=send_stream)   
t1.start()

clientsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
clientsocket.connect(('localhost', 8089))

text = ALProxy("ALTextToSpeech", ROBO_IP, PORT)
text.say("connected")

move = ALProxy("ALBehaviorManager", ROBO_IP, PORT)

posture = ALProxy("ALRobotPosture", ROBO_IP, PORT)
posture.goToPosture("Stand", 0.5)

cameraProxy = ALProxy("ALVideoDevice", ROBO_IP, PORT)
subscriber = cameraProxy.subscribeCamera("demo", 0, 2, 13, 5) # params: mod_name, cam_idx, resolution_idx (2: 640x480), colors_idx, fps

print(clientsocket.recv(1024).decode())

cap = cv2.VideoCapture(0)


while True:   

    print("ritorno qui")
    correct = False
    
    while correct is False:            

        gesture = str(clientsocket.recv(1024).decode())
        print("iniz:", gesture)        

        if gesture != "favicon.ico" and gesture != "recog":
            
            print("gesto scelto:", gesture)
            text.say(gesture)
            behavior = "pepper_choregraphe-27c329/" + gesture
            move.runBehavior(behavior)    

            # ritorno a posiz originale
            posture.goToPosture("Stand", 0.7)   
            
            # il client riceve questa var inutile quanod finita l'esecuzione di Pepper
            foo = str(clientsocket.recv(1024).decode())
            print("foo:", foo)

            result = str(clientsocket.recv(1024).decode())
            print("resu:", result)            

            if result == "corretto":
                text.say(result)
                correct = True

            elif result == "sbagliato":
                text.say(result)
                pass

    print("res correct")
    

#clientsocket.close()



