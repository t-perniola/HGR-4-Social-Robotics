# coding=utf-8
import socket
import cv2
import struct
import pickle
#import threading
import numpy as np
from naoqi import ALProxy

ROBO_IP = "127.0.0.1"
PEPPER_IP = "192.168.206.245"
PORT = 9559

## get NAOqi module ALVideoDevice with proxy
cameraProxy = ALProxy("ALVideoDevice", ROBO_IP, PORT)

def get_sub():
    # subscribe top camera
    AL_kTopCamera = 0
    AL_kQVGA = 1          # risoluzione: 320x240
    AL_kBGRColorSpace = 13
    captureDevice = cameraProxy.subscribeCamera(
        "f", AL_kTopCamera, AL_kQVGA, AL_kBGRColorSpace, 15)

    # create image
    width = 320
    height = 240
    image = np.zeros((height, width, 3), np.uint8)

    return width, height, image, captureDevice

boolean = False

def send_stream():
    while boolean is False:        
        print("eccoci")

        # get image
        width, height, image, captureDevice = get_sub()
        result = cameraProxy.getImageRemote(captureDevice)
        cameraProxy.releaseImage(captureDevice)
        
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

            data = pickle.dumps(image) 
            clientsocket.sendall(struct.pack("L", len(data))+data)   

def send_stream_pc():
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
    
#t1 = threading.Thread(target=send_stream)   
#t1.start()

clientsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
clientsocket.connect(('localhost', 8089))

text = ALProxy("ALTextToSpeech", ROBO_IP, PORT)
text.say("connected")

move = ALProxy("ALBehaviorManager", ROBO_IP, PORT)

posture = ALProxy("ALRobotPosture", ROBO_IP, PORT)
posture.goToPosture("Stand", 0.5)

# if virtual pepper
subscriber = cameraProxy.subscribeCamera("demo", 0, 2, 13, 5) # params: mod_name, cam_idx, resolution_idx (2: 640x480), colors_idx, fps
cap = cv2.VideoCapture(0)

# recv welcome message
print(clientsocket.recv(1024).decode())

# enter loop
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
            
            # il client riceve questa var inutile quando è finita l'esecuzione di Pepper
            #foo = str(clientsocket.recv(1024).decode())
            #print("foo:", foo)

            send_stream_pc()

            result = str(clientsocket.recv(1024).decode())
            print("resu:", result)            

            if result == "corretto":
                text.say(result)
                correct = True

            elif result == "sbagliato":
                text.say(result)
                pass

    print("res correct")    
    clientsocket.close()



