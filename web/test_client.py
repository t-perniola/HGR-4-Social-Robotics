# coding=utf-8
import socket
from naoqi import ALProxy

ROBO_IP = "127.0.0.1"
PORT = 9559

clientsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
clientsocket.connect(('localhost', 8089))

text = ALProxy("ALTextToSpeech", ROBO_IP, PORT)
text.say("connected")

move = ALProxy("ALBehaviorManager", ROBO_IP, PORT)

posture = ALProxy("ALRobotPosture", ROBO_IP, PORT)
posture.goToPosture("Stand", 0.5)

print(clientsocket.recv(1024).decode())

while True:
    gesture = str(clientsocket.recv(1024).decode())
    print(gesture)

    if gesture != "favicon.ico":
        print("gesto scelto:", gesture)
        text.say(gesture)
        behavior = "pepper_choregraphe-27c329/" + gesture
        move.runBehavior(behavior)    

        # ritorno a posiz originale
        posture.goToPosture("Stand", 0.7)

        # avvio fotocamera pc, simulando quella di pepper
        

#clientsocket.close()