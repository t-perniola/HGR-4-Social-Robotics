from naoqi import ALProxy

ROBO_IP = "127.0.0.1"
PORT = 9559

# init
test = ALProxy("ALTextToSpeech", ROBO_IP, PORT) # mi connetto al Pepper virtuale
test.say("Hello, world!")

test = ALProxy("ALBehaviorManager", ROBO_IP, PORT)
test.runBehavior("pepper_hgr-650b9f/hello")

#motion = ALProxy("ALMotion", ROBO_IP, PORT)
#motion.moveTo(1.0, 0.0, 0.0)

