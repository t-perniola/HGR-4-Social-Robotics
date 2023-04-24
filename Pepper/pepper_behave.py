from naoqi import ALProxy

ROBO_IP = "127.0.0.1"
PEPPER_IP = "192.168.206.245"
PORT = 9559

# init
test = ALProxy("ALTextToSpeech", ROBO_IP, PORT) # mi connetto al Pepper virtuale
test.say("Saluto!")

tablet = ALProxy("ALTabletService", ROBO_IP, PORT)
#tablet.showImage("https://em-content.zobj.net/source/noto-emoji-animations/344/waving-hand_1f44b.gif")
#tablet.hideImage()

test.say("Ciao a te!")

test = ALProxy("ALBehaviorManager", ROBO_IP, PORT)
test.runBehavior("pepper_choregraphe-27c329/saluto")

#motion = ALProxy("ALMotion", ROBO_IP, PORT)
#motion.moveTo(1.0, 0.0, 0.0)



