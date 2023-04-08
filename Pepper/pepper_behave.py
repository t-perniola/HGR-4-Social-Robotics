from naoqi import ALProxy

ROBO_IP = "127.0.0.1"
PEPPER_IP = "192.168.71.245"
PORT = 9559

# init
test = ALProxy("ALTextToSpeech", PEPPER_IP, PORT) # mi connetto al Pepper virtuale
#test.say("Hello, world!")

test = ALProxy("ALBehaviorManager", PEPPER_IP, PORT)
#test.runBehavior("pepper_choregraphe-27c329/preghiera")

motion = ALProxy("ALMotion", ROBO_IP, PORT)
#motion.moveTo(1.0, 0.0, 0.0)

tablet = ALProxy("ALTabletService", PEPPER_IP, PORT)
tablet.showImage("https://staticfanpage.akamaized.net/wp-content/uploads/sites/6/2020/03/significato-emoji-strane-doppio-senso-WhatsApp-2.jpg")
tablet.hideImage()

