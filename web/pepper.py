from naoqi import ALProxy
tts = ALProxy("ALTextToSpeech", "192.168.214.245", 9559)
tts.say("Hello, world!")