from naoqi import ALProxy
from PIL import Image

ROBO_IP = "127.0.0.1"
PEPPER_IP = "192.168.104.245"
PORT = 9559

cameraProxy = ALProxy("ALVideoDevice", PEPPER_IP, PORT)
subscriber = cameraProxy.subscribeCamera("demo", 0, 1, 13, 10) # params: mod_name, cam_idx, resolution_idx (2: 640x480), colors_idx, fps

img = cameraProxy.getImageRemote(subscriber)
cameraProxy.releaseImage(subscriber)

# Get the image size and pixel array.
imageWidth = img[0]
imageHeight = img[1]
array = img[6]
image_string = str(bytearray(array))

# Create a PIL Image from our pixel array.
im = Image.frombytes("RGB", (imageWidth, imageHeight), image_string)
im.show()

# Save the image.
#im.save("camImage.png", "PNG")

cameraProxy.unsubscribe(subscriber)
