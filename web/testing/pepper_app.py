import cv2
import numpy as np
from naoqi import ALProxy
from flask import Flask, render_template, Response
from PIL import Image

pepper_app = Flask(__name__)

ROBO_IP = "127.0.0.1"
PORT = 9559

cameraProxy = ALProxy("ALVideoDevice", ROBO_IP, PORT)
subscriber = cameraProxy.subscribeCamera("demo", 0, 2, 13, 5) # params: mod_name, cam_idx, resolution_idx (2: 640x480), colors_idx, fps
#cap = cv2.VideoCapture(0) 

# Use NGROK
#run_with_ngrok(app)

# Define ROUTES
@pepper_app.route("/")
def home():    
    return render_template("home.html")

@pepper_app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@pepper_app.route('/webcam')
def webcam():
    return render_template("webcam.html")


def gen_frames():  

    i = 0

    while i < 5:
        img = cameraProxy.getImageRemote(subscriber)
        cameraProxy.releaseImage(subscriber)

        # Get the image size and pixel array.
        imageWidth = img[0]
        imageHeight = img[1]
        array = img[6]
        image_string = str(bytearray(array))

        # Create a PIL Image from our pixel array.
        im = Image.frombytes("RGB", (imageWidth, imageHeight), image_string)
        #im.show()

        i+=1
    
        ret, buffer = cv2.imencode('.jpg', np.array(im))
        image_string = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image_string/jpeg\r\n\r\n' + image_string + b'\r\n')  
        
def test_gen_frames(): 
    cap = cv2.VideoCapture(0) 
    while True:
        success, frame = cap.read()  # read the camera frame
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n') 


if __name__ == "__main__":
    pepper_app.run(debug=True)
