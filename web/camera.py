import cv2
from imutils.video import WebcamVideoStream

class VideoCamera(object):

    # inizializzazione
    def __init__(self):        # 0 = camera del nostro sistema
        self.stream = WebcamVideoStream(src=0).start()

    # stoppiamo il flusso di acquisizione dei frame
    def __del__(self):
        self.stream.stop()

    def get_frame(self):
        image = self.stream.read()

        # codifichiamo immagini
        ret, jpeg = cv2.imencode(".jpg", image)
        data = []
        
        # convertiamole in bytes in modo tale da poterle "esportare"
        data.append(jpeg.tobytes()) 
        return data