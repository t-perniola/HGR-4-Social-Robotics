# coding=utf-8
# import dependancies
import cv2
import numpy as np
import socket
import pickle
import struct 
import threading
from naoqi import ALProxy

# Indirizzo IP e Porta default Pepper
PEPPER_IP = "192.168.206.245"
ROBO_IP = "127.0.0.1"
PORT = 9559

## get NAOqi module ALVideoDevice with proxy
cameraProxy = ALProxy("ALVideoDevice", ROBO_IP, PORT)

# facciamo concentrare Pepper su di noi, appena riconosciuti
#engageProxy = ALProxy("ALBasicAwareness", ROBO_IP, PORT)
#engageProxy.setEngagementMode("FullyEngaged")

# accedo alle funzionalità del tablet
#tabletProxy = ALProxy("ALTabletService", ROBO_IP, PORT)

# prendo moduli per far parlare pepper e usare i behaviors
speechProxy = ALProxy("ALTextToSpeech", ROBO_IP, PORT) 
behaveProxy = ALProxy("ALBehaviorManager", ROBO_IP, PORT)

## SOCKET ##
clientsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
clientsocket.connect(('localhost', 8089))

labels_hagrid = np.array(["ok", "peace", "rock", "victory", "mano del gaucho", "stop", "mute", "like", "dislike"])

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

# funzione che manda lo stream video collegandosi alla fotocamera di Pepper
def send_stream():
    while True:

        # get image
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

            data = pickle.dumps(image) 
            clientsocket.sendall(struct.pack("L", len(data))+data)   

        # exit by [ESC]
        if cv2.waitKey(10) & 0xFF == ord('q'):            
            break    

    cameraProxy.unsubscribe(captureDevice)
    clientsocket.close()

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
    clientsocket.close()

def all_same(items):
    return all(x == items[0] for x in items)

# funzione target del thread avviato prima dell'invio dello stream:
# - ha come obiettivo quello di ricevere tutte le prediction prodotte in output dal modello..
#    .. nel mentre che riceve i frame mandati da questo stesso modulo
def recv_prediction():
    
    predicted = False
    predictions_list = []

    while predicted is False:   
        pred_received = str(clientsocket.recv(1024).decode())

        if len(pred_received) < 10: # evito di inserire le previsioni concatenate fra loro come un'unica previsione ricevute mentre Pepper esegue gesto (es. "bacibacibaci")
            predictions_list.append(pred_received) # inserisco tutte le predizioni ritornate dal server
            print(predictions_list)        

            if len(predictions_list) > 20:   # se la lista delle predizioni contiene più di 20 elem li prendo
                last_elems = predictions_list[-20:]  
                print("acquisendo...")

                if all_same(last_elems): # se gli ultimi 20 elem sono tutti uguali, allora la predizione sarà in un certo senso "stabile" ..
                    real_prediction = last_elems[-1]  # .. cioè il sistema sarà abbastanza sicuro della sua predizione
                    real_prediction = str(real_prediction)  # lo trasformo dal formato unicode in stringa normale
                    print("REAL PRED:", real_prediction)  # tal predizione finale sarà restituita in output
                    #predicted = True
                    
                    # mando a Pepper la prediction del modello
                    send_pred_to_Pepper(real_prediction)

                    predictions_list = [] # reinizializzo la lista così le prediction prese circa nel mentre pepper esegue gesto non vengono contate
                    print("should be empty:", predictions_list)

            else: print("Wait...") 

# funzione che manda predicition a Pepper, che:
    # - la ripeterà a voce
    # - mostrerà un emoji sul tablet
    # - la eseguirà runnando un behavior su choregraphe
def send_pred_to_Pepper(prediction):
    print("Pepper sta per eseguire il seguente gesto:", prediction)  
    speechProxy.say(prediction)   
    name = ""  

    # a seconda del gesto riconosciuto, pepper eseguirà il gesto e farà vedere sul tablet un'immagine relativa
    if prediction == "preghiera":
        name = "preghiera"
        #tabletProxy.showImage("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT51IPgNafE0brl5o3cOPhlL9SMcu5X2mHmGQ&usqp=CAU")
    elif prediction == "applauso":
        name = "applauso"
        #tabletProxy.showImage("https://media2.giphy.com/media/ZdNlmHHr7czumQPvNE/giphy.gif?cid=6c09b952mgwzit77f8n3ljt65ff5wzqxxtvpbjjpbk3j6ntm&rid=giphy.gif&ct=g")
    elif prediction == "saluto":
        name = "saluto"
        #tabletProxy.showImage("https://em-content.zobj.net/source/noto-emoji-animations/344/waving-hand_1f44b.gif")
    elif prediction == "baci":
        name = "baci"
        #tabletProxy.showImage("https://i.pinimg.com/474x/bf/e3/3d/bfe33dccf22de08366698befb5f4027c.jpg")
    elif prediction == "peace":
        name = "peace"
        #tabletProxy.showImage("https://www.dictionary.com/e/wp-content/uploads/2018/08/victory-hand.png")
    elif prediction == "victory":
        name = "victory"
        #tabletProxy.showImage("https://www.dictionary.com/e/wp-content/uploads/2018/08/victory-hand.png")
    elif prediction == "rock":
        name = "rock"
        #tabletProxy.showImage("https://cdn.shopify.com/s/files/1/1061/1924/products/Sign_Of_The_Horns_Emoji_Icon_ios10_grande.png?v=1571606093")
    elif prediction == "chiamata":
        name = "chiamata"
        #tabletProxy.showImage("https://img.vavel.com/c1200_675_ronaldinho-48-4-v1417611739-4529021265.jpg")
    elif prediction == "ok":
        name = "ok"
        #tabletProxy.showImage("https://images.emojiterra.com/google/android-oreo/512px/1f44c.png")
    elif prediction == "stop":
        name = "stop"
        #tabletProxy.showImage("https://cdn3.iconfinder.com/data/icons/emoticon-emoji-1/50/TalktotheHand-512.png")
    elif prediction == "mute":
        name = "mute"
        #tabletProxy.showImage("https://i.pinimg.com/474x/53/f6/61/53f6616a3c4979122c545c68025ce40d.jpg")
    elif prediction == "like":
        name = "like"
        #tabletProxy.showImage("https://images.emojiterra.com/google/android-10/512px/1f44d.png")
    elif prediction == "dislike":
        name = "dislike"
        #tabletProxy.showImage("https://media.tenor.com/hqJZKOzhoXMAAAAC/emoji-dislike.gif")

    # se gesto appartiene ad HaGRID
    if name in labels_hagrid:
        print("non posso replicare questo gesto!")
        speechProxy.say("non posso replicare questo gesto!")
    else:
        behavior = "pepper_choregraphe-27c329/" + name
        behaveProxy.runBehavior(behavior)

    #tabletProxy.hideImage()

# avvio thread in modo che client sia pronto a ricevere le prediction
t = threading.Thread(target=recv_prediction)
t.start()

send_stream_pc()
