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
PEPPER_IP = "192.168.71.245"
PORT = 9559

## get NAOqi module ALVideoDevice with proxy
cameraProxy = ALProxy("ALVideoDevice", PEPPER_IP, PORT)

# facciamo concentrare Pepper su di noi, appena riconosciuti
engageProxy = ALProxy("ALBasicAwareness", PEPPER_IP, PORT)
engageProxy.setEngagementMode("FullyEngaged")

# accedo alle funzionalità del tablet
tabletProxy = ALProxy("ALTabletService", PEPPER_IP, PORT)

# prendo moduli per far parlare pepper e usare i behaviors
speechProxy = ALProxy("ALTextToSpeech", PEPPER_IP, PORT) 
behaveProxy = ALProxy("ALBehaviorManager", PEPPER_IP, PORT)

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
        if cv2.waitKey(33) == 27:
            break     
        
    cameraProxy.unsubscribe(captureDevice)

def all_same(items):
    return all(x == items[0] for x in items)

# funzione target del thread avviato prima dell'invio dello stream:
# - ha come obiettivo quello di ricevere tutte le prediction prodotte in output dal modello..
#    .. nel mentre che riceve i frame mandati da questo stesso modulo
def recv_prediction():
    
    predicted = True
    predictions_list = []

    while predicted is True:              
        predictions_list.append(clientsocket.recv(1024).decode()) # inserisco tutte le predizioni ritornate dal server
        print(predictions_list)

        if len(predictions_list) > 20:   # se la lista delle predizioni contiene più di 20 elem li prendo
            last_elems = predictions_list[-20:]  
            print("acquisendo...")

            if all_same(last_elems): #and predicted is False:   # se gli ultimi 20 elem sono tutti uguali, allora la predizione sarà in un certo senso "stabile" ..
                real_prediction = last_elems[-1]  # .. cioè il sistema sarà abbastanza sicuro della sua predizione
                real_prediction = str(real_prediction)  # lo trasformo dal formato unicode in stringa normale
                print("REAL PRED:", real_prediction)  # tal predizione finale sarà restituita in output
                #predicted = False
                
                # mando a Pepper la prediction del modello
                send_pred_to_Pepper(real_prediction)

                predictions_list = [] # reinizializzo la lista così le prediction prese circa nel mentre pepper esegue gesto non vengono contate

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
        #tabletProxy.showImage("data:image/png;base64,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")
    elif prediction == "victory":
        name = "victory"
        #tabletProxy.showImage("data:image/png;base64,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")
    elif prediction == "rock":
        name = "rock"
        #tabletProxy.showImage("data:image/jpeg;base64,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")
    elif prediction == "mano del gaucho":
        name = "mano del gaucho"
        #tabletProxy.showImage("https://img.vavel.com/c1200_675_ronaldinho-48-4-v1417611739-4529021265.jpg")
    elif prediction == "ok":
        name = "ok"
        #tabletProxy.showImage("data:image/png;base64,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")
    elif prediction == "stop":
        name = "stop"
        #tabletProxy.showImage("data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEAAkGBxISEhUTERIVFhUXGRUYFRUXFhUXFRYVFRUWFhUVFRcYHSggGBolHRUXITEhJSkrLi4uFx8zODMsNygtLisBCgoKDg0OGxAQGi0lICUtLS0tLS0tLS4tLS0tLS0tLS0tLS0vLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLf/AABEIALgBEgMBIgACEQEDEQH/xAAcAAACAgMBAQAAAAAAAAAAAAABAgADBAUGBwj/xABAEAABAwIDBQYCCQIGAQUAAAABAAIRAyEEEjEFIkFRYQYTMnGBoZHBBxRCUmKx0eHwkvEzcoKissJTIyQ0RNL/xAAaAQEAAgMBAAAAAAAAAAAAAAAAAgMBBAUG/8QAMxEAAgECAgYKAQQDAQAAAAAAAAECAxEEIQUSMUGRwRNRYXGBobHR4fAyIiRS8RQzQgb/2gAMAwEAAhEDEQA/APa6jwRAN0lIZTeyjaZbc8EXOz2CAFVuYyLp2PAEE3StdksfNA0ib80AKbCDJFk9Xe0uoauaw4oNGS5QBonKINkjmEmQLJnNz3HkiKoFuVkAajwRA1S0Rl1sgKeW54IuOfTggBUGYyLp2vAETdK12Sx80DSJv6oAU2kGTYJ6u9pdQ1M1hx/ug0ZNeKANI5RBsqywzMWmfRO5ma4VVbGU6Yh7mt4XIHTijyVwld2RfVcCIFylo7szZao7ewzD/iZvJrj7xCrrdpqB4P8A6R+q1njMOsnUjxResNWf/D4M3FVpJkXCcOERN4j1Wlpdp8OBBz/0/oVfR2vh3GRWaLzDpb/yAUoYqhJ2jNcUYeHqrbF8GZ9JpBk2CatvRF0rKoqDdM9ZBHsmbua8eXT+6vKRqTgBBsVXkMzFpn0TFma4R70eH0+SANVwIgXKWjuzNlAzLcqO39OHzQC1WkmRcKwvERN4j1QFTLYpe7Pi9fmgJSaWmTYI1t6Iui5+awQbua8eSAam8AQbFVBhmYtM+ic081xxR70Hd9PkgJVOYQLqUt3WyDWZLlRwz6cEBb3reaCq7g8wogI2oXWPFFzclx7pqjQBI1SUjJ3roAtbnufKyBqEWHBSsYNrKxjQRJF0AppBtxwStOex4ckKbiSAdE9UR4bIAOdksPO6IpA3vzUoiRvXVbnEGBogGbULrHii4ZNOPNNUaAJGqWlfxXQEa3Pc+VkDUIt6KVTBtZY+Px1Oi3O/U6AauPQKMpKKcpOyRmMXJ2W0yXsDRm5c9Fo9odo6bbNHeOHIw31defRaLaO1atcwSQzgwaHz+8VfgtiOdd+6OX2v2XFr6UnN6mHXi/bd48DpU8FCC1qz8Pu3w4lOJ21iKlg8tbyZu++p+KpobMqvuGOM8TafU6rpsLgadPwtE8zc/E6eiFba1Bjg19em1xIAaXtBJOgAm5Wk8PKq71ptv08XyXEu/wAlRVqUbL7uRqKfZ6odSwepJ9grR2cP/kH9J/Vb+UJVqwtDqfF8rFTxNXr8jQu7OO4Pb8CFjVdg1RoA7yP6wunlU4vGU6Tc9V7WNsMziGiTYXKxLC0Gt/F87iOKqr+jkTRqUjO+w8xLfcarYYbtFWbAqRUA/wBLviBB9Qt7RxVOoNx7Hj8JDh7LFxeyKT9BkPTT1H6QoRp1aOdCfh9yfiix1oVMqsfvqZ2z9sU6lmmD911j+h9Fse6Hi46/NcFjtmPp3ItwcNP2K2Wyu0TmkMrkubpn4gfi+8PfzXQw2lE5aldar693j1d+w16uBy1qTuurf4e206lr81j7Iu3NOPPohnaWhzCCDEEcQeSNG85r+a7Bzwtp5rn2Sd4fDw0+SlVxBgaKwtETF490AHMy3Hug3f14ckKRJMG4RrWjLbyQAdULbBMaQG9x1RpNBEnVVBxmOE+yAdrs1j7KOOTTjzTVAAJFktG/iv5oBfrB6KK7u28gogKWUyDJFk1U5rNuiaua0aoBuS5ugDSOUQbJH0yTIFkS3PcW4JhVy2jRAF7wRA1S0hl8VlBSy35KE57C0IAVG5jIunbUAEHVKHZLa8VO6m863QC02EGSLJ6hzeG6neZrc1j4vENoML3achqTwA6lYk1FXewyk27Io2ltJuHZvXefC3n1PIdVx5NTEVJdvOPwA6cgEK9Z+Iq5jdzrAcAOAHQfqVvsJQbSb7ud5fJeXxWKlip2vaC+3fb6cTsU6aw8euT+27vUXC4SnQaXuIsJc91g0DWJ0HVef9qfpbYwmngWh5FjWfPd/wChti/zkDzXPdsO0WI2xifqWBBNBpvBhtQg3qVHf+McBx1vaO57H/R1hsGA+qBWr65nDdYfwN4eevVbMY06EV0iz3R5v78UNuTv5nEYLZm2tqHNVqVW0zxqE0qcfhpNjN6i/Ndt2b+jDCYctfWJrVBBE7tMEcQwa/6iV3AKkqueKlLJZLqWRm7WwslSUmZSVV0hCw8rC2xsujiqZpV2B7DwuCCNCCLg9QsrMhKdKZSPLdtfRS9hNTAVyCNGPJa4dG1G3+IPmufpdrdr7MeGYnO9ugbXGYO/yVhcn1PkvcpWPjsHSrMNOsxr2Gxa4Ag/FXxxbeVRay8+JJtvac92S7d4XaAyD/060XoviTzLDo8e/MBbHaWyQd6mPNv/AOf0XmXbX6NX4ecTs4vIac3dAnvGRfNSdqY5a8uS6j6M+2/16maNcgYmmL8O8aLZwOB4Ec/NYr06dSGvDNecfj7mITcJZfDN3sbazsO6DJpnxN4jmW9enFdmKoqBrmGQRMjquV2xgQQXtF/tDn181X2d2r3L8j/A7/a7n5c/is6PxrpSVGo/07n1fHp2LZPE0FWj0kF+reuv59TsmPDRBsUgpmZi0z6JjTzX0U7z7MdPkvRHJDUcHCBcoUt3xWUDMl9VDv6WhALUYXGRorC8RHGI9UBUy25Id1G9PX5oAU25TJsEaozeG6hfntooDk1vKAr7l3L8lFb9ZHJRAQ0g244INOex9klMkm8wrK1hu+yAD3ZLDzumbSBueKFESN73Vb3GbTCAZtQuseKLxkuPdM+ALRPRJRv4vdAFjc9z5WQNQi3DRCqYO7p0VrGiLxKAxsTiKVKC97Wk6ZiBPNcl2i2p31QMaQWM0jRzuLuvIevNJtompiKma+WGjoAB8yT6rVPZlcvO47HOo5UVkk+Nvk7GFw0aaVS93b1+DOweLbSJOUuPTgOPqh2jpPxuEqUaFXunPhpcRJAkZmkdRb1VNBsk+avwRyvI5gH4f3XNpS1ZWW1Zl9aKab3lfZXYWF2XSFIOBqOu95G84+mjeS6SniGu0Mg6HgVzdcZqjyecfAR8lkbMdGZvCQfj/ZSdRzk9Z3e0rlRUYXXYdBKMqprrI5lV0hTqluZTMqsymZOkFi3MpmVWZTMnSCxZKEpMykp0g1R5XDbT7BD69TxuEq9y4OBqtyy14+1HIuFj8dV2soZlKFeULuL2q3gZsMSuc2hh8riBpqPIroCVqNqXd5AfM/NVTndF9G6kbrsztMup92fEznqW8D6afBb00hGbjr81wGzsV3NVr+AMOHNps79fQLuw4zrafSP0XqdGYnpqNntjl7Ph6M52No9HUutjz9/vaM1+axRfuaceaNUAC2vRCjec3uuiaYW0w4SUoqE24aIVSQbadFYWiOEx6ygA9uW490GjPrw5IUiSd7TqjWt4fZAH6uOqKpzu5lRAWvqAiBqlpDLd1ke6y3nRQuz20QC1BmMtunZUAEHVAOyW14qd1mvOqAVlMgydE9U5vDdDvc1o1UAyX1lAGkcog2SOpkmQLKnGYljWF7zlaLcyTwA5m6x9nbapVd1khwGjhBI5jVQdWCkoNq73E1Tm4uSTst5z+0x/7mr5g/FjT81rMe2CFnYirmxFU/iI9G7o/wCKx9ptsvF15fvJdTb9WdyF1FJ/xXoJghqmNqg8j/PZTZ+pUxtnA9VWnatYlLNPu5D0WyXeZ/NJSeG1CCYn5f3VuBvPmfzXD/Sw09zScDBbWFwYImnUFj5wrMPHpMRqX25GKn4vuPT6brBYu0NrUMOM1esymPxOAJ8hqfRfO7drYrT6zXjl31SP+SOEwlStUhjX1KjuQLnHqTy6ldKOhne855di5vYaqlrbEfR2FxbKrG1KTw9jhLXNMgjoVdmXNdhNlVMLg2Uqtn5nvLZnLmMhsi3UxxJXQSuHXcYVJRg7pPJk0i3MtftjbWHwrQ/EVW0wTAzHU8gOKy5XAfSt2WrYxlKrQGZ1LMDTm5a6CS2eIgKeF6OpVUakrJ7zEk0skdvs3a1DENz0KrKjebXA/HksyV8vClUoPg56VQf5mP8AkYWaNq4lwh2JrkcjWqEfDMuxLQrveE8u72K41L7UfQe0dvYWgQK2IpUydA54zfDWOuizw6bj4r512RsitiHZaFNzzxI8I6vcbD1XvGx8OaGHo0Scxp02MLuBLWgGOllqaQwtPCxjaV2/t7FqTZn1asBaytcyVkvMrHqBcpTbL4Kxg1mrtOz2L7zDtk7wlnw8PsQuOqhbvsc+9RvLK8DykH5LuaHq6tfV/kmufLzKsdDWot9WfLmdNTaWmTYI1d7w3hQuz20UG51leqOIFjw0QdUgYZmLa+iY0819JU7yd2OnyQBqODhAuVKRy+KyAZkvqoRn6QgH75vP81En1bqogFbULjB4p3jLcIvAi0T01SUtd73/AHQDMbmufJK6qQYGgUqa7unT9k7AIvE9dUBHUw0SOCVhzWKWnMiZjronq/h9v2QGh7X2psaNM8nzyuhaDCv7upTf91wJ/wAsw72JXSdpWTh5OrXtN9YO7/2XNVmy1ea0pN08XGS7Hwb9jsYOzoJd64/DMfZr5cSdTc+ZWftJu6tZs0w5bjGNlq4td6tZM2r5pmFss3/nJPtMaeiq2UblZO1RZJZYgjbd2cgbN0Pr+axtp0GuID2hwOoIBB9Fk7K0Ve0vEPNIO2IYex93Iw6XZDAG5wtKf8oA+C3OBwNKi3LRpspt5NaGj2Rwp3VcFq1q1STcZSbXeyCGlSUFFRczYMqSgolxYxcds+jWEVqTHjk5oP5rVt7H4AGfqtL+kEfDRb1BWwrVIK0ZNdzYK6NBrAGsaGtGgaAAPQIlOUhUb3zZkrcqnq9yperIkkYdVZ/ZZ8V4+81zT8Wn5LBqrL7Nf/Jb5P8A+JXV0c7YiHejFdXoyXY/Q7V7Mtwgzf14IU5nemOunujW4Zfb9l7Q8+B9QtMDRMaYAnjqpTAi8T11VYJnjE9YhAMx2axRecmnFGpEbuvT9lKX4vf90Anfnoor4b09kEBS2mQZOgT1DmsEO9zWiJUy5L6+yANN2WxSOpkmRoU2XPfThzU73LaNEAXvDhA1QpjLcqd1lvMwpOe2keqAwtt089GpHBhPq3e+S5OmJau3qCAWG8g+9lxeBG7BXnNPq2pPv5e508BL9El2rz/o1uHEVYW7qiWrVYikRVafRbljCRouBiXfUkjem95qNnDfd5rN2i3dVuH2aWuLuayauELhCVHKVVSSdiLqRve5rNkeEJdr6Sthh8F3YgLA20Nw+SRf7m9trCaby6i/Z7parq9drBLvQcT5LXbNxIFKdeQ5lW0qBcczrn8ug6KFWilUk5bL8SNPOKbEqY2ofCA0fEqYXGPJIces/qjinAWGqswGFi51Ov6Kx9GoXcV2FkrauwzmGyKiC0CBEEVEACkKYpCpIyI5VPVjiqahV0SSMaqs/sqya8/da4n4tHzWuqlbrsdTvUdzysB85J/6rsaLg5YiHffgmyOJdqEu63E6mo4OEDVCnu68VMmS+qnj6R6r2BwAPYXGRonNQERx09UveZbRMId3G9PWPdASm3KZKNQZtFM+e2ik5Os+iATuXKJ/rHT3UQBdTAEjUIUzmsUrJm8x1mE9bTd9v2QAqOy2CLaYIk6lSjpva9f3Vb5m0x0mEAWVC4wdCneMuiNSItE9NUtL8Xv+6ANNua5XHtblq1G8nvA8sxhddU13dOn7LlMQIxFUfin4gH5rh6fj+3T7eT9jewLzkuz76mTToN1IV3egKgSUpavKqs4R/Sjc1bvMyhiQj9aCw1E/zKo6KJlHEArVbYbmaY5FZajqc6hY/wAicmm9xKKUHdGg2FhiWNLuQtylbqq7K1O1gGiwNq1oaVKc5YitfrZONskti5HFbc7fUMNWew031HssYLWsBImMxMzfktQ/6XKzrUsJTHQve8/7Q1d3sfs3hGzU+r0i9xJL3MDnEuMky6Tqt9TptbZrQPIAfkuhUxWDg9V0XK2V27eWZRacnfWt4HkrPpUxgu/C0Y8qrPckrZYH6W6Z/wAbCuHWnUa//a4N/NelLExezKFURVo0nj8dNjvzCqeLwMtuHt3S/ozqTX/XkajZfbfAVyA2uGOP2aoNMyeALt0nyJXQyuW2h9H+z6sxSNInjScW/wC0y32Wro9mNoYAzgMSK1Mf/Xq2BHIAnLPUFhWHh8HW/wBNRxfVPZ4S3ElrLad2SlcVrNh7aGIa4OpupVmQKtF9nMnRwkDMwwYctg4rSqUp05uE1ZosWYriqKhVrysWo5TgixIpquXZ9nsJkw7SfEZf8fD7ALk9nYXvqrafAnePJou79PULuwDPGJ9I/Rem0LQzlVfcvV8jR0jUtFU/F8hmOLjBRqbunFGrEbuvTX2Qpfi9J/dd85QWMDhJ1SCoSY4aKVJm0x00VhiOEx6ygBUblEhCmM2vBLTmd7Trp7pqv4fb9kA/cNUVG9191EBa6oHCBqUKYy3KndZbzopmz20QEe3NcItqACDqEubJbXij3Wa86oBWUy0ydAnqHNYId7mtESpGS+soA03ZbHzXL7ZZGJJ+81p9sv8A1XT5c99OH8+K57tMyKlN3QsJ8oI/Ny5ml6evhJdln525m3gnarbrTXPkGiFa6mqsK6yyoXlqMFKCNyTaZgPbCVZdZixWi60a1LUlYsjK6LaVNX92npNTkLpUqCUSlzzMCs2Fzu2Kkua3mfyuulxdgVytTergch+Z/ZQoU1Gs5dSuXJ/ofDibvDthoCtlVgppXNebuWWGlCUJSylgMSgSlLkpcs2M2A4Cc0CQCAYEhpIJaDqASAY6BI5yjnKl7lalckkR7ljVXo1HrZdntld8/O7/AA2/7nfd8ufwW9hsPKrNQjtf25mU404uUtiNp2Z2YW0+8I3n6cw3h8dfgt+agiOOnrol7zLbVTuvtT1j3XtKNKNKChHYvvmcCrUdSbk94GNLTJRqb+nBTPntop4Osq0rCx4aIOqQUyDPDVN3ea+kqd7O7HT5IAvdmsEKZy68VMmS+qkZ+kIB+/CiT6v19kUAjKhJg6J6oy3CLyItE9NUlKx3vdANSGa5SOqEGBojVud32TsIi8T7oAPpgCRqhSObxJaYMiZjromq38PsgA92UwFgbewuegSBvNh49PF7ErY0bDe16qt7TOhj2hQqU1Ug4S2NW4koTcJKS3HMbPrSFs2OWnxVD6vVLPsm7D+Hl5jT4c1nUa0rw9pYeq6czsTSktaOxmTUWGdVkl6xahuqMVJOzIwRm0inc5YtOolr4kNGq2YVlqkdTMo2nWhq0GzmS9z+v5K3HYk1DAT0AGiAot6kH1v0NlU8kvEzcybMsYPRzrT1CdjIzJS5U50peigLFxckc9Umoq3VFNQJJFr3qh9RI6ot1svs695Dq4LW23dHEdfuj38lu4bCTrS1YL2Xf97iNScKUdab+e773mJsbZTsQ68imPE7n+FvXrwXZtotpNDWDKAIA8k4Y0NDWAACAA3gOkI0bTm916vB4OGHjZZve/u44uIxMq0uzcgsYHCTqk7wzHCY9FKoM2mOmisJEcJj1lbhrgqNDRI1Qpb3i4IUgQd7TqjWvGX2QC1HlpgaKw0xE8dfVCmRF4nrqqwDPGJ9IQDMdmMHRGqcvhRqEEbuvRClbxe6ATvnc0Fk5m8x7IICptMtueCLznsFO9zWjVQtyX1QBY7LY+aU0ybjiiG576cFO9y2jRAE1A6w4oNGS5R7rLedEAc9tIQEe3NceSIqACDrogXZLa8Ue6m863QGHj9nNqNipoLgjxNPMLn6uzq9LQGo0faYL+rdfhK6sVc1uang6ytLFYCjifzWfWtpfRxE6eSzXUzj27RGhseRsfgo7Gt5rrKlBtS7mg8IIB/NVfVaLTAo0+U5Gz+S5Ev/AD13/sy7vk2ljYfxfE5VmIc8xSa5x/CJ+J4LMPZ6q9s1agYTo0DN8TNvSV03d5BI4cPZEHP0hbmH0LQpfleT4fPmVzx03+Ct5v28jha+xa9K/dlw+8zeHwFx8FhiqvRi/JbVVVsDTqXe1ruN2g+6hW0NCTvCVu/P25lsNItZTjwy8tnoefism71da7YOGef8Mt6h7vyJhV1uzFAcan9Tfm0rSehq+5x4v2NhY+i+vh7M5bvUpqrrKXZegRMv9XD5AK6jsbDtMCiDwlxc71hxIUo6FrN5uK4+1vMw8fRWxPy9zjaeZ5hjS48mgk/ALZ4Xs7WdBqRTB53d/SPmV2FOg2mN0ADkAAPZMN/pHz/st6joelHOpJy8l7+aNeppGb/BW8/jyNfs/YtOldgk/edd3pwHotj3gjLx09dEC/LbVTuvtT1+a60IRgtWKsjQlKUneTuwNYW3KL9/Thz6qB+a2ih3Os/JSIha8NEFL3ZnNw1+aIp5r6Kd79mOnyQBe/NYIM3NePJQsyX1UG/0hABzC640TGoCI46IGpltrCndRvT1+aADG5blF4z6cFA/PbRQnJ1lAL9XPRRN9Z6KIAvpgCRqlpHNZ11FEAKhymG2TspgiTqiogK2PJMHRPVGXw2UUQEpNzCXXSOqEGAbKKICx7ABI1S0jm8V1FEAKjspgWCdrARPHVRRAV03lxg6Jqgy+G386qKIA02hwk3KQ1DMTaY9FFEA72hokapae94rx/OCKiAFRxaYFgmDBE8Yn1RUQCU3Fxg6KVN3w2n+cUVEAWMDhJ1Sd4ZibTHogogLKjQ0SLFLT3vFeP5wRUQAqPLTA0TGmInjE+qKiASm4uMG4UqbvhtP84oqIAsYHCTqkDzMTbT0QUQFlRoaJFihTGbxX/nRFRAN3LeX5qKKID//2Q==")
    elif prediction == "mute":
        name = "mute"
        #tabletProxy.showImage("https://i.pinimg.com/474x/53/f6/61/53f6616a3c4979122c545c68025ce40d.jpg")
    elif prediction == "like":
        name = "like"
        #tabletProxy.showImage("https://images.emojiterra.com/google/android-10/512px/1f44d.png")
    elif prediction == "dislike":
        name = "dislike"
        #tabletProxy.showImage("https://media.tenor.com/hqJZKOzhoXMAAAAC/emoji-dislike.gif")

    if name in labels_hagrid:
        print("non posso replicare questo gesto!")
        speechProxy.say("non posso replicare questo gesto!")
    else:
        behavior = "pepper_choregraphe-27c329/" + name
        speechProxy.say(name)
        behaveProxy.runBehavior(behavior)

    tabletProxy.hideImage()

# avvio thread in modo che client sia pronto a ricevere le prediction
t = threading.Thread(target=recv_prediction)
t.start()

send_stream()
