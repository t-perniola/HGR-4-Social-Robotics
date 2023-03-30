# coding=utf-8
import cv2
import socket
import pickle
import struct 

cap = cv2.VideoCapture(0)
clientsocket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
clientsocket.connect(('localhost', 8089))

while cap.isOpened():
    ret, frame = cap.read()
    #cv2.imshow("frame", frame)
    data = pickle.dumps(frame) 
    clientsocket.sendall(struct.pack("L", len(data))+data)

    # Break gracefully
    if cv2.waitKey(10) & 0xFF == ord('q'): #se premiamo q -> quit
        break

cap.release()
cv2.destroyAllWindows()

#serverName = 'localhost'
#serverPort = 65432
# create TCP socket
#clientSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# connect socket to remote server at (serverName, serverPort)
#clientSocket.connect((serverName, serverPort))

# Send sentence into socket, no need to specify server IP and port
#clientSocket.send("ciauuu")
# read reply message from socket into modifiedMessage string
#recvSentence = clientSocket.recv(1024)
# Print out received modifiedMessage string
#print(recvSentence)

# Close socket
#clientSocket.close()