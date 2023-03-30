import cv2
import socket
import pickle
import struct ### new code

clientsocket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
clientsocket.connect(('localhost', 8089))

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