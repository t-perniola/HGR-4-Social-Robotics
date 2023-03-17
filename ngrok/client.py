import socket

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(("6.tcp.eu.ngrok.io", 12010))

print(client.recv(1024).decode())
client.send("ziopera".encode())