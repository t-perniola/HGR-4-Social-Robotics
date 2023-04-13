import socket

PORT = 8089

while True:
    c = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    c.connect(('localhost', PORT))

    print(c.recv(1024).decode())

    c.close()

