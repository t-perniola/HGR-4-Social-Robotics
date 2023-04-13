import socket

HOST = ''
PORT = 8089

def init():
    conn, addr = s.accept()  
    print('Got connection from', addr)

    conn.send('Thank you for connecting'.encode())

    #conn.close()

    return conn

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print('Socket created')

s.bind((HOST, PORT))
print('Socket bind complete')
s.listen(1)
print('Socket now listening')

i = 0
while i < 9:       
    conn = init()
    print("conn", conn)
    

    