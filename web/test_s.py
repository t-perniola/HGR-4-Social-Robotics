import socket

HOST = ''
PORT = 8089

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print('Socket created')

s.bind((HOST, PORT))
print('Socket bind complete')
s.listen(1)
print('Socket now listening')

while True:

    def init():
        conn, addr = s.accept()  
        print('Got connection from', addr)

        conn.send('Thank you for connecting'.encode())

        #conn.close()

        return conn
    
    conn = init()
    print("conn", conn)
    conn.close()

    