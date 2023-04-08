# coding=utf-8
import socket
import pickle
import struct
#import pepper_server as server_ut
from flask import Flask, render_template

# Declarations
test = Flask(__name__)

HOST = ''
PORT = 8089

# definisco funzione di inizializzazione socket
def init():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print('Socket created')

    s.bind((HOST, PORT))
    print('Socket bind complete')
    s.listen(10)
    print('Socket now listening')

    conn, addr = s.accept()  
    print('Got connection from', addr)

    conn.send('Thank you for connecting'.encode())

    return conn

# avvio socket
conn = init()

# Define ROUTES
@test.route("/")
def pepper():       
    return render_template("pepper.html")

@test.route("/<name>") # tramite richiesta GET, ottenuta dal bottone premuto ...
def gesture(name):  # ... ottengo il gesto corrispondente scelto
    conn.send(name.encode())
    return render_template("web_control.html", gesture = name)

if __name__ == "__main__":
    test.run()


