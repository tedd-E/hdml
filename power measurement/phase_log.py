# first of all import the socket library
import socket            
import time

PHASES_FILE = open("./phase.txt", "w+")

HOST = '169.254.220.80'  # Standard loopback interface address (localhost)
PORT = 6000        # Port to listen on (non-privileged ports are > 1023)

trainstart = []
trainend = []
sendstart = []
sendend = []

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    conn, addr = s.accept()
    with conn:
        print('Connected by', addr)
        counter = 0
        while True:
            data = conn.recv(2048).decode()
            if data:
                info = data
                print(info)
                
                if info=="finished":
                    break
                
                elif info=="trainstart":
                    trainstart.append(time.time())
                    
                elif info=="trainend":
                    trainend.append(time.time())
                    
                elif info=="sendstart":
                    sendstart.append(time.time())
                    
                elif info=="sendend":
                    sendend.append(time.time())
                
                else:
                    print("ERROR!!!! using automatic collection")
                    if len(trainend) < len(trainstart):
                        trainend.append(time.time())
                    elif len(sendstart) < len(trainend):
                        sendstart.append(time.time())
                    elif len(sendend) < len(sendstart):
                        sendend.append(time.time())
                    else:
                        trainstart.append(time.time())

        PHASES_FILE.write("Train Start\n")
        for x in trainstart : PHASES_FILE.write(str(x) + "\n")

        PHASES_FILE.write("Train End\n")
        for x in trainend : PHASES_FILE.write(str(x) + "\n")

        PHASES_FILE.write("Send Start\n")
        for x in sendstart : PHASES_FILE.write(str(x) + "\n")

        PHASES_FILE.write("Send end\n")
        for x in sendend : PHASES_FILE.write(str(x) + "\n")
