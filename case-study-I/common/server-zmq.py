#ubuntu@UNIKLU-VIE:~/zara/trafficGenerator$ python3.6 server-zmq.py
#
#   Hello World server in Python
#   Binds REP socket to tcp://*:5555
#   Expects b"Hello" from client, replies with b"World"
#
import subprocess as commands
import time
import zmq

context = zmq.Context()
socket_rep = context.socket(zmq.REP)
socket_rep.bind("tcp://194.182.172.55:5201")

#socket_req = context.socket(zmq.REQ)
#socket_req.connect("tcp://194.182.181.170:5201")

while True:
    #  Wait for next request from client
    message = socket_rep.recv()
    print("%s" % message)
    ips = str(message).split(' ')
    print (ips[0])
    tup = commands.getstatusoutput("docker run encod")
    #print (type(tup[1]))
    splittt = str(tup[1]).splitlines()
    #  Do some 'work'
    time.sleep(1)
    b = bytes(splittt[len(splittt)-1], 'utf-8')
    #  Send reply back to client
    socket_rep.send(b)
    socket_req = context.socket(zmq.REQ)
    socket_req.connect("tcp://"+ips[0][2:]+":5201")
    b2 = bytes("{} {} {} {}".format(ips[1], ips[2], ips[3], ips[4]), 'utf-8')
    socket_req.send(b2)
    message = socket_req.recv()
    print("Received reply [%s]" % (message))
socket_rep.close()