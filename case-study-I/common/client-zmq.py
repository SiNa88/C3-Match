#
#   Hello World client in Python
#   Connects REQ socket to tcp://localhost:5555
#   Sends "Hello" to server, expects "World" back
#
import sys
import zmq

context = zmq.Context()

#  Socket to talk to server
print("Connecting to hello world server…")
socket = context.socket(zmq.REQ)
#socket.connect("tcp://194.182.172.55:5201")
#print (len(sys.argv))
if (len(sys.argv) != 7):
	print("Please enter six IP addresses.")
	sys.exit()
ip_0 = sys.argv[1]
ip_1 = sys.argv[2]
ip_2 = sys.argv[3]
ip_3 = sys.argv[4]
ip_4 = sys.argv[5]
ip_5 = sys.argv[6]
socket.connect("tcp://"+ip_0+":5201")
#  Do 10 requests, waiting each time for a response
#for request in range(10):
print("Sending request...")
b = bytes("{} {} {} {} {}".format(ip_1, ip_2, ip_3, ip_4, ip_5), 'utf-8')
socket.send(b)

#  Get the reply.
message = socket.recv()
print("Received reply [%s]" % (message))
