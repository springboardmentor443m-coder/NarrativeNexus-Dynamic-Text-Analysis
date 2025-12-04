import socket
import sys

sock = socket.socket()
result = sock.connect_ex(('localhost', 8000))
sock.close()

if result == 0:
    print('[OK] Port 8000 is OPEN - Server is running!')
    sys.exit(0)
else:
    print('[ERROR] Port 8000 is CLOSED - Server may not be running')
    sys.exit(1)
