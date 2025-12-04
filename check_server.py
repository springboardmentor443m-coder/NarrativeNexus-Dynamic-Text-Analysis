import socket
import time
time.sleep(2)
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
result = sock.connect_ex(('127.0.0.1', 8000))
if result == 0:
    print("SUCCESS: Server is running on port 8000")
else:
    print("FAILED: Server is not responding")
sock.close()
