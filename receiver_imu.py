import socket, json
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("127.0.0.1", 8765))
print("listening on udp://127.0.0.1:8765")
while True:
    data, _ = sock.recvfrom(4096)
    m = json.loads(data.decode("utf-8"))
    print(m)  # {'t_us': ..., 'q':[w,x,y,z], 'qos': N}
