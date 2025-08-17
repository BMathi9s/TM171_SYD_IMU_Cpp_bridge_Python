# receiver_imu.py
import socket, json

HOST, PORT = "127.0.0.1", 8765
print(f"listening on udp://{HOST}:{PORT}", flush=True)

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# Big receive buffer in case the sender bursts
sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
sock.bind((HOST, PORT))

try:
    while True:
        data, addr = sock.recvfrom(65507)  # max UDP payload
        # try UTF-8 first; fall back to latin-1 just to avoid crashes
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = data.decode("latin-1", errors="ignore")

        # Some senders add '\n' between messages; handle 1..N JSONs per datagram
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                # Uncomment this to debug malformed frames:
                # print("bad json:", repr(line), flush=True)
                continue

            # Print exactly what you get (RPY, quat, raw, combo)
            print(msg, flush=True)

except KeyboardInterrupt:
    pass
finally:
    sock.close()
    print("bye", flush=True)
