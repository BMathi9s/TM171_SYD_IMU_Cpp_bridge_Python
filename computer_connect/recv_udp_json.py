import socket, json, time, argparse, select
#python3 recv_udp_json.py --bind 0.0.0.0 --port 9101



# timedatectl status
# sudo timedatectl set-ntp true     # or: sudo apt install chrony


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bind", default="0.0.0.0", help="Bind IP on Sim PC")
    ap.add_argument("--port", type=int, default=9101, help="UDP port to listen")
    ap.add_argument("--print-every", type=float, default=0.5, help="Print period (s)")
    args = ap.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # Small-ish receive buffer so backlog doesn't grow into seconds
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 64 * 1024)
    except OSError:
        pass
    sock.bind((args.bind, args.port))
    sock.setblocking(False)

    print(f"[recv] listening on {(args.bind, args.port)}")
    last = None
    pkts = 0
    last_print = time.time()

    try:
        while True:
            # Drain everything available right now
            ready, _, _ = select.select([sock], [], [], 0.05)
            if ready:
                while True:
                    try:
                        data, addr = sock.recvfrom(2048)
                    except BlockingIOError:
                        break
                    pkts += 1
                    try:
                        last = (json.loads(data.decode("utf-8").strip()), addr)
                    except json.JSONDecodeError:
                        continue

            now = time.time()
            if now - last_print >= args.print_every:
                if last is not None:
                    msg, addr = last
                    age_ms = None
                    if "t_ms" in msg:
                        age_ms = int(now * 1000) - int(msg["t_ms"])
                    print(f"[recv] pkts={pkts:4d}/~{args.print_every:.1f}s  "
                          f"last_seq={msg.get('seq')}  age={age_ms} ms  from={addr[0]}")
                else:
                    print(f"[recv] no packets yet…")
                pkts = 0
                last_print = now
    except KeyboardInterrupt:
        print("\n[recv] bye")

if __name__ == "__main__":
    main()
