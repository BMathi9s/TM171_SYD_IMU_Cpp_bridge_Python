import socket, json, time, argparse, math, os, random

#python send_udp_json.py --dest 10.10.0.2 --port 9101 --hz 100
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="10.10.0.2", help="Receiver IP (Sim PC)")
    ap.add_argument("--port", type=int, default=9101, help="Receiver UDP port")
    ap.add_argument("--hz", type=float, default=100.0, help="Send rate")
    args = ap.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dest = (args.dest, args.port)

    period = 1.0 / max(args.hz, 1.0)
    seq = 0
    next_t = time.perf_counter()
    last_print = time.time()
    sent = 0

    print(f"[sender] → {dest}  at ~{args.hz:.1f} Hz")
    try:
        while True:
            now = time.perf_counter()
            if now < next_t:
                time.sleep(next_t - now)
            next_t += period

            t_ms = int(time.time() * 1000)
            # Dummy data just to exercise bandwidth (replace with real later)
            # 12 finger values + xyz + rpy (normalized)
            phase = time.time() * 0.5
            fingers = [math.sin(phase + i*0.2) for i in range(12)]
            xyz = [math.sin(phase)*0.2, math.cos(phase)*0.2, math.sin(phase*0.7)*0.2]
            rpy = [0.1*math.sin(phase), 0.1*math.cos(phase), 0.1*math.sin(phase*0.5)]

            msg = {
                "type": "AH01",
                "seq": seq,
                "t_ms": t_ms,
                "f": [round(x, 4) for x in fingers],
                "xyz": [round(x, 4) for x in xyz],
                "rpy": [round(x, 4) for x in rpy],
            }
            packet = (json.dumps(msg) + "\n").encode("utf-8")
            sock.sendto(packet, dest)
            seq += 1
            sent += 1

            if time.time() - last_print > 1.0:
                print(f"[sender] sent {sent} pkts in last 1s")
                sent = 0
                last_print = time.time()
    except KeyboardInterrupt:
        print("\n[sender] bye")

if __name__ == "__main__":
    main()
