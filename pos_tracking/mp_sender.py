#!/usr/bin/env python3


#python mp_sender.py --dest 10.10.0.2 --port 9101 --show --draw
#no gui
#python mp_sender.py --dest 10.10.0.2 --port 9101 

"""
MediaPipe → UDP sender (human-readable JSON)

Sends:
  type: "AH01"
  seq : uint
  t_ms: sender wall-clock (ms)
  f   : 12 floats in [-1,1]  (thumb CMC,MCP,IP, index MCP,PIP,DIP, middle MCP,PIP,DIP, ring MCP,PIP,DIP)
  xyz : 3 floats in [-1,1]   (nx, ny, nz)

Run:
  python mp_sender.py --dest 10.10.0.2 --port 9101
"""
import socket, json, time, argparse
import cv2

from hand_position_tracker import hand_position_tracker  # your module

def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="10.10.0.2", help="Receiver IP (Sim PC)")
    ap.add_argument("--port", type=int, default=9101, help="Receiver UDP port")
    ap.add_argument("--show", action="store_true", help="Show camera UI (slower)")
    ap.add_argument("--draw", action="store_true", help="Draw landmarks on the UI")
    ap.add_argument("--name", default="MP→UDP", help="Window name if --show")
    ap.add_argument("--w", type=int, default=1280)
    ap.add_argument("--h", type=int, default=720)
    return ap

def clamp01(v):
    return float(max(-1.0, min(1.0, v)))

def main():
    args = build_parser().parse_args()

    # UDP setup
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dest = (args.dest, args.port)
    print(f"[sender] → {dest}  (best-effort, no fixed rate)")

    # Tracker bring-up (with your wizards)
    tracker = hand_position_tracker(
        color_w=args.w, color_h=args.h,
        tag_size_m=0.072,
        bake_adjustments_at_capture=True
    )
    tracker.start()

    MAIN_WIN = args.name
    res_xyz  = tracker.startup_menu(window_name=MAIN_WIN if args.show else None)
    if not res_xyz:
        print("[init] XYZ step cancelled; exiting.")
        tracker.stop(); return

    res_hand = tracker.hand_startup_menu(window_name=MAIN_WIN if args.show else None)
    if not res_hand:
        print("[init] Hand step cancelled; exiting.")
        tracker.stop(); return

    seq = 0
    last_print_t = time.time()
    sent_in_last = 0

    try:
        while True:
            # one tracker step (this is the pacing)
            color_bgr, depth_vis = tracker.process(draw_landmarks=args.draw)
            if color_bgr is None:
                continue

            # (optional) HUD/UI
            if args.show:
                tracker.draw_hud(color_bgr)
                cv2.imshow(MAIN_WIN, color_bgr)
                if depth_vis is not None:
                    cv2.imshow("Depth (vis)", depth_vis)
                # keep UI responsive but don't stall the loop
                cv2.waitKey(1)

            # Read normalized data
            nx, ny, nz = tracker.get_normalized_xyz()  # [-1,1]
            n_thumb  = tracker.get_normalized_flexion_thumb()   # CMC,MCP,IP
            n_index  = tracker.get_normalized_flexion_index()   # MCP,PIP,DIP
            n_middle = tracker.get_normalized_flexion_middle()  # MCP,PIP,DIP
            n_ring   = tracker.get_normalized_flexion_ring()    # MCP,PIP,DIP
            # (pinky intentionally ignored)

            fingers = [
                clamp01(n_thumb["CMC"]),  clamp01(n_thumb["MCP"]),  clamp01(n_thumb["IP"]),
                clamp01(n_index["MCP"]),  clamp01(n_index["PIP"]),  clamp01(n_index["DIP"]),
                clamp01(n_middle["MCP"]), clamp01(n_middle["PIP"]), clamp01(n_middle["DIP"]),
                clamp01(n_ring["MCP"]),   clamp01(n_ring["PIP"]),   clamp01(n_ring["DIP"]),
            ]
            xyz = [clamp01(nx), clamp01(ny), clamp01(nz)]

            msg = {
                "type": "AH01",
                "seq": seq,
                "t_ms": int(time.time() * 1000),
                "f": [round(x, 5) for x in fingers],
                "xyz": [round(x, 5) for x in xyz],
            }
            packet = (json.dumps(msg) + "\n").encode("utf-8")
            sock.sendto(packet, dest)
            seq += 1
            sent_in_last += 1

            # light stats
            now = time.time()
            if now - last_print_t >= 1.0:
                print(f"[sender] ~{sent_in_last} pkts/s  (UI={'on' if args.show else 'off'}, draw={'on' if args.draw else 'off'})")
                sent_in_last = 0
                last_print_t = now

            # hotkeys (only when showing UI)
            if args.show:
                k = cv2.waitKey(1) & 0xFF
                if k in (ord('q'), ord('Q')):
                    break
                if k in (ord('h'), ord('H')):
                    tracker.hand_calibration_wizard(window_name=MAIN_WIN)

    except KeyboardInterrupt:
        print("\n[sender] Ctrl-C")
    finally:
        tracker.stop()
        if args.show:
            cv2.destroyAllWindows()
        print("[sender] done.")

if __name__ == "__main__":
    main()
