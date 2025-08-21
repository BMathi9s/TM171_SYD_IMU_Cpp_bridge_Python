#!/usr/bin/env python3
"""
Sim PC runner:
- Receives MediaPipe -> UDP (hand flexions + normalized XYZ) on UDP 9101
- Reads IMU (normalized RPY) from local ImuUdpClient
- Drives DexSuite: Allegro (JOINT_POSITION), Arm (OSC_POSE)
- Non-blocking: latest-sample-wins; never blocks the sim loop
- RUNS AT 40Hz (25ms per loop iteration)

Run:
  python sim_osc_from_udp.py --bind 0.0.0.0 --hand-port 9101 --stale 0.6

Hotkeys:
  I : zero IMU (set current RPY as zero)
  Q : quit
"""

import argparse, json, socket, select, threading, time
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import torch
import cv2  # only for key handling

# --- Your modules ---
# from TM171_SYD_IMU_Cpp_bridge_Python.imu_client import ImuUdpClient
import dexsuite as ds

# ======= Config knobs ======= #
PRINT_EVERY_S   = 0.05   # console print period
STALE_TIMEOUT_S = 0.60   # if no new UDP sample for this long, fall back to zeros
TARGET_HZ = 500           # Target loop frequency
TARGET_DT = 1.0 / TARGET_HZ  # 0.025 seconds = 25ms
# Optional gains if you want to scale normalized [-1,1] to phys units for OSC:
POS_GAIN = 1.0   # keep 1.0 if your controller expects normalized [-1,1] directly
ROT_GAIN = 1.0   # same for r,p,y
# =========================== #


def clamp01(v: float) -> float:
    return float(max(-1.0, min(1.0, v)))


@dataclass
class LatestPacket:
    seq: int = -1
    t_ms: int = 0
    recv_time: float = 0.0  # local arrival time
    # 12 finger floats: [th_CMC, th_MCP, th_IP, idx_MCP, idx_PIP, idx_DIP, mid_MCP, mid_PIP, mid_DIP, ring_MCP, ring_PIP, ring_DIP]
    f: Tuple[float, ...] = field(default_factory=lambda: tuple([0.0]*12))
    # normalized xyz
    xyz: Tuple[float, float, float] = (0.0, 0.0, 0.0)


class HandUdpReceiver(threading.Thread):
    """Non-blocking UDP receiver; stores only the latest packet."""
    def __init__(self, bind_ip: str, port: int, buf_kb: int = 64):
        super().__init__(daemon=True)
        self.bind_ip = bind_ip
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, buf_kb * 1024)
        except OSError:
            pass
        self.sock.bind((bind_ip, port))
        self.sock.setblocking(False)

        self.latest = LatestPacket()
        self._lock = threading.Lock()
        self._run = True
        self.pkts = 0
        self.lost = 0
        self._expect = None

    def stop(self):
        self._run = False

    def get_latest(self) -> Tuple[LatestPacket, float, int, int]:
        """Returns a snapshot of latest + freshness (s) + pkt stats (pkts,lost)."""
        with self._lock:
            lp = LatestPacket(
                seq=self.latest.seq,
                t_ms=self.latest.t_ms,
                recv_time=self.latest.recv_time,
                f=self.latest.f,
                xyz=self.latest.xyz,
            )
            pkts = self.pkts
            lost = self.lost
        fresh_s = time.monotonic() - lp.recv_time if lp.recv_time > 0 else 1e9
        return lp, fresh_s, pkts, lost

    def run(self):
        while self._run:
            ready, _, _ = select.select([self.sock], [], [], 0.01)
            if not ready:
                continue
            while True:
                try:
                    data, _ = self.sock.recvfrom(4096)
                except BlockingIOError:
                    break
                except OSError:
                    break
                self.pkts += 1
                try:
                    msg = json.loads(data.decode("utf-8").strip())
                except json.JSONDecodeError:
                    continue

                if not isinstance(msg, dict):
                    continue
                if msg.get("type") != "AH01":
                    continue
                f = msg.get("f", [])
                xyz = msg.get("xyz", [])
                seq = int(msg.get("seq", -1))
                t_ms = int(msg.get("t_ms", 0))
                if not (isinstance(f, list) and len(f) == 12 and isinstance(xyz, list) and len(xyz) == 3):
                    continue

                # track loss
                if self._expect is not None and seq > self._expect:
                    self.lost += (seq - self._expect)
                self._expect = seq + 1

                # clamp values
                f = tuple(clamp01(float(x)) for x in f)
                xyz = (clamp01(float(xyz[0])), clamp01(float(xyz[1])), clamp01(float(xyz[2])))

                with self._lock:
                    self.latest.seq = seq
                    self.latest.t_ms = t_ms
                    self.latest.recv_time = time.monotonic()
                    self.latest.f = f
                    self.latest.xyz = xyz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bind", default="0.0.0.0", help="IP to bind UDP hand receiver")
    ap.add_argument("--hand-port", type=int, default=9101, help="UDP port for hand/xyz input")
    ap.add_argument("--stale", type=float, default=STALE_TIMEOUT_S, help="Seconds before hand stream considered stale")
    ap.add_argument("--hz", type=float, default=TARGET_HZ, help="Target loop frequency (Hz)")
    args = ap.parse_args()

    # Update target timing based on command line arg
    target_dt = 1.0 / args.hz
    print(f"[timing] Target frequency: {args.hz} Hz ({target_dt*1000:.1f}ms per loop)")

    # === 0) IMU bring-up (local, independent port) ===
    # imu = ImuUdpClient(smooth_alpha=0.0)
    # imu.wait_for_first_sample(timeout_s=1.0)
    # imu.zero_current_rpy()
    #[ x: 175.3799773, y: -3.9643635, z: 89.8886347 ]
    # === 1) DexSuite env ===
    env = ds.make(
        "reach",
        manipulator     = "franka",
        gripper         = "allegro",
        arm_control     = "ik_pose",
        gripper_control = "JOINT_POSITION",
        render_mode     = "human",
    )
    env.reset()

    hand_ctrl = env.robot.hand_ctrl
    arm_ctrl  = env.robot.arm_ctrl
    device    = env.robot.device

    # Allegro mapping (KEEP)
    JOINTS_THUMB  = [3, 7, 11, 15]  # [base, 2nd, 3rd, tip]
    JOINTS_INDEX  = [0, 4, 8, 12]
    JOINTS_MIDDLE = [1, 5, 9, 13]
    JOINTS_RING   = [2, 6, 10, 14]
    BASE_ROTATION_JOINTS = [3, 0, 1, 2]

    # === 2) Start UDP receiver ===
    rx = HandUdpReceiver(args.bind, args.hand_port)
    rx.start()
    print(f"[net] listening UDP on {(args.bind, args.hand_port)} for AH01")

    # last-applied values (used if no new packet)
    last_seq_applied = -1
    th_CMC=th_MCP=th_IP=idx_MCP=idx_PIP=idx_DIP=mid_MCP=mid_PIP=mid_DIP=ring_MCP=ring_PIP=ring_DIP = 0.0
    nx = ny = nz = 0.0

    last_print = time.monotonic()
    
    # === TIMING VARIABLES FOR 40Hz ===
    loop_start_time = time.monotonic()
    loop_count = 0
    timing_stats = {"min_dt": float('inf'), "max_dt": 0.0, "total_time": 0.0}

    try:
        while True:
            loop_iter_start = time.monotonic()
            
            # ---- 1) Pull latest UDP snapshot (non-blocking) ----
            latest, fresh_s, pkts, lost = rx.get_latest()

            if latest.seq >= 0 and latest.seq != last_seq_applied:
                # New data: update local command state
                (th_CMC, th_MCP, th_IP,
                 idx_MCP, idx_PIP, idx_DIP,
                 mid_MCP, mid_PIP, mid_DIP,
                 ring_MCP, ring_PIP, ring_DIP) = latest.f
                nx, ny, nz = latest.xyz
                last_seq_applied = latest.seq

            # If stream stale, optionally fall back to zeros for safety
            stale = (fresh_s > max(0.05, args.stale))
            if stale:
                # Keep base rotations at 0; fingers & xyz already default to last values.
                # If you prefer to relax fingers when stale, uncomment:
                # th_CMC=th_MCP=th_IP=idx_MCP=idx_PIP=idx_DIP=mid_MCP=mid_PIP=mid_DIP=ring_MCP=ring_PIP=ring_DIP = 0.0
                # nx = ny = nz = 0.0
                pass

            # ---- 2) Read IMU normalized RPY every step (lightweight) ----
            # nr, np_, nyaw = imu.get_rpy_normalized()

            # ---- 3) Build actions ----
            # OSC_POSE tuple (you previously noted order is (r,p,y,x,y,z))
            arm_action = torch.tensor(
                [0,0,0,0,0,0] ,
                dtype=torch.float32, device=device
            )

            hand_action = torch.zeros(hand_ctrl.dof, device=device, dtype=torch.float32)
            # Zero base rotation joints
            for j in BASE_ROTATION_JOINTS:
                hand_action[j] = 0.0

            # Thumb
            hand_action[JOINTS_THUMB[1]]  = clamp01(th_CMC)
            hand_action[JOINTS_THUMB[2]]  = clamp01(th_MCP)
            hand_action[JOINTS_THUMB[3]]  = clamp01(th_IP)
            # Index
            hand_action[JOINTS_INDEX[1]]  = clamp01(idx_MCP)
            hand_action[JOINTS_INDEX[2]]  = clamp01(idx_PIP)
            hand_action[JOINTS_INDEX[3]]  = clamp01(idx_DIP)
            # Middle
            hand_action[JOINTS_MIDDLE[1]] = clamp01(mid_MCP)
            hand_action[JOINTS_MIDDLE[2]] = clamp01(mid_PIP)
            hand_action[JOINTS_MIDDLE[3]] = clamp01(mid_DIP)
            # Ring
            hand_action[JOINTS_RING[1]]   = clamp01(ring_MCP)
            hand_action[JOINTS_RING[2]]   = clamp01(ring_PIP)
            hand_action[JOINTS_RING[3]]   = clamp01(ring_DIP)

            # ---- 4) Step sim ----
            obs, reward, terminated, truncated, info = env.step((arm_action, hand_action))

            # ---- 5) Light prints & hotkeys ----
            now = time.monotonic()
            if now - last_print >= PRINT_EVERY_S:
                actual_hz = loop_count / (now - loop_start_time) if loop_count > 0 else 0
                avg_dt = timing_stats["total_time"] / loop_count if loop_count > 0 else 0
                print(f"[loop] fresh={fresh_s*1000:4.0f} ms  pkts={pkts}  lost={lost}  "
                      f"seq={latest.seq}  xyz=({nx:+.2f},{ny:+.2f},{nz:+.2f})  "
                    #   f"rpy=({nr:+.2f},{np_:+.2f},{nyaw:+.2f})  stale={stale}  "
                      f"hz={actual_hz:.1f}  dt={avg_dt*1000:.1f}ms")
                last_print = now

            k = cv2.waitKey(1) & 0xFF
            if k in (ord('q'), ord('Q')):
                break
            # if k in (ord('i'), ord('I')):
            #     imu.zero_current_rpy()

            # ---- 6) RATE LIMITING: Sleep to maintain target Hz ----
            loop_iter_end = time.monotonic()
            loop_duration = loop_iter_end - loop_iter_start
            
            # Update timing stats
            timing_stats["min_dt"] = min(timing_stats["min_dt"], loop_duration)
            timing_stats["max_dt"] = max(timing_stats["max_dt"], loop_duration)
            timing_stats["total_time"] += loop_duration
            loop_count += 1
            
            # Sleep for remaining time to hit target frequency
            sleep_time = target_dt - loop_duration
            if sleep_time > 0:
                time.sleep(sleep_time)
            # Note: if loop_duration > target_dt, we're already running slow, no sleep needed

    except KeyboardInterrupt:
        print("\n[run] Interrupted.")
    finally:
        print("[exit] Cleaning up…")
        if loop_count > 0:
            total_runtime = time.monotonic() - loop_start_time
            actual_avg_hz = loop_count / total_runtime
            print(f"[timing] Final stats: {actual_avg_hz:.1f} Hz avg, "
                  f"dt range: {timing_stats['min_dt']*1000:.1f}-{timing_stats['max_dt']*1000:.1f}ms")
        rx.stop(); rx.join(timeout=1.0)
        env.close()
        cv2.destroyAllWindows()
        print("[exit] Done.")


if __name__ == "__main__":
    main()