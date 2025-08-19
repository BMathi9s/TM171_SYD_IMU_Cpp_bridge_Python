#!/usr/bin/env python3
import os, sys, time, logging
import numpy as np
import torch
import cv2

# --- Your modules ---
from pos_tracking.hand_position_tracker import hand_position_tracker
from imu_client import ImuUdpClient

# --- DexSuite / Genesis ---
import dexsuite as ds

# Silence Genesis FPS spam
genesis_logger = logging.getLogger('genesis')
class _FPSFilter(logging.Filter):
    def filter(self, record): return 'FPS' not in record.getMessage()
genesis_logger.addFilter(_FPSFilter())
genesis_logger.setLevel(logging.WARNING)

class _Silence:
    def __enter__(self):
        self._stderr, self._stdout = sys.stderr, sys.stdout
        sys.stderr = open(os.devnull, 'w'); sys.stdout = open(os.devnull, 'w')
        return self
    def __exit__(self, *_):
        sys.stderr.close(); sys.stdout.close()
        sys.stderr, sys.stdout = self._stderr, self._stdout

def clamp01(v):  # actually [-1,1]
    return float(max(-1.0, min(1.0, v)))

def main():
    # === 0) IMU bring-up ===
    imu = ImuUdpClient(smooth_alpha=0.0)
    imu.wait_for_first_sample(timeout_s=2.0)
    imu.zero_current_rpy()   # make current pose the zero

    # === 1) DexSuite env with IK pose on the arm, joint-pos on Allegro ===
    print("[init] Starting DexSuite with IK pose controller…")
    with _Silence():
        env = ds.make(
            "reach",
            manipulator     = "franka",
            gripper         = "allegro",
            arm_control     = "IK_POSE",         # <— absolute IK pose controller
            gripper_control = "JOINT_POSITION",  # Allegro joints as before
            render_mode     = "human",
        )
        env.reset()

    hand_ctrl = env.robot.hand_ctrl
    arm_ctrl  = env.robot.arm_ctrl
    device    = env.robot.device
    dt        = 1.0 / 60.0

    # Sanity: IK pose should be 6 dof (x,y,z,r,p,y)
    if arm_ctrl.dof != 6:
        print(f"[warn] Arm controller dof is {arm_ctrl.dof}, expected 6 for IK pose.")

    # === 2) Allegro joint mapping (KEEP) ===
    JOINTS_THUMB  = [3, 7, 11, 15]  # [base, 2nd, 3rd, tip]
    JOINTS_INDEX  = [0, 4, 8, 12]
    JOINTS_MIDDLE = [1, 5, 9, 13]
    JOINTS_RING   = [2, 6, 10, 14]
    BASE_ROTATION_JOINTS = [3, 0, 1, 2]  # zero every frame (thumb/index/middle/ring base)

    # === 3) Tracker bring-up (with your menus) ===
    tracker = hand_position_tracker(
        color_w=1280, color_h=720,
        tag_size_m=0.072,
        bake_adjustments_at_capture=True
    )
    tracker.start()

    MAIN_WIN = "Main"
    res_xyz  = tracker.startup_menu(window_name=MAIN_WIN)
    if not res_xyz:
        print("[init] XYZ step cancelled; exiting.")
        tracker.stop()
        with _Silence(): env.close()
        return

    res_hand = tracker.hand_startup_menu(window_name=MAIN_WIN)
    if not res_hand:
        print("[init] Hand step cancelled; exiting.")
        tracker.stop()
        with _Silence(): env.close()
        return

    print("[run] Driving Allegro fingers from tracker; EEF from normalized XYZ + IMU RPY.")

    last_print = 0.0
    try:
        while True:
            # --- Tracker frame/update + HUD ---
            color_bgr, depth_vis = tracker.process(draw_landmarks=True)
            if color_bgr is None:
                continue
            tracker.draw_hud(color_bgr)

            cv2.imshow(MAIN_WIN, color_bgr)
            if depth_vis is not None:
                cv2.imshow("Depth (vis)", depth_vis)

            # --- Read normalized hand flexions ---
            n_index  = tracker.get_normalized_flexion_index()   # dict: MCP,PIP,DIP
            n_middle = tracker.get_normalized_flexion_middle()
            n_ring   = tracker.get_normalized_flexion_ring()
            n_thumb  = tracker.get_normalized_flexion_thumb()   # dict: CMC,MCP,IP

            # --- Read normalized XYZ (EEF position target) ---
            nx, ny, nz = tracker.get_normalized_xyz()  # each in [-1, 1]

            # --- Read normalized orientation from IMU (RPY) ---
            nr, np_, nyaw = imu.get_rpy_normalized()   # each in [-1, 1]

            # --- Build actions ---
            # Arm IK pose expects [x, y, z, r, p, y] (we're sending normalized as requested)
            arm_action = torch.tensor(
                [clamp01(nx), clamp01(ny), clamp01(nz),
                 clamp01(nr), clamp01(np_), clamp01(nyaw)],
                dtype=torch.float32, device=device
            )

            # Allegro hand joints: 16-dof normalized [-1,1]
            hand_action = torch.zeros(hand_ctrl.dof, device=device, dtype=torch.float32)

            # Zero the base rotation joints every frame
            for j in BASE_ROTATION_JOINTS:
                hand_action[j] = 0.0

            # Thumb (ignore pinky entirely)
            hand_action[JOINTS_THUMB[1]]  = clamp01(n_thumb['CMC'])
            hand_action[JOINTS_THUMB[2]]  = clamp01(n_thumb['MCP'])
            hand_action[JOINTS_THUMB[3]]  = clamp01(n_thumb['IP'])

            # Index
            hand_action[JOINTS_INDEX[1]]  = clamp01(n_index['MCP'])
            hand_action[JOINTS_INDEX[2]]  = clamp01(n_index['PIP'])
            hand_action[JOINTS_INDEX[3]]  = clamp01(n_index['DIP'])

            # Middle
            hand_action[JOINTS_MIDDLE[1]] = clamp01(n_middle['MCP'])
            hand_action[JOINTS_MIDDLE[2]] = clamp01(n_middle['PIP'])
            hand_action[JOINTS_MIDDLE[3]] = clamp01(n_middle['DIP'])

            # Ring
            hand_action[JOINTS_RING[1]]   = clamp01(n_ring['MCP'])
            hand_action[JOINTS_RING[2]]   = clamp01(n_ring['PIP'])
            hand_action[JOINTS_RING[3]]   = clamp01(n_ring['DIP'])

            # --- Step sim ---
            with _Silence():
                obs, reward, terminated, truncated, info = env.step((arm_action, hand_action))

            # --- Light console prints @ ~10 Hz ---
            now = time.time()
            if now - last_print > 0.10:
                last_print = now
                print(
                    f"EEF n-XYZ=({nx:+.2f},{ny:+.2f},{nz:+.2f})  "
                    f"n-RPY=({nr:+.2f},{np_:+.2f},{nyaw:+.2f})  |  "
                    f"Index(M,P,D)=({n_index['MCP']:+.2f},{n_index['PIP']:+.2f},{n_index['DIP']:+.2f})  "
                    f"Middle=({n_middle['MCP']:+.2f},{n_middle['PIP']:+.2f},{n_middle['DIP']:+.2f})  "
                    f"Ring=({n_ring['MCP']:+.2f},{n_ring['PIP']:+.2f},{n_ring['DIP']:+.2f})  "
                    f"Thumb(C,M,IP)=({n_thumb['CMC']:+.2f},{n_thumb['MCP']:+.2f},{n_thumb['IP']:+.2f})"
                )

            # --- Hotkeys ---
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q')):
                break
            if key in (ord('h'), ord('H')):
                tracker.hand_calibration_wizard(window_name=MAIN_WIN)
            if key in (ord('i'), ord('I')):
                imu.zero_current_rpy()  # set current orientation as new zero

            time.sleep(dt)

    except KeyboardInterrupt:
        print("\n[run] Interrupted.")
    finally:
        print("[exit] Cleaning up…")
        tracker.stop()
        with _Silence(): env.close()
        print("[exit] Done.")

if __name__ == "__main__":
    main()
