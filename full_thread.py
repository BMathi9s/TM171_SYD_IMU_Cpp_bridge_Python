#!/usr/bin/env python3
import time, threading
import numpy as np
import torch
import cv2

# --- Your modules ---
from TM171_SYD_IMU_Cpp_bridge_Python.pos_tracking.hand_position_tracker import hand_position_tracker
from TM171_SYD_IMU_Cpp_bridge_Python.imu_client import ImuUdpClient

# --- DexSuite / Genesis ---
import dexsuite as ds


# ================================ knobs ================================= #
SHOW_UI          = True     # set False to remove all imshow() overhead
UI_MAX_FPS       = 20       # cap UI draw rate
DRAW_LANDMARKS   = False    # True = draw hands overlay in the worker thread
STALE_TIMEOUT_S  = 0.6      # if inputs older than this, send zeros to hand
PRINT_EVERY_S    = 0.5      # console prints
# ======================================================================== #


def clamp01(v):  # actually maps to [-1, 1] clamp
    return float(max(-1.0, min(1.0, v)))


class SharedState:
    """Lock-protected snapshot of the latest tracker outputs."""
    def __init__(self):
        self.lock = threading.Lock()
        # normalized finger flexions (dicts with MCP/PIP/DIP or CMC/MCP/IP)
        self.n_index  = {"MCP": 0.0, "PIP": 0.0, "DIP": 0.0}
        self.n_middle = {"MCP": 0.0, "PIP": 0.0, "DIP": 0.0}
        self.n_ring   = {"MCP": 0.0, "PIP": 0.0, "DIP": 0.0}
        self.n_thumb  = {"CMC": 0.0, "MCP": 0.0, "IP":  0.0}
        # normalized xyz
        self.nx = 0.0; self.ny = 0.0; self.nz = 0.0
        # optional UI frames
        self.color_bgr = None
        self.depth_vis = None
        # freshness
        self.seq   = 0
        self.stamp = 0.0


class TrackerWorker(threading.Thread):
    """Runs MediaPipe tracking in a background thread and publishes latest values."""
    def __init__(self, tracker, shared: SharedState, draw_landmarks=DRAW_LANDMARKS, show_ui=SHOW_UI):
        super().__init__(daemon=True)
        self.tracker = tracker
        self.shared  = shared
        self.draw_landmarks = draw_landmarks
        self.show_ui  = show_ui
        self._run_evt = threading.Event(); self._run_evt.set()   # running
        self._pause_evt = threading.Event(); self._pause_evt.set()  # not paused

    def pause(self):
        self._pause_evt.clear()

    def resume(self):
        self._pause_evt.set()

    def stop(self):
        self._run_evt.clear()
        self._pause_evt.set()  # in case paused, allow exit

    def run(self):
        while self._run_evt.is_set():
            self._pause_evt.wait()  # block if paused (wizard, etc.)
            # one tracker step
            color_bgr, depth_vis = self.tracker.process(draw_landmarks=self.draw_landmarks)
            if color_bgr is None:
                continue

            # grab normalized values
            n_index  = self.tracker.get_normalized_flexion_index()
            n_middle = self.tracker.get_normalized_flexion_middle()
            n_ring   = self.tracker.get_normalized_flexion_ring()
            n_thumb  = self.tracker.get_normalized_flexion_thumb()
            nx, ny, nz = self.tracker.get_normalized_xyz()

            # publish latest snapshot
            with self.shared.lock:
                self.shared.n_index  = dict(n_index)
                self.shared.n_middle = dict(n_middle)
                self.shared.n_ring   = dict(n_ring)
                self.shared.n_thumb  = dict(n_thumb)
                self.shared.nx, self.shared.ny, self.shared.nz = float(nx), float(ny), float(nz)
                if self.show_ui:
                    # copy frames to avoid reuse by tracker
                    self.shared.color_bgr = color_bgr.copy()
                    self.shared.depth_vis = (depth_vis.copy() if depth_vis is not None else None)
                self.shared.seq   += 1
                self.shared.stamp  = time.monotonic()


def main():
    # === IMU (kept here; it’s light) ===
    imu = ImuUdpClient(smooth_alpha=0.0)
    imu.wait_for_first_sample(timeout_s=1.0)
    imu.zero_current_rpy()

    # === DexSuite env ===
    env = ds.make(
        "reach",
        manipulator     = "franka",
        gripper         = "allegro",
        arm_control     = "osc_pose",         # using your OSC pose controller
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
    BASE_ROTATION_JOINTS = [3, 0, 1, 2]  # zero per frame

    # === Tracker bring-up on main thread (menus/wizards use OpenCV UI) ===
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
        tracker.stop(); env.close(); return

    res_hand = tracker.hand_startup_menu(window_name=MAIN_WIN)
    if not res_hand:
        print("[init] Hand step cancelled; exiting.")
        tracker.stop(); env.close(); return

    # === Start the background worker ===
    shared = SharedState()
    worker = TrackerWorker(tracker, shared, draw_landmarks=DRAW_LANDMARKS, show_ui=SHOW_UI)
    worker.start()

    print("[run] Threaded: Allegro fingers ← tracker; EEF held at zeros (OSC).")
    last_print = 0.0
    last_ui = 0.0

    try:
        while True:
            # Snapshot inputs (non-blocking, minimal time under lock)
            with shared.lock:
                n_index  = dict(shared.n_index)
                n_middle = dict(shared.n_middle)
                n_ring   = dict(shared.n_ring)
                n_thumb  = dict(shared.n_thumb)
                nx, ny, nz = shared.nx, shared.ny, shared.nz
                img = shared.color_bgr
                dep = shared.depth_vis
                age = time.monotonic() - shared.stamp

            # Arm action (OSC tuple): you’re fixing EEF for latency tests → zeros
            # If/when you want to use IMU+XYZ, pack as (r,p,y, x,y,z) per your controller.
            arm_action = torch.zeros(6, dtype=torch.float32, device=device)

            # Allegro hand (16-dof) — if inputs stale, hold last (or drive 0s)
            hand_action = torch.zeros(hand_ctrl.dof, device=device, dtype=torch.float32)
            if age <= STALE_TIMEOUT_S:
                # zero base rotation joints
                for j in BASE_ROTATION_JOINTS:
                    hand_action[j] = 0.0
                # Thumb (no pinky)
                hand_action[JOINTS_THUMB[1]]  = clamp01(n_thumb["CMC"])
                hand_action[JOINTS_THUMB[2]]  = clamp01(n_thumb["MCP"])
                hand_action[JOINTS_THUMB[3]]  = clamp01(n_thumb["IP"])
                # Index
                hand_action[JOINTS_INDEX[1]]  = clamp01(n_index["MCP"])
                hand_action[JOINTS_INDEX[2]]  = clamp01(n_index["PIP"])
                hand_action[JOINTS_INDEX[3]]  = clamp01(n_index["DIP"])
                # Middle
                hand_action[JOINTS_MIDDLE[1]] = clamp01(n_middle["MCP"])
                hand_action[JOINTS_MIDDLE[2]] = clamp01(n_middle["PIP"])
                hand_action[JOINTS_MIDDLE[3]] = clamp01(n_middle["DIP"])
                # Ring
                hand_action[JOINTS_RING[1]]   = clamp01(n_ring["MCP"])
                hand_action[JOINTS_RING[2]]   = clamp01(n_ring["PIP"])
                hand_action[JOINTS_RING[3]]   = clamp01(n_ring["DIP"])
            else:
                # Inputs too old → keep base joints 0; others already 0
                pass

            # Step sim (tight loop; no sleeps)
            obs, reward, terminated, truncated, info = env.step((arm_action, hand_action))

            # UI (throttled)
            if SHOW_UI and (time.monotonic() - last_ui) >= (1.0 / max(UI_MAX_FPS, 1)):
                last_ui = time.monotonic()
                if img is not None:
                    cv2.imshow(MAIN_WIN, img)
                if dep is not None:
                    cv2.imshow("Depth (vis)", dep)
                cv2.waitKey(1)

            # Light console prints
            now = time.monotonic()
            if now - last_print > PRINT_EVERY_S:
                last_print = now
                print(f"age={age*1000:5.0f} ms  "
                      f"Idx(M,P,D)=({n_index['MCP']:+.2f},{n_index['PIP']:+.2f},{n_index['DIP']:+.2f})  "
                      f"Th(C,M,IP)=({n_thumb['CMC']:+.2f},{n_thumb['MCP']:+.2f},{n_thumb['IP']:+.2f})")

            # Hotkeys (wizard runs on main thread; pause worker to avoid contention)
            k = cv2.waitKey(1) & 0xFF if SHOW_UI else 255
            if k in (ord('q'), ord('Q')):
                break
            if k in (ord('h'), ord('H')):
                worker.pause()
                tracker.hand_calibration_wizard(window_name=MAIN_WIN)
                worker.resume()
            if k in (ord('i'), ord('I')):
                imu.zero_current_rpy()

    except KeyboardInterrupt:
        print("\n[run] Interrupted.")
    finally:
        print("[exit] Cleaning up…")
        worker.stop(); worker.join(timeout=1.0)
        tracker.stop()
        env.close()
        if SHOW_UI:
            cv2.destroyAllWindows()
        print("[exit] Done.")


if __name__ == "__main__":
    main()
