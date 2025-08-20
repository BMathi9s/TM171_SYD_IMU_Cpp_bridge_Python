#!/usr/bin/env python3
import os, sys, time, logging
import numpy as np
import torch

# === Glove I/O (from glove.py) ===
try:
    from glove import (
        serial,  # type: ignore
        find_arduino_device,
        guided_calibration,
        read_sensor_frame,
        get_normalized_values,
        SENSOR_MAP,
        NUM_SENSORS,
    )
except Exception as e:
    print("[fatal] Could not import glove helpers. Ensure glove.py is on PYTHONPATH.")
    raise

# === DexSuite / Genesis ===
import dexsuite as ds


def clamp01(v: float) -> float:  # actually clamps to [-1,1]
    return float(max(-1.0, min(1.0, v)))


def main():
    print("[init] Starting DexSuite with IK pose controller…")
    env = ds.make(
        "reach",
        manipulator     = "franka",
        gripper         = "allegro",
        arm_control     = "osc_pose",        # 6-DoF pose (we will send zeros)
        gripper_control = "JOINT_POSITION",  # Allegro joints
        render_mode     = "human",
    )
    env.reset()

    hand_ctrl = env.robot.hand_ctrl
    arm_ctrl  = env.robot.arm_ctrl
    device    = env.robot.device

    # Sanity: IK pose should be 6 dof (x,y,z,r,p,y)
    if arm_ctrl.dof != 6:
        print(f"[warn] Arm controller dof is {arm_ctrl.dof}, expected 6 for IK pose.")

    # === Allegro joint indices ===
    # Order: [0..15] map to Allegro joints; each finger has 4 joints (base, MCP, PIP, DIP/IP)
    JOINTS_THUMB  = [3, 7, 11, 15]  # [base rot, CMC, MCP, IP]
    JOINTS_INDEX  = [0, 4, 8, 12]   # [base rot, MCP, PIP, DIP]
    JOINTS_MIDDLE = [1, 5, 9, 13]
    JOINTS_RING   = [2, 6, 10, 14]
    BASE_ROTATION_JOINTS = [3, 0, 1, 2]  # zero every frame

    # === 0) Glove bring-up + port calibration wizard ===
    # Auto-detect port; if not found, prompt the user.
    port = None
    try:
        port = find_arduino_device()
    except Exception:
        port = None

    if not port:
        print("Could not auto-detect Arduino device.")
        print("Common Linux serial ports: /dev/ttyUSB0,/dev/ttyUSB1,/dev/ttyACM0,/dev/ttyACM1")
        port = input("Enter your serial port (e.g., /dev/ttyUSB0): ").strip()
        if not port:
            print("[init] No serial port specified. Exiting.")
            env.close()
            return

    baud = 115200
    try:
        ser = serial.Serial(port, baud, timeout=1)  # type: ignore
        time.sleep(2.0)
    except Exception as e:
        print(f"[fatal] Failed to open serial port {port}: {e}")
        env.close()
        return

    print(f"[glove] Connected to {port} @ {baud} baud.")
    print("[glove] Starting guided port calibration wizard (finger-by-finger).")
    try:
        mins, maxs = guided_calibration(ser)
    except KeyboardInterrupt:
        print("\n[glove] Calibration interrupted by user. Exiting.")
        ser.close()
        env.close()
        return

    # === 1) Run: fingers from glove; arm zeros ===
    print("[run] Driving Allegro fingers from GLOVE; arm pose set to zeros (testing mode).")
    last_print = 0.0
    try:
        while True:
            # Read a complete frame and normalize to [-1,1]
            raw_vals = read_sensor_frame(ser)
            norm = get_normalized_values(raw_vals, mins, maxs)  # dict[int] -> float|None

            # Build 16-dof Allegro action
            hand_action = torch.zeros(hand_ctrl.dof, device=device, dtype=torch.float32)

            # Zero the base rotation joints every frame
            for j in BASE_ROTATION_JOINTS:
                hand_action[j] = 0.0

            # Map per-finger. Glove sensors:
            # thumb: 4 sensors -> we'll use first 3 as [CMC,MCP,IP]; sensor 3 ignored
            # index/middle/ring: 3 sensors -> [MCP,PIP,DIP]
            # pinky: ignored (no pinky on Allegro hand)
            def nval(sid: int) -> float:
                v = norm.get(sid, 0.0)
                return 0.0 if v is None else clamp01(float(v))

            # Thumb mapping (glove sensors 0,1,2,3)
            hand_action[JOINTS_THUMB[1]] = nval(SENSOR_MAP["thumb"][0])  # CMC
            hand_action[JOINTS_THUMB[2]] = nval(SENSOR_MAP["thumb"][1])  # MCP
            hand_action[JOINTS_THUMB[3]] = nval(SENSOR_MAP["thumb"][2])  # IP
            # Index
            hand_action[JOINTS_INDEX[1]]  = nval(SENSOR_MAP["index"][0])   # MCP
            hand_action[JOINTS_INDEX[2]]  = nval(SENSOR_MAP["index"][1])   # PIP
            hand_action[JOINTS_INDEX[3]]  = nval(SENSOR_MAP["index"][2])   # DIP
            # Middle
            hand_action[JOINTS_MIDDLE[1]] = nval(SENSOR_MAP["middle"][0])
            hand_action[JOINTS_MIDDLE[2]] = nval(SENSOR_MAP["middle"][1])
            hand_action[JOINTS_MIDDLE[3]] = nval(SENSOR_MAP["middle"][2])
            # Ring
            hand_action[JOINTS_RING[1]]   = nval(SENSOR_MAP["ring"][0])
            hand_action[JOINTS_RING[2]]   = nval(SENSOR_MAP["ring"][1])
            hand_action[JOINTS_RING[3]]   = nval(SENSOR_MAP["ring"][2])

            # Arm: zeros [x,y,z,r,p,y]  (note: osc_pose action order is [r,p,y, x,y,z] in the user's prior code)
            arm_action = torch.tensor([0,0,0, 0,0,0], dtype=torch.float32, device=device)

            # Step sim
            obs, reward, terminated, truncated, info = env.step((arm_action, hand_action))

            # Prints @ ~2 Hz
            now = time.time()
            if now - last_print > 0.5:
                last_print = now
                def fmt(v): 
                    return f"{v:+.2f}" if v is not None else "  NA"
                # Small summary by finger
                t0, t1, t2 = [norm.get(i) for i in SENSOR_MAP["thumb"][:3]]
                i0, i1, i2 = [norm.get(i) for i in SENSOR_MAP["index"]]
                m0, m1, m2 = [norm.get(i) for i in SENSOR_MAP["middle"]]
                r0, r1, r2 = [norm.get(i) for i in SENSOR_MAP["ring"]]
                print(
                    "Thumb(CMC,MCP,IP)=({},{},{}) | "
                    "Index(M,P,D)=({},{},{}) | "
                    "Middle(M,P,D)=({},{},{}) | "
                    "Ring(M,P,D)=({},{},{})".format(
                        fmt(t0), fmt(t1), fmt(t2),
                        fmt(i0), fmt(i1), fmt(i2),
                        fmt(m0), fmt(m1), fmt(m2),
                        fmt(r0), fmt(r1), fmt(r2),
                    )
                )

    except KeyboardInterrupt:
        print("\n[run] Interrupted.")
    finally:
        print("[exit] Cleaning up…")
        try:
            ser.close()
        except Exception:
            pass
        env.close()
        print("[exit] Done.")


if __name__ == "__main__":
    main()
