# hand_rs_mp_imu.py
# Fuse RealSense+MediaPipe Hands with IMU orientation for a palm-tangent arrow.
# Keys:
#   q = quit
#   d = define baseline: zero IMU + capture palm forward from landmarks (index/pinky MCP → forward)

import time
from collections import defaultdict
import cv2
import numpy as np
import pyrealsense2 as rs
import mediapipe as mp

# ---- IMU (UDP) ----
from imu_client import ImuUdpClient

# ========= Tunables =========
COLOR_W, COLOR_H, FPS = 640, 480, 30
DEPTH_RANGE_M = (0.25, 1.2)
EMA_ALPHA     = 0.35
K_NEIGH       = 3
SHOW_Z_AT     = mp.solutions.hands.HandLandmark.WRIST
ARROW_LEN_M   = 0.14   # 14 cm arrow
# ============================

def depth_at_m(depth_frame, u, v, k=3):
    u = int(round(u)); v = int(round(v))
    w = depth_frame.get_width(); h = depth_frame.get_height()
    r = k // 2; vals = []
    for yy in range(max(0, v - r), min(h, v + r + 1)):
        for xx in range(max(0, u - r), min(w, u + r + 1)):
            d = depth_frame.get_distance(xx, yy)
            if d and np.isfinite(d):
                vals.append(d)
    if not vals: return 0.0
    return float(np.median(vals))

def deproject(intr, u, v, z_m):
    pt = rs.rs2_deproject_pixel_to_point(intr, [float(u), float(v)], float(z_m))
    return np.array(pt, dtype=np.float32)  # [X,Y,Z] meters, camera coords

def project_px(intr, P3):
    u, v = rs.rs2_project_point_to_pixel(intr, [float(P3[0]), float(P3[1]), float(P3[2])])
    return int(round(u)), int(round(v))

# Rotation helpers (RPY in degrees → rotation matrix; ZYX convention: Rz*y*Rx)
def rpy_deg_to_R(r_deg, p_deg, y_deg):
    r = np.deg2rad(r_deg); p = np.deg2rad(p_deg); y = np.deg2rad(y_deg)
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    # ZYX: Rz * Ry * Rx
    Rz = np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]],dtype=np.float32)
    Ry = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]],dtype=np.float32)
    Rx = np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]],dtype=np.float32)
    return (Rz @ Ry @ Rx).astype(np.float32)

# ---- RealSense setup ----
pipe = rs.pipeline()
cfg  = rs.config()
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, FPS)
cfg.enable_stream(rs.stream.color, COLOR_W, COLOR_H, rs.format.bgr8, FPS)
profile = pipe.start(cfg)
align = rs.align(rs.stream.color)

# Optional IR projector
try:
    depth_sensor = profile.get_device().first_depth_sensor()
    if depth_sensor.supports(rs.option.emitter_enabled):
        depth_sensor.set_option(rs.option.emitter_enabled, 1)
    if depth_sensor.supports(rs.option.laser_power):
        depth_sensor.set_option(rs.option.laser_power,
                                min(depth_sensor.get_option_range(rs.option.laser_power).max, 150))
except Exception:
    pass

# Depth filters
spatial  = rs.spatial_filter()
temporal = rs.temporal_filter()
holefill = rs.hole_filling_filter()

# Intrinsics for color-aligned depth
color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
intr = color_stream.get_intrinsics()

# ---- MediaPipe Hands ----
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
    model_complexity=1
)
mp_draw = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles

# Smoothing state per hand/landmark
ema_3d = defaultdict(dict)

# ---- IMU client ----
imu = ImuUdpClient(smooth_alpha=0.1)  # a bit of smoothing is fine
imu.wait_for_first_sample(2.0)
# imu.invert_axis("pitch") 

# Baseline state (captured on 'd')
has_baseline = False
R0 = np.eye(3, dtype=np.float32)         # IMU rotation at baseline
fwd0_cam = np.array([0,0,1], np.float32) # baseline "forward" (palm tangent, toward fingers) in camera frame

print("Press 'd' to set IMU zero + capture palm-forward baseline; 'q' to quit.")

try:
    while True:
        frames = pipe.wait_for_frames()
        frames = align.process(frames)

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Depth filtering
        df = depth_frame
        df = spatial.process(df)
        df = temporal.process(df)
        df = holefill.process(df)
        df = df.as_depth_frame() or depth_frame

        # Images
        color_bgr = np.asanyarray(color_frame.get_data())
        color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

        # MediaPipe
        res = hands.process(color_rgb)

        # Depth vis side window
        depth_raw = np.asanyarray(df.get_data())
        depth_vis = cv2.convertScaleAbs(depth_raw, alpha=0.03)

        if res.multi_hand_landmarks:
            for h_i, (lms, handed) in enumerate(zip(res.multi_hand_landmarks, res.multi_handedness or [])):
                label = handed.classification[0].label if handed.classification else "Hand"

                # Draw 2D landmarks
                mp_draw.draw_landmarks(
                    color_bgr, lms, mp_hands.HAND_CONNECTIONS,
                    mp_style.get_default_hand_landmarks_style(),
                    mp_style.get_default_hand_connections_style()
                )

                # 3D landmarks (meters, camera coords)
                xyz_list = []
                for j, lm in enumerate(lms.landmark):
                    u = lm.x * COLOR_W
                    v = lm.y * COLOR_H
                    z = depth_at_m(df, u, v, k=K_NEIGH)
                    if z <= 0 or not (DEPTH_RANGE_M[0] <= z <= DEPTH_RANGE_M[1]):
                        prev = ema_3d[h_i].get(j, None)
                        if prev is None:
                            xyz_list.append(None)
                            continue
                        xyz_list.append(prev.copy())
                        continue
                    xyz = deproject(intr, u, v, z)
                    prev = ema_3d[h_i].get(j, None)
                    smoothed = xyz if prev is None else (EMA_ALPHA * prev + (1.0 - EMA_ALPHA) * xyz)
                    ema_3d[h_i][j] = smoothed
                    xyz_list.append(smoothed)

                # Wrist annotation: print X,Y,Z in meters
                jz = int(SHOW_Z_AT)
                if 0 <= jz < len(xyz_list) and xyz_list[jz] is not None:
                    u_px = int(lms.landmark[jz].x * COLOR_W)
                    v_px = int(lms.landmark[jz].y * COLOR_H)
                    X, Y, Z = map(float, xyz_list[jz])  # meters, camera coords
                    
                    line1_y = v_px - 6
                    cv2.putText(
                        color_bgr, f"{label} X={X:.3f}m Y={Y:.3f}m Z={Z:.3f}m",
                        (u_px + 6, line1_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA
                    )
                    
                    # line 2: RPY in deg (blue)
                    rpy = imu.get_rpy_deg()
                    if rpy is not None:
                        roll, pitch, yaw = rpy
                        line2_y = line1_y + 18
                        cv2.putText(
                            color_bgr, f"RPY [deg] R={roll:.1f} P={pitch:.1f} Y={yaw:.1f}",
                            (u_px + 6, line2_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA
                        )
                    
                    

                # Draw IMU-driven arrow if we have a baseline and IMU data
                if has_baseline and xyz_list[jz] is not None:
                    rpy = imu.get_rpy_deg()
                    if rpy is not None:
                        # Current IMU rotation
                        Rc = rpy_deg_to_R(*rpy)      # current
                        Rdelta = Rc @ np.linalg.inv(R0)  # relative to baseline
                        fwd_cam = (Rdelta @ fwd0_cam.reshape(3,1)).reshape(3)
                        # Normalize
                        n = np.linalg.norm(fwd_cam)
                        if n > 1e-6:
                            fwd_cam = fwd_cam / n
                            origin = xyz_list[jz]
                            tip    = origin + ARROW_LEN_M * fwd_cam
                            u0, v0 = project_px(intr, origin)
                            u1, v1 = project_px(intr, tip)
                            cv2.arrowedLine(color_bgr, (u0, v0), (u1, v1), (0,0,255), 2, tipLength=0.25)
                            cv2.putText(color_bgr, "IMU forward", (u0+8, v0-8),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,255), 1, cv2.LINE_AA)

                # Handle key for baseline capture **inside** per-frame loop
                key = cv2.waitKey(1) & 0xFF
                if key == ord('d'):
                    # 1) Zero IMU at current pose
                    imu.zero_current_rpy()
                    # Read back (could be near 0/0/0 now); save R0 from raw (pre-offset) storage:
                    # Use get_rpy_deg() + offsets reset is ok because zero_current_rpy captured raw.
                    # Reconstruct R0 as identity to use pure deltas from now on:
                    R0 = rpy_deg_to_R(0.0, 0.0, 0.0)

                    # 2) Capture palm-forward baseline from landmarks
                    WRIST     = int(mp_hands.HandLandmark.WRIST)
                    INDEX_MCP = int(mp_hands.HandLandmark.INDEX_FINGER_MCP)
                    PINKY_MCP = int(mp_hands.HandLandmark.PINKY_MCP)

                    pw = xyz_list[WRIST]
                    pi = xyz_list[INDEX_MCP] if INDEX_MCP < len(xyz_list) else None
                    pp = xyz_list[PINKY_MCP] if PINKY_MCP < len(xyz_list) else None
                    if pw is not None and pi is not None and pp is not None:
                        across = pi - pp
                        na = np.linalg.norm(across)
                        if na > 1e-6:
                            across /= na
                            center = 0.5*(pi + pp)
                            to_fingers = center - pw
                            nf = np.linalg.norm(to_fingers)
                            if nf > 1e-6:
                                fwd0_cam = to_fingers / nf      # baseline forward in camera frame
                                has_baseline = True
                                print("Baseline set: IMU zeroed; palm-forward captured.")

                elif key == ord('q'):
                    raise KeyboardInterrupt

        # Show
        cv2.imshow("D435i - Color + MP Hands + IMU arrow", color_bgr)
        cv2.imshow("D435i - Depth (vis)", depth_vis)

except KeyboardInterrupt:
    pass
finally:
    pipe.stop()
    cv2.destroyAllWindows()
    hands.close()
    imu.close()
