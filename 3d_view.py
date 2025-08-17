import time
from collections import defaultdict

import cv2
import numpy as np
import pyrealsense2 as rs
import mediapipe as mp

# ========= Tunables =========
COLOR_W, COLOR_H, FPS = 640, 480, 30
DEPTH_RANGE_M = (0.25, 1.2)     # accept landmarks only if depth is within [min,max]
EMA_ALPHA     = 0.8            # 3D smoothing (0=no smooth, 1=very sticky)
K_NEIGH       = 3               # odd >=1; median over k×k neighborhood for depth sampling
SHOW_Z_AT     = mp.solutions.hands.HandLandmark.WRIST  # which landmark to annotate with Z
# ============================

def depth_at_m(depth_frame, u, v, k=3):
    """Median depth (m) in k×k window around (u,v); ignores zeros/NaNs."""
    u = int(round(u)); v = int(round(v))
    w = depth_frame.get_width(); h = depth_frame.get_height()
    r = k // 2
    vals = []
    for yy in range(max(0, v - r), min(h, v + r + 1)):
        for xx in range(max(0, u - r), min(w, u + r + 1)):
            d = depth_frame.get_distance(xx, yy)  # already meters
            if d and np.isfinite(d):
                vals.append(d)
    if not vals:
        return 0.0
    return float(np.median(vals))

def deproject(intr, u, v, z_m):
    pt = rs.rs2_deproject_pixel_to_point(intr, [float(u), float(v)], float(z_m))
    return np.array(pt, dtype=np.float32)  # [X,Y,Z] meters, camera coords

# ---- RealSense setup ----
pipe = rs.pipeline()
cfg  = rs.config()
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, FPS)
cfg.enable_stream(rs.stream.color, COLOR_W, COLOR_H, rs.format.bgr8, FPS)
profile = pipe.start(cfg)

# Align depth → color coordinates
align = rs.align(rs.stream.color)

# Optional: enable IR projector (helps on low-texture scenes)
try:
    depth_sensor = profile.get_device().first_depth_sensor()
    if depth_sensor.supports(rs.option.emitter_enabled):
        depth_sensor.set_option(rs.option.emitter_enabled, 1)
    if depth_sensor.supports(rs.option.laser_power):
        # Typical range ~0..360; increase if your scene is dim/low-texture
        depth_sensor.set_option(rs.option.laser_power, min(depth_sensor.get_option_range(rs.option.laser_power).max, 150))
except Exception:
    pass

# Depth filters (reduce speckle/holes)
spatial  = rs.spatial_filter()      # edge-preserving blur
temporal = rs.temporal_filter()     # temporal smoothing
holefill = rs.hole_filling_filter() # fill small gaps

# Intrinsics for deprojection (color-aligned depth uses color intrinsics)
color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
intr = color_stream.get_intrinsics()

# ---- MediaPipe Hands (Solutions API for easy setup) ----
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

# Smoothing state: per-hand (0/1), per-landmark (0..20) → 3D EMA
ema_3d = defaultdict(dict)

t0 = time.time()
frame_idx = 0

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

        # Numpy images for display / MP
        color_bgr = np.asanyarray(color_frame.get_data())
        color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

        # Run MediaPipe
        res = hands.process(color_rgb)

        # Depth visualization (8-bit) for a side window
        depth_raw = np.asanyarray(df.get_data())
        depth_vis = cv2.convertScaleAbs(depth_raw, alpha=0.03)

        if res.multi_hand_landmarks:
            for h_i, (lms, handed) in enumerate(zip(res.multi_hand_landmarks, res.multi_handedness or [])):
                label = handed.classification[0].label if handed.classification else "Hand"

                # Draw 2D
                mp_draw.draw_landmarks(
                    color_bgr, lms, mp_hands.HAND_CONNECTIONS,
                    mp_style.get_default_hand_landmarks_style(),
                    mp_style.get_default_hand_connections_style()
                )

                # Compute true 3D
                xyz_list = []
                for j, lm in enumerate(lms.landmark):
                    u = lm.x * COLOR_W
                    v = lm.y * COLOR_H
                    z = depth_at_m(df, u, v, k=K_NEIGH)  # meters

                    # Range-gate (reject implausible/noisy points)
                    if z <= 0 or not (DEPTH_RANGE_M[0] <= z <= DEPTH_RANGE_M[1]):
                        # fall back to previous EMA if available; else skip
                        prev = ema_3d[h_i].get(j, None)
                        if prev is None:
                            xyz_list.append(None)
                            continue
                        xyz_list.append(prev.copy())
                        continue

                    xyz = deproject(intr, u, v, z)

                    # EMA smoothing
                    prev = ema_3d[h_i].get(j, None)
                    if prev is None:
                        smoothed = xyz
                    else:
                        smoothed = EMA_ALPHA * prev + (1.0 - EMA_ALPHA) * xyz
                    ema_3d[h_i][j] = smoothed
                    xyz_list.append(smoothed)

                # Annotate one landmark’s Z (e.g., WRIST)
                jz = int(SHOW_Z_AT)
                if 0 <= jz < len(xyz_list) and xyz_list[jz] is not None:
                    u = int(lms.landmark[jz].x * COLOR_W)
                    v = int(lms.landmark[jz].y * COLOR_H)
                    z_m = float(xyz_list[jz][2])
                    cv2.putText(color_bgr, f"{label} Z={z_m:.3f} m", (u + 6, v - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

        # Show
        cv2.imshow("D435i - Color + MP Hands", color_bgr)
        cv2.imshow("D435i - Depth (vis)", depth_vis)

        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_idx += 1

finally:
    pipe.stop()
    cv2.destroyAllWindows()
    hands.close()
