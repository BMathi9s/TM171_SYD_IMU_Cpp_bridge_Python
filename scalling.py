#!/usr/bin/env python3
"""
Hand → RealSense depth → World → [-1,1] mapping with AprilTag board.
Hotkeys:
  C  : capture/lock camera→world using visible AprilTag(s) (avg ~30 frames)
  [ ]: yaw rotate world by -/+90° around +Z (to align +X with your arrow)
  X/Y/Z: flip that axis (if signs feel wrong)
  1..6: set x_min,x_max,y_min,y_max,z_min,z_max from current wrist pos
  S  : save calib+limits to desk_calib.yaml
  L  : load calib+limits from desk_calib.yaml
  Q  : quit
"""
import time, os, math, yaml
from collections import defaultdict, deque

import cv2
import numpy as np
import mediapipe as mp
import pyrealsense2 as rs

# ======== Tunables ========
COLOR_W, COLOR_H, FPS = 640, 480, 30
TAG_SIZE_M   = 0.072     # your printed tags are 80 mm
DEPTH_RANGE  = (0.25, 1.8)  # accept depth [m]
K_NEIGH      = 5        # median window for depth sampling
EMA_ALPHA    = 0.75     # 3D smoothing (0 none, 1 very sticky)
POSE_AVG_FR  = 30       # frames to average when capturing pose
CFG_FILE     = "desk_calib.yaml"
SHOW_LM      = mp.solutions.hands.HandLandmark.WRIST
# ==========================

# ---------- Helpers ----------
def depth_at_m(depth_frame, u, v, k=5):
    """Median depth (m) in k×k window around (u,v); ignores zeros/NaNs."""
    u = int(round(u)); v = int(round(v))
    w = depth_frame.get_width(); h = depth_frame.get_height()
    r = k // 2
    vals = []
    for yy in range(max(0, v - r), min(h, v + r + 1)):
        for xx in range(max(0, u - r), min(w, u + r + 1)):
            d = depth_frame.get_distance(xx, yy)  # meters
            if d and np.isfinite(d):
                vals.append(d)
    if not vals: return 0.0
    return float(np.median(vals))

def rs_intrinsics_to_cvK_D(intr):
    K = np.array([[intr.fx, 0, intr.ppx],
                  [0, intr.fy, intr.ppy],
                  [0,       0,       1]], dtype=np.float32)
    # RealSense typically Brown–Conrady (k1,k2,p1,p2,k3)
    D = np.array(intr.coeffs[:5], dtype=np.float32)
    return K, D

def rodrigues_to_R(rvec):
    R, _ = cv2.Rodrigues(rvec.astype(np.float64))
    return R.astype(np.float32)

def invert_rt(R, t):
    Rt = R.T
    tt = -Rt @ t
    return Rt, tt

def yaw_deg_R(deg):
    th = math.radians(deg)
    c, s = math.cos(th), math.sin(th)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float32)

def normalize_unit(p, pmin, pmax):
    p = np.asarray(p, np.float32)
    pmin = np.asarray(pmin, np.float32)
    pmax = np.asarray(pmax, np.float32)
    span = np.maximum(pmax - pmin, 1e-6)
    pn = 2.0*(p - pmin)/span - 1.0
    return np.clip(pn, -1.0, 1.0)

def draw_bar(img, x, y, w, h, val, label):
    # val in [-1,1]
    cv2.rectangle(img, (x,y), (x+w, y+h), (80,80,80), 1)
    cx = int(x + (val+1)*0.5*w)
    cv2.line(img, (cx,y), (cx,y+h), (0,255,0), 2)
    cv2.putText(img, f"{label}:{val:+.2f}", (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

# ---------- RealSense ----------
pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, FPS)
cfg.enable_stream(rs.stream.color, COLOR_W, COLOR_H, rs.format.bgr8, FPS)
profile = pipe.start(cfg)

align = rs.align(rs.stream.color)

# Enable projector if available
try:
    depth_sensor = profile.get_device().first_depth_sensor()
    if depth_sensor.supports(rs.option.emitter_enabled):
        depth_sensor.set_option(rs.option.emitter_enabled, 1)
    if depth_sensor.supports(rs.option.laser_power):
        rng = depth_sensor.get_option_range(rs.option.laser_power)
        depth_sensor.set_option(rs.option.laser_power, min(rng.max, 180))
except Exception:
    pass

# Basic depth filters
spatial  = rs.spatial_filter()
temporal = rs.temporal_filter()
holefill = rs.hole_filling_filter()

color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
intr = color_stream.get_intrinsics()
Kcv, Dcv = rs_intrinsics_to_cvK_D(intr)

# ---------- AprilTag (OpenCV ArUco) ----------
# Use the AprilTag 36h11 dictionary in OpenCV
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
aruco_params = cv2.aruco.DetectorParameters()
aruco = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

def detect_tag_pose(bgr):
    corners, ids, _ = aruco.detectMarkers(bgr)
    if ids is None or len(ids)==0:
        return None
    # Choose the largest marker in view (most stable)
    areas = [cv2.contourArea(c.astype(np.float32)) for c in corners]
    k = int(np.argmax(areas))
    rvecs, tvecs, _obj = cv2.aruco.estimatePoseSingleMarkers([corners[k]], TAG_SIZE_M, Kcv, Dcv)
    rvec = rvecs[0].reshape(3,1).astype(np.float32)
    tvec = tvecs[0].reshape(3,1).astype(np.float32)
    return corners[k], int(ids[k][0]), rvec, tvec

# ---------- MediaPipe Hands ----------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.6, min_tracking_confidence=0.6,
                       model_complexity=1)
mp_draw  = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles

ema_3d = defaultdict(dict)

# ---------- World/Calib state ----------
have_pose = False
R_world_cam = np.eye(3, dtype=np.float32)
t_world_cam = np.zeros((3,1), dtype=np.float32)
yaw_steps = 0
axis_flip = np.array([1.0, 1.0, 1.0], dtype=np.float32)  # flips for x,y,z

# limits
lims = {
    "x_min": None, "x_max": None,
    "y_min": None, "y_max": None,
    "z_min": None, "z_max": None,
}

def save_cfg(path):
    data = {
        "R_world_cam": R_world_cam.tolist(),
        "t_world_cam": t_world_cam.flatten().tolist(),
        "yaw_steps": int(yaw_steps),
        "axis_flip": axis_flip.tolist(),
        "limits": {k:(None if v is None else float(v)) for k,v in lims.items()},
    }
    with open(path, "w") as f:
        yaml.safe_dump(data, f)
    print(f"[saved] {path}")

def load_cfg(path):
    global R_world_cam, t_world_cam, yaw_steps, axis_flip, lims, have_pose
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    R_world_cam = np.array(data["R_world_cam"], dtype=np.float32)
    t_world_cam = np.array(data["t_world_cam"], dtype=np.float32).reshape(3,1)
    yaw_steps   = int(data.get("yaw_steps", 0))
    axis_flip   = np.array(data.get("axis_flip", [1,1,1]), dtype=np.float32)
    lims = {k: (None if v is None else float(v)) for k,v in data.get("limits", lims).items()}
    have_pose = True
    print(f"[loaded] {path}")
    
    


# ----------- Main loop -----------
pose_stack = deque(maxlen=POSE_AVG_FR)

try:
    while True:
        frames = pipe.wait_for_frames()
        frames = align.process(frames)

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # depth filters
        df = depth_frame
        df = spatial.process(df)
        df = temporal.process(df)
        df = holefill.process(df)
        df = df.as_depth_frame() or depth_frame

        color_bgr = np.asanyarray(color_frame.get_data())
        color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

        # ---- AprilTag pose (for viz and capture) ----
        det = detect_tag_pose(color_bgr)
        if det is not None:
            corners, tag_id, rvec, tvec = det
            # draw marker
            cv2.aruco.drawDetectedMarkers(color_bgr, [corners], np.array([[tag_id]], dtype=np.int32))
            # draw axis of marker (in camera view)
            cv2.drawFrameAxes(color_bgr, Kcv, Dcv, rvec, tvec, TAG_SIZE_M*0.5)

        # ---- Hands ----
        res = hands.process(color_rgb)
        depth_raw = np.asanyarray(df.get_data())
        depth_vis = cv2.convertScaleAbs(depth_raw, alpha=0.03)

        # ---- If capturing pose (press C), average several frames ----
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('c'), ord('C')):
            pose_stack.clear()
            print("[pose] capturing… keep the tag board steady")
            # grab multiple frames quickly
            for _ in range(POSE_AVG_FR):
                frames2 = pipe.wait_for_frames()
                frames2 = align.process(frames2)
                color2 = np.asanyarray(frames2.get_color_frame().get_data())
                det2 = detect_tag_pose(color2)
                if det2 is None: continue
                _, _, rvec2, tvec2 = det2
                R_cm = rodrigues_to_R(rvec2)   # marker→camera
                t_cm = tvec2                   # marker→camera
                # camera→marker
                R_mc, t_mc = invert_rt(R_cm, t_cm)
                pose_stack.append( (R_mc, t_mc) )
                time.sleep(0.005)
            if len(pose_stack) >= 5:
                # average rotation via quaternion; translation via mean
                Rs = np.stack([p[0] for p in pose_stack], axis=0)
                ts = np.stack([p[1].reshape(3) for p in pose_stack], axis=0)
                # simple chordal mean for R
                M = np.zeros((3,3), dtype=np.float64)
                for R in Rs:
                    M += R.astype(np.float64).T @ R.astype(np.float64)
                # eigenvector with largest eigenvalue
                _, vecs = np.linalg.eigh(M)
                # rebuild mean R as average of rotations (fallback to last if numerical issue)
                R_mean = Rs[-1]
                t_mean = ts.mean(axis=0).reshape(3,1).astype(np.float32)

                # base: world = marker; then apply yaw & flips
                R_adj = yaw_deg_R(90*yaw_steps) @ np.diag(axis_flip)
                R_world_cam = (R_adj @ R_mean).astype(np.float32)
                t_world_cam = (R_adj @ t_mean).astype(np.float32)
                have_pose = True
                print("[pose] locked. use [ or ] to rotate by 90°, X/Y/Z to flip axes.  !!! bug to fix its a bit tricky to to the flip coz it also change the yaw, so play with both")
            else:
                print("[pose] failed (not enough frames). Make sure tags are visible.")

        # rotate/flips
        if key == ord('['): yaw_steps = (yaw_steps - 1) % 4
        if key == ord(']'): yaw_steps = (yaw_steps + 1) % 4
        if key in (ord('x'),ord('X')): axis_flip[0] *= -1
        if key in (ord('y'),ord('Y')): axis_flip[1] *= -1
        if key in (ord('z'),ord('Z')): axis_flip[2] *= -1
        if key in (ord('s'),ord('S')): save_cfg(CFG_FILE)
        if key in (ord('l'),ord('L')) and os.path.exists(CFG_FILE): load_cfg(CFG_FILE)
        if key in (ord('q'),ord('Q')): break

        # recompute adjusted R (for viz) every frame in case yaw/flip changed
        if have_pose:
            R_adj = yaw_deg_R(90*yaw_steps) @ np.diag(axis_flip)
            Rwc = (R_adj @ R_world_cam).astype(np.float32)
            twc = (R_adj @ t_world_cam).astype(np.float32)

        # ---- Compute 3D for wrist, transform to world, normalize ----
        nvec = (0.0, 0.0, 0.0)
        world_xyz = None

        if res.multi_hand_landmarks:
            lms = res.multi_hand_landmarks[0]
            mp.solutions.drawing_utils.draw_landmarks(
                color_bgr, lms, mp_hands.HAND_CONNECTIONS,
                mp_style.get_default_hand_landmarks_style(),
                mp_style.get_default_hand_connections_style()
            )
            # choose landmark (wrist)
            lm = lms.landmark[int(SHOW_LM)]
            u = lm.x * COLOR_W
            v = lm.y * COLOR_H
            z = depth_at_m(df, u, v, k=K_NEIGH)
            if z>0 and DEPTH_RANGE[0] <= z <= DEPTH_RANGE[1]:
                # deproject using RealSense intrinsics
                p_cam = np.array(rs.rs2_deproject_pixel_to_point(intr, [float(u), float(v)], float(z)), dtype=np.float32).reshape(3,1)
                # EMA smoothing
                prev = ema_3d[0].get('wrist')
                if prev is None:
                    p_cam_s = p_cam
                else:
                    p_cam_s = EMA_ALPHA*prev + (1.0-EMA_ALPHA)*p_cam
                ema_3d[0]['wrist'] = p_cam_s

                if have_pose:
                    p_world = (Rwc @ p_cam_s + twc).reshape(3)
                    world_xyz = p_world.copy()

                    # set limits via hotkeys
                    if key == ord('1'): lims["x_min"] = p_world[0]
                    if key == ord('2'): lims["x_max"] = p_world[0]
                    if key == ord('3'): lims["y_min"] = p_world[1]
                    if key == ord('4'): lims["y_max"] = p_world[1]
                    if key == ord('5'): lims["z_min"] = p_world[2]
                    if key == ord('6'): lims["z_max"] = p_world[2]

                    # normalize if we have all limits
                    if all(v is not None for v in lims.values()):
                        pmin = [lims["x_min"], lims["y_min"], lims["z_min"]]
                        pmax = [lims["x_max"], lims["y_max"], lims["z_max"]]
                        n = normalize_unit(p_world, pmin, pmax)
                        nvec = (float(n[0]), float(n[1]), float(n[2]))

                # annotate on image
                cv2.circle(color_bgr, (int(u), int(v)), 6, (0,255,0), 2)
                cv2.putText(color_bgr, f"z={z:.3f} m", (int(u)+6, int(v)-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        # ---- HUD ----
        h, w = color_bgr.shape[:2]
        # bars
        draw_bar(color_bgr, 10, h-30, 180, 18, nvec[0], "nx (+fwd)")
        draw_bar(color_bgr, 200, h-30, 180, 18, nvec[1], "ny (+left)")
        draw_bar(color_bgr, 390, h-30, 180, 18, nvec[2], "nz (+up)")

        # text
        
        
        ypos = [20]
        def put(s):
            cv2.putText(color_bgr, s, (10, ypos[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            ypos[0] += 18

        put(f"Pose: {'LOCKED' if have_pose else '—'}  yaw_steps={yaw_steps}  flip={axis_flip.astype(int).tolist()}")
        # ... same calls as before ...
        # show
        cv2.imshow("Color + Hands + World HUD", color_bgr)
        cv2.imshow("Depth (vis)", depth_vis)

finally:
    pipe.stop()
    cv2.destroyAllWindows()
    hands.close()
