# desk_hand_tracker.py
import os, time, math, yaml
from collections import defaultdict, deque

import cv2
import numpy as np
import mediapipe as mp
import pyrealsense2 as rs
import re  


import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


#TODO when clicking on c is doesnt lock the camera pose when running to the world space anymore, gotta fix that, but its not a big deal
#TODO improve fingger calibration


class hand_position_tracker:
    """
    RealSense color+depth + MediaPipe wrist → camera 3D → world 3D (from AprilTag board) → [-1,1].
    This version matches your current behavior:
      - Capture ('C') bakes yaw/flip into the stored pose (as in your script).
      - Each frame, the same yaw/flip are applied again before transforming the point (your current flow).
    If you later want the mathematically "clean" single-application, we can toggle a flag.

    Public API (most relevant):
      start(), stop(), process()               # run the stream; call process() every frame
      capture_pose()                           # capture camera→world from AprilTag board
      set_yaw_steps(n), add_yaw_step(±1)       # rotate world by 90° steps
      flip_axis('x'|'y'|'z')                   # toggle sign on an axis
      set_limit(axis, 'min'|'max', value)      # set limits numerically
      set_limit_from_current(axis, 'min'|'max')# capture limit at current wrist location
      save_calibration(path), load_calibration(path)

      get_world_xyz()      -> (x,y,z) or None
      get_normalized_xyz() -> (nx,ny,nz) in [-1,1]
      get_limits()         -> dict with x_min/x_max/...
      have_pose()          -> bool

    Optional HUD helpers:
      draw_hud(color_bgr, world_xyz, nvec)     # bars + text overlay

    Hotkeys (example usage script shows how to bind):
      C capture pose; [ ] rotate; X/Y/Z flip; 1..6 set limits; S save; L load; Q quit
    """

    # ---- Defaults / Tunables ----
    DEFAULT_W = 848
    DEFAULT_H = 480
    DEFAULT_FPS = 60

    def __init__(
        self,
        color_w=DEFAULT_W,
        color_h=DEFAULT_H,
        fps=DEFAULT_FPS,
        tag_size_m=0.072,                 # you printed 80 mm → 0.08 m
        depth_range=(0.25, 2.2),
        k_neigh=5,                       # median patch for depth sampling
        ema_alpha=0.75,                  # 3D smoothing
        pose_avg_frames=100,              # frames to average on capture
        cfg_file= None,
        show_landmark=mp.solutions.hands.HandLandmark.WRIST,
        bake_adjustments_at_capture=True # keep "as-is" behavior
    ):
        # Config
        self.COLOR_W = color_w
        self.COLOR_H = color_h
        self.FPS = fps
        self.TAG_SIZE_M = float(tag_size_m)
        self.DEPTH_RANGE = depth_range
        self.K_NEIGH = int(k_neigh)
        self.EMA_ALPHA = float(ema_alpha)
        self.POSE_AVG_FR = int(pose_avg_frames)
        
        
        self.PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.CALIB_DIR = os.path.join(self.PROJ_ROOT, "calibration", "world_calibration")
        os.makedirs(self.CALIB_DIR, exist_ok=True)

        # Default config path: <project>/calibration/desk_calib.yaml
        if cfg_file is None:
            cfg_file = os.path.join(self.CALIB_DIR, "desk_calib.yaml")
        self.CFG_FILE = cfg_file
        
                # ---- Hand-flexion calibration (separate from XYZ/pose) ----
        # Folder: TransducerM_Lib_Protocol_CPP\calibration\hand_calibration
        self.HAND_CALIB_DIR = os.path.join(self.PROJ_ROOT, "calibration", "hand_calibration")
        os.makedirs(self.HAND_CALIB_DIR, exist_ok=True)

        # Open/Close angle snapshots (degrees) captured from self._angles
        self._hand_open  = None   # same nested dict shape as self._angles
        self._hand_close = None

        # Min usable range (deg). If |close - open| < threshold ⇒ normalized = 0 for safety.
        self.HAND_MIN_RANGE_DEG = 5.0

        
        
        
        self.SHOW_LM = show_landmark
        self.BAKE_AT_CAPTURE = bool(bake_adjustments_at_capture)

        # RS state
        self.pipe = None
        self.align = None
        self.depth_filters = {}
        self.intr = None
        self.Kcv = None
        self.Dcv = None

        # AprilTag detector (OpenCV's AprilTag 36h11)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        # MediaPipe
        base_opts = mp_python.BaseOptions(
            model_asset_path="hand_landmarker.task",  # put the .task file next to your script
            delegate=mp_python.BaseOptions.Delegate.GPU
        )
        hl_opts = mp_vision.HandLandmarkerOptions(
            base_options=base_opts,
            num_hands=1,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6,
            # running_mode omitted → defaults to IMAGE
        )
        self.hands = mp_vision.HandLandmarker.create_from_options(hl_opts)

        # World/Calib state
        self._have_pose = False
        self.R_world_cam = np.eye(3, dtype=np.float32)
        self.t_world_cam = np.zeros((3,1), dtype=np.float32)
        self.yaw_steps = 0
        self.axis_flip = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        self.lims = {
            "x_min": None, "x_max": None,
            "y_min": None, "y_max": None,
            "z_min": None, "z_max": None,
        }
        
        

        # Runtime caches
        self._ema_3d = defaultdict(dict)
        self._pose_stack = deque(maxlen=self.POSE_AVG_FR)
        self._last_world_xyz = None
        self._last_norm_xyz = (0.0, 0.0, 0.0)
        
        
        #finger joint pose
        
        # --- Angles (degrees): most recent frame, flexion-only ---
        # Fingers: { 'MCP': float, 'PIP': float, 'DIP': float }
        # Thumb:   { 'CMC': float, 'MCP': float, 'IP':  float }
        self._angles = {
            'index':  {'MCP': 0.0, 'PIP': 0.0, 'DIP': 0.0},
            'middle': {'MCP': 0.0, 'PIP': 0.0, 'DIP': 0.0},
            'ring':   {'MCP': 0.0, 'PIP': 0.0, 'DIP': 0.0},
            'pinky':  {'MCP': 0.0, 'PIP': 0.0, 'DIP': 0.0},
            'thumb':  {'CMC': 0.0, 'MCP': 0.0, 'IP':  0.0},
        }
        # Per-landmark EMA (camera-frame 3D points), keyed by lm idx
        self._ema_pts = {}
        self.ANGLE_EMA_ALPHA = 0.7      # angle smoothing (EMA in angle-space)
        self.PT_EMA_ALPHA    = 0.75     # 3D point smoothing (already similar for wrist)
        self.SHOW_FLEX_PANEL = True
        self.FLEX_PANEL_SCALE = 1.0  # 1.0 = default bar size; set 0.8 for smaller, 1.2 for larger

        


    # ---------- Vector helpers for joint angles ----------
    @staticmethod
    def _unit(v):
        v = np.asarray(v, np.float32).reshape(-1)
        n = np.linalg.norm(v)
        return v / max(n, 1e-9)

    @staticmethod
    def _project_to_plane(v, n_hat):
        # remove component along plane normal n_hat
        v = np.asarray(v, np.float32).reshape(-1)
        n_hat = hand_position_tracker._unit(n_hat)
        return v - np.dot(v, n_hat) * n_hat

    @staticmethod
    def _angle_between(v1, v2):
        # robust unsigned angle in radians
        v1 = hand_position_tracker._unit(v1)
        v2 = hand_position_tracker._unit(v2)
        c = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
        s = float(np.linalg.norm(np.cross(v1, v2)))
        return math.atan2(s, c)
    
    def _draw_mini_bar(self, img, x, y, w, h, val, label, thickness=1):
        # Frame
        cv2.rectangle(img, (x, y), (x + w, y + h), (160, 160, 160), thickness)
        # Cursor at [-1,1] mapped across width
        cx = int(x + int((val + 1.0) * 0.5 * w))
        cv2.line(img, (cx, y), (cx, y + h), (0, 200, 255), max(1, thickness))
        # Label
        cv2.putText(img, f"{label}:{val:+.2f}", (x, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)

    def _draw_flexion_panel(self, img, corner="br"):
        """
        15 normalized finger flexion bars:
        Thumb (CMC/MCP/IP) + Index/Middle/Ring/Pinky (MCP/PIP/DIP)
        Values are in [-1, 1]. Set self.FLEX_PANEL_SCALE to resize.
        """
        n = self.get_all_flexions_normalized()

        # Row spec: (short_label, values_dict, joint_order_list)
        rows = [
            ("Thb", n["thumb"],  ["CMC", "MCP", "IP"]),
            ("Idx", n["index"],  ["MCP", "PIP", "DIP"]),
            ("Mid", n["middle"], ["MCP", "PIP", "DIP"]),
            ("Rng", n["ring"],   ["MCP", "PIP", "DIP"]),
            ("Pky", n["pinky"],  ["MCP", "PIP", "DIP"]),
        ]

        h_img, w_img = img.shape[:2]
        s = float(getattr(self, "FLEX_PANEL_SCALE", 1.0))
        bar_w, bar_h = int(90 * s), int(12 * s)   # size per bar
        pad_x, pad_y = int(10 * s), int(14 * s)   # spacing between bars
        block_pad    = int(12 * s)                # panel inset from edges

        cols = 3  # three joints per row
        panel_w = cols * bar_w + (cols - 1) * pad_x
        panel_h = len(rows) * (bar_h + pad_y)

        # Anchor
        if corner == "br":
            x0 = w_img - panel_w - block_pad
            y0 = h_img - panel_h - block_pad
        elif corner == "tr":
            x0 = w_img - panel_w - block_pad
            y0 = block_pad + 40
        elif corner == "bl":
            x0 = block_pad
            y0 = h_img - panel_h - block_pad
        else:  # 'tl'
            x0 = block_pad
            y0 = block_pad + 40

        # Background
        overlay = img.copy()
        cv2.rectangle(overlay, (x0 - 6, y0 - 22), (x0 + panel_w + 6, y0 + panel_h + 6), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)

        # Title
        cv2.putText(img, "Finger Flex (norm)", (x0, y0 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Draw rows
        for r, (short, vals, joint_order) in enumerate(rows):
            y = y0 + r * (bar_h + pad_y)
            # Row label
            cv2.putText(img, short, (x0 - 40, y + bar_h),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            # Bars: iterate over joint order
            for c, j in enumerate(joint_order):
                x = x0 + c * (bar_w + pad_x)
                val = float(vals.get(j, 0.0))
                self._draw_mini_bar(img, x, y, bar_w, bar_h, val, j, thickness=1)


    @staticmethod
    def _ema(prev, new, alpha):
        return (alpha * prev + (1.0 - alpha) * new) if prev is not None else new

    # ======== Config-name helpers ========
    @staticmethod
    def _valid_basename(name: str) -> bool:
        # letters, digits, underscore, dash only
        return bool(re.fullmatch(r"[A-Za-z0-9_-]+", name))
    
    def _landmarks_cam3d(self, df, lms):
        """
        Returns np.array shape (21, 3) of camera-frame 3D points (meters),
        or None if too many depths are missing.
        """
        pts = np.zeros((21, 3), np.float32)
        ok_count = 0
        for i, lm in enumerate(lms.landmark):
            u = float(lm.x) * self.COLOR_W
            v = float(lm.y) * self.COLOR_H
            z = self._depth_at_m(df, u, v, k=self.K_NEIGH)
            if not (z > 0 and self.DEPTH_RANGE[0] <= z <= self.DEPTH_RANGE[1]):
                pts[i, :] = np.nan
                continue
            p = np.array(rs.rs2_deproject_pixel_to_point(self.intr, [u, v], z), dtype=np.float32).reshape(3)
            # EMA smooth each landmark in 3D
            prev = self._ema_pts.get(i)
            p_s  = self._ema(prev, p, self.PT_EMA_ALPHA)
            self._ema_pts[i] = p_s
            pts[i, :] = p_s
            ok_count += 1
        if ok_count < 12:  # too few
            return None
        return pts



    def _compute_flexions_deg(self, P):
        """
        P: (21,3) camera-frame 3D landmarks (meters).
        Updates self._angles (degrees) with flexion-only angles.
        """
        # Indices as in MediaPipe
        W  = 0
        # Thumb: 1,2,3,4; Index: 5,6,7,8; Middle: 9,10,11,12; Ring: 13,14,15,16; Pinky: 17,18,19,20

        # Basic sanity: if any critical point is nan, skip
        def ok(i): return np.isfinite(P[i]).all()

        # Palm normal: use wrist(0), index_mcp(5), pinky_mcp(17)
        if ok(0) and ok(5) and ok(17):
            v1 = P[5] - P[0]
            v2 = P[17] - P[0]
            n_palm = self._unit(np.cross(v2, v1))  # outward normal (pick a consistent winding)
        else:
            n_palm = np.array([0, 0, 1], np.float32)

        def ema_angle(key1, key2, new_deg):
            prev = self._angles[key1][key2]
            # EMA in angle space (in degrees). Keep it simple since ranges are small.
            self._angles[key1][key2] = float(self._ema(prev, new_deg, self.ANGLE_EMA_ALPHA))

        def joint_angle_flexion(parent_vec, child_vec, ray_for_plane=None):
            """
            Compute flexion between parent_vec and child_vec after projecting both
            into a 'sagittal' plane. If ray_for_plane is provided, build the plane
            from the palm normal and this finger ray; otherwise project using palm normal only.
            Returns degrees in [0, 180].
            """
            if ray_for_plane is not None and np.linalg.norm(ray_for_plane) > 0:
                # sagittal plane normal ~ cross(finger_ray, palm_normal)
                n_plane = self._unit(np.cross(self._unit(ray_for_plane), n_palm))
                v1p = self._project_to_plane(parent_vec, n_plane)
                v2p = self._project_to_plane(child_vec,  n_plane)
            else:
                # fallback: just project out of palm normal
                v1p = self._project_to_plane(parent_vec, n_palm)
                v2p = self._project_to_plane(child_vec,  n_palm)
            rad = self._angle_between(v1p, v2p)
            return math.degrees(rad)

        # ---- Four fingers: MCP/PIP/DIP flexion ----
        fingers = {
            'index':  (5,6,7,8),
            'middle': (9,10,11,12),
            'ring':   (13,14,15,16),
            'pinky':  (17,18,19,20),
        }
        for name, (mcp, pip, dip, tip) in fingers.items():
            if not (ok(W) and ok(mcp) and ok(pip) and ok(dip) and ok(tip)):
                continue
            # Bones (child - joint)
            v_meta = P[mcp] - P[W]        # approximate metacarpal
            v_prox = P[pip] - P[mcp]
            v_mid  = P[dip] - P[pip]
            v_dist = P[tip] - P[dip]
            # Finger ray for sagittal plane
            ray = v_prox
            # Flexions
            a_mcp = joint_angle_flexion(v_meta, v_prox, ray_for_plane=ray)
            a_pip = joint_angle_flexion(v_prox, v_mid,  ray_for_plane=ray)
            a_dip = joint_angle_flexion(v_mid,  v_dist, ray_for_plane=ray)
            ema_angle(name, 'MCP', a_mcp)
            ema_angle(name, 'PIP', a_pip)
            ema_angle(name, 'DIP', a_dip)

        # ---- Thumb: CMC/MCP/IP flexion ----
        # Landmarks: CMC=1, MCP=2, IP=3, TIP=4
        if ok(1) and ok(2) and ok(3) and ok(4) and ok(W):
            v_meta_t = P[1] - P[W]    # wrist→CMC as metacarpal proxy
            v_prox_t = P[2] - P[1]    # CMC→MCP
            v_mid_t  = P[3] - P[2]    # MCP→IP
            v_dist_t = P[4] - P[3]    # IP→TIP
            ray_t    = v_prox_t

            a_cmc = joint_angle_flexion(v_meta_t, v_prox_t, ray_for_plane=ray_t)
            a_mcp = joint_angle_flexion(v_prox_t, v_mid_t,  ray_for_plane=ray_t)
            a_ip  = joint_angle_flexion(v_mid_t,  v_dist_t, ray_for_plane=ray_t)
            ema_angle('thumb', 'CMC', a_cmc)
            ema_angle('thumb', 'MCP', a_mcp)
            ema_angle('thumb', 'IP',  a_ip)
            
    # ---------- Flexion getters (degrees) ----------
    def get_flexion_index(self):
        """Return {'MCP':deg, 'PIP':deg, 'DIP':deg} (degrees)."""
        return dict(self._angles['index'])

    def get_flexion_middle(self):
        return dict(self._angles['middle'])

    def get_flexion_ring(self):
        return dict(self._angles['ring'])

    def get_flexion_pinky(self):
        return dict(self._angles['pinky'])

    def get_flexion_thumb(self):
        """Return {'CMC':deg, 'MCP':deg, 'IP':deg} (degrees)."""
        return dict(self._angles['thumb'])

    def get_all_flexions(self):
        """Convenience snapshot of all fingers (degrees)."""
        return {k: dict(v) for k, v in self._angles.items()}
    
    
    
    # ---------- Normalized flexion getters [-1,1] ----------
    def get_normalized_flexion_index(self):
        return dict(self._normalize_all_fingers(
            {"index": self._angles["index"]},
            self._hand_open, self._hand_close
        )["index"])

    def get_normalized_flexion_middle(self):
        return dict(self._normalize_all_fingers(
            {"middle": self._angles["middle"]},
            self._hand_open, self._hand_close
        )["middle"])

    def get_normalized_flexion_ring(self):
        return dict(self._normalize_all_fingers(
            {"ring": self._angles["ring"]},
            self._hand_open, self._hand_close
        )["ring"])

    def get_normalized_flexion_pinky(self):
        return dict(self._normalize_all_fingers(
            {"pinky": self._angles["pinky"]},
            self._hand_open, self._hand_close
        )["pinky"])

    def get_normalized_flexion_thumb(self):
        return dict(self._normalize_all_fingers(
            {"thumb": self._angles["thumb"]},
            self._hand_open, self._hand_close
        )["thumb"])

    def get_all_flexions_normalized(self):
        """Full snapshot of normalized flexions (degrees→[-1,1]) for all fingers."""
        return self._normalize_all_fingers(self._angles, self._hand_open, self._hand_close)

    
        # ---------- Hand-flexion calibration API ----------
    def record_hand_open(self):
        """Capture current flexion angles (deg) as the 'open hand' calibration."""
        self._hand_open = self._angles_snapshot()
        return {k: dict(v) for k, v in self._hand_open.items()}

    def record_hand_close(self):
        """Capture current flexion angles (deg) as the 'closed hand' calibration."""
        self._hand_close = self._angles_snapshot()
        return {k: dict(v) for k, v in self._hand_close.items()}

    def save_hand_calibration(self, path: str | None = None):
        """
        Save open/close angle sets to YAML under .../calibration/hand_calibration/.
        Prompts for a name if not provided. Does NOT touch XYZ/pose calibration.
        """
        if self._hand_open is None or self._hand_close is None:
            raise RuntimeError("Hand calibration incomplete. Call record_hand_open() and record_hand_close() first.")
        if path is None:
            base = self._prompt_config_basename("save hand calibration")
            if base is None:
                print("[hand] Save cancelled.")
                return None
            path = self._resolve_hand_cfg_path(base)
        else:
            path = self._resolve_hand_cfg_path(path)

        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "type": "hand_flexion_calibration",
            "min_range_deg": float(self.HAND_MIN_RANGE_DEG),
            "open":  self._hand_open,
            "close": self._hand_close,
        }
        with open(path, "w") as f:
            yaml.safe_dump(data, f)
        print(f"[hand] Saved hand calibration: {path}")
        return path

    def load_hand_calibration(self, path: str | None = None):
        """
        Load open/close angle sets from YAML under .../calibration/hand_calibration/.
        Prompts for a name if not provided.
        """
        if path is None:
            base = self._prompt_config_basename("load hand calibration")
            if base is None:
                print("[hand] Load cancelled.")
                return None
            path = self._resolve_hand_cfg_path(base)
        else:
            path = self._resolve_hand_cfg_path(path)

        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}

        self._hand_open  = data.get("open",  None)
        self._hand_close = data.get("close", None)
        # optional: respect file's min range, else keep current
        mrd = data.get("min_range_deg", None)
        if isinstance(mrd, (int, float)) and np.isfinite(mrd):
            self.HAND_MIN_RANGE_DEG = float(mrd)
        print(f"[hand] Loaded hand calibration: {path}")
        return path


    
    def _resolve_cfg_path(self, path_or_base: str) -> str:
        """If given a bare name (no .yaml or folder), drop into CALIB_DIR and add .yaml."""
        p = path_or_base.strip()
        # Add .yaml if missing
        if not (p.lower().endswith(".yaml")):
            p = f"{p}.yaml"
        # If absolute or already points somewhere, keep it; else put in calibration/
        if os.path.isabs(p):
            return p
        return os.path.join(self.CALIB_DIR, p)
    # ---------- Hand-calibration helpers ----------
    def _resolve_hand_cfg_path(self, path_or_base: str) -> str:
        """
        Resolve a hand-calibration YAML path under HAND_CALIB_DIR.
        Accepts bare names (adds .yaml) or absolute/relative paths.
        """
        p = (path_or_base or "").strip()
        if not p:
            # Will be handled by callers that prompt the user
            return os.path.join(self.HAND_CALIB_DIR, "hand_default.yaml")
        if not p.lower().endswith(".yaml"):
            p = f"{p}.yaml"
        if os.path.isabs(p):
            return p
        # If caller passed a relative path that includes folders, keep it; else drop in HAND_CALIB_DIR
        base_dir = self.HAND_CALIB_DIR if (os.path.dirname(p) == "") else ""
        return os.path.join(base_dir, p)

    def _angles_snapshot(self):
        """Deep-copy the current degrees dictionary (self._angles)."""
        return {finger: {j: float(a) for j, a in joints.items()} for finger, joints in self._angles.items()}

    @staticmethod
    def _norm_clip(x):
        return float(max(-1.0, min(1.0, x)))

    def _normalize_joint(self, a_now, a_open, a_close):
        """
        Map degrees → [-1,1] so that open -> -1, close -> +1.
        Safety: if range < HAND_MIN_RANGE_DEG or any missing -> 0.
        """
        if a_open is None or a_close is None:
            return 0.0
        rng = float(a_close - a_open)
        if not np.isfinite(a_now) or not np.isfinite(rng) or abs(rng) < float(self.HAND_MIN_RANGE_DEG):
            return 0.0
        # n = -1 at open, +1 at close, linear in between; clamp
        n = 2.0 * ((float(a_now) - float(a_open)) / rng) - 1.0
        return self._norm_clip(n)

    def _normalize_all_fingers(self, angles_now, calib_open, calib_close):
        """
        Vectorized normalization for the full nested dict.
        Missing/bad joints -> 0 by design.
        """
        out = {}
        for finger, joints in angles_now.items():
            out[finger] = {}
            for jname, a_now in joints.items():
                a_open  = calib_open.get(finger, {}).get(jname, None) if calib_open else None
                a_close = calib_close.get(finger, {}).get(jname, None) if calib_close else None
                out[finger][jname] = self._normalize_joint(a_now, a_open, a_close)
        return out

    def _prompt_config_basename(self, action: str):
        """
        Prompt until we get a valid basename (no '.yaml'), or user cancels.
        action: 'load' or 'save'
        Returns basename string or None if cancelled.
        """
        while True:
            base = input(
                f"Enter config name to {action} (no .yaml; use letters/digits/_/-), "
                "or 'cancel': "
            ).strip()
            low = base.lower()
            if low in ("cancel", "c", "q", "quit"):
                return None
            if low.endswith(".yaml"):
                print("Please omit the '.yaml' extension — it will be added automatically.")
                continue
            if not self._valid_basename(base):
                print("Invalid name. Use only letters, digits, underscores, or dashes (no spaces or dots).")
                continue
            return base
        
        
    # ---------- Overlay UI helpers (non-blocking) ----------
    def _overlay_text_input(self, window_name, prompt, subhint="", initial_text=""):
        """
        On-screen name entry. Returns typed string or None if cancelled (ESC).
        Avoids terminal input() so OpenCV never blocks.
        """
        import cv2, string, time
        allowed = string.ascii_letters + string.digits + "_-."
        text = initial_text
        while True:
            color_bgr, depth_vis = self.process(draw_landmarks=True)
            if color_bgr is None:
                continue
            y = 20
            cv2.putText(color_bgr, prompt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2); y += 24
            if subhint:
                cv2.putText(color_bgr, subhint, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1); y += 20
            cv2.putText(color_bgr, f"> {text}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2); y += 26
            cv2.putText(color_bgr, "[ENTER]=OK   [BACKSPACE]=del   [ESC]=cancel", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1)
            cv2.imshow(window_name, color_bgr)
            # if depth_vis is not None:
            #     cv2.imshow("Depth (vis)", depth_vis)

            k = cv2.waitKey(1) & 0xFF
            if k in (13, 10):   # ENTER
                return text.strip() or None
            if k == 27:         # ESC
                return None
            if k in (8, 127):   # BACKSPACE / DEL
                text = text[:-1]
                continue
            if 32 <= k <= 126:
                ch = chr(k)
                if ch in allowed:
                    text += ch

    def _overlay_flash(self, window_name, message, ms=900):
        """Quick toast-style message."""
        import cv2, time
        t0 = time.time()
        while (time.time() - t0) * 1000 < ms:
            color_bgr, depth_vis = self.process(draw_landmarks=True)
            if color_bgr is None:
                continue
            cv2.putText(color_bgr, message, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.imshow(window_name, color_bgr)
            # if depth_vis is not None:
            #     cv2.imshow("Depth (vis)", depth_vis)
            cv2.waitKey(1)




    # ======== Public getters/setters ========
    def have_pose(self) -> bool:
        return bool(self._have_pose)

    def get_world_xyz(self):
        return None if self._last_world_xyz is None else tuple(map(float, self._last_world_xyz))

    def get_normalized_xyz(self):
        return tuple(map(float, self._last_norm_xyz))

    def get_limits(self):
        return {k: (None if v is None else float(v)) for k, v in self.lims.items()}

    def set_yaw_steps(self, steps: int):
        self.yaw_steps = int(steps) % 4

    def add_yaw_step(self, delta: int):
        self.yaw_steps = (self.yaw_steps + int(delta)) % 4

    def flip_axis(self, axis: str):
        i = {"x":0,"y":1,"z":2}[axis.lower()]
        self.axis_flip[i] *= -1.0

    def set_limit(self, axis: str, which: str, value: float):
        key = f"{axis.lower()}_{which.lower()}"
        if key not in self.lims:
            raise KeyError(f"Bad limit key: {key}")
        self.lims[key] = float(value)

    def set_limit_from_current(self, axis: str, which: str):
        if self._last_world_xyz is None:
            return
        idx = {"x":0, "y":1, "z":2}[axis.lower()]
        self.set_limit(axis, which, self._last_world_xyz[idx])

    def save_calibration(self, path=None):
        if path is None:
            path = self.CFG_FILE
        path = self._resolve_cfg_path(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "R_world_cam": self.R_world_cam.tolist(),
            "t_world_cam": self.t_world_cam.flatten().tolist(),
            "yaw_steps": int(self.yaw_steps),
            "axis_flip": self.axis_flip.tolist(),
            "limits": {k:(None if v is None else float(v)) for k,v in self.lims.items()},
            "tag_size_m": float(self.TAG_SIZE_M),
            "bake_at_capture": bool(self.BAKE_AT_CAPTURE),
        }
        with open(path, "w") as f:
            yaml.safe_dump(data, f)
        self.CFG_FILE = path
        return path

    def load_calibration(self, path=None):
        if path is None:
            path = self.CFG_FILE
        path = self._resolve_cfg_path(path)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        self.R_world_cam = np.array(data["R_world_cam"], dtype=np.float32)
        self.t_world_cam = np.array(data["t_world_cam"], dtype=np.float32).reshape(3,1)
        self.yaw_steps   = int(data.get("yaw_steps", 0))
        self.axis_flip   = np.array(data.get("axis_flip", [1,1,1]), dtype=np.float32)
        if "limits" in data:
            self.lims = {k: (None if v is None else float(v)) for k,v in data["limits"].items()}
        self.TAG_SIZE_M = float(data.get("tag_size_m", self.TAG_SIZE_M))
        self.BAKE_AT_CAPTURE = bool(data.get("bake_at_capture", self.BAKE_AT_CAPTURE))
        self._have_pose = True
        self.CFG_FILE = path
        return path

    # ======== Lifecycle ========
    def start(self):
        # RealSense streams
        self.pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, self.COLOR_W, self.COLOR_H, rs.format.z16, self.FPS)
        cfg.enable_stream(rs.stream.color, self.COLOR_W, self.COLOR_H, rs.format.bgr8, self.FPS)
        profile = self.pipe.start(cfg)

        import time
        t0 = time.time()
        frames = None
        while frames is None:
            frames = self.pipe.poll_for_frames()     # returns None until something arrives
            if frames:
                break
            if time.time() - t0 > 10.0:
                raise RuntimeError("Camera started but no frames in 3s. Check USB3 and close RealSense Viewer.")
            time.sleep(0.01)


        self.align = rs.align(rs.stream.color)
        
        

        # Projector ON (if supported)
        try:
            depth_sensor = profile.get_device().first_depth_sensor()
            if depth_sensor.supports(rs.option.emitter_enabled):
                depth_sensor.set_option(rs.option.emitter_enabled, 1)
            if depth_sensor.supports(rs.option.laser_power):
                rng = depth_sensor.get_option_range(rs.option.laser_power)
                depth_sensor.set_option(rs.option.laser_power, min(rng.max, 180))
        except Exception:
            pass

        # Depth filters
        self.depth_filters["spatial"]  = rs.spatial_filter()
        self.depth_filters["temporal"] = rs.temporal_filter()
        self.depth_filters["holefill"] = rs.hole_filling_filter()

        # Intrinsics
        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        self.intr = color_stream.get_intrinsics()
        self.Kcv, self.Dcv = self._rs_intrinsics_to_cvK_D(self.intr)
        
        print("[pose] locked. use [ or ] to rotate by 90°, X/Y/Z to flip axes.  !!! bug to fix its a bit tricky to to the flip coz it also change the yaw, so play with both")
        print("[pose] use 1..6 to set limits for wrist position, S to save, L to load.")
        print("[pose] press C to capture again.")
        print("[pose]you need to go throught 1 to 6 to set the limits, then you can use the wrist position to normalize the pose")
        print("[pose]to fix: the flip also change the yaw, so play with both")


    def stop(self):
        try:
            if self.pipe is not None:
                self.pipe.stop()
        finally:
            self.pipe = None
            self.hands.close()
            cv2.destroyAllWindows()

    # ======== Core processing ========
    
    
    # ======== Core processing ========
def process(self, draw_landmarks=True):
    """
    Pull one frame pair, run tag detect (for viz), run hand detect (GPU via Tasks),
    update world/normalized XYZ, and return (color_bgr, depth_vis).
    """
    # Grab and align frames
    frames = self.pipe.wait_for_frames()
    frames = self.align.process(frames)

    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    if not depth_frame or not color_frame:
        return None, None

    # Depth filtering (keep as-is)
    df = depth_frame
    df = self.depth_filters["spatial"].process(df)
    df = self.depth_filters["temporal"].process(df)
    df = self.depth_filters["holefill"].process(df)
    df = df.as_depth_frame() or depth_frame

    # Color image
    color_bgr = np.asanyarray(color_frame.get_data())
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

    # ---------- AprilTag (for drawing only; pose capture has its own averaging) ----------
    det = self._detect_tag_pose(color_bgr)
    if det is not None:
        corners, tag_id, rvec, tvec = det
        cv2.aruco.drawDetectedMarkers(color_bgr, [corners], np.array([[tag_id]], dtype=np.int32))
        cv2.drawFrameAxes(color_bgr, self.Kcv, self.Dcv, rvec, tvec, self.TAG_SIZE_M * 0.5)

    # ---------- Hands (GPU via MediaPipe Tasks) ----------
    # Build MP image and run detector
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=color_rgb)
    res = self.hands.detect(mp_img)

    # Adapt Tasks output to a minimal object compatible with our downstream code
    lms = None
    if res and len(res.hand_landmarks) > 0:
        # Tiny structs to mimic .landmark[i].x/.y/.z and container with .landmark
        class _Lm:
            __slots__ = ("x", "y", "z")
            def __init__(self, x, y, z=0.0):
                self.x, self.y, self.z = x, y, z

        class _Lms:
            __slots__ = ("landmark",)
            def __init__(self, arr):
                self.landmark = arr

        pts = [_Lm(p.x, p.y, getattr(p, "z", 0.0)) for p in res.hand_landmarks[0]]
        lms = _Lms(pts)

        # Optional drawing using the same connections spec
        if draw_landmarks:
            self.mp_draw.draw_landmarks(
                color_bgr,
                lms,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_style.get_default_hand_landmarks_style(),
                self.mp_style.get_default_hand_connections_style()
            )

    # Depth visualization
    depth_raw = np.asanyarray(df.get_data())
    depth_vis = cv2.convertScaleAbs(depth_raw, alpha=0.03)

    world_xyz = None
    nvec = (0.0, 0.0, 0.0)

    if lms is not None:
        # === All 21 points in camera frame (3D) for finger flexion ===
        P_cam = self._landmarks_cam3d(df, lms)
        if P_cam is not None:
            # Compute flexion angles (degrees), updates self._angles
            self._compute_flexions_deg(P_cam)

        # Wrist-specific depth & world mapping (existing behavior)
        lm = lms.landmark[int(self.SHOW_LM)]
        u = float(lm.x) * self.COLOR_W
        v = float(lm.y) * self.COLOR_H
        z = self._depth_at_m(df, u, v, k=self.K_NEIGH)

        if z > 0 and self.DEPTH_RANGE[0] <= z <= self.DEPTH_RANGE[1]:
            p_cam = np.array(
                rs.rs2_deproject_pixel_to_point(self.intr, [u, v], z),
                dtype=np.float32
            ).reshape(3, 1)

            # EMA smoothing
            prev = self._ema_3d[0].get('wrist')
            p_cam_s = p_cam if prev is None else self.EMA_ALPHA * prev + (1.0 - self.EMA_ALPHA) * p_cam
            self._ema_3d[0]['wrist'] = p_cam_s

            if self._have_pose:
                # Adjusted R,t each frame (matches your current flow)
                R_adj = self._yaw_deg_R(90 * self.yaw_steps) @ np.diag(self.axis_flip)
                Rwc = (R_adj @ self.R_world_cam).astype(np.float32)
                twc = (R_adj @ self.t_world_cam).astype(np.float32)

                p_world = (Rwc @ p_cam_s + twc).reshape(3)
                world_xyz = p_world.copy()

                # Normalize if limits are set
                if all(v is not None for v in self.lims.values()):
                    pmin = [self.lims["x_min"], self.lims["y_min"], self.lims["z_min"]]
                    pmax = [self.lims["x_max"], self.lims["y_max"], self.lims["z_max"]]
                    n = self._normalize_unit(p_world, pmin, pmax)
                    nvec = (float(n[0]), float(n[1]), float(n[2]))

            # Annotate depth at wrist pixel
            cv2.circle(color_bgr, (int(u), int(v)), 6, (0, 255, 0), 2)
            cv2.putText(color_bgr, f"z={z:.3f} m", (int(u) + 6, int(v) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    self._last_world_xyz = world_xyz
    self._last_norm_xyz = nvec
    return color_bgr, depth_vis

    # ======== HUD helpers (optional) ========
    @staticmethod
    def draw_bars(img, x, y, w, h, val, label):
        cv2.rectangle(img, (x,y), (x+w, y+h), (80,80,80), 1)
        cx = int(x + (val+1.0)*0.5*w)
        cv2.line(img, (cx,y), (cx,y+h), (0,255,0), 2)
        cv2.putText(img, f"{label}:{val:+.2f}", (x, y-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    def draw_hud(self, color_bgr):
        h, w = color_bgr.shape[:2]
        nx, ny, nz = self._last_norm_xyz
        self.draw_bars(color_bgr, 10,  h-30, 180, 18, nx, "nx (+fwd)")
        self.draw_bars(color_bgr, 200, h-30, 180, 18, ny, "ny (+left)")
        self.draw_bars(color_bgr, 390, h-30, 180, 18, nz, "nz (+up)")

        def put(img, s, y):
            cv2.putText(img, s, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            return y + 18

        y0 = 20
        y0 = put(color_bgr, f"Pose: {'LOCKED' if self._have_pose else '—'}  yaw_steps={self.yaw_steps}  flip={self.axis_flip.astype(int).tolist()}", y0)
        if self._last_world_xyz is not None:
            x,y,z = self._last_world_xyz
            y0 = put(color_bgr, f"World xyz [m]: {x:+.3f}, {y:+.3f}, {z:+.3f}", y0)
            y0 = put(color_bgr, f"Norm  n  [-1,1]: {nx:+.2f}, {ny:+.2f}, {nz:+.2f}", y0)
        L = self.lims
        y0 = put(color_bgr, f"Limits x:[{L['x_min']},{L['x_max']}], y:[{L['y_min']},{L['y_max']}], z:[{L['z_min']},{L['z_max']}]", y0)
        put(color_bgr, "Hotkeys: C capture  [ ] rotate  X/Y/Z flip  1..6 set lims  S save  L load  Q quit", y0)
        
        if getattr(self, "SHOW_FLEX_PANEL", True):
            self._draw_flexion_panel(color_bgr, corner="br")

    # ======== Internals (helpers) ========
    def _detect_tag_pose(self, bgr):
        corners, ids, _ = self.aruco.detectMarkers(bgr)
        if ids is None or len(ids) == 0:
            return None
        areas = [cv2.contourArea(c.astype(np.float32)) for c in corners]
        k = int(np.argmax(areas))
        rvecs, tvecs, _obj = cv2.aruco.estimatePoseSingleMarkers([corners[k]], self.TAG_SIZE_M, self.Kcv, self.Dcv)
        rvec = rvecs[0].reshape(3,1).astype(np.float32)
        tvec = tvecs[0].reshape(3,1).astype(np.float32)
        return corners[k], int(ids[k][0]), rvec, tvec

    @staticmethod
    def _depth_at_m(depth_frame, u, v, k=5):
        u = int(round(u)); v = int(round(v))
        w = depth_frame.get_width(); h = depth_frame.get_height()
        r = k // 2
        vals = []
        for yy in range(max(0, v - r), min(h, v + r + 1)):
            for xx in range(max(0, u - r), min(w, u + r + 1)):
                d = depth_frame.get_distance(xx, yy)
                if d and np.isfinite(d):
                    vals.append(d)
        if not vals:
            return 0.0
        return float(np.median(vals))

    @staticmethod
    def _rs_intrinsics_to_cvK_D(intr):
        K = np.array([[intr.fx, 0, intr.ppx],
                      [0, intr.fy, intr.ppy],
                      [0,       0,       1]], dtype=np.float32)
        D = np.array(intr.coeffs[:5], dtype=np.float32)  # k1,k2,p1,p2,k3
        return K, D

    @staticmethod
    def _rodrigues_to_R(rvec):
        R, _ = cv2.Rodrigues(rvec.astype(np.float64))
        return R.astype(np.float32)

    @staticmethod
    def _invert_rt(R, t):
        Rt = R.T
        tt = -Rt @ t
        return Rt, tt

    @staticmethod
    def _yaw_deg_R(deg):
        th = math.radians(deg)
        c, s = math.cos(th), math.sin(th)
        return np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float32)

    @staticmethod
    def _normalize_unit(p, pmin, pmax):
        p = np.asarray(p, np.float32)
        pmin = np.asarray(pmin, np.float32)
        pmax = np.asarray(pmax, np.float32)
        span = np.maximum(pmax - pmin, 1e-6)
        pn = 2.0*(p - pmin)/span - 1.0
        return np.clip(pn, -1.0, 1.0)
    
    
    # ======== NEW: startup menu & calibration wizard ========

    def startup_menu(self, window_name="Startup"):
        """
        On-screen menu for XYZ calibration:
          L = Load existing (name prompt overlay)
          N = New (launches calibration wizard in this same window)
          ESC = cancel
        Returns 'loaded', 'created', or False if cancelled.
        """
        import cv2, os
        while True:
            color_bgr, depth_vis = self.process(draw_landmarks=True)
            if color_bgr is None:
                continue

            y = 24
            cv2.putText(color_bgr, "=== XYZ Calibration ===", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2); y += 30
            cv2.putText(color_bgr, "[L] Load existing   [N] New (open wizard)   [ESC] Cancel", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.imshow(window_name, color_bgr)
            # if depth_vis is not None:
            #     cv2.imshow("Depth (vis)", depth_vis)

            k = cv2.waitKey(1) & 0xFF
            if k == 27:  # ESC
                return False

            # Load existing
            if k in (ord('l'), ord('L')):
                base = self._overlay_text_input(window_name, "Load XYZ calib (name only):", subhint="Folder: .../calibration/")
                if not base:
                    self._overlay_flash(window_name, "[load] Cancelled.", ms=700)
                    continue
                cfg_path = self._resolve_cfg_path(base)
                if not os.path.exists(cfg_path):
                    self._overlay_flash(window_name, "[load] File not found.", ms=900)
                    continue
                try:
                    self.load_calibration(cfg_path)
                    self._overlay_flash(window_name, f"[loaded XYZ] {os.path.basename(cfg_path)}", ms=900)
                    return "loaded"
                except FileNotFoundError:
                    self._overlay_flash(window_name, "[load] File disappeared.", ms=900)

            # New: run wizard in-place
            if k in (ord('n'), ord('N')):
                ok = self.calibration_wizard(window_name=window_name)
                if ok:
                    # User should press 'S' inside the wizard to save. We just continue.
                    self._overlay_flash(window_name, "[XYZ] Created. (Use 'S' in wizard to save.)", ms=1000)
                    return "created"
                else:
                    self._overlay_flash(window_name, "[XYZ] Wizard cancelled.", ms=900)

    def hand_startup_menu(self, window_name="Main"):
        """
        On-screen menu for Hand Flexion calibration:
          L = Load existing (name prompt overlay)
          N = New (launches hand wizard; O=open, P=close, S=save; returns immediately after S)
          ESC = skip/cancel
        Returns 'loaded', 'created', or False if cancelled.
        """
        import cv2, os
        while True:
            color_bgr, depth_vis = self.process(draw_landmarks=True)
            if color_bgr is None:
                continue

            y = 24
            cv2.putText(color_bgr, "=== Hand Flexion Calibration ===", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2); y += 30
            cv2.putText(color_bgr, "[L] Load existing   [N] New (open wizard)   [ESC] Skip", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.imshow(window_name, color_bgr)
            # if depth_vis is not None:
            #     cv2.imshow("Depth (vis)", depth_vis)

            k = cv2.waitKey(1) & 0xFF
            if k == 27:  # ESC
                return False

            # Load existing
            if k in (ord('l'), ord('L')):
                base = self._overlay_text_input(window_name, "Load HAND calib (name only):",
                                                subhint="Folder: .../calibration/hand_calibration/")
                if not base:
                    self._overlay_flash(window_name, "[hand] Load cancelled.", ms=700)
                    continue
                path = self._resolve_hand_cfg_path(base)
                if not os.path.exists(path):
                    self._overlay_flash(window_name, "[hand] File not found.", ms=900)
                    continue
                try:
                    self.load_hand_calibration(path)
                    self._overlay_flash(window_name, "[hand] Loaded.", ms=800)
                    return "loaded"
                except FileNotFoundError:
                    self._overlay_flash(window_name, "[hand] File disappeared.", ms=900)

            # New: launch hand wizard
            if k in (ord('n'), ord('N')):
                ok = self.hand_calibration_wizard(window_name=window_name)
                if ok:
                    self._overlay_flash(window_name, "[hand] Created & saved.", ms=900)
                    return "created"
                else:
                    # Wizard returns False on ESC or if not both O&P captured before Enter.
                    self._overlay_flash(window_name, "[hand] Wizard cancelled.", ms=900)



    def calibration_wizard(self, window_name="Calibration Wizard"):
        """
        Interactive, on-screen wizard to:
        - capture pose (press C),
        - adjust orientation ([ ] X Y Z),
        - set limits (1..6),
        - save (S),
        - finish (ENTER) or cancel (ESC).
        Returns True if a usable calibration exists when finishing.
        """
        print("\n=== Calibration Wizard ===")
        print("Instructions:")
        print("  • Show the AprilTag board; press 'C' to capture pose (hold steady).")
        print("  • Align axes: '[' ']' rotate 90°, 'X' 'Y' 'Z' flip axes.")
        print("  • Set workspace limits from current wrist: 1..6 (x_min,x_max,y_min,y_max,z_min,z_max).")
        print("  • 'S' to save calibration, ENTER to finish, ESC to cancel.\n")

        # simple overlay helper
        def _put(img, s, y):
            import cv2
            cv2.putText(img, s, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            return y + 18

        while True:
            color_bgr, depth_vis = self.process(draw_landmarks=True)
            if color_bgr is None:
                continue

            # draw HUD + wizard text
            self.draw_hud(color_bgr)
            y = 90
            y = _put(color_bgr, "[Wizard] C=capture  [ ][ ]=rotate  X/Y/Z=flip", y)
            y = _put(color_bgr, "          1..6=set limits  S=save  ENTER=finish  ESC=cancel", y)
            y = _put(color_bgr, f"Pose: {'LOCKED' if self._have_pose else '—'}   Limits set: {all(v is not None for v in self.lims.values())}", y)

            cv2.imshow(window_name, color_bgr)
            # if depth_vis is not None:
            #     cv2.imshow("Depth (vis)", depth_vis)

            key = cv2.waitKey(1) & 0xFF
            if key in (13, 10):  # ENTER/RETURN
                # Must have pose and limits to be considered complete
                if self._have_pose and all(v is not None for v in self.lims.values()):
                    return True
                else:
                    print("[wizard] Need pose (C) and all 6 limits (1..6) before finishing.")
            if key == 27:  # ESC
                return False

            # same controls as your main loop
            if key in (ord('c'), ord('C')):
                ok = self.capture_pose()
                print("[pose] locked" if ok else "[pose] failed; show the board steady")
            if key == ord('['):
                self.add_yaw_step(-1)
            if key == ord(']'):
                self.add_yaw_step(+1)
            if key in (ord('x'), ord('X')):
                self.flip_axis('x')
            if key in (ord('y'), ord('Y')):
                self.flip_axis('y')
            if key in (ord('z'), ord('Z')):
                self.flip_axis('z')
            if key == ord('1'):
                self.set_limit_from_current('x', 'min')
            if key == ord('2'):
                self.set_limit_from_current('x', 'max')
            if key == ord('3'):
                self.set_limit_from_current('y', 'min')
            if key == ord('4'):
                self.set_limit_from_current('y', 'max')
            if key == ord('5'):
                self.set_limit_from_current('z', 'min')
            if key == ord('6'):
                self.set_limit_from_current('z', 'max')
            if key in (ord('s'), ord('S')):
                base = self._overlay_text_input(
                    window_name,
                    "Save XYZ calibration name ('.yaml' auto):",
                    subhint="Folder: .../calibration/",
                    initial_text=""
                )
                if base is None:
                    self._overlay_flash(window_name, "[save] Cancelled.", ms=700)
                    continue
                cfg_path = self._resolve_cfg_path(base)
                if os.path.exists(cfg_path):
                    # ask to overwrite
                    ask = "File exists. Press 'Y' to overwrite or any key to cancel."
                    y0 = 20
                    while True:
                        color_bgr, depth_vis = self.process(draw_landmarks=True)
                        if color_bgr is None: continue
                        y = y0
                        cv2.putText(color_bgr, ask, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2); y += 24
                        cv2.imshow(window_name, color_bgr)
                        # if depth_vis is not None: cv2.imshow("Depth (vis)", depth_vis)
                        kk = cv2.waitKey(1) & 0xFF
                        if kk in (ord('y'), ord('Y')):
                            break
                        if kk != 255:
                            cfg_path = None
                            break
                if cfg_path:
                    p = self.save_calibration(cfg_path)
                    self.CFG_FILE = cfg_path
                    self._overlay_flash(window_name, f"[saved XYZ] {os.path.basename(p)}", ms=900)


    def hand_calibration_wizard(self, window_name="Hand Flexion Calibration"):
        """
        Press:
          O = record OPEN hand (fingers extended/relaxed),
          P = record CLOSED hand (make a firm fist),
          S = save (prompts for name in hand_calibration/),
          L = load (prompts),
          ENTER = finish, ESC = cancel.
        """
        print("\n=== Hand Flexion Calibration ===")
        print("  • Show your hand to the camera.")
        print("  • Press 'O' to capture OPEN; 'P' to capture CLOSE.")
        print("  • 'S' to save, 'L' to load, ENTER to finish, ESC to cancel.\n")

        def _put(img, s, y):
            cv2.putText(img, s, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            return y + 18

        open_ok = self._hand_open  is not None
        close_ok = self._hand_close is not None

        while True:
            color_bgr, depth_vis = self.process(draw_landmarks=True)
            if color_bgr is None:
                continue

            y = 20
            y = _put(color_bgr, f"[O] Open:  {'OK' if open_ok else '—'}", y)
            y = _put(color_bgr, f"[P] Close: {'OK' if close_ok else '—'}", y)
            y = _put(color_bgr, f"Min range: {self.HAND_MIN_RANGE_DEG:.1f} deg (too small ⇒ normalized=0)", y)
            y = _put(color_bgr, "[S] Save  [L] Load  [ENTER] Finish  [ESC] Cancel", y)

            # (Optional) Show index MCP normalized live (once both captured)
            if open_ok and close_ok:
                idx_n = self.get_normalized_flexion_index()
                y = _put(color_bgr, f"Index MCP/PIP/DIP (norm): {idx_n['MCP']:+.2f} {idx_n['PIP']:+.2f} {idx_n['DIP']:+.2f}", y)

            cv2.imshow(window_name, color_bgr)
            # if depth_vis is not None: cv2.imshow("Depth (vis)", depth_vis)

            key = cv2.waitKey(1) & 0xFF
            if key in (13, 10):  # ENTER
                return open_ok and close_ok
            if key == 27:       # ESC
                return False
            if key in (ord('o'), ord('O')):
                self.record_hand_open()
                open_ok = True
                print("[hand] Captured OPEN.")
            if key in (ord('p'), ord('P')):
                self.record_hand_close()
                close_ok = True
                print("[hand] Captured CLOSE.")
            if key in (ord('s'), ord('S')):
                # Only allow save once both OPEN and CLOSE captured
                if not (open_ok and close_ok):
                    self._overlay_flash(window_name, "[hand] Capture OPEN (O) and CLOSE (P) first.", ms=1100)
                    continue
                base = self._overlay_text_input(
                    window_name,
                    "Save HAND calibration name ('.yaml' auto):",
                    subhint="Folder: .../calibration/hand_calibration/",
                    initial_text=""
                )
                if base is None:
                    self._overlay_flash(window_name, "[hand] Save cancelled.", ms=700)
                    continue
                path = self._resolve_hand_cfg_path(base)
                if os.path.exists(path):
                    ask = "File exists. Press 'Y' to overwrite or any key to cancel."
                    while True:
                        color_bgr, depth_vis = self.process(draw_landmarks=True)
                        if color_bgr is None: continue
                        cv2.putText(color_bgr, ask, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                        cv2.imshow(window_name, color_bgr)
                        # if depth_vis is not None: cv2.imshow("Depth (vis)", depth_vis)
                        kk = cv2.waitKey(1) & 0xFF
                        if kk in (ord('y'), ord('Y')):
                            break
                        if kk != 255:
                            path = None
                            break
                if path:
                    try:
                        self.save_hand_calibration(path)
                        self._overlay_flash(window_name, f"[saved hand] {os.path.basename(path)}", ms=900)
                        # Immediately finish so main loop starts printing
                        return True
                    except RuntimeError as e:
                        self._overlay_flash(window_name, f"[hand] Save failed", ms=900)
            if key in (ord('l'), ord('L')):
                base = self._overlay_text_input(
                    window_name,
                    "Load HAND calibration (type name without .yaml):",
                    subhint="Folder: .../calibration/hand_calibration/",
                    initial_text=""
                )
                if base is None:
                    self._overlay_flash(window_name, "[hand] Load cancelled.", ms=700)
                    continue
                path = self._resolve_hand_cfg_path(base)
                if not os.path.exists(path):
                    self._overlay_flash(window_name, "[hand] File not found.", ms=1000)
                    continue
                self.load_hand_calibration(path)
                open_ok  = self._hand_open  is not None
                close_ok = self._hand_close is not None
                self._overlay_flash(window_name, "[hand] Loaded.", ms=700)

            if key in (ord('l'), ord('L')):
                self.load_hand_calibration(None)
                open_ok  = self._hand_open  is not None
                close_ok = self._hand_close is not None
