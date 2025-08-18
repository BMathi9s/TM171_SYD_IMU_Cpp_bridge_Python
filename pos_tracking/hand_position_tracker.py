# desk_hand_tracker.py
import os, time, math, yaml
from collections import defaultdict, deque

import cv2
import numpy as np
import mediapipe as mp
import pyrealsense2 as rs
import re  


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
    DEFAULT_W = 640
    DEFAULT_H = 480
    DEFAULT_FPS = 30

    def __init__(
        self,
        color_w=DEFAULT_W,
        color_h=DEFAULT_H,
        fps=DEFAULT_FPS,
        tag_size_m=0.08,                 # you printed 80 mm → 0.08 m
        depth_range=(0.25, 1.8),
        k_neigh=5,                       # median patch for depth sampling
        ema_alpha=0.75,                  # 3D smoothing
        pose_avg_frames=30,              # frames to average on capture
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
        self.CALIB_DIR = os.path.join(self.PROJ_ROOT, "calibration")
        os.makedirs(self.CALIB_DIR, exist_ok=True)

        # Default config path: <project>/calibration/desk_calib.yaml
        if cfg_file is None:
            cfg_file = os.path.join(self.CALIB_DIR, "desk_calib.yaml")
        self.CFG_FILE = cfg_file
        
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
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False, max_num_hands=1,
            min_detection_confidence=0.6, min_tracking_confidence=0.6,
            model_complexity=1
        )
        self.mp_draw  = mp.solutions.drawing_utils
        self.mp_style = mp.solutions.drawing_styles

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



    # ======== Config-name helpers ========
    @staticmethod
    def _valid_basename(name: str) -> bool:
        # letters, digits, underscore, dash only
        return bool(re.fullmatch(r"[A-Za-z0-9_-]+", name))
    
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

    # @staticmethod
    # def _to_yaml_path(basename: str) -> str:
    #     return f"{basename}.yaml"


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
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, self.FPS)
        cfg.enable_stream(rs.stream.color, self.COLOR_W, self.COLOR_H, rs.format.bgr8, self.FPS)
        profile = self.pipe.start(cfg)

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
    def process(self, draw_landmarks=True):
        """
        Pull one frame pair, run tag detect (for viz), run hand detect,
        update world/normalized XYZ, and return (color_bgr, depth_vis).
        """
        frames = self.pipe.wait_for_frames()
        frames = self.align.process(frames)

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            return None, None

        # Filters
        df = depth_frame
        df = self.depth_filters["spatial"].process(df)
        df = self.depth_filters["temporal"].process(df)
        df = self.depth_filters["holefill"].process(df)
        df = df.as_depth_frame() or depth_frame

        color_bgr = np.asanyarray(color_frame.get_data())
        color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

        # Tag pose (for drawing; capture_pose() does its own averaging)
        det = self._detect_tag_pose(color_bgr)
        if det is not None:
            corners, tag_id, rvec, tvec = det
            cv2.aruco.drawDetectedMarkers(color_bgr, [corners], np.array([[tag_id]], dtype=np.int32))
            cv2.drawFrameAxes(color_bgr, self.Kcv, self.Dcv, rvec, tvec, self.TAG_SIZE_M*0.5)

        # Hands
        res = self.hands.process(color_rgb)
        depth_raw = np.asanyarray(df.get_data())
        depth_vis = cv2.convertScaleAbs(depth_raw, alpha=0.03)

        world_xyz = None
        nvec = (0.0, 0.0, 0.0)

        if res.multi_hand_landmarks:
            lms = res.multi_hand_landmarks[0]
            if draw_landmarks:
                self.mp_draw.draw_landmarks(
                    color_bgr, lms, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_style.get_default_hand_landmarks_style(),
                    self.mp_style.get_default_hand_connections_style()
                )
            # pick wrist
            lm = lms.landmark[int(self.SHOW_LM)]
            u = lm.x * self.COLOR_W
            v = lm.y * self.COLOR_H
            z = self._depth_at_m(df, u, v, k=self.K_NEIGH)

            if z > 0 and self.DEPTH_RANGE[0] <= z <= self.DEPTH_RANGE[1]:
                p_cam = np.array(rs.rs2_deproject_pixel_to_point(self.intr, [float(u), float(v)], float(z)), dtype=np.float32).reshape(3,1)

                # EMA smoothing
                prev = self._ema_3d[0].get('wrist')
                p_cam_s = p_cam if prev is None else self.EMA_ALPHA*prev + (1.0-self.EMA_ALPHA)*p_cam
                self._ema_3d[0]['wrist'] = p_cam_s

                if self._have_pose:
                    # Adjusted R,t each frame (matches your current flow)
                    R_adj = self._yaw_deg_R(90*self.yaw_steps) @ np.diag(self.axis_flip)
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

                # annotate depth at wrist pixel
                cv2.circle(color_bgr, (int(u), int(v)), 6, (0,255,0), 2)
                cv2.putText(color_bgr, f"z={z:.3f} m", (int(u)+6, int(v)-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        self._last_world_xyz = world_xyz
        self._last_norm_xyz = nvec
        return color_bgr, depth_vis

    # ======== Pose capture ========
    def capture_pose(self):
        """Hold the tag board steady and call this once; averages several frames."""
        self._pose_stack.clear()
        for _ in range(self.POSE_AVG_FR):
            frames = self.pipe.wait_for_frames()
            frames = self.align.process(frames)
            color = np.asanyarray(frames.get_color_frame().get_data())
            det = self._detect_tag_pose(color)
            if det is None:
                continue
            _, _, rvec, tvec = det
            R_cm = self._rodrigues_to_R(rvec)    # marker→camera
            t_cm = tvec                          # marker→camera
            # camera→marker
            R_mc, t_mc = self._invert_rt(R_cm, t_cm)
            self._pose_stack.append((R_mc, t_mc))
            time.sleep(0.004)

        if len(self._pose_stack) < 5:
            return False

        Rs = np.stack([p[0] for p in self._pose_stack], axis=0)
        ts = np.stack([p[1].reshape(3) for p in self._pose_stack], axis=0)
        R_mean = Rs[-1]  # (simple fallback; a full rotation avg can be added)
        t_mean = ts.mean(axis=0).reshape(3,1).astype(np.float32)

        if self.BAKE_AT_CAPTURE:
            # base: world = marker; then apply yaw & flips immediately (matches your current code)
            R_adj = self._yaw_deg_R(90*self.yaw_steps) @ np.diag(self.axis_flip)
            self.R_world_cam = (R_adj @ R_mean).astype(np.float32)
            self.t_world_cam = (R_adj @ t_mean).astype(np.float32)
        else:
            # store RAW, apply yaw/flip only at runtime (mathematically cleaner)
            self.R_world_cam = R_mean.astype(np.float32)
            self.t_world_cam = t_mean.astype(np.float32)

        self._have_pose = True
        return True

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

    def startup_menu(self, path=None):
        """
        Console menu:
        [1] Load calibration (prompts for name; appends .yaml)
        [2] Create new calibration (runs wizard; save prompts for name)
        Returns 'loaded' or 'created'.
        """
        while True:
            print("\n=== Start ===")
            print("[1] Load calibration")
            print("[2] Create new calibration")
            choice = input("Select 1 or 2: ").strip()
            if choice == "1":
                base = self._prompt_config_basename("load")
                if base is None:
                    continue
                cfg_path = self._resolve_cfg_path(base)
                if not os.path.exists(cfg_path):
                    print(f"[load] '{cfg_path}' not found. Try again or choose [2] to create one.")
                    continue
                try:
                    p = self.load_calibration(cfg_path)
                    print(f"[loaded] {p}")
                    return "loaded"
                except FileNotFoundError:
                    print(f"[load] '{cfg_path}' not found. (Race condition?)")
            elif choice == "2":
                ok = self.calibration_wizard()
                if ok:
                    print("[created] Calibration captured.")
                    return "created"
                else:
                    print("[wizard] Cancelled or failed. Try again.")
            else:
                print("Please type 1 or 2.")


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
            if depth_vis is not None:
                cv2.imshow("Depth (vis)", depth_vis)

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
                while True:
                    base = self._prompt_config_basename("save")
                    if base is None:
                        print("[save] Cancelled.")
                        break
                    cfg_path = self._resolve_cfg_path(base)
                    if os.path.exists(cfg_path):
                        resp = input(f"'{cfg_path}' exists. Overwrite? [y/N]: ").strip().lower()
                        if resp != "y":
                            print("Choose another name.")
                            continue
                    p = self.save_calibration(cfg_path)
                    self.CFG_FILE = cfg_path  # remember as default for next time
                    print(f"[saved] {p}")
                    break

