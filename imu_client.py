# imu_client.py
import socket, json, threading, time, math
from typing import Optional, Tuple

def _wrap_angle_deg(a: float) -> float:
    # [-180, 180)
    a = (a + 180.0) % 360.0 - 180.0
    return a

class ImuUdpClient:
    """
    Listens to IMU bridge UDP JSON (127.0.0.1:8765 by default).
    Keeps the latest rpy/quaternion/raw and provides getters.
    Allows setting an RPY offset (e.g., zero at current pose).
    Optional EMA smoothing (per axis) for RPY.
    """
    def __init__(self,
                 host: str = "127.0.0.1",
                 port: int = 8765,
                 recv_buf: int = 1 << 20,
                 smooth_alpha: float = 0.0  # 0 = off, 0.2..0.5 = moderate smoothing
                 ):
        self.host = host
        self.port = port
        self.smooth_alpha = float(smooth_alpha)
        self._have_offset = False

        # latest values (protected by a lock)
        self._lock = threading.Lock()
        self._rpy_deg: Optional[Tuple[float, float, float]] = None
        self._quat: Optional[Tuple[float, float, float, float]] = None
        self._raw: Optional[Tuple[Tuple[float,float,float],
                                  Tuple[float,float,float],
                                  Tuple[float,float,float]]] = None
        self._t_us: Optional[int] = None

        # offsets (applied to RPY getter)
        self._r_off = 0.0
        self._p_off = 0.0
        self._y_off = 0.0
        

        # EMA state
        self._ema_r = None
        self._ema_p = None
        self._ema_y = None

        # thread control
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, recv_buf)
        self._sock.bind((self.host, self.port))
        self._sock.settimeout(0.2)
        # quick and dirty swap
        self._axis_order = [0, 1, 2]       # roll=0, pitch=1, yaw=2
        self._axis_signs = [1.0, 1.0, 1.0] # no inversion
        
        
        self._running = True
        self._thr = threading.Thread(target=self._run, daemon=True)
        self._thr.start()

    # ---------- public API ----------

    def close(self):
        self._running = False
        try:
            self._thr.join(timeout=1.0)
        except Exception:
            pass
        try:
            self._sock.close()
        except Exception:
            pass
    
    def swap_axes(self, a: str, b: str):
        """Swap two axes ('roll','pitch','yaw')."""
        name_to_idx = {"roll": 0, "pitch": 1, "yaw": 2}
        if a not in name_to_idx or b not in name_to_idx:
            raise ValueError("Axes must be 'roll', 'pitch', or 'yaw'")
        ia, ib = name_to_idx[a], name_to_idx[b]
        self._axis_order[ia], self._axis_order[ib] = self._axis_order[ib], self._axis_order[ia]

    def invert_axis(self, axis: str):
        """Invert an axis ('roll','pitch','yaw')."""
        name_to_idx = {"roll": 0, "pitch": 1, "yaw": 2}
        if axis not in name_to_idx:
            raise ValueError("Axis must be 'roll', 'pitch', or 'yaw'")
        i = name_to_idx[axis]
        self._axis_signs[i] *= -1.0

    def get_timestamp_us(self) -> Optional[int]:
        with self._lock:
            return self._t_us

    # --- adjust getters to respect mapping ---
    def get_rpy_deg(self):
        """Return (roll, pitch, yaw) in degrees with offset, mapping, and inversion."""
        with self._lock:
            if self._rpy_deg is None:
                return None
            r, p, y = self._rpy_deg
            vals = [r - self._r_off, p - self._p_off, y - self._y_off]

        # apply order and sign
        ordered = [vals[i] * self._axis_signs[j] for j, i in enumerate(self._axis_order)]
        return tuple(_wrap_angle_deg(v) for v in ordered)


    def get_quat(self) -> Optional[Tuple[float, float, float, float]]:
        """Return (w, x, y, z) as latest from the stream. (Offsets are not applied to quat.)"""
        with self._lock:
            return self._quat

    def get_raw(self):
        """Return (acc, gyro, mag) as tuples if available; else None."""
        with self._lock:
            return self._raw

    def set_rpy_offset(self, roll0: float = 0.0, pitch0: float = 0.0, yaw0: float = 0.0):
        """Set fixed offsets (deg) that will be subtracted in get_rpy_deg()."""
        with self._lock:
            self._r_off, self._p_off, self._y_off = float(roll0), float(pitch0), float(yaw0)
            
    def set_rpy_offset(self, roll0: float = 0.0, pitch0: float = 0.0, yaw0: float = 0.0):
        """Set fixed offsets (deg) that will be subtracted in get_rpy_deg()."""
        with self._lock:
            self._r_off, self._p_off, self._y_off = float(roll0), float(pitch0), float(yaw0)
            self._have_offset = True

    def zero_current_rpy(self):
        """Use the current raw RPY as zero so future get_rpy_deg() returns ~0,0,0 now."""
        with self._lock:
            if self._rpy_deg is None:
                return
            r, p, y = self._rpy_deg  # raw stored values (no offset applied here)
            self._r_off, self._p_off, self._y_off = r, p, y
            self._have_offset = True



    def get_rpy_normalized(self):
        """Normalized [-1,1] with mapping/inversion."""
        if not self._have_offset:
            raise RuntimeError("Call zero_current_rpy() or set_rpy_offset(...) first.")
        rpy = self.get_rpy_deg()
        if rpy is None:
            raise RuntimeError("No RPY data yet.")
        def _nz(v): 
            v = max(-180.0, min(180.0, v))
            return v / 180.0
        return tuple(_nz(v) for v in rpy)
    
    
    def get_rpy_normalized_withrobot_pose(self,r= 175.3799773, p= -3.9643635, y= 89.8886347):
        """Return normalized [-1,1] RPY where:
        (IMU_read - imu_offset [+ mapping/inversion]) + robot_start_pose.
        Robot start pose is in degrees.
        """
        if not self._have_offset:
            raise RuntimeError("Call zero_current_rpy() or set_rpy_offset(...) first.")

        # 1) This already returns (r,p,y) with your IMU offsets applied and axes/signs mapped,
        #    wrapped to [-180, 180). See get_rpy_deg().
        rpy = self.get_rpy_deg()
        if rpy is None:
            raise RuntimeError("No RPY data yet.")

        # 2) Robot's actual start pose (degrees).
        robot_start_deg = (r, p, y)

        # 3) Add robot start, then wrap back into [-180, 180).
        r = _wrap_angle_deg(rpy[0] + robot_start_deg[0])
        p = _wrap_angle_deg(rpy[1] + robot_start_deg[1])
        y = _wrap_angle_deg(rpy[2] + robot_start_deg[2])

        # 4) Normalize to [-1, 1]
        def _nz(v):
            # clamp just in case, then scale
            v = max(-180.0, min(180.0, v))
            return v / 180.0

        return (_nz(r), _nz(p), _nz(y))


    def wait_for_first_sample(self, timeout_s: float = 2.0) -> bool:
        """Block up to timeout_s until first RPY (or quat/raw) arrives. Returns True if ready."""
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            with self._lock:
                if self._rpy_deg is not None or self._quat is not None or self._raw is not None:
                    return True
            time.sleep(0.01)
        return False
        # ---------- listener thread ----------

    def _run(self):
        buf = bytearray(65507)
        view = memoryview(buf)
        while self._running:
            try:
                n, _ = self._sock.recvfrom_into(view)
            except socket.timeout:
                continue
            except OSError:
                break

            if n <= 0:
                continue
            try:
                txt = view[:n].tobytes().decode("utf-8", errors="ignore")
            except Exception:
                continue

            for line in txt.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue

                t_us = int(msg.get("t_us", int(time.time()*1e6)))

                if "rpy_deg" in msg:
                    r, p, y = msg["rpy_deg"]
                    # optional EMA smoothing
                    if self.smooth_alpha > 0.0:
                        a = self.smooth_alpha
                        self._ema_r = r if self._ema_r is None else (a*self._ema_r + (1-a)*r)
                        self._ema_p = p if self._ema_p is None else (a*self._ema_p + (1-a)*p)
                        self._ema_y = y if self._ema_y is None else (a*self._ema_y + (1-a)*y)
                        r, p, y = self._ema_r, self._ema_p, self._ema_y

                    with self._lock:
                        self._rpy_deg = (float(r), float(p), float(y))
                        self._t_us    = t_us

                elif "q" in msg:
                    w, x, y, z = msg["q"]
                    with self._lock:
                        self._quat = (float(w), float(x), float(y), float(z))
                        self._t_us = t_us

                else:
                    # raw-only or combo
                    acc = tuple(msg.get("acc") or ())
                    gyr = tuple(msg.get("gyro") or ())
                    mag = tuple(msg.get("mag") or ())
                    if len(acc)==3 or len(gyr)==3 or len(mag)==3:
                        with self._lock:
                            self._raw = (acc if len(acc)==3 else None,
                                         gyr if len(gyr)==3 else None,
                                         mag if len(mag)==3 else None)
                            self._t_us = t_us
        # end loop
