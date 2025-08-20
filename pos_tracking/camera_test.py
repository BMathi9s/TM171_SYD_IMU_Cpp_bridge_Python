import pyrealsense2 as rs

def start_rs_60fps():
    wanted = [
        (848, 480, 60),
        (640, 480, 60),
    ]
    pipe = rs.pipeline()
    cfg  = rs.config()

    # Try “good” pairs first
    for (w, h, fps) in wanted:
        try:
            cfg.disable_all_streams()
            cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
            cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
            profile = pipe.start(cfg)
            align = rs.align(rs.stream.color)
            print(f"Started color/depth {w}x{h}@{fps} successfully")
            return pipe, align, profile
        except Exception as e:
            try:
                pipe.stop()
            except Exception:
                pass
            pipe = rs.pipeline()
    # If we got here, enumerate what the camera actually supports to debug
    ctx = rs.context()
    dev = ctx.query_devices()[0]
    sensors = dev.query_sensors()
    print("Available streams:")
    for s in sensors:
        for p in s.get_stream_profiles():
            try:
                v = p.as_video_stream_profile()
                print(v.stream_type(), v.format(), v.width(), v.height(), v.fps())
            except Exception:
                pass
    raise RuntimeError("No 60 FPS color+depth combo available. Check USB3 / bandwidth.")

pipe, align, profile = start_rs_60fps()
