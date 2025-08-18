# demo_run.py
import time
import cv2
from TransducerM_Lib_Protocol_CPP.pos_tracking.hand_position_tracker import hand_position_tracker   # <- class name

def main():
    tracker = hand_position_tracker(
        tag_size_m=0.072,
        bake_adjustments_at_capture=True
    )
    tracker.start()

    # NEW: menu (load/create)
    tracker.startup_menu()  # loads if available, otherwise runs the wizard

    last_print = 0.0
    try:
        while True:
            color_bgr, depth_vis = tracker.process(draw_landmarks=True)
            if color_bgr is None:
                continue

            # HUD
            tracker.draw_hud(color_bgr)

            # Show
            cv2.imshow("Color + Hands + World HUD", color_bgr)
            if depth_vis is not None:
                cv2.imshow("Depth (vis)", depth_vis)

            # Print values to terminal at ~10 Hz
            now = time.time()
            if now - last_print > 0.10:
                last_print = now
                nx, ny, nz = tracker.get_normalized_xyz()
                wxyz = tracker.get_world_xyz()
                if wxyz is not None:
                    print(f"World [m]: x={wxyz[0]:+.3f}, y={wxyz[1]:+.3f}, z={wxyz[2]:+.3f} | "
                          f"Norm [-1,1]: nx={nx:+.2f}, ny={ny:+.2f}, nz={nz:+.2f}")

            # Basic hotkeys still work if you want them during runtime
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q')):
                break
            if key in (ord('l'), ord('L')):
                try:
                    tracker.load_calibration()
                    print("[loaded] calibration")
                except FileNotFoundError:
                    print("[load] no calib file found")
            if key in (ord('s'), ord('S')):
                p = tracker.save_calibration()
                print(f"[saved] {p}")

    finally:
        tracker.stop()

if __name__ == "__main__":
    main()
