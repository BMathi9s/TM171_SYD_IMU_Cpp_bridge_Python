# demo_run.py
import time
import cv2
from euro_hand_position_tracker import hand_position_tracker

def main():
    tracker = hand_position_tracker(
        color_w=848, color_h=480,fps=60,
        tag_size_m=0.072,
        bake_adjustments_at_capture=True
    )
    tracker.start()

    MAIN_WIN = "Main"  # use one window for everything

    # === 1) XYZ startup menu (Load or New) ===
    # Non-blocking, on-screen UI; launches the XYZ wizard if you choose New.
    res_xyz = tracker.startup_menu(window_name=MAIN_WIN)  # returns "loaded" or "created" (or False if cancelled)
    if not res_xyz:
        print("[init] XYZ step cancelled; exiting.")
        tracker.stop()
        return

    # === 2) Hand flexion startup (Load or New) ===
    # If New, it opens the hand wizard; press O (open), P (close), then S (save) and it returns immediately.
    res_hand = tracker.hand_startup_menu(window_name=MAIN_WIN)  # returns "loaded" or "created" (or False if cancelled)
    if not res_hand:
        print("[init] Hand step cancelled; exiting.")
        tracker.stop()
        return

    last_print = 0.0
    try:
        while True:
            color_bgr, depth_vis = tracker.process(draw_landmarks=True)
            if color_bgr is None:
                continue

            tracker.draw_hud(color_bgr)

            # Show (single window)
            cv2.imshow(MAIN_WIN, color_bgr)
            if depth_vis is not None:
                cv2.imshow("Depth (vis)", depth_vis)

            # Print values to terminal at ~10 Hz
            now = time.time()
            if now - last_print > 0.10:
                last_print = now
                nx, ny, nz = tracker.get_normalized_xyz()
                wxyz = tracker.get_world_xyz()

                # Degrees
                index_flex  = tracker.get_flexion_index()
                middle_flex = tracker.get_flexion_middle()
                ring_flex   = tracker.get_flexion_ring()
                pinky_flex  = tracker.get_flexion_pinky()
                thumb_flex  = tracker.get_flexion_thumb()

                # Normalized [-1,1]
                n_index  = tracker.get_normalized_flexion_index()
                n_middle = tracker.get_normalized_flexion_middle()
                n_ring   = tracker.get_normalized_flexion_ring()
                n_pinky  = tracker.get_normalized_flexion_pinky()
                n_thumb  = tracker.get_normalized_flexion_thumb()

                if wxyz is not None:
                    print("looping")
                    # print(f"World [m]: x={wxyz[0]:+.3f}, y={wxyz[1]:+.3f}, z={wxyz[2]:+.3f}")
                    print(f"Norm [-1,1]: nx={nx:+.2f}, ny={ny:+.2f}, nz={nz:+.2f}")

                    # print(f"Index(deg):  MCP={index_flex['MCP']:+.1f}° PIP={index_flex['PIP']:+.1f}° DIP={index_flex['DIP']:+.1f}°")
                    # print(f"Middle(deg): MCP={middle_flex['MCP']:+.1f}° PIP={middle_flex['PIP']:+.1f}° DIP={middle_flex['DIP']:+.1f}°")
                    # print(f"Ring(deg):   MCP={ring_flex['MCP']:+.1f}° PIP={ring_flex['PIP']:+.1f}° DIP={ring_flex['DIP']:+.1f}°")
                    # print(f"Pinky(deg):  MCP={pinky_flex['MCP']:+.1f}° PIP={pinky_flex['PIP']:+.1f}° DIP={pinky_flex['DIP']:+.1f}°")
                    # print(f"Thumb(deg):  CMC={thumb_flex['CMC']:+.1f}° MCP={thumb_flex['MCP']:+.1f}° IP={thumb_flex['IP']:+.1f}°")

                    print(f"Index(norm):  MCP={n_index['MCP']:+.2f} PIP={n_index['PIP']:+.2f} DIP={n_index['DIP']:+.2f}")
                    print(f"Middle(norm): MCP={n_middle['MCP']:+.2f} PIP={n_middle['PIP']:+.2f} DIP={n_middle['DIP']:+.2f}")
                    print(f"Ring(norm):   MCP={n_ring['MCP']:+.2f} PIP={n_ring['PIP']:+.2f} DIP={n_ring['DIP']:+.2f}")
                    print(f"Pinky(norm):  MCP={n_pinky['MCP']:+.2f} PIP={n_pinky['PIP']:+.2f} DIP={n_pinky['DIP']:+.2f}")
                    print(f"Thumb(norm):  CMC={n_thumb['CMC']:+.2f} MCP={n_thumb['MCP']:+.2f} IP={n_thumb['IP']:+.2f}")

            # Hotkeys (optional convenience)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q')):
                break
            if key in (ord('h'), ord('H')):   # re-run hand wizard anytime
                tracker.hand_calibration_wizard(window_name=MAIN_WIN)

    finally:
        tracker.stop()

if __name__ == "__main__":
    main()
