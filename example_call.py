from imu_client import ImuUdpClient
import time

imu = ImuUdpClient(smooth_alpha=0.0)

# ensure we have data before zeroing or normalizing
imu.wait_for_first_sample(timeout_s=2.0)

# imu.swap_axes("roll", "pitch")  # swap roll & pitch
# imu.invert_axis("yaw")          # flip yaw sign

# set your zero **to the current pose** (this is what you want)
imu.zero_current_rpy()

try:
    while True:
        rpy  = imu.get_rpy_deg()
        norm = imu.get_rpy_normalized()  # requires offset (we just set it)
        if rpy is not None:
            roll, pitch, yaw = rpy
            print(f"RPY [deg]: roll={roll:.2f}, pitch={pitch:.2f}, yaw={yaw:.2f}")
            nr, np_, ny = norm
            print(f"RPY [norm]: roll={nr:.2f}, pitch={np_:.2f}, yaw={ny:.2f}")
        time.sleep(0.05)
except KeyboardInterrupt:
    pass
finally:
    imu.close()
