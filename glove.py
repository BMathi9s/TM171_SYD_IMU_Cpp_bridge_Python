import serial
import time
import os
import yaml
import threading
import sys
from typing import List, Dict, Optional

# Sensor mapping: 4 on thumb, 3 per other finger
SENSOR_MAP: Dict[str, List[int]] = {
    "thumb":  [0, 1, 2, 3],
    "index":  [4, 5, 6],
    "middle": [7, 8, 9],
    "ring":   [10, 11, 12],
    "pinky":  [13, 14, 15],
}
FINGER_ORDER = ["thumb", "index", "middle", "ring", "pinky"]
NUM_SENSORS = 16

# ---------- Serial helpers ----------

def parse_line(line: str) -> tuple[int, float]:
    """Parse a single line like '>Joint_X:value' or 'Joint_X:value'."""
    line = line.lstrip('>')  # Remove leading '>' if present
    if not line.startswith("Joint_"):
        raise ValueError(f"Invalid line format: {line!r}")
    try:
        sensor_id = int(line.split('_')[1].split(':')[0])
        value = float(line.split(':')[1])
        return sensor_id, value
    except (ValueError, IndexError):
        raise ValueError(f"Failed to parse line: {line!r}")

def read_sensor_frame(ser: serial.Serial) -> List[float]:
    """Read one complete frame of 16 sensor values."""
    readings = [None] * NUM_SENSORS
    valid_count = 0
    start_time = time.time()
    
    while valid_count < NUM_SENSORS and (time.time() - start_time) < 5.0:  # 5s timeout
        raw = ser.readline().decode(errors="ignore").strip()
        if not raw:
            continue
        try:
            sensor_id, value = parse_line(raw)
            if -1000000 < value < 1000000:  # Filter extreme values (e.g., -2147483392)
                readings[sensor_id] = value
                valid_count += 1
            else:
                readings[sensor_id] = None  # Mark invalid for this frame
                print(f"Warning: Invalid value for sensor {sensor_id}: {value}")
        except ValueError as e:
            print(f"Error parsing line: {e}")
            continue
    if valid_count < NUM_SENSORS:
        print(f"Warning: Incomplete frame, only {valid_count}/{NUM_SENSORS} values received")
    return readings

# ---------- Normalization ----------

def get_normalized_values(raw_values: List[float], mins: List[float], maxs: List[float]) -> Dict[int, Optional[float]]:
    """Normalize raw sensor values to [-1, 1] using min/max calibration."""
    normalized = {}
    for i in range(NUM_SENSORS):
        if raw_values[i] is not None and maxs[i] > mins[i]:  # Valid value and span
            span = maxs[i] - mins[i]
            normalized[i] = 2 * (raw_values[i] - mins[i]) / span - 1
        else:
            normalized[i] = None  # Invalid value or zero span
    return normalized

# ---------- Table display ----------

def print_sensor_table(readings: List[float], mins: List[float] = None, maxs: List[float] = None) -> None:
    """Print a formatted table of sensor values grouped by finger, with optional min/max spans."""
    # Clear terminal for Linux/Unix systems
    os.system('clear')
    
    print("Finger Joint Sensor Values")
    print("=" * 60)
    print(f"{'Finger':<10} {'Sensors':<50}")
    print("-" * 60)
    
    for finger in FINGER_ORDER:
        sensor_ids = SENSOR_MAP[finger]
        values = [f"{readings[i]:6.1f}" if readings[i] is not None else "   N/A" for i in sensor_ids]
        sensor_str = ", ".join(f"s{i}:{val}" for i, val in zip(sensor_ids, values))
        if mins and maxs:
            spans = [f"{(maxs[i] - mins[i]):.1f}" if maxs[i] > -1e308 and mins[i] < 1e308 else "N/A" for i in sensor_ids]
            span_str = ", ".join(f"s{i}Δ:{span}" for i, span in zip(sensor_ids, spans))
            print(f"{finger.capitalize():<10} {sensor_str:<50}")
            print(f"{'':<10} {span_str:<50}")
        else:
            print(f"{finger.capitalize():<10} {sensor_str:<50}")
    print("=" * 60)
    print("Press Ctrl+C to quit, or Enter during calibration to proceed.")

# ---------- Calibration ----------

def calibrate_finger(
    finger: str,
    ser: serial.Serial,
    mins: List[float],
    maxs: List[float],
    duration: float = 10.0
) -> None:
    """Calibrate a single finger by moving it through its full range."""
    idxs = SENSOR_MAP[finger]
    print("\n" + "=" * 60)
    print(f"Calibrating {finger.upper()} — move ALL its joints through full range.")
    print(f"Calibration will run for {duration} seconds or until you press ENTER.")
    print("=" * 60)

    # Waiter thread to capture Enter
    done = threading.Event()
    def waiter():
        input()
        done.set()
    threading.Thread(target=waiter, daemon=True).start()

    start_time = time.time()
    last_print = 0.0
    while not done.is_set() and (time.time() - start_time) < duration:
        vals = read_sensor_frame(ser)
        # Update only this finger's sensors
        for i in idxs:
            v = vals[i]
            if v is not None:  # Skip invalid readings
                mins[i] = min(mins[i], v)
                maxs[i] = max(maxs[i], v)

        # Show current raw values and spans
        now = time.time()
        if now - last_print >= 0.00005:  # Update every 0.5s
            print_sensor_table(vals, mins, maxs)
            last_print = now
        time.sleep(0.1)  # Sample every 100ms

    # Sanity check: any zero-span sensor on this finger?
    zeroes = [i for i in idxs if (maxs[i] - mins[i]) <= 1e-6]
    if zeroes:
        print(f"⚠️  Sensors with ~zero span on {finger}: {zeroes} — consider recalibrating this finger.")

def guided_calibration(ser: serial.Serial) -> tuple[List[float], List[float]]:
    """Perform guided calibration for all fingers."""
    mins = [float("inf")] * NUM_SENSORS
    maxs = [float("-inf")] * NUM_SENSORS

    print("\nStarting GUIDED calibration (finger-by-finger).")
    print("Tip: move slowly to capture extremes; wiggle near endpoints.\n")

    # Check for faulty sensors
    print("Checking for faulty sensors...")
    initial_vals = read_sensor_frame(ser)
    for i in range(NUM_SENSORS):
        if initial_vals[i] is None or abs(initial_vals[i]) > 1000000:
            print(f"Warning: Sensor {i} may be faulty (reading: {initial_vals[i]}). Calibration may be inaccurate.")

    for finger in FINGER_ORDER:
        calibrate_finger(finger, ser, mins, maxs)

    print("\nCalibration complete for all fingers.")
    print("\nFinal spans per sensor:")
    for i in range(NUM_SENSORS):
        span = maxs[i] - mins[i] if maxs[i] > -1e308 and mins[i] < 1e308 else 0.0
        print(f"s{i}: min={mins[i]:.2f}, max={maxs[i]:.2f}, span={span:.2f}")
    return mins, maxs

# ---------- Save calibration ----------

def save_calibration(path: str, mins: List[float], maxs: List[float]) -> None:
    """Save min/max values to a YAML file."""
    calibration = {
        str(i): {
            "min": mins[i] if mins[i] < 1e308 else None,
            "max": maxs[i] if maxs[i] > -1e308 else None,
            "valid": mins[i] < 1e308 and maxs[i] > -1e308
        } for i in range(NUM_SENSORS)
    }
    with open(path, "w") as f:
        yaml.dump(calibration, f, default_flow_style=False)

# ---------- Device detection ----------

def find_arduino_device():
    """Try to auto-detect Arduino device on Linux."""
    common_ports = [
        "/dev/ttyUSB0", "/dev/ttyUSB1", "/dev/ttyUSB2",
        "/dev/ttyACM0", "/dev/ttyACM1", "/dev/ttyACM2",
        "/dev/ttyS0", "/dev/ttyS1"
    ]
    
    print("Searching for Arduino device...")
    for port in common_ports:
        if os.path.exists(port):
            try:
                # Try to open the port briefly to test
                test_ser = serial.Serial(port, 115200, timeout=1)
                test_ser.close()
                print(f"Found potential device at: {port}")
                return port
            except:
                continue
    return None

# ---------- Main ----------

def main():
    # Try to auto-detect serial port, fallback to manual specification
    port = find_arduino_device()
    
    if not port:
        print("Could not auto-detect Arduino device.")
        print("Common Linux serial ports:")
        print("  - /dev/ttyUSB0, /dev/ttyUSB1 (USB-to-serial adapters)")
        print("  - /dev/ttyACM0, /dev/ttyACM1 (Arduino Uno, Nano)")
        print("  - /dev/ttyS0, /dev/ttyS1 (built-in serial ports)")
        print("\nTo find your device, try: ls /dev/tty*")
        port = input("Enter your serial port (e.g., /dev/ttyUSB0): ").strip()
        if not port:
            print("No port specified. Exiting.")
            sys.exit(1)
    
    baud = 115200  # Matches glove_ESPNOWsetup(mac, 115200)
    
    try:
        ser = serial.Serial(port, baud, timeout=1)
        time.sleep(2)  # Wait for serial connection to stabilize
    except Exception as e:
        print(f"❌ Failed to open serial port {port}: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check if the device is connected: ls /dev/tty*")
        print("2. Check permissions: sudo chmod 666 /dev/ttyUSB0")
        print("3. Add user to dialout group: sudo usermod -a -G dialout $USER")
        print("   (then logout and login again)")
        sys.exit(1)

    print(f"✅ Connected to {port} @ {baud} baud. Starting calibration...")

    try:
        # Perform calibration
        mins, maxs = guided_calibration(ser)
        
        # Save calibration to YAML
        yaml_path = "finger_sensor_calibration.yaml"
        try:
            save_calibration(yaml_path, mins, maxs)
            print(f"\n💾 Calibration saved to {yaml_path}")
        except Exception as e:
            print(f"⚠️ Could not save calibration: {e}")

        # Stream normalized values
        print("\nStreaming normalized values in [-1, 1]. Press Ctrl+C to quit.\n")
        while True:
            vals = read_sensor_frame(ser)
            norm = get_normalized_values(vals, mins, maxs)
            norm_list = [norm[i] for i in range(NUM_SENSORS)]  # Convert dict to list for table
            print_sensor_table(norm_list, mins, maxs)
            time.sleep(0.5)  # 0.5s delay
    except KeyboardInterrupt:
        print("\nExiting. Bye!")
    finally:
        ser.close()
        
        
        
        
        

def monitor_hz(ser: serial.Serial, duration: float = 30.0) -> None:
    """Monitor and display the Hz (frequency) of incoming sensor data."""
    print(f"\n{'='*60}")
    print("MONITORING HZ OUTPUT")
    print(f"Duration: {duration} seconds")
    print(f"{'='*60}")
    
    frame_count = 0
    line_count = 0
    start_time = time.time()
    last_display = start_time
    display_interval = 1.0  # Update display every second
    
    # Rolling window for more accurate Hz calculation
    frame_times = []
    line_times = []
    window_size = 10  # Keep last 10 measurements
    
    try:
        while (time.time() - start_time) < duration:
            # Try to read one complete frame
            frame_start = time.time()
            readings = [None] * NUM_SENSORS
            valid_count = 0
            frame_line_count = 0
            
            # Read until we get a complete frame or timeout
            frame_timeout = 2.0
            while valid_count < NUM_SENSORS and (time.time() - frame_start) < frame_timeout:
                try:
                    raw = ser.readline().decode(errors="ignore").strip()
                    if not raw:
                        continue
                    
                    line_times.append(time.time())
                    line_count += 1
                    frame_line_count += 1
                    
                    sensor_id, value = parse_line(raw)
                    if -1000000 < value < 1000000 and readings[sensor_id] is None:
                        readings[sensor_id] = value
                        valid_count += 1
                        
                except ValueError:
                    continue  # Skip invalid lines
            
            if valid_count >= NUM_SENSORS * 0.8:  # At least 80% of sensors read
                frame_times.append(time.time())
                frame_count += 1
            
            # Keep rolling window
            current_time = time.time()
            frame_times = [t for t in frame_times if current_time - t <= window_size]
            line_times = [t for t in line_times if current_time - t <= window_size]
            
            # Display stats every second
            if current_time - last_display >= display_interval:
                elapsed = current_time - start_time
                
                # Calculate Hz rates
                overall_frame_hz = frame_count / elapsed if elapsed > 0 else 0
                overall_line_hz = line_count / elapsed if elapsed > 0 else 0
                
                # Rolling window Hz (more accurate for current rate)
                if len(frame_times) > 1:
                    window_duration = frame_times[-1] - frame_times[0]
                    window_frame_hz = (len(frame_times) - 1) / window_duration if window_duration > 0 else 0
                else:
                    window_frame_hz = 0
                    
                if len(line_times) > 1:
                    window_duration = line_times[-1] - line_times[0]
                    window_line_hz = (len(line_times) - 1) / window_duration if window_duration > 0 else 0
                else:
                    window_line_hz = 0
                
                # Clear and display stats
                os.system('clear')
                print(f"{'='*60}")
                print("GLOVE Hz MONITORING")
                print(f"{'='*60}")
                print(f"Elapsed Time: {elapsed:.1f}s / {duration:.1f}s")
                print(f"")
                print(f"FRAME STATISTICS (complete 16-sensor readings):")
                print(f"  Total Frames:     {frame_count}")
                print(f"  Overall Frame Hz: {overall_frame_hz:.2f} Hz")
                print(f"  Current Frame Hz: {window_frame_hz:.2f} Hz (last {window_size}s)")
                print(f"")
                print(f"LINE STATISTICS (individual sensor readings):")
                print(f"  Total Lines:      {line_count}")
                print(f"  Overall Line Hz:  {overall_line_hz:.2f} Hz")
                print(f"  Current Line Hz:  {window_line_hz:.2f} Hz (last {window_size}s)")
                print(f"")
                print(f"Expected rates:")
                print(f"  - Line Hz should be ~16x Frame Hz (16 sensors per frame)")
                print(f"  - Typical glove rates: 10-100 Hz for frames")
                print(f"")
                print("Press Ctrl+C to stop monitoring...")
                
                last_display = current_time
                
    except KeyboardInterrupt:
        print(f"\n\nMonitoring stopped by user.")
    
    # Final summary
    total_elapsed = time.time() - start_time
    final_frame_hz = frame_count / total_elapsed if total_elapsed > 0 else 0
    final_line_hz = line_count / total_elapsed if total_elapsed > 0 else 0
    
    print(f"\n{'='*60}")
    print("FINAL STATISTICS")
    print(f"{'='*60}")
    print(f"Total Duration:   {total_elapsed:.2f} seconds")
    print(f"Total Frames:     {frame_count}")
    print(f"Total Lines:      {line_count}")
    print(f"Average Frame Hz: {final_frame_hz:.2f} Hz")
    print(f"Average Line Hz:  {final_line_hz:.2f} Hz")
    print(f"Lines per Frame:  {line_count/frame_count:.1f}" if frame_count > 0 else "Lines per Frame:  N/A")


# ---------- Modified Main with Hz Monitor Option ----------

def main__():
    # Try to auto-detect serial port, fallback to manual specification
    port = find_arduino_device()
    
    if not port:
        print("Could not auto-detect Arduino device.")
        print("Common Linux serial ports:")
        print("  - /dev/ttyUSB0, /dev/ttyUSB1 (USB-to-serial adapters)")
        print("  - /dev/ttyACM0, /dev/ttyACM1 (Arduino Uno, Nano)")
        print("  - /dev/ttyS0, /dev/ttyS1 (built-in serial ports)")
        print("\nTo find your device, try: ls /dev/tty*")
        port = input("Enter your serial port (e.g., /dev/ttyUSB0): ").strip()
        if not port:
            print("No port specified. Exiting.")
            sys.exit(1)
    
    baud = 115200  # Matches glove_ESPNOWsetup(mac, 115200)
    
    try:
        ser = serial.Serial(port, baud, timeout=1)
        time.sleep(2)  # Wait for serial connection to stabilize
    except Exception as e:
        print(f"❌ Failed to open serial port {port}: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check if the device is connected: ls /dev/tty*")
        print("2. Check permissions: sudo chmod 666 /dev/ttyUSB0")
        print("3. Add user to dialout group: sudo usermod -a -G dialout $USER")
        print("   (then logout and login again)")
        sys.exit(1)

    print(f"✅ Connected to {port} @ {baud} baud.")
    
    # Menu for operation mode
    print("\nSelect operation mode:")
    print("1. Calibration and monitoring")
    print("2. Hz monitoring only")
    choice = input("Enter choice (1 or 2): ").strip()
    
    try:
        if choice == "2":
            duration = input("Monitor duration in seconds (default 30): ").strip()
            duration = float(duration) if duration else 30.0
            monitor_hz(ser, duration)
        else:
            # Original calibration flow
            print("Starting calibration...")
            mins, maxs = guided_calibration(ser)
            
            # Save calibration to YAML
            yaml_path = "finger_sensor_calibration.yaml"
            try:
                save_calibration(yaml_path, mins, maxs)
                print(f"\n💾 Calibration saved to {yaml_path}")
            except Exception as e:
                print(f"⚠️ Could not save calibration: {e}")

            # Stream normalized values
            print("\nStreaming normalized values in [-1, 1]. Press Ctrl+C to quit.\n")
            while True:
                vals = read_sensor_frame(ser)
                norm = get_normalized_values(vals, mins, maxs)
                norm_list = [norm[i] for i in range(NUM_SENSORS)]  # Convert dict to list for table
                print_sensor_table(norm_list, mins, maxs)
                time.sleep(0.5)  # 0.5s delay
                
    except KeyboardInterrupt:
        print("\nExiting. Bye!")
    finally:
        ser.close()

if __name__ == "__main__":
    main__()
    
    
    
    
    
