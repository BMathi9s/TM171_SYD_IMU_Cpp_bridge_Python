// imu_udp_bridge_linux.cpp — continuous-stream reader for TM171 (Linux)
// Build:
//   g++ -O2 -std=gnu++17 -I EasyProfile imu_udp_bridge_linux.cpp EasyProfile/*.cpp -o imu_udp_bridge
// Run:
//   ./imu_udp_bridge                 # interactive → type /dev/ttyUSB0 or 'list'
//   ./imu_udp_bridge /dev/ttyACM0   # direct open
//
// If you get a permissions error opening the port, do (each boot):
//   sudo chmod a+rw /dev/ttyUSB0
//
// Notes:
// - UDP out goes to 127.0.0.1:8765 (same as Windows version).
// - Baud default 115200 (change BAUD below if needed).
// - Port scan lists ttyUSB/ttyACM/ttyS. The program probes to verify they open.

#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <unistd.h>
#include <fcntl.h>
#include <termios.h>
#include <glob.h>
#include <errno.h>

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <iostream>
#include <chrono>
#include <thread>

// EasyProfile
#include "EasyProfile/EasyObjectDictionary.h"
#include "EasyProfile/EasyProfile.h"

// ================== CONFIG ==================
static const char*  DEFAULT_PORT = "/dev/ttyUSB0";
static const unsigned BAUD       = 115200;      // must match device UART setting
static const int     UDP_PORT    = 8765;        // local UDP out (127.0.0.1)
// Output channels (same flags as your Windows build)
#define ENABLE_OUT_RPY    1
#define ENABLE_OUT_QUAT   0
#define ENABLE_OUT_RAW    0
#define ENABLE_OUT_COMBO  0
// ============================================

static inline long long now_ms() {
  using clk = std::chrono::steady_clock;
  return std::chrono::duration_cast<std::chrono::milliseconds>(clk::now().time_since_epoch()).count();
}
static inline long long now_us() {
  using clk = std::chrono::steady_clock;
  return std::chrono::duration_cast<std::chrono::microseconds>(clk::now().time_since_epoch()).count();
}
static inline void tiny_sleep() {
  std::this_thread::sleep_for(std::chrono::milliseconds(0));
}

// Map plain integer baud to termios speed_t (extend if you need more)
static speed_t baud_to_speed(unsigned baud) {
  switch (baud) {
    case 9600:   return B9600;
    case 19200:  return B19200;
    case 38400:  return B38400;
    case 57600:  return B57600;
    case 115200: return B115200;
    // Add cases for higher bauds if needed (or implement termios2 for custom rates)
    default:     return B115200;
  }
}

// Open and configure a POSIX serial port, return fd or -1 on error.
static int open_serial(const char* dev, unsigned baud) {
  int fd = ::open(dev, O_RDWR | O_NOCTTY | O_NONBLOCK);
  if (fd < 0) {
    if (errno == EACCES || errno == EPERM) {
      std::fprintf(stderr,
        "Permission denied opening %s. Try:\n  sudo chmod a+rw %s\n",
        dev, dev);
    } else {
      std::perror("open");
      std::fprintf(stderr, "Failed opening %s (errno=%d)\n", dev, errno);
    }
    return -1;
  }

  termios tio{};
  if (tcgetattr(fd, &tio) != 0) {
    std::perror("tcgetattr");
    ::close(fd);
    return -1;
  }

  // Raw 8N1, no flow control
  cfmakeraw(&tio);
  tio.c_cflag |= (CLOCAL | CREAD);
  tio.c_cflag &= ~CSTOPB;     // 1 stop bit
  tio.c_cflag &= ~PARENB;     // no parity
  tio.c_cflag &= ~CRTSCTS;    // no HW flow control
  tio.c_cflag &= ~CSIZE;  tio.c_cflag |= CS8; // 8 data bits

  speed_t spd = baud_to_speed(baud);
  cfsetispeed(&tio, spd);
  cfsetospeed(&tio, spd);

  // Non-blocking-ish: return promptly with whatever's available; short inter-byte timeout
  tio.c_cc[VMIN]  = 0;  // 0 = return immediately with any available data
  tio.c_cc[VTIME] = 1;  // 0.1s overall read timeout (tenths of a second)

  if (tcsetattr(fd, TCSANOW, &tio) != 0) {
    std::perror("tcsetattr");
    ::close(fd);
    return -1;
  }

  tcflush(fd, TCIOFLUSH);
  // clear O_NONBLOCK for normal reads if you like; we keep it with VMIN/VTIME
  return fd;
}

// Scan common Linux serial device patterns and keep the ones that can open
static std::vector<std::string> scan_ports() {
  std::vector<std::string> out;
  auto glob_add = [&](const char* pattern){
    glob_t g{};
    if (glob(pattern, 0, nullptr, &g) == 0) {
      for (size_t i = 0; i < g.gl_pathc; ++i) out.emplace_back(g.gl_pathv[i]);
    }
    globfree(&g);
  };
  glob_add("/dev/ttyUSB*");
  glob_add("/dev/ttyACM*");
  glob_add("/dev/ttyS*");

  std::vector<std::string> ok;
  for (auto& p : out) {
    int fd = open_serial(p.c_str(), BAUD);
    if (fd >= 0) { ok.push_back(p); ::close(fd); }
  }
  return ok;
}

int main(int argc, char** argv) {
  // === UDP socket (127.0.0.1:UDP_PORT) ===
  int sock = ::socket(AF_INET, SOCK_DGRAM, 0);
  if (sock < 0) { std::perror("socket"); return 1; }
  sockaddr_in dst{}; dst.sin_family = AF_INET; dst.sin_port = htons(UDP_PORT);
  dst.sin_addr.s_addr = htonl(INADDR_LOOPBACK); // 127.0.0.1

  // === EasyProfile ===
  EasyObjectDictionary eOD;
  EasyProfile eP(&eOD);

  // === Choose serial port ===
  std::string port = DEFAULT_PORT;
  if (argc >= 2 && argv[1] && argv[1][0]) port = argv[1];

  int ser_fd = -1;
  for (;;) {
    std::cout << "\nEnter serial port (default " << port
              << "), or type 'list' to scan:\n> ";
    std::string in;
    std::getline(std::cin, in);

    if (!in.empty()) {
      if (in == "list" || in == "LIST") {
        auto ports = scan_ports();
        if (ports.empty()) {
          std::cout << "No serial ports detected (or accessible).\n"
                       "If your device is plugged in, try:\n"
                       "  ls /dev/ttyUSB* /dev/ttyACM*\n"
                       "  sudo chmod a+rw /dev/ttyUSB0\n";
        } else {
          std::cout << "Detected ports: ";
          for (size_t i = 0; i < ports.size(); ++i)
            std::cout << ports[i] << (i + 1 < ports.size() ? ", " : "\n");
        }
        continue;
      } else {
        port = in; // user entered a path like /dev/ttyUSB0
      }
    }

    ser_fd = open_serial(port.c_str(), BAUD);
    if (ser_fd >= 0) {
      std::printf("Opened %s @ %u\n", port.c_str(), BAUD);
      break;
    }

    std::fprintf(stderr,
      "Failed to open %s @ %u. If this is a permission issue, run:\n"
      "  sudo chmod a+rw %s\n", port.c_str(), BAUD, port.c_str());

    std::cout << "Try another port? (y/n): ";
    std::string yn; std::getline(std::cin, yn);
    if (!yn.empty() && (yn[0] == 'n' || yn[0] == 'N')) return 1;
  }

  // === Streaming loop ===
  char rxbuf[1024];
  uint64_t pkt_count = 0;
  long long last_log = now_ms();

  for (;;) {
    // POSIX read with VMIN/VTIME behavior (see open_serial)
    ssize_t rd = ::read(ser_fd, rxbuf, sizeof(rxbuf));
    if (rd > 0) {
      Ep_Header header;
      if (EP_SUCC_ == eP.On_RecvPkg(rxbuf, (int)rd, &header)) {
        switch (header.cmd) {
          case EP_CMD_RPY_: {
            Ep_RPY rpy;
            if (EP_SUCC_ == eOD.Read_Ep_RPY(&rpy)) {
              auto scale_if_needed = [](float v){
                return (std::fabs(v) > 720.f ? v * 1e-2f : v);
              };
              float roll  = scale_if_needed(rpy.roll);
              float pitch = scale_if_needed(rpy.pitch);
              float yaw   = scale_if_needed(rpy.yaw);
              long long tus = now_us();
              #if ENABLE_OUT_RPY
              char msg[192];
              int m = std::snprintf(msg, sizeof(msg),
                  "{\"t_us\":%lld,\"rpy_deg\":[%.4f,%.4f,%.4f]}\n",
                  tus, (double)roll, (double)pitch, (double)yaw);
              sendto(sock, msg, m, 0, (sockaddr*)&dst, sizeof(dst));
              #endif
              pkt_count++;
            }
          } break;

          case EP_CMD_Q_S1_E_: {
            Ep_Q_s1_e q;
            if (EP_SUCC_ == eOD.Read_Ep_Q_s1_e(&q)) {
              long long tus = now_us();
              #if ENABLE_OUT_QUAT
              char msg[192];
              int m = std::snprintf(msg, sizeof(msg),
                  "{\"t_us\":%lld,\"q\":[%.9f,%.9f,%.9f,%.9f]}\n",
                  tus, (double)q.q[0], (double)q.q[1],
                  (double)q.q[2], (double)q.q[3]);
              sendto(sock, msg, m, 0, (sockaddr*)&dst, sizeof(dst));
              #endif
              pkt_count++;
            }
          } break;

          case EP_CMD_Raw_GYRO_ACC_MAG_: {
            Ep_Raw_GyroAccMag r;
            if (EP_SUCC_ == eOD.Read_Ep_Raw_GyroAccMag(&r)) {
              long long tus = now_us();
              #if ENABLE_OUT_RAW
              char msg[256];
              int m = std::snprintf(msg, sizeof(msg),
                  "{\"t_us\":%lld,\"acc\":[%.6f,%.6f,%.6f],"
                  "\"gyro\":[%.6f,%.6f,%.6f],\"mag\":[%.6f,%.6f,%.6f]}\n",
                  tus, (double)r.acc[0], (double)r.acc[1], (double)r.acc[2],
                  (double)r.gyro[0], (double)r.gyro[1], (double)r.gyro[2],
                  (double)r.mag[0],  (double)r.mag[1],  (double)r.mag[2]);
              sendto(sock, msg, m, 0, (sockaddr*)&dst, sizeof(dst));
              #endif
              pkt_count++;
            }
          } break;

          case EP_CMD_COMBO_: {
            Ep_Combo c;
            if (EP_SUCC_ == eOD.Read_Ep_Combo(&c)) {
              long long tus = now_us();
              #if ENABLE_OUT_COMBO
              char msg[512];
              int m = std::snprintf(msg, sizeof(msg),
                  "{\"t_us\":%lld,"
                  "\"q\":[%.7f,%.7f,%.7f,%.7f],"
                  "\"acc\":[%.6f,%.6f,%.6f],"
                  "\"gyro\":[%.6f,%.6f,%.6f],"
                  "\"mag\":[%.6f,%.6f,%.6f],\"qos\":%u}\n",
                  tus,
                  (double)(c.q1*1e-7f),(double)(c.q2*1e-7f),
                  (double)(c.q3*1e-7f),(double)(c.q4*1e-7f),
                  (double)(c.ax*1e-5f),(double)(c.ay*1e-5f),(double)(c.az*1e-5f),
                  (double)(c.wx*1e-5f),(double)(c.wy*1e-5f),(double)(c.wz*1e-5f),
                  (double)(c.mx*1e-3f),(double)(c.my*1e-3f),(double)(c.mz*1e-3f),
                  (unsigned)c.sysState.bits.qos);
              sendto(sock, msg, m, 0, (sockaddr*)&dst, sizeof(dst));
              #endif
              pkt_count++;
            }
          } break;

          default: break;
        }
      }
    }

    long long now = now_ms();
    if (now - last_log >= 1000) {
      std::fprintf(stderr, "rate: %llu Hz\n", (unsigned long long)pkt_count);
      pkt_count = 0;
      last_log = now;
    }

    tiny_sleep();
  }

  // (unreached) cleanup
  // close(ser_fd); close(sock);
  return 0;
}
