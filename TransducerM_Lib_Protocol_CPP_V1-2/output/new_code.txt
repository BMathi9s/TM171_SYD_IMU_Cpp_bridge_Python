// imu_udp_bridge.cpp — continuous-stream reader for TM171
// Build (MinGW/TDM-GCC or similar):
//   g++ -O2 -std=gnu++17 -I EasyProfile imu_udp_bridge.cpp EasyProfile/*.cpp -o imu_udp_bridge.exe -lws2_32

// Sockets FIRST
#include <winsock2.h>
#include <ws2tcpip.h>
// Then Windows (for serial)
#include <windows.h>

#include <cstdio>
#include <cstdint>
#include <chrono>
#include <cstring>
#include <cmath>   // for std::fabs

// EasyProfile
#include "EasyProfile/EasyObjectDictionary.h"
#include "EasyProfile/EasyProfile.h"

#ifdef _MSC_VER
#pragma comment(lib, "Ws2_32.lib")
#endif


#define ENABLE_OUT_RPY    1
#define ENABLE_OUT_QUAT   0
#define ENABLE_OUT_RAW    0
#define ENABLE_OUT_COMBO  0

// ================== CONFIG ==================
static const char*  COM_PORT = "COM7";      // e.g., "COM7" or "\\\\.\\COM7"
static const DWORD  BAUD     = 115200;      // must match the app's UART setting
static const int    UDP_PORT = 8765;        // local UDP out
// ============================================

static inline long long now_ms() {
    using clk = std::chrono::steady_clock;
    return std::chrono::duration_cast<std::chrono::milliseconds>(clk::now().time_since_epoch()).count();
}
static inline long long now_us() {
    using clk = std::chrono::steady_clock;
    return std::chrono::duration_cast<std::chrono::microseconds>(clk::now().time_since_epoch()).count();
}

static void print_last_error(const char* where) {
    DWORD e = GetLastError();
    LPVOID lpMsg = nullptr;
    FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                   NULL, e, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                   (LPSTR)&lpMsg, 0, NULL);
    std::fprintf(stderr, "%s: GetLastError=%lu (%s)\n", where, (unsigned long)e,
                 lpMsg ? (char*)lpMsg : "unknown");
    if (lpMsg) LocalFree(lpMsg);
}

static HANDLE open_serial(const char* comPort, DWORD baud) {
    char path[64];
    if (std::strncmp(comPort, "\\\\.\\" , 4) == 0)
        std::snprintf(path, sizeof(path), "%s", comPort);
    else
        std::snprintf(path, sizeof(path), "\\\\.\\" "%s", comPort);

    std::printf("Opening serial: %s @ %lu\n", path, (unsigned long)baud);

    HANDLE h = CreateFileA(path, GENERIC_READ | GENERIC_WRITE, 0, nullptr,
                           OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) { print_last_error("CreateFileA"); return h; }

    // 8N1, no flow control
    DCB dcb{}; dcb.DCBlength = sizeof(dcb);
    if (!GetCommState(h, &dcb)) { print_last_error("GetCommState"); CloseHandle(h); return INVALID_HANDLE_VALUE; }
    dcb.BaudRate = baud;
    dcb.ByteSize = 8;
    dcb.Parity   = NOPARITY;
    dcb.StopBits = ONESTOPBIT;
    dcb.fOutxCtsFlow = FALSE; dcb.fOutxDsrFlow = FALSE;
    dcb.fOutX = FALSE; dcb.fInX = FALSE;
    if (!SetCommState(h, &dcb)) { print_last_error("SetCommState"); CloseHandle(h); return INVALID_HANDLE_VALUE; }

    // Fast/non-blocking-ish timeouts (no 20ms stalls)
    COMMTIMEOUTS to{};
    to.ReadIntervalTimeout         = 1;
    to.ReadTotalTimeoutConstant    = 0;
    to.ReadTotalTimeoutMultiplier  = 0;
    to.WriteTotalTimeoutConstant   = 0;
    to.WriteTotalTimeoutMultiplier = 0;
    if (!SetCommTimeouts(h, &to)) { print_last_error("SetCommTimeouts"); CloseHandle(h); return INVALID_HANDLE_VALUE; }

    SetupComm(h, 1<<20, 1<<20);
    PurgeComm(h, PURGE_RXCLEAR | PURGE_TXCLEAR);

    // Some USB-serial chips like DTR/RTS asserted
    EscapeCommFunction(h, SETDTR);
    EscapeCommFunction(h, SETRTS);

    return h;
}

int main() {
    // UDP out
    WSADATA wsa; WSAStartup(MAKEWORD(2,2), &wsa);
    SOCKET sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    sockaddr_in dst{}; dst.sin_family = AF_INET; dst.sin_port = htons(UDP_PORT);
    dst.sin_addr.s_addr = htonl(INADDR_LOOPBACK); // 127.0.0.1

    // EasyProfile
    EasyObjectDictionary eOD;
    EasyProfile eP(&eOD);

    // Serial
    HANDLE ser = open_serial(COM_PORT, BAUD);
    if (ser == INVALID_HANDLE_VALUE) {
        std::fprintf(stderr, "Failed to open %s @ %lu\n", COM_PORT, (unsigned long)BAUD);
        return 1;
    }
    std::printf("Opened %s @ %lu\n", COM_PORT, (unsigned long)BAUD);

    // Continuous-stream read: drain, parse, UDP
    char rxbuf[1024];

    uint64_t pkt_count = 0;
    long long last_log = now_ms();

    for (;;) {
        // Drain the serial input fully each loop
        DWORD errs = 0; COMSTAT cs{};
        ClearCommError(ser, &errs, &cs);
        while (cs.cbInQue > 0) {
            DWORD want = cs.cbInQue > sizeof(rxbuf) ? sizeof(rxbuf) : cs.cbInQue;
            DWORD rd = 0;
            if (!ReadFile(ser, rxbuf, want, &rd, nullptr) || rd == 0) break;

            Ep_Header header;
            if (EP_SUCC_ == eP.On_RecvPkg(rxbuf, (int)rd, &header)) {
                switch (header.cmd) {
                    case EP_CMD_RPY_: {
                        Ep_RPY rpy;
                            if (EP_SUCC_ == eOD.Read_Ep_RPY(&rpy)) {
                                auto scale_if_needed = [](float v){
                                    return (std::fabs(v) > 720.f ? v * 1e-2f : v); // centideg -> deg if needed
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
                                    tus, (double)q.q[0], (double)q.q[1], (double)q.q[2], (double)q.q[3]);
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
                                    "{\"t_us\":%lld,\"acc\":[%.6f,%.6f,%.6f],\"gyro\":[%.6f,%.6f,%.6f],\"mag\":[%.6f,%.6f,%.6f]}\n",
                                    tus, (double)r.acc[0], (double)r.acc[1], (double)r.acc[2],
                                        (double)r.gyro[0],(double)r.gyro[1],(double)r.gyro[2],
                                        (double)r.mag[0], (double)r.mag[1], (double)r.mag[2]);
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
                                    (double)(c.q1*1e-7f),(double)(c.q2*1e-7f),(double)(c.q3*1e-7f),(double)(c.q4*1e-7f),
                                    (double)(c.ax*1e-5f),(double)(c.ay*1e-5f),(double)(c.az*1e-5f),
                                    (double)(c.wx*1e-5f),(double)(c.wy*1e-5f),(double)(c.wz*1e-5f),
                                    (double)(c.mx*1e-3f),(double)(c.my*1e-3f),(double)(c.mz*1e-3f),
                                    (unsigned)c.sysState.bits.qos);
                                sendto(sock, msg, m, 0, (sockaddr*)&dst, sizeof(dst));
                        #endif
                                pkt_count++;
                            }
                        } break;

                    default: break; // ignore other types
                }
            }
            ClearCommError(ser, &errs, &cs); // keep draining
        }

        // Once-per-second rate log (stderr)
        long long now = now_ms();
        if (now - last_log >= 1000) {
            std::fprintf(stderr, "rate: %llu Hz\n", (unsigned long long)pkt_count);
            pkt_count = 0;
            last_log = now;
        }

        Sleep(0); // tiny yield
    }

    // (never reached)
    return 0;
}
