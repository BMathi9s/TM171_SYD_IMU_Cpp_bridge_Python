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

#include <string>
#include <iostream>
#include <vector>


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



static std::vector<std::string> scan_com_ports(int first=1, int last=64, DWORD baud=BAUD) {
    std::vector<std::string> found;
    for (int i = first; i <= last; ++i) {
        std::string p = "COM" + std::to_string(i);
        HANDLE h = open_serial(p.c_str(), baud);
        if (h != INVALID_HANDLE_VALUE) {
            found.push_back(p);
            CloseHandle(h); // just probing; release it
        }
        // On failure we do nothing—port likely not present/in use
    }
    return found;
}


int main(int argc, char** argv) {
    // UDP out (unchanged)
    WSADATA wsa; WSAStartup(MAKEWORD(2,2), &wsa);
    SOCKET sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    sockaddr_in dst{}; dst.sin_family = AF_INET; dst.sin_port = htons(UDP_PORT);
    dst.sin_addr.s_addr = htonl(INADDR_LOOPBACK); // 127.0.0.1

    // EasyProfile (unchanged)
    EasyObjectDictionary eOD;
    EasyProfile eP(&eOD);

    // ===== Choose COM port =====
    std::string com = "COM7"; // default matches your previous hardcoded value
    if (argc >= 2 && argv[1] && argv[1][0]) {
        com = argv[1];
    }

    HANDLE ser = INVALID_HANDLE_VALUE;
    for (;;) {
        std::cout << "\nEnter COM port (default " << com
                  << "), or type 'list' to scan available ports:\n> ";
        std::string in;
        std::getline(std::cin, in);

        if (!in.empty()) {
            if (in == "list" || in == "LIST") {
                auto ports = scan_com_ports();
                if (ports.empty()) {
                    std::cout << "No COM ports detected (or all busy). "
                                 "If your device is plugged in, check drivers or try another USB port.\n";
                } else {
                    std::cout << "Detected ports: ";
                    for (size_t i = 0; i < ports.size(); ++i) {
                        std::cout << ports[i] << (i+1<ports.size()? ", " : "\n");
                    }
                }
                // loop back to prompt
                continue;
            } else {
                com = in; // user supplied something like "COM7" or "\\\\.\\COM7"
            }
        }

        ser = open_serial(com.c_str(), BAUD);
        if (ser != INVALID_HANDLE_VALUE) {
            std::printf("Opened %s @ %lu\n", com.c_str(), (unsigned long)BAUD);
            break;
        }

        std::fprintf(stderr,
            "Failed to open %s @ %lu. Ensure the device is connected and not in use.\n",
            com.c_str(), (unsigned long)BAUD);

        std::cout << "Try another port? (y/n): ";
        std::string yn; std::getline(std::cin, yn);
        if (!yn.empty() && (yn[0]=='n' || yn[0]=='N')) {
            return 1;
        }
        // otherwise loop and prompt again
    }

    // ===== Rest of your original loop (unchanged) =====
    char rxbuf[1024];
    uint64_t pkt_count = 0;
    long long last_log = now_ms();

    for (;;) {
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
            ClearCommError(ser, &errs, &cs);
        }

        long long now = now_ms();
        if (now - last_log >= 1000) {
            std::fprintf(stderr, "rate: %llu Hz\n", (unsigned long long)pkt_count);
            pkt_count = 0;
            last_log = now;
        }

        Sleep(0);
    }

    return 0;
}