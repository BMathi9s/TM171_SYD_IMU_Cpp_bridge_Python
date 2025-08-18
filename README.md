# TM171_SYD_IMU_Cpp_bridge_Python
This code reads the imu using the c++ library and sent it a python scripts to be used

! Make sure to have a windows device with the imu app from 
https://www.syd-dynamics.com/download-center/

set the right Hz and enable the right output

Also if u wish to re-compile the code to output the right output for you : 


// === OUTPUT TO UDP TOGGLES === in the 
#define ENABLE_OUT_RPY    1
#define ENABLE_OUT_QUAT   0
#define ENABLE_OUT_RAW    0
#define ENABLE_OUT_COMBO  0



![alt text](<IMU_settings.png>)



Windows - > 

imu_udp_bridge.exe 
-> this .exe include a list port

Linux -> 

scan port 

sudo chmod a+rw /dev/ttyUSB0 #that port



then -> 

to start the listening -> receiver_imu.py



tag generator 
https://chaitanyantr.github.io/apriltag.html







