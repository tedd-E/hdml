#!/usr/bin/python3

'''
A simple power reading demo for powermeter
'''
import os
import time
import socket
from powermeter import PowerMeter

UDP_IP = "169.254.200.28" # local IP
UDP_PORT = 30000
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

"""
On the other RPi, use the following script to wrap the workload to be measured:
The raw power log will be saved in file_name.txt.

UDP_IP = "169.254.200.28" # local IP
UDP_PORT = 30000
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

sock.sendto(b's,file_name', (UDP_IP, UDP_PORT))
st = time.time()
WORKLOAD_TO_RUN
ed = time.time()
sock.sendto(b't,', (UDP_IP, UDP_PORT))
print('execution time: {} s'.format(ed - st))
"""

def pwr_callback(pwr):
    '''
    Callback function that reads power from module.
    This function is responsible for recording time stamps
    that receive the power measurements.

    Args:
        pwr (float): the power measurement in W.

    Attributes:
        pwr_callback.start_time (float): the time that first data comes in, seconds
        pwr_callback.pwr_data: [time stamp (s), power (W)]
    '''
    if pwr_callback.start_time is None:
        pwr_callback.start_time = time.time()

    pwr_callback.pwr_data.append(
        [float(time.time() - pwr_callback.start_time), pwr]
    )

pwr_callback.pwr_data = []
pwr_callback.start_time = None


def main():
    '''
    main function
    Start measurement for 10s, save traces to PWR_FILE,
    and return all power values in pwr_callback.pwr_data.
    '''
    while True:
        # wait for the signal
        data, addr = sock.recvfrom(20)
        print(data)
        data = str(data, 'UTF-8').split(',')
        if data[0] == 's': # start power measure
            filename = data[1] + '.txt'
            pm = PowerMeter(filename)
            pm.run(pwr_callback)
        elif data[0] == 't': # stop power measure
            pm.stop()
        else:
            print('Invalid signal!')

if __name__ == '__main__':
    main()