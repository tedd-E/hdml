#!/usr/bin/python3

'''
A simple power reading demo for powermeter
'''
import os
import time

from powermeter import PowerMeter

PWR_FILE = "./power.txt"
MEASURE_TIME = 10
MSG = "Collecting power measurements for {} seconds...\r\n".format(
        MEASURE_TIME)
MSG += "Check {} for detailed traces afterwards.".format(PWR_FILE)

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
    pm = PowerMeter(PWR_FILE)
    pm.run(pwr_callback)

    print(MSG)
    time.sleep(MEASURE_TIME)

    pm.stop()

    for p in pwr_callback.pwr_data:
        print(p)

if __name__ == '__main__':
    main()