#!/usr/bin/python3

# Yeseong Kim, UCSD 2017-2018,
# Reading total power consumption from HIOKI 3334 multimeter
# This is forked from the old serial_measurement script.
import sys
import os
import threading
import serial, select
import time
import signal
from functools import reduce

PORT = "/dev/ttyUSB0"

def get_time_milli():
    return int(time.time() * 1000)

class PowerMeter(object):
    def __init__(self, filename):
        self.filename = filename
        self.loop_thread = None
        self.f = None
        self.callback = None

    def log(self, string):
        if self.f is None:
            print(string)
        else:
            self.f.write(string)
            self.f.write("\n")

    def init_psu(self):
        psu = serial.Serial(PORT, 9600, timeout=0.1)
        psu.flush()
        psu.flushInput()
        psu.flushOutput()

        psu.write(b"\r\n")
        #time.sleep(1)
        #psu.write(b"*IDN?\r\n")
        #psu.write(b":MEASure?\r\n")

        psu.write(b":HEAD OFF\r\n") # no header

        #psu.write(b":DATAout:ITEM 2,0\r\n"); item = "current" # for current
        #psu.write(b":DATAout:ITEM 4,0\r\n"); item = "power" # for power
        #psu.write(b":DATAout:ITEM 1,0\r\n"); item = "volt" # for voltage
        psu.write(b":DATAout:ITEM 35,0\r\n"); item = "volt,curr,pf" # for more details
        #psu.write(b":DATAout:ITEM?\r\n")
        #time.sleep(1)

        #psu.write(b":CURR:RANG 0.1\r\n") # Current range to minimum
        #psu.write(b":CURR:RANG 1.0\r\n") # Current range to minimum

        return psu, item

    # ** Recommend not to use this function as an external function **
    # ** Instead, use run(self) for async running
    def run_in_loop(self):
        # Initialize PSU and header
        psu, item = self.init_psu()
        extra_item_size = item.count(',')
        if extra_item_size > 0:
            item += ",mul"
        self.log("time_stamp," + item)

        # Start of the loop
        self.is_loop_running = True
        bNeedtoMeasure = True
        read_idx = 0
        line = ""
        bTypeError = False
        while self.is_loop_running:
            if bNeedtoMeasure == True:
                psu.write(b":MEAS?\r\n")
                bNeedtoMeasure = False

            try:
                read_char = psu.read()
                read_char = read_char.decode('UTF-8')
            except serial.serialutil.SerialException:
                break
            except (select.error, v):
                break

            try:
                line += "%c" % read_char
            except TypeError:
                bTypeError = True
                continue

            read_idx += 1
            if (read_idx == 12+11*extra_item_size):
                # +00.150E+3 or
                # +0122.9E+0;+058.93E-3;+00.487E+0
                if bTypeError == False:
                    output_str = str(get_time_milli())

                    sublist = line.split(";")
                    vallist = []
                    for subdata in sublist: # Parsing "XXE+YY" format
                        idx = subdata.find("E")
                        exp_part = 10 ** int(subdata[idx+1:])
                        val = float(subdata[:idx]) * exp_part
                        vallist.append(val)
                        output_str += "," + str(val)

                    if extra_item_size > 0:
                        # multiply everything: power
                        mul = reduce(lambda x, y: x*y, vallist)
                        output_str += "," + str(mul)
                        if self.callback is not None:
                            self.callback(mul)

                    self.log(output_str)

                read_idx = 0
                line = ""
                bTypeError = False
                bNeedtoMeasure = True
        # End of the loop

        psu.close()

    # Start asynchronous measurement
    # The created loop can be terminated using stop()
    def run(self, callback=None):
        # check whether the port exists
        if not os.path.exists(PORT):
            raise Exception('Please connect the power meter to host!')

        # Create file
        if self.filename is not None:
            self.f = open(self.filename, "w+")
        else:
            self.f = None
        self.callback = callback

        # Start thread
        self.loop_thread = threading.Thread(target=self.run_in_loop)
        self.loop_thread.start()

    # Finish the created asynchronous measurement
    def stop(self):
        # Stop running thread
        assert(self.loop_thread is not None)
        self.is_loop_running = False
        self.loop_thread.join()

        # Flush file
        if self.f is not None:
            self.f.close()
