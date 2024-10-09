import numpy as np
import casperfpga
import sys
import time

firmware = "./mistral_firmware_current.fpg"
ip = "192.168.41.38"

katcp_port = 7147
roach = casperfpga.katcp_fpga.KatcpFpga(ip, timeout=120.)

if roach.is_connected():

    print("Roach connected")
    print("ip="+ str(ip))
    print("port="+str(katcp_port))
    print("Uploading firmware")

    roach.upload_to_ram_and_program(firmware)
    print("done")
print("Testing communication with estimate_fpga_clock:")

clock = roach.estimate_fpga_clock()
print(clock)
print("done")


