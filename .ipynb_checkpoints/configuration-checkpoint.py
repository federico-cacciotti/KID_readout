###############################################################################
#                                                                             #
#                  CONFIGURATION FILE FOR THE ROACH2 READOUT                  #
#                                  Version 0.1                                #
#                                                                             #
###############################################################################

from pathlib import Path

################  CHANGE PARAMETERS HERE  #################
####  FREQUENCY SWEEP PARAMETERS  ####
vna_step = 1.25e3    # VNA sweep resolution in Hz, values lower than 1.25kHz are not allowed
sweep_step = 1.25e3  # Target sweep resolution in Hz, values lower than 1.25kHz are not allowed
sweep_span = 15.0e3  # Target sweep half span in Hz, sweep goes from [f_0-sweep_span, f_0+sweep_span]
sweep_offset = 0.0   # Target sweep frequency offset. It adds an offset to each tone frequency

# Arduino valiable attenuators values
att_RFOUT = 18.0  # dB
att_RFIN  = 10.0  # dB

OFF_RESONANCE_TONES = [100.0, 300.0, 450,0]       # list of frequencies in MHz to append at the end of a Target sweep, leave empty for no tones
skip_tones_attenuation = 0     # sets no attenuation for the first number of tones defined by this parameter
baseline_attenuation = -44.8   # power lever for a flat comb in dBm (-44.8 is the highest value for the MISTRAL ROACH)

LO = 453.5                     # local oscillator frequency in MHz
mixer_const = 2.               # LO multiplier for the mixer. Set this to 1 for MISTRAL-like ROACH, 2 for OLIMPO and COSMO-like ROACH
roach_ip = '192.168.41.38'     # this is the IP address of the ROACH. Verify it with $arp0
data_socket = Path("enp1s0f2") # data socket port (not the PPC socket port)

####  PATHS  ####
datadir = Path("/home/mew/data/data_logger/log_kids/")  # directory where dirfiles are saved
setupdir = Path("/home/mew/data/setup/kids/")           # directory where array configurations are saved 
path_configuration = Path("/home/mew/data/setup/kids/sweeps/target/current/") # default configuration file. It is the current folder, with the latest array configuration.
transfer_function_file = Path("/src/roach2_readout/transfunc_polyfit_coefficients.npy") # transfer function file
log_dir = Path("/src/roach2_readout/client_logs/")               # directory where the client log file is saved
path_format = Path("/src/roach2_readout/format_files")           # directory containing the format files for the dirfile standard
bitstream = Path("/src/roach2_readout/benchmark_firmware.fpg")   # October 2017 from Sam Gordon

# DEPRECATED
#valon_port = Path("/dev/ttyUSB")   # port for the valon
#arduino_port = Path("/dev/ttyACM") # port for the Arduino variable attenuator
folder_frequencies = Path("/home/mew/data/setup/kids/sweeps")