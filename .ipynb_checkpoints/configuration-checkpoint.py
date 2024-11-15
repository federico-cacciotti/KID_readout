###############################################################################
#                                                                             #
#                  CONFIGURATION FILE FOR THE ROACH2 READOUT                  #
#                                  Version 0.1                                #
#                                                                             #
###############################################################################

from pathlib import Path

####  FREQUENCY SWEEP PARAMETERS  ####
VNA_STEP = 1.25e3      # VNA sweep resolution in Hz, values lower than 1.25kHz are not allowed
TARGET_STEP = 1.25e3   # Target sweep resolution in Hz, values lower than 1.25kHz are not allowed
TARGET_HSPAN = 150.0e3 # Target sweep half span in Hz, sweep goes from [f_0-sweep_span, f_0+sweep_span]
TARGET_OFFSET = 0.0    # Target sweep frequency offset in Hz. It adds an offset to each tone frequency

# Arduino valiable attenuators values
ATT_RFOUT = 15.0  # dB
ATT_RFIN  = 0.0   # dB

OFF_RESONANCE_TONES = [100.0, 300.0, 450,0]       # list of frequencies in MHz to append at the end of a Target sweep, leave empty for no tones
NUMBER_OF_TEST_TONES = 400     # number of the test comb tones

LO = 290.0                        # local oscillator frequency in MHz
MIXER_CONST = 2.                  # LO multiplier for the mixer. Set this to 1 for MISTRAL-like ROACH, 2 for OLIMPO and COSMO-like ROACH
ROACH_IP = "192.168.41.38"        # this is the IP address of the ROACH. Verify it with $arp0
DATA_SOCKET = "enp1s0f2"          # data socket port (not the PPC socket port)
ACCUMULATION_LENGTH = 2**20       # 2**16: 3906.25Hz (not stable), 2**17: 1953.125Hz, 2**20: 244.14Hz, 2**21: 144.07Hz, 2**22: 61.014Hz
ADC_MAX_AMPLITUDE = 2**31-1       # maximum amplitude of the ADC
NUMPY_RANDOM_SEED = 23578         # seed of the numpy.random module
WAVEMAX = 1.1543e-5 *2            # maximum amplitude value allowed by summing up all the tones amplitudes
DAC_SAMPLING_FREQUENCY = 512.0e6  # sampling frequency of the DAC in Hz
FPGA_SAMPLING_FREQUENCY = 256.0e6 # sampling frequency of the FPGA in Hz
FFT_LENGTH = 1024                 # Fast Fourier Transform length
LUT_BUFFER_LENGTH = 2**21         # length of the LUT buffer
DDS_SHIFT = 318                   # a specific number that changes with the firmware (it was 305)
READOUT_BANDWIDTH = 512.0e6       # total readout bandwidth in Hz

UDP_PACKET_LENGHT = 8234          # UDP packet length in bytes
UDP_PACKET_HEADER = 42            # UDP packet header in bytes



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
skip_tones_attenuation = 0     # sets no attenuation for the first number of tones defined by this parameter
baseline_attenuation = -44.8   # power lever for a flat comb in dBm (-44.8 is the highest value for the MISTRAL ROACH)