from pathlib import Path

LO = 453.5 #
locroach_ip = '192.168.41.38' #ip of the Roach. Veal oscillator frequency in MHz
mixer_const = 2. #The mixer requires twice the intended LO. Set this to 1 for MISTRAL, 2 for OLIMPO and COSMO

vna_step = 1.25e3
sweep_step = 1.25e3 #step for the target sweep in Hz. Min=1.25 kHz. 
sweep_span = 15.0e3 #half span of the target sweep i.e. it goes from target-span to target+span
sweep_offset = 0.0 #frequency offset for the target sweep.

roach_ip = '192.168.41.38' #ip of the Roach. Verify it with $arp0/home/mew/data/setup/kids/sweeps/target/20230822_102833

data_socket = Path("enp1s0f2") #data socket. NOT THE PPC SOCKET/home/mew/data/setup/kids/sweeps/vna/20231003_GP2v2_140mK_1

# default arduino attenuators values
att_RFOUT = 18.0 #dB
att_RFIN = 10.0 #dB

skip_tones_attenuation = 0 #It will not calculate the attenuations for the first 5 target freqs in the target_freqs.dat file. Set to 0 to 

#valon_port = Path("/dev/ttyUSB") #port for the valon. /home/mew/data/setup/kids/sweeps/target/20230822_102833
#arduino_port = Path("/dev/ttyACM") #Port for the Arduino variable attenuator

baseline_attenuation = -44.8 #dBm --> this is the max power level for a flat comb

datadir = Path("/home/mew/data/data_logger/log_kids/") #directory where dirfiles are saved
setupdir = Path("/home/mew/data/setup/kids/") #directory where array conf/home/mew/data/setup/kids/sweeps/vna/20231003_GP2v2_140mK_1igurations are saved 
path_configuration = Path("/home/mew/data/setup/kids/sweeps/target/current/") #default configuration file. It is the current folder, with the latest array configuration.
transfer_function_file = Path("/src/roach2_readout/transfunc_polyfit_coefficients.npy") #transfer function file
folder_frequencies = Path("/src/parameters/current")#/home/mistral/src/parameters/OLIMPO_150_480_CURRENT/") #update to MISTRAL and not olimpo. Serve la cartella CURRENT!
log_dir = Path("/src/roach2_readout/client_logs/")
path_format = Path("/src/roach2_readout/format_files")

