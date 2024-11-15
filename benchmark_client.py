#!/usr/bin/pythonimport valon_synth9 as valon_synth This software is a work in progress. It is a console interface designed 
# to operate the BLAST-TNG ROACH2 firmware. 
#
# Copyright (C) May 23, 2016  Gordon, Sam <sbgordo1@asu.edu Author: Gordon, Sam <sbgordo1@asu.edu>
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import matplotlib, time, struct
import numpy as np
import shutil
#matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import casperfpga 
#import corr
from myQdr import Qdr as myQdr
import types
import logging
import glob  
import os
import sys
import benchmark_valon_synth9 as valon_synth
from socket import *
from scipy import signal
import benchmark_find_kids as fk
import subprocess
from benchmark_pipeline import pipeline
from tqdm import tqdm
from pathlib import Path
import variable_attenuator as vatt

import configuration as conf

version = "0.1"
version_string = "\033[35mBenchmark readout client\nVersion {:s}\033[0m\n".format(version)

np.set_printoptions(threshold=sys.maxsize)

class roachInterface(object):

    def __init__(self):
        self.clearScreen()
        
        # setting up timestring for file naming
        self.timestring = "%04d%02d%02d_%02d%02d%02d" % (time.localtime()[0],time.localtime()[1], time.localtime()[2], time.localtime()[3], time.localtime()[4], time.localtime()[5])
        self.today = "%04d%02d%02d" % (time.localtime()[0],time.localtime()[1], time.localtime()[2])
        string = 'dirfile_'+self.timestring

        # initialize logging file
        log_filename = (conf.log_dir / (self.timestring+".log")).as_posix()
        self.log_filename = log_filename
        rootLogger = logging.getLogger()
        rootLogger.setLevel(logging.INFO)
        fileFormatter = logging.Formatter("[%(asctime)s]:%(levelname)s:%(message)s")
        streamFormatter = logging.Formatter("%(message)s")
        fileHandler = logging.FileHandler(log_filename)
        fileHandler.setFormatter(fileFormatter)
        fileHandler.setLevel(logging.INFO)
        consoleHandler = logging.StreamHandler(sys.stdout)
        consoleHandler.setLevel(logging.INFO)
        consoleHandler.setFormatter(streamFormatter)
        
        if len(rootLogger.handlers) < 1:
            rootLogger.addHandler(consoleHandler)
        
        rootLogger.addHandler(fileHandler)
        logging.info(version_string)

        self.do_transf = False #provvisorio! Da sistemare.
        
        self.test_comb, self.delta = np.linspace(-0.5*conf.READOUT_BANDWIDTH, 0.5*conf.READOUT_BANDWIDTH, num=conf.NUMBER_OF_TEST_TONES, retstep=True)
        
        # setting up paths
        self.datadir = conf.datadir
        self.setupdir = conf.setupdir
        self.vnadir = self.setupdir / "sweeps" / "vna"
        self.vnacurrentdir = self.vnadir / "current"
        self.targetdir = self.setupdir / "sweeps" / "target"
        self.targetcurrentdir = self.targetdir / "current"
        self.dirfilecurrentdir = self.datadir / "dirfile_kids_current" 
        # NOT USED self.transfer_function_file = conf.transfer_function_file
        self.path_configuration = conf.path_configuration
        # NOT USED self.logdir = conf.log_dir

        print("")
        logging.info("Paths:")
        logging.info("\tLogfile: " + log_filename)
        logging.info("\tDirfiles: " + self.datadir.as_posix())
        logging.info("\tDirfile current: " + self.dirfilecurrentdir.as_posix())
        logging.info("\tSweeps: " + self.setupdir.as_posix())
        logging.info("\tVNAs: " + self.vnadir.as_posix())
        logging.info("\tVNA current: " + self.vnacurrentdir.as_posix())
        logging.info("\tTargets: " + self.targetdir.as_posix())
        logging.info("\tTarget current: " + self.targetcurrentdir.as_posix())
        logging.info("\tpath_configuration: " + self.path_configuration.as_posix())
        logging.info("\tDirfile name: " + string)
        
        # setting up socket
        logging.info("\n\t\033[33m+++  Setting up socket  +++\033[0m")
        self.s = socket(AF_PACKET, SOCK_RAW, htons(3))
        self.s.setsockopt(SOL_SOCKET, SO_RCVBUF, conf.UDP_PACKET_LENGHT)
        self.s.bind((conf.DATA_SOCKET, 3)) #must be the NET interface. NOT PPC INTERFACE! Do not change the 3.
        logging.info("Data socket: "+conf.DATA_SOCKET)

        # setting up ROACH
        self.divconst = 1.0 # not really necessary anymore since we only use MISTRAL-like ROACH and not OLIMPO ROACH, that needs twice the LO frequency (set to 2.0 for the OLIMPO ROACH!)
        self.center_freq = conf.LO
        self.global_attenuation = 1.0 # need to implement the class for Arduino variable attenuator!
         
        #self.zeros = signal.firwin(27, 1.5e6, window="hanning", nyq=0.5*conf.FPGA_SAMPLING_FREQUENCY)
        #self.zeros = signal.firwin(29, 1.5e3, window="hanning", nyq=0.5*conf.FPGA_SAMPLING_FREQUENCY)
        #self.zeros = self.zeros[1:-1]
        self.zeros = signal.firwin(23, 10.0e3, window="hanning", nyq=0.5*conf.FPGA_SAMPLING_FREQUENCY)

        # setting up Valon, for some god forsaken reason, the VALON changes serial port. Here we cycle through the serial ports until it connects and sets the LO frequency correctly.
        logging.info("\n\t\033[33m+++  Connecting to VALON...  +++\033[0m")
        port_attempt = 0
        while(not self.connect_to_valon(port="/dev/ttyUSB{:d}".format(port_attempt))):
            port_attempt += 1
            if port_attempt == 10:
                logging.inf0("Can't connect to Valon. Exiting program.")
                sys.exit(0)

        # setting up Arduino attenuators
        logging.info("\n\t\033[33m+++  Connecting to Arduino attenuators...  +++\033[0m")
        port_attempt = 0
        while(not self.connect_to_arduino(port="/dev/ttyACM{:d}".format(port_attempt))):
            port_attempt += 1
            if port_attempt == 10:
                logging.inf0("Can't connect to Arduino. Exiting program.")
                sys.exit(0)

        #self.nchan = len(self.raw_chan[0])
        logging.info("\n\t\033[33m+++  Activating ROACH interface  +++\033[0m")
        self.fpga = casperfpga.katcp_fpga.KatcpFpga(conf.ROACH_IP ,timeout=1200.)

        self.dac_freq_res = 0.5*(conf.DAC_SAMPLING_FREQUENCY/conf.LUT_BUFFER_LENGTH)

        # setting up sampling frequency
        self.accum_freq = conf.FPGA_SAMPLING_FREQUENCY/(conf.ACCUMULATION_LENGTH - 1)
        logging.info("Sampling frequency: {:.2f}Hz".format(self.accum_freq))
        
        np.random.seed(conf.NUMPY_RANDOM_SEED)
        self.phases = np.random.uniform(0., 2.*np.pi, 2000)

        # check if off resonance tones are specifed in the configuration file
        if len(conf.OFF_RESONANCE_TONES) > 0:
            self.ADD_OFF_RESONANCE_TONES = True
        else:
            self.ADD_OFF_RESONANCE_TONES = False

        self.main_prompt = "\n\t\033[33mBenchmark ROACH Readout\033[0m\n\t\033[35mChoose number and press Enter\033[0m"
        self.main_opts = ["Initialize",
                "Write test comb",
                "Write saved bb frequencies",
                "Write saved RF frequencies",
                "Print packet info to screen (UDP)",
                "VNA sweep, plot and locate",
                "Locate resonances",
                "Target sweep and plot",
                "Set global attenuation",
                "Set path with freqs, centers, radii, rotations",
                "Save dirfile for all channels in complex format",
                "Plot VNA sweep",
                "Plot Target sweep",
                "Exit"]

    def clearScreen(self):
        '''
        Clears the console

        Parameters
        -------
            None
            
        Returns
        -------
            None
        '''
        
        print("\033[H\033[2J")
    
    def connect_to_valon(self, port, set_to_default=True):
        '''
        Allows the connection with the VALON

		Parameters
		-------
		port: string
			Path to device port

		set_to_defaults: boolean, default is True
			If True sets the Valon frequencies to default

        Returns
        -------
            None
        '''
        time.sleep(0.5)
        try:
            self.valon_port = port
            self.v1 = valon_synth.Synthesizer(self.valon_port)
            logging.info("Success!")
            logging.info("Connected at port "+self.valon_port)
            
            if set_to_default:
                logging.info("Setting defaults...")
                self.v1.set_frequency(2, self.center_freq, 0.01)
                self.v1.set_frequency(1, 512., 0.01)
                    
            logging.info("Current synth freqs:")
            logging.info("Synth1: {:f} MHz".format(self.v1.get_frequency(1)))
            logging.info("Synth2: {:f} MHz".format(self.v1.get_frequency(2)))
            return True
            
        except OSError:
            logging.info("Port "+self.valon_port+" doesn't work")
            pass
        return False


    def connect_to_arduino(self, port, set_to_default=True):
        '''
        Allows the connection with the Arduino attenuators

        Parameters
		-------
		port: string
			Path to device port

		set_to_defaults: boolean, default is True
			If True sets the attenuations to default

        Returns
        -------
            None
        '''
        
        time.sleep(0.5)
        try:
            self.arduino_port = port
            self.att = vatt.Attenuator(self.arduino_port)
            logging.info("Success!")
            logging.info("Connected at port "+self.arduino_port)
                
            if set_to_default:
                logging.info("Setting defaults...")
                self.att.set_att(1, conf.ATT_RFOUT)
                self.att.set_att(2, conf.ATT_RFIN)
                atts = self.att.get_att()

            logging.info("Current attenuation levels:")
            logging.info("(RF_OUT, RF_IN) = " + str(self.att.get_att()))
            return True
           
        except OSError:
            logging.info("Port "+self.arduino_port+" doesn't work")
            pass
        return False

    def array_configuration(self):
        '''
        Takes radii, centers and rotations and loads them in the RoachInterface class.
        It also writes the formatfile.

        Parameters
        -------
            None
            
        Returns
        -------
            None
        '''
        
        logging.info("Running array_configuration")

        logging.info("Setting freqs, centers, radii and rotations from:" + self.path_configuration.as_posix())
        logging.info("Loading target frequencies")
		
        try:
			logging.info("Trying to load target_freqs_new.dat")
			self.cold_array_rf = np.loadtxt(self.path_configuration.as_posix()+'/target_freqs_new.dat')
        except IOError:
			logging.warning("target_freqs_new.dat not found. Trying target_freqs.dat")
			self.cold_array_rf = np.loadtxt(self.path_configuration.as_posix()+'/target_freqs.dat')
        logging.info("done")

        self.cold_array_bb = (((self.cold_array_rf) - (self.center_freq)/self.divconst))*1.0e6
        self.centers = np.load(self.path_configuration.as_posix() + '/centers.npy')
        self.rotations = np.load(self.path_configuration.as_posix() + '/rotations.npy')
        self.radii = np.load(self.path_configuration.as_posix() + '/radii.npy')
        
        logging.info("Reading freqs, centers, rotations and radii from {:s}\n".format(self.path_configuration.as_posix()))

        self.make_format(path_current = True)

    def lpf(self, zeros):
        zeros *= conf.ADC_MAX_AMPLITUDE
        for i in range(len(zeros)/2 + 1):
            coeff = np.binary_repr(int(zeros[i]), 32)
            coeff = int(coeff, 2)
            self.fpga.write_int('FIR_h'+str(i), coeff)
        return 

    def upload_fpg(self):
        '''
        Uploads the FPGA firmware

        Parameters
        -------
            None
            
        Returns
        -------
            None
        '''
        
        logging.info("Connecting to ROACH2 on ip " + conf.ROACH_IP)
        t1 = time.time()
        timeout = 10
        while not self.fpga.is_connected():
			if (time.time()-t1) > timeout:
				logging.error("Connection with roach timed out")
				raise Exception("Connection timeout to roach")
				return 1

        time.sleep(0.1)
        
        if (self.fpga.is_connected()):
            logging.info("Connection established")
            self.fpga.upload_to_ram_and_program(conf.bitstream.as_posix())
        else:
			logging.critical("FPGA not connected")
			return 1 
        logging.info('Uploaded' + str(conf.bitstream.as_posix()))
        
        time.sleep(3)
        return 0

    def qdrCal(self):    
    # Calibrates the QDRs. Run after writing to QDR.      
        logging.info("Resetting DAC")
        self.fpga.write_int('dac_reset', 1)
        logging.info('DAC on')
        bFailHard = False
        calVerbosity = 0 # set to 1 to see calibration results on console
        qdrMemName = 'qdr0_memory'
        qdrNames = ['qdr0_memory','qdr1_memory']
        
        fpga_clock = self.fpga.estimate_fpga_clock()
        logging.info("Fpga Clock Rate: " + str(fpga_clock))
        
        #self.fpga.get_system_information()
        results = {}
        
        for qdr in self.fpga.qdrs:
            #logging.info(qdr)
            mqdr = myQdr.from_qdr(qdr)
            results[qdr.name] = mqdr.qdr_cal2(fail_hard=bFailHard ,verbosity=calVerbosity)

        if calVerbosity == 1:
            logging.info('qdr cal results: ' + str(results))
        
        for qdrName in ['qdr0','qdr1']:
            if not results[qdr.name]:
                logging.critical('Calibration Failed')
                break
        

    # calibrates QDR and initializes GbE block
    def initialize(self):
        res1 = self.upload_fpg()

        if res1 == 1:
            logging.info("firmware uploaded succesfully")
            return 1

        # do we want to upload the firmware each time?
        # this is the destination IP of the packets
        #self.dest_ip = 192*(2**24) + 168*(2**16) + 40*(2**8) + 2
        self.dest_ip = 192*(2**24) + 168*(2**16) + 49*(2**8) + 2
        self.fabric_port = 60000

        # setting up the Gigabit Ethernet interface
        try:
            self.fpga.write_int('GbE_tx_destip', self.dest_ip)
            self.fpga.write_int('GbE_tx_destport', self.fabric_port)
            self.fpga.write_int('downsamp_sync_accum_len', conf.ACCUMULATION_LENGTH-1)
            self.fpga.write_int('PFB_fft_shift', 2**9 -1)
            self.fpga.write_int('dds_shift', conf.DDS_SHIFT)
            self.lpf(self.zeros)    #parametri filtro lp
            #self.save_path = '/mnt/iqstream/'
            self.qdrCal()
            logging.info("QDR calibrated")
            self.initialize_GbE()
            logging.info("GbE initialized")
            return res1 + 0
        except:
            logging.error("Calibration failed")
            return 1
        

    def fft_bin_index(self, freqs, fft_len, samp_freq):
	    # returns the fft bin index for a given frequency, fft length, and sample frequency
        bin_index = np.round((freqs/samp_freq)*fft_len).astype('int')
        return bin_index

    def read_mixer_snaps(self, shift, chan, mixer_out = True):
	    # returns snap data for the dds mixer inputs and outputs
        self.fpga.write_int('dds_shift', shift)
        if (chan % 2) > 0: # if chan is odd
            self.fpga.write_int('DDC_snap_chan_select', (chan - 1) / 2)
        else:
            self.fpga.write_int('DDC_snap_chan_select', chan/1)
        self.fpga.write_int('DDC_snap_rawfftbin_ctrl', 0)
        self.fpga.write_int('DDC_snap_mixerout_ctrl', 0)
        self.fpga.write_int('DDC_snap_rawfftbin_ctrl', 1)
        self.fpga.write_int('DDC_snap_mixerout_ctrl', 1)
        mixer_in = np.fromstring(self.fpga.read('DDC_snap_rawfftbin_bram', 16*2**10),dtype='>i2').astype('float')
        mixer_in /= 2.0**15
        if mixer_out:
            mixer_out = np.fromstring(self.fpga.read('DDC_snap_mixerout_bram', 8*2**10),dtype='>i2').astype('float')
            mixer_out /= 2.0**14
            return mixer_in, mixer_out
        else:
            return mixer_in

    def return_shift(self, chan):
	    # Returns the dds shift
        dds_spec = np.abs(np.fft.rfft(self.I_dds[chan::conf.FFT_LENGTH], conf.FFT_LENGTH))
        dds_index = np.where(np.abs(dds_spec) == np.max(np.abs(dds_spec)))[0][0]
        print('Finding LUT shift...')
        for i in range(conf.FFT_LENGTH/2):
            mixer_in = self.read_mixer_snaps(i, chan, mixer_out = False)
            I0_dds_in = mixer_in[2::8]    
            #I0_dds_in[np.where(I0_dds_in > 32767.)] -= 65535.
            snap_spec = np.abs(np.fft.rfft(I0_dds_in, conf.FFT_LENGTH))
            snap_index = np.where(np.abs(snap_spec) == np.max(np.abs(snap_spec)))[0][0]
            if dds_index == snap_index:
                print('LUT shift:', i)
                shift = i
                break
        return shift

    def freq_comb(self, freqs, samp_freq, resolution, random_phase = True, DAC_LUT = True, apply_transfunc = False):
        # Generates a frequency comb for the DAC or DDS look-up-tables. DAC_LUT = True for the DAC LUT. Returns I and Q 
        
        freqs = np.round(freqs/self.dac_freq_res)*self.dac_freq_res
        amp_full_scale = (2**15 - 1)
        
        if DAC_LUT:
            
            logging.info('freq comb uses DAC_LUT')
            
            fft_len = conf.LUT_BUFFER_LENGTH
            bins = self.fft_bin_index(freqs, fft_len, samp_freq)
            #np.random.seed(333)
            phase = self.phases[0:len(bins)]
            
            try:
                if apply_transfunc == True:
                    logging.info("apply_transfunc=True")
                    self.calc_transfunc(freqs+self.center_freq*1.e6)
                elif apply_transfunc == False:
                    logging.info("apply_transfunc=False. Setting all amps to 1")
                    self.amps = np.ones(len(freqs))
            except:
                logging.info("Something went wrong with apply_transfunc. Setting all amps to 1")
                self.amps = np.ones(len(freqs))

            '''
            if apply_transfunc:
                self.amps = self.get_transfunc(path_current = True)
            else:
                try:
                    if self.target_sweep_flag == True:
                        self.amps = self.amps
                    if self.test_comb_flag == True:
                        self.amps = self.test_comb_amps
                except: 
                    self.amps = np.ones(len(freqs))
            '''

            if len(bins) != len(self.amps): 
				self.amps = 1

            if not random_phase:
                phase = np.load('/mnt/iqstream/last_phases.npy') 
            
            self.spec = np.zeros(fft_len, dtype='complex')
            self.spec[bins] = self.amps*np.exp(1j*(phase))
            wave = np.fft.ifft(self.spec)
            #waveMax = np.max(np.abs(wave))
            waveMax = conf.WAVEMAX
            
            I = (wave.real/waveMax)*(amp_full_scale)*self.global_attenuation  
            Q = (wave.imag/waveMax)*(amp_full_scale)*self.global_attenuation  
            np.save("/home/mew/wave_I.npy", I)
            np.save("/home/mew/wave_Q.npy", Q)
        else:
            fft_len = (conf.LUT_BUFFER_LENGTH/conf.FFT_LENGTH)
            bins = self.fft_bin_index(freqs, fft_len, samp_freq)
            spec = np.zeros(fft_len, dtype='complex')
            amps = np.array([1.]*len(bins))
            phase = 0.0
            spec[bins] = amps*np.exp(1j*(phase))
            wave = np.fft.ifft(spec)
            #wave = signal.convolve(wave,signal.hanning(3), mode = 'same')
            waveMax = np.max(np.abs(wave))
            I = (wave.real/waveMax)*(amp_full_scale)
            Q = (wave.imag/waveMax)*(amp_full_scale)
        return I, Q    

    def select_bins(self, freqs):
	    # Calculates the offset from each bin center, to be used as the DDS LUT frequencies, and writes bin numbers to RAM
        bins = self.fft_bin_index(freqs, conf.FFT_LENGTH, conf.DAC_SAMPLING_FREQUENCY)
        bin_freqs = bins*conf.DAC_SAMPLING_FREQUENCY/conf.FFT_LENGTH
        bins[bins<0] += conf.FFT_LENGTH
        self.freq_residuals = freqs - bin_freqs
        #for i in range(len(freqs)):
        #	print "bin, fbin, freq, offset:", bins[i], bin_freqs[i]/1.0e6, freqs[i]/1.0e6, self.freq_residuals[i]
        ch = 0
        for fft_bin in bins:
            self.fpga.write_int('bins', fft_bin)
            self.fpga.write_int('load_bins', 2*ch + 1)
            self.fpga.write_int('load_bins', 0)
            ch += 1
        return 

    def define_DDS_LUT(self, freqs):
		# Builds the DDS look-up-table from I and Q given by freq_comb. freq_comb is called with the sample rate equal to the sample rate for a single FFT bin. There are two bins returned for every fpga clock, so the bin sample rate is 256 MHz / half the fft length  
        self.select_bins(freqs)
        I_dds, Q_dds = np.array([0.]*(conf.LUT_BUFFER_LENGTH)), np.array([0.]*(conf.LUT_BUFFER_LENGTH))
        for m in range(len(self.freq_residuals)):
            I, Q = self.freq_comb(np.array([self.freq_residuals[m]]), conf.FPGA_SAMPLING_FREQUENCY/(conf.FFT_LENGTH/2.), self.dac_freq_res, random_phase = False, DAC_LUT = False)
            I_dds[m::conf.FFT_LENGTH] = I
            Q_dds[m::conf.FFT_LENGTH] = Q
        return I_dds, Q_dds

    def pack_luts(self, freqs, transfunc = False):
    # packs the I and Q look-up-tables into strings of 16-b integers, in preparation to write to the QDR. Returns the string-packed look-up-tables
        if transfunc:
			self.I_dac, self.Q_dac = self.freq_comb(freqs, conf.DAC_SAMPLING_FREQUENCY, self.dac_freq_res, random_phase = True, apply_transfunc = True)
        else:
			self.I_dac, self.Q_dac = self.freq_comb(freqs, conf.DAC_SAMPLING_FREQUENCY, self.dac_freq_res, random_phase = True)

        self.I_dds, self.Q_dds = self.define_DDS_LUT(freqs)
        self.I_lut, self.Q_lut = np.zeros(conf.LUT_BUFFER_LENGTH*2), np.zeros(conf.LUT_BUFFER_LENGTH*2)
        self.I_lut[0::4] = self.I_dac[1::2]         
        self.I_lut[1::4] = self.I_dac[0::2]
        self.I_lut[2::4] = self.I_dds[1::2]
        self.I_lut[3::4] = self.I_dds[0::2]
        self.Q_lut[0::4] = self.Q_dac[1::2]         
        self.Q_lut[1::4] = self.Q_dac[0::2]
        self.Q_lut[2::4] = self.Q_dds[1::2]
        self.Q_lut[3::4] = self.Q_dds[0::2]
        
        logging.info("String packing LUT")
        
        I_lut_packed = self.I_lut.astype('>h').tostring()
        Q_lut_packed = self.Q_lut.astype('>h').tostring()
        
        logging.info("Done")
        
        return I_lut_packed, Q_lut_packed

    
    def writeQDR(self, freqs, transfunc = False):
        '''
        Writes packed LUTs to QDR

        Parameters
        -------
            freqs: list or numpy array
                Set of frequencies in Hz in the range (-0.5*conf.READOUT_BANDWIDTH,+0.5*conf.READOUT_BANDWIDTH)MHz
                
            transfunc: boolean, default is False
                If True a set of custom attenumations will be computed
                
        Returns
        -------
            None
        '''
     
        if transfunc:
            logging.info('Applying transfer function')
            I_lut_packed, Q_lut_packed = self.pack_luts(freqs, transfunc = True)
        else:
			I_lut_packed, Q_lut_packed = self.pack_luts(freqs, transfunc = False)

        self.fpga.write_int('dac_reset', 1)
        self.fpga.write_int('dac_reset', 0)
        logging.info('Writing DAC and DDS LUTs to QDR, this might take a while...')
        self.fpga.write_int('start_dac', 0)
        self.fpga.blindwrite('qdr0_memory', I_lut_packed, 0)
        self.fpga.blindwrite('qdr1_memory', Q_lut_packed, 0)
        self.fpga.write_int('start_dac', 1)
        self.fpga.write_int('downsamp_sync_accum_reset', 0)
        self.fpga.write_int('downsamp_sync_accum_reset', 1)
        logging.info('Done')
        return 

    
    def read_QDR_katcp(self):
	    # Reads out QDR buffers with KATCP, as 16-b signed integers.    
        self.QDR0 = np.fromstring(self.fpga.read('qdr0_memory', 8 * 2**20), dtype='>i2')
        self.QDR1 = np.fromstring(self.fpga.read('qdr1_memory', 8* 2**20), dtype='>i2')
        self.I_katcp = self.QDR0.reshape(len(self.QDR0)/4., 4.)
        self.Q_katcp = self.QDR1.reshape(len(self.QDR1)/4., 4.)
        self.I_dac_katcp = np.hstack(zip(self.I_katcp[:, 1], self.I_katcp[:, 0]))
        self.Q_dac_katcp = np.hstack(zip(self.Q_katcp[:, 1], self.Q_katcp[:, 0]))
        self.I_dds_katcp = np.hstack(zip(self.I_katcp[:, 3], self.I_katcp[:, 2]))
        self.Q_dds_katcp = np.hstack(zip(self.Q_katcp[:, 3], self.Q_katcp[:, 2]))
        return        


    def read_accum_snap(self):
        # Reads the avgIQ buffer. Returns I and Q as 32-b signed integers     
        self.fpga.write_int('accum_snap_accum_snap_ctrl', 0)
        self.fpga.write_int('accum_snap_accum_snap_ctrl ', 1)
        accum_data = np.fromstring(self.fpga.read('accum_snap_accum_snap_bram', 16*2**11), dtype = '>i').astype('float')
        I0 = accum_data[0::4]    
        Q0 = accum_data[1::4]    
        I1 = accum_data[2::4]    
        Q1 = accum_data[3::4]    
        I = np.hstack(zip(I0, I1))
        Q = np.hstack(zip(Q0, Q1))
        return I, Q    

    def add_out_of_res(self, path, tones):

        freqs = np.loadtxt(path, unpack=True, usecols=0)
        freqs = np.append(freqs, tones)
        freqs = np.sort(freqs)

        np.savetxt(path, np.transpose(freqs))
	'''
    def calc_transfunc(self, freqs):
    
        #Da modificare:
        #- Overfittare la funzione di trasferimento. O con un poly di grado superiore o con interpolazione
        #- Salvare le attenuazioni in un qualche file amps.npy
        
        freqs = freqs/1.e6

        self.transfunc_parameters_file = "/home/mistral/src/mistral_readout_dev/transfunc_polyfit_coefficients.npy"
        #ricordati di mettere il file dei coefficienti in mistral_readout
        logging.info("Calculating transfunction from parameters file "+self.transfunc_parameters_file)
        logging.info("baseline_attenuation="+str(self.baseline_attenuation))
        poly_par = np.load(self.transfunc_parameters_file)
        logging.info("Polynomial parameters="+str(poly_par))

        attenuations = np.poly1d(poly_par)(freqs)

        logging.info("prima di 10**")
        logging.info(attenuations)
        attenuations = 10.**((self.baseline_attenuation-attenuations)/20.)
        logging.info("dopo 10**")
        logging.info(attenuations)
        
        logging.info("Setting attenuation to 1 for "+str(conf.skip_tones_attenuation)+" out of resonance tones")
        
        attenuations[0:int(conf.skip_tones_attenuation)] = 1

        logging.info("Attenuations=")
        logging.info(attenuations)
        bad_attenuations = np.argwhere(np.floor(attenuations))
        logging.info("Checking for bad attenuations")
        logging.info(np.array2string(bad_attenuations))

        if bad_attenuations.size > 0:
            logging.info("Warning: invalid attenuations found. Setting them to 1.0 by default.")
            attenuations[bad_attenuations] = 1.0
        
        table = np.array([freqs,attenuations])

        #np.savetxt(path,np.transpose(table)) 

        #logging.info(attenuations)
        
        self.amps = attenuations
        logging.info("attenuations = "+str(self.amps))

        #for f in freqs:
        #    attenuation = np.poly1d(poly_par)(f)
        #    attenuation = 10 **((self.baseline_attenuation-attenuation)/20)
	'''

    def get_transfunc(self, path_current=False):


        nchannel = len(self.cold_array_bb)
        channels = range(nchannel)
        tf_path = self.setupdir+"transfer_functions/"
        if path_current:
                tf_dir = self.timestring
        else:
                tf_dir = raw_input('Insert folder for TRANSFER_FUNCTION (e.g. '+self.timestring+'): ')
        save_path = os.path.join(tf_path, tf_dir)
        print("save TF in "+save_path)
        try:
                os.mkdir(save_path)
        except OSError:
                pass
        #command_cleanlink = "rm -f "+tf_path+'current'
        os.remove(tf_path+'current')
        #command_linkfile = "ln -f -s " + save_path +" "+ tf_path+'current'
        os.link(src=save_path, dest=tf_path+'current')


        by_hand = raw_input("set TF by hand (y/n)?")
        if by_hand == 'y':
			transfunc = np.zeros(nchannel)+1
			chan = input("insert channel to change")
			value = input("insert values to set [0, 1]")
			transfunc[chan] = value
        else:

			#mag_array = np.zeros((100, len(self.test_comb)))
			mag_array = np.zeros((100, nchannel))+1
			for i in range(100):
				packet = self.s.recv(conf.UDP_PACKET_LENGHT)
				data = np.fromstring(packet[conf.UDP_PACKET_HEADER:], dtype = '<i').astype('float')
				packet_count = (np.fromstring(packet[-4:], dtype = '>I'))
				for chan in channels:
					if (chan % 2) > 0:
						I = data[1024 + ((chan - 1) / 2)]    
						Q = data[1536 + ((chan - 1) /2)]    
					else:
						I = data[0 + (chan/2)]    
						Q = data[512 + (chan/2)]    

					mags = np.sqrt(I**2 + Q**2)
					if mags ==0:
						mags = 1
					print(chan, mags)
					mag_array[i, chan] = mags     #[2:len(self.test_comb)+2]
			transfunc = np.mean(mag_array, axis = 0)
			transfunc = 1./ (transfunc / np.max(transfunc))

        np.save(save_path+'/last_transfunc.npy',transfunc)
        return transfunc


    def initialize_GbE(self):
        # Configure GbE Block. Run immediately after calibrating QDR.
        self.fpga.write_int('GbE_tx_rst', 0)
        self.fpga.write_int('GbE_tx_rst', 1)
        self.fpga.write_int('GbE_tx_rst', 0)
        return

    def stream_UDP(self, chan, Npackets):
        self.fpga.write_int('GbE_pps_start', 1)
        count = 0
        while count < Npackets:
            packet = self.s.recv(conf.UDP_PACKET_LENGHT)
            data = np.fromstring(packet[conf.UDP_PACKET_HEADER:], dtype = '<i').astype('float')
            forty_two = (np.fromstring(packet[-16:-12], dtype = '>I'))
            pps_count = (np.fromstring(packet[-12:-8], dtype = '>I'))
            time_stamp = np.round((np.fromstring(packet[-8:-4], dtype = '>I').astype('float')/conf.FPGA_SAMPLING_FREQUENCY)*1.0e3,3)
            packet_count = (np.fromstring(packet[-4:], dtype = '>I'))
            if (chan % 2) > 0:
                I = data[1024 + ((chan - 1) / 2)]    
                Q = data[1536 + ((chan - 1) /2)]    
            else:
                I = data[0 + (chan/2)]    
                Q = data[512 + (chan/2)]    
            phase = np.arctan2([Q],[I])
            #print forty_two, pps_count, time_stamp, packet_count, phase
            count += 1
        return 

    def dirfile_complex(self, nchannel):
        
        channels = range(nchannel)
        self.fpga.write_int('GbE_pps_start', 1)
        save_path = self.datadir / ("dirfile_"+self.timestring)
     
        logging.info( "log data in "+save_path.as_posix())
        
        try:
            logging.info("Creating folder:"+save_path.as_posix())
            os.mkdir(save_path.as_posix())
        except OSError:
            logging.warning("Folder already existing: appending to previous dirfile:"+save_path.as_posix())
            pass

        if os.path.exists(self.dirfilecurrentdir.as_posix()):
            logging.info("\nRemoving old symlink to current directory...")
            os.remove(self.dirfilecurrentdir.as_posix()) # that removes the symlink
        logging.info("Updating symlink to current directory...")
        os.symlink(save_path.as_posix(), self.dirfilecurrentdir.as_posix())

        shutil.copy(conf.path_format.as_posix() + "/format", save_path.as_posix() + "/format")
        #shutil.copy(conf.path_format.as_posix() + "/format_complex", save_path + "/format") # prima c'era questa linea F.C.
        #shutil.copy(conf.path_complex.as_posix() + "/format_extra", save_path + "/format_extra")

        self.make_format_complex()
        
        shutil.copy(self.datadir.as_posix() + "/format_complex_extra", save_path.as_posix() + "/format_complex_extra")
        nfo_I = map(lambda x: save_path.as_posix() + "/chI_" + str(x).zfill(3), range(nchannel))
        nfo_Q = map(lambda y: save_path.as_posix() + "/chQ_" + str(y).zfill(3), range(nchannel))
        fo_I = map(lambda x: open(x, "ab"), nfo_I)
        fo_Q = map(lambda y: open(y, "ab"), nfo_Q)
        fo_time = open(save_path.as_posix() + "/time", "ab")

        fo_count = open(save_path.as_posix() + "/packet_count", "ab")	
        count = 0
        try:
            logging.info("Writing format_time file")
            self.make_format_time(save_path.as_posix())
            
            logging.info("Acquisition started")
            while True:
                packet = self.s.recv(conf.UDP_PACKET_LENGHT)
                if(len(packet) == conf.UDP_PACKET_LENGHT):

                    ts = time.time()
                    data_bin = packet[conf.UDP_PACKET_HEADER:]
                    packet_count_bin = packet[-1]+packet[-2]+packet[-3]+packet[-4]#packet[-4:]
                    
                    #ts = ts0 + (packet_count_bin.astype("int32") - packet_count_0) ) / self.accum_freq
                    
                    for chan in channels:
                        if (chan % 2) > 0:   # odd channels
							I = data_bin[(1024+((chan-1) /2))*4 : (1024+((chan-1) /2))*4+4 ] 
							Q = data_bin[(1536+((chan-1) /2))*4 : (1536+((chan-1) /2))*4+4 ] 
                        else:                # even channels
							I = data_bin[(   0+ (chan/2))*4 : (   0+(chan /2))*4+4 ]  
							Q = data_bin[( 512+ (chan/2))*4 : ( 512+(chan /2))*4+4 ]

                        fo_I[chan].write(I)
                        fo_Q[chan].write(Q)
                    count += 1
                    fo_time.write(struct.pack('d', ts))
                    fo_count.write(packet_count_bin)
                    fo_count.flush()
                    if(count/20 == count/20.):
                        fo_time.flush()
                        map(lambda x: (x.flush()), fo_I)
                        map(lambda x: (x.flush()), fo_Q)
                else:
                    logging.warning("Incomplete packet received, length="+ str(len(packet)))
        except KeyboardInterrupt:
            logging.info("Acquisition stopped")
            pass
        for chan in channels:
                fo_I[chan].close()
                fo_Q[chan].close()
        fo_time.close()
        fo_count.close()
      
        logging.info("Saving logfile in target sweep folder: "+save_path.as_posix())
        shutil.copy(self.log_filename, save_path.as_posix() + "/client_log.log")
            
        return 



    def kst_UDP(self, chan, time_interval):
        Npackets = np.int(time_interval * self.accum_freq)
        self.fpga.write_int('GbE_pps_start', 1)
        count = 0
        f_chan = open('/home/olimpo/home/data/ch' + str(chan), 'ab')
        f_time = open('/home/olimpo/home/data/time' + str(chan), 'ab')
        while count < Npackets:
            ts = time.time()
            packet = self.s.recv(conf.UDP_PACKET_LENGHT)
            data = np.fromstring(packet[conf.UDP_PACKET_HEADER:], dtype = '<i').astype('float')
            if (chan % 2) > 0:
                I = data[1024 + ((chan - 1) / 2)]    
                Q = data[1536 + ((chan - 1) /2)]    
            else:
                I = data[0 + (chan/2)]    
                Q = data[512 + (chan/2)]    
            phase = np.arctan2([Q], [I])
            f_chan.write(struct.pack('f', phase))
            f_time.write(struct.pack('d', ts))
            f_time.flush()
            f_chan.flush()
            count += 1
        f_chan.close()
        f_time.close()
        return 

    def IQ_grad(self, dark_sweep_path, plot_chan): 
        lo_freqs, I_dark, Q_dark = self.open_stored(dark_sweep_path)
        bb_freqs, delta_f = np.linspace(-200.0e6, 200.0e6, 500, retstep=True)
        #bb_freqs = np.load('/mnt/iqstream/last_bb_freqs.npy')
        channels = np.arange(len(bb_freqs))
        delta_lo = 5e3
        i_index = [np.where(np.abs(np.diff(I_dark[:,chan])) == np.max(np.abs(np.diff(I_dark[:,chan]))))[0][0] for chan in channels]
        q_index = [np.where(np.abs(np.diff(Q_dark[:,chan])) == np.max(np.abs(np.diff(Q_dark[:,chan]))))[0][0] for chan in channels]
        di_df = np.array([(I_dark[:,chan][i_index[chan] + 1] - I_dark[:,chan][i_index[chan] - 1])/(2*delta_lo) for chan in channels])
        dq_df = np.array([(Q_dark[:,chan][q_index[chan] + 1] - Q_dark[:,chan][q_index[chan] - 1])/(2*delta_lo) for chan in channels])
        I0 = np.array([I_dark[:,chan][i_index[chan]] for chan in channels])
        Q0 = np.array([Q_dark[:,chan][q_index[chan]] for chan in channels])
        rf_freqs = np.array([200.0e6 + bb_freqs[chan] for chan in channels])
        return di_df[plot_chan], dq_df[plot_chan], rf_freqs[plot_chan]

    def plot_stream_UDP(self, chan):
        dark_sweep_path = '/mnt/iqstream/vna_sweeps/scaled2'
        di_df, dq_df, rf_freq = self.IQ_grad(dark_sweep_path, chan)
        Npackets = 244
        self.fpga.write_int('GbE_pps_start', 1)
        fig = plt.figure(num= None, figsize=(18,12), dpi=80, facecolor='w', edgecolor='w')
        plt.suptitle('1s stream: Channel ' + str(chan) + ', Freq = ' + str(np.round(rf_freq/1.0e6,3)) + ' MHz', fontsize = 20)
        # channel phase
        plot1 = fig.add_subplot(211)
        plot1.set_ylabel('rad')
        line1, = plot1.plot(np.arange(Npackets), np.zeros(Npackets), 'k-', linewidth = 1)
        plt.grid()
        # df
        plot2 = fig.add_subplot(212)
        plot2.set_ylabel('Hz')
        line2, = plot2.plot(np.arange(Npackets), np.zeros(Npackets), 'b-', linewidth = 1)
        plt.grid()
        plt.xlabel('Packet #', fontsize = 20)
        plt.show(block = False)
        stop = 1.0e6
        count = 0
        phases = np.zeros(Npackets)
        delta_I = np.zeros(Npackets)
        delta_Q = np.zeros(Npackets)
        df = np.zeros(Npackets)
        chan_freq = rf_freq
        while count < stop:
			packet_count = 0
			while packet_count < Npackets:
				packet = self.s.recv(conf.UDP_PACKET_LENGHT)
				data = np.fromstring(packet[conf.UDP_PACKET_HEADER:], dtype = '<i').astype('float')
				#data = np.fromstring(packet,dtype = '<i').astype('float')
				data /= 2.0**17
				data /= (conf.ACCUMULATION_LENGTH/512.)
				ts = (np.fromstring(packet[-4:], dtype = '<i').astype('float')/ conf.FPGA_SAMPLING_FREQUENCY)*1.0e3 # ts in ms
				if (chan % 2) > 0:
					I = data[1024 + ((chan - 1) / 2)]    
					Q = data[1536 + ((chan - 1) /2)]    
				else:
					I = data[0 + (chan/2)]    
					Q = data[512 + (chan/2)]    
				phases[packet_count] = np.arctan2([Q], [I])
				if (count and packet_count) == 0:
					I0 = I
					Q0 = Q
				delta_I = I - I0 	
				delta_Q = Q - Q0 	
				df[packet_count] = ((delta_I * di_df) + (delta_Q * dq_df)) / (di_df**2 + dq_df**2)
				packet_count += 1
			avg_phase = np.round(np.mean(phases),5)
			avg_df = np.round(np.mean(df[1:]))
			avg_dfbyf = avg_df / chan_freq

			plot1.set_ylim((np.min(phases) - 1.0e-3, np.max(phases)+1.0e-3))
			plot2.set_ylim((np.min(df[1:]) - 1.0e-3, np.max(df[1:])+1.0e-3))
			line1.set_ydata(phases)
			line2.set_ydata(df)
			plot1.set_title('Phase, avg =' + str(avg_phase) + ' rad', fontsize = 18)
			plot2.set_title('Delta f, avg =' + str(avg_df) + 'Hz' + ', avg df/f = ' + str(avg_dfbyf), fontsize = 18)
			plt.draw()
			count += 1
        return 

    def vna_sweep(self, do_plot=True):

        '''
        Function performing a VNA sweep to roughly identify resonances. 
        '''

        logging.info("\nStarting VNA sweep with parameters")
        logging.info("- Comb:\t{:d} tones".format(conf.NUMBER_OF_TEST_TONES))
        logging.info("- Span:\t[{:.2f}, {:.2f}] MHz".format(conf.LO-0.5*conf.READOUT_BANDWIDTH*1.0e-6, conf.LO+0.5*conf.READOUT_BANDWIDTH*1.0e-6))
        logging.info("- Step:\t{:.2f} kHz".format(conf.VNA_STEP*1.0e-3))

        self.do_transf = True

        center_freq = self.center_freq*1.e6*conf.MIXER_CONST
           
        sweep_dir = Path(raw_input('\nInsert a new VNA sweep name (e.g. YYYYMMDD_EXPERIMENT_PROTOTYPE_NOTES_TEMPERATURE): '))
        
        save_path = self.vnadir / sweep_dir
        logging.info("\nCreating save folder at path: " + save_path.as_posix())
        os.mkdir(save_path.as_posix())

        if os.path.exists(self.vnacurrentdir.as_posix()):
            logging.info("\nRemoving old symlink to current directory...")
            os.remove(self.vnacurrentdir.as_posix()) # that removes the symlink
        logging.info("Updating symlink to current directory...")
        os.symlink(save_path.as_posix(), self.vnacurrentdir.as_posix())

        self.v1.set_frequency(2, center_freq/1.0e6, 0.01)
        span = self.delta

        start = center_freq - (span/2) * conf.MIXER_CONST
        stop  = center_freq + (span/2) * conf.MIXER_CONST
        step  = conf.VNA_STEP * conf.MIXER_CONST
        sweep_freqs = np.arange(start, stop, step)
        self.sweep_freqs = np.round(sweep_freqs/step)*step

        logging.info("\nSaving bb_freqs.npy at path:\t\t" + (save_path / "bb_freqs.npy").as_posix())
        np.save((save_path / "bb_freqs.npy").as_posix(), self.test_comb)
        logging.info("Saving sweep_freqs.npy at path:\t\t" + (save_path / "sweep_freqs.npy").as_posix())
        np.save((save_path / "sweep_freqs.npy").as_posix(), self.sweep_freqs)
        
        #self.calc_transfunc(self.test_comb-center_freq)
        
        self.writeQDR(self.test_comb, transfunc=True)
            
        logging.info("\nStarting VNA sweep...")
        progress_bar = tqdm(self.sweep_freqs, desc="Starting VNA sweep")
        for freq in progress_bar:
            if self.v1.set_frequency(2, freq/(1.0e6), 0.01):
                self.store_UDP(100, freq, save_path.as_posix(), channels=conf.NUMBER_OF_TEST_TONES)
                self.v1.set_frequency(2, center_freq/(1.0e6), 0.01) # LO
                progress_bar.set_description_str("Base band frequency {:.2f} kHz".format(((freq*1.0e-6)*0.5-conf.LO)*1.0e3))
                progress_bar.refresh()
            
        if do_plot:
            logging.info("\nPlotting VNA sweep")
            self.plot_vna(save_path.as_posix())
        
        logging.info("Saving logfile in VNA sweep folder: " + save_path.as_posix())
        shutil.copy(self.log_filename, (save_path / "client_log.log").as_posix())

        return 


    def target_sweep(self, do_plot=True):

        '''
        Function used for tuning. It spans a small range of frequencies around known resonances.
        '''
        
        logging.info("\nStarting target sweep with parameters")
        logging.info("- Half span:\t{:.2f} kHz".format(conf.TARGET_HSPAN*1.0e-3))
        logging.info("- Step freq:\t{:.2f} kHz".format(conf.TARGET_STEP*1.0e-3))
        if self.ADD_OFF_RESONANCE_TONES:
            logging.info("- Off res.:  \t{:s} MHz".format(str(conf.OFF_RESONANCE_TONES)))
        else:
            logging.info("- Off res.:  \tno off resonance tones specified")

        #self.target_sweep_flag = True
        center_freq = (self.center_freq * 1.0e6 + conf.TARGET_OFFSET) * conf.MIXER_CONST
        self.print_useful_paths()        
        vna_path = raw_input('\nAbsolute path to VNA sweep dir ? ')

        save_path = self.targetdir / self.timestring
        raw_input('Data will be saved to '+save_path.as_posix()+'\nPress enter to start...')
        
        self.path_configuration = save_path
        
        #try:
        logging.info("\nCreating save folder at path: " + save_path.as_posix())
        os.mkdir(save_path.as_posix())
        #except OSError:
		#	logging.info("Failed to create new folder: folder already existing")
		#	pass
        
        logging.info("Loading target_freqs_new.dat at "+ vna_path + "/target_freqs_new.dat")
        try:
            self.target_freqs = np.loadtxt(vna_path + '/target_freqs_new.dat')
            
        except IOError:
            logging.info("Failed to load target_freqs_new.dat. Trying with target_freqs.dat")
            try:
                self.target_freqs, self.amps = np.loadtxt(os.path.join(vna_path, 'target_freqs.dat'), unpack=True)
                logging.info("Loaded target_freqs.dat file with freqs and amps")
            except:
                self.target_freqs= np.loadtxt(os.path.join(vna_path, 'target_freqs.dat'), unpack=True)
                logging.info("Loaded target_freqs.dat file without amps")
        
        if os.path.exists(self.targetcurrentdir.as_posix()):
            logging.info("\nRemoving old symlink to current directory...")
            os.remove(self.targetcurrentdir.as_posix()) # that removes the symlink
        logging.info("Updating symlink to current directory...")
        os.symlink(save_path.as_posix(), self.targetcurrentdir.as_posix())
        
        logging.info("\nSaving target_freqs.dat at path:\t" + (save_path / "target_freqs.dat").as_posix())
        np.savetxt((save_path / "target_freqs.dat").as_posix(), self.target_freqs)
        
        self.bb_target_freqs = ((self.target_freqs*1.0e6) - center_freq/conf.MIXER_CONST)
        upconvert = (self.bb_target_freqs + center_freq) / 1.0e6
        #logging.info("RF tones: " + np.array2string(upconvert))
        self.v1.set_frequency(2, center_freq / (1.0e6), 0.01) # LO
        #logging.info("Target baseband freqs (MHz): "+ str(self.bb_target_freqs/1.0e6))
        
        span = conf.TARGET_HSPAN * conf.MIXER_CONST
        start = center_freq - (span)  # era (span/2)
        stop = center_freq + (span)   # era (span/2) 
        step = conf.TARGET_STEP * conf.MIXER_CONST #1.25e3 * 2.       # era senza   
        sweep_freqs = np.arange(start, stop, step)
        sweep_freqs = np.round(sweep_freqs/step)*step
        nsteps = len(sweep_freqs)
        #logging.info("LO freqs: " + str(sweep_freqs))
        
        logging.info("Saving bb_freqs.dat at path:\t\t" + (save_path / "bb_freqs.dat").as_posix())
        np.savetxt((save_path / "bb_freqs.dat").as_posix(), self.bb_target_freqs)
        logging.info("Saving sweep_freqs.dat at path:\t\t" + (save_path / "sweep_freqs.dat").as_posix())
        np.savetxt((save_path / "sweep_freqs.dat").as_posix(), sweep_freqs)

        self.do_transf = True

        if self.do_transf == True:
            self.writeQDR(self.bb_target_freqs, transfunc=True)
        else:
            self.writeQDR(self.bb_target_freqs)
        
        logging.info("\nStarting target sweep")
        progress_bar = tqdm(sweep_freqs, desc="Starting target sweep")
        for freq in progress_bar:
            if self.v1.set_frequency(2, freq/(1.0e6), 0.01):
                self.store_UDP(100, freq, save_path.as_posix(), channels=len(self.bb_target_freqs)) 
                self.v1.set_frequency(2, center_freq/(1.0e6), 0.01) # LO
                progress_bar.set_description_str("Base band frequency {:.2f} kHz".format(((freq*1.0e-6)*0.5-conf.LO)*1.0e3))
                progress_bar.refresh()
        
        logging.info("\nRunning pipline.py on " + save_path.as_posix())
        # locate centers, rotations, and resonances
        pipeline(save_path.as_posix())
        # includes make_format	
        self.array_configuration() 

        if do_plot:
			logging.info("\nPlotting target sweep")
			self.plot_targ(save_path.as_posix())
        
        logging.info("Saving logfile in target sweep folder: " + save_path.as_posix())
        shutil.copy(self.log_filename, (save_path / "client_log.log").as_posix())

        return
    '''
    def sweep_lo(self, Npackets_per = 100, channels = None, span = 2.0e6, save_path = '/sweeps/vna'):
        center_freq = self.center_freq*1.e6
        for freq in self.sweep_freqs:
            print('Sweep freq = {:f}'.format(freq/1.0e6))
            if self.v1.set_frequency(2, freq/1.0e6, 0.01): 
                self.store_UDP(Npackets_per, freq, save_path, channels=channels) 
        self.v1.set_frequency(2, center_freq / (1.0e6), 0.01) # LO
        return
    '''

    def store_UDP(self, Npackets, LO_freq, save_path, skip_packets=2, channels = None):
        channels = np.arange(channels)
        I_buffer = np.empty((Npackets + skip_packets, len(channels)))
        Q_buffer = np.empty((Npackets + skip_packets, len(channels)))
        #self.fpga.write_int('GbE_pps_start', 1)
        count = 0
        while count < Npackets + skip_packets:
            packet = self.s.recv(conf.UDP_PACKET_LENGHT)
            if(len(packet) == conf.UDP_PACKET_LENGHT):
                #data = np.fromstring(packet,dtype = '<i').astype('float')
                data = np.frombuffer(packet[conf.UDP_PACKET_HEADER:], dtype = '<i').astype('float')
                ts = (np.frombuffer(packet[-4:], dtype = '<i').astype('float')/ conf.FPGA_SAMPLING_FREQUENCY)*1.0e3 # ts in ms
                odd_chan = channels[1::2]
                even_chan = channels[0::2]
                I_odd = data[1024 + ((odd_chan - 1) /2)]    
                Q_odd = data[1536 + ((odd_chan - 1) /2)]    
                I_even = data[0 + (even_chan/2)]    
                Q_even = data[512 + (even_chan/2)]    
                even_phase = np.arctan2(Q_even, I_even)
                odd_phase = np.arctan2(Q_odd, I_odd)
                if len(channels) % 2 > 0:
					if len(I_odd) > 0:
						I = np.hstack(zip(I_even[:len(I_odd)], I_odd))
						Q = np.hstack(zip(Q_even[:len(Q_odd)], Q_odd))
						I = np.hstack((I, I_even[-1]))    
						Q = np.hstack((Q, Q_even[-1]))
					else:
						I = I_even[0]
						Q = Q_even[0]
					I_buffer[count] = I
					Q_buffer[count] = Q
                else:
					I = np.hstack(zip(I_even, I_odd))
					Q = np.hstack(zip(Q_even, Q_odd))
					I_buffer[count] = I
					Q_buffer[count] = Q
                count += 1
            else: 
                print("Incomplete packet received of length {:d}".format(len(packet)))
                
        I_file = 'I' + str(LO_freq)
        Q_file = 'Q' + str(LO_freq)
        np.save(os.path.join(save_path,I_file), np.mean(I_buffer[skip_packets:], axis = 0)) 
        np.save(os.path.join(save_path,Q_file), np.mean(Q_buffer[skip_packets:], axis = 0)) 
        return 

    def ADC_to_dB(self, I, Q):
        mag = np.sqrt(I**2 + Q**2)
        mag /= conf.ADC_MAX_AMPLITUDE
        mag /= ((conf.ACCUMULATION_LENGTH - 1) / (conf.FFT_LENGTH/2))
        mag = 20*np.log10(mag)
        return mag

    def plot_vna(self, path):
        sweep_freqs, Is, Qs = ri.open_stored(path)
        sweep_freqs = np.load(path + '/sweep_freqs.npy')
        bb_freqs = np.load(path + '/bb_freqs.npy')
        rf_freqs = np.zeros((len(bb_freqs), len(sweep_freqs)))
        for chan in range(len(bb_freqs)):
			rf_freqs[chan] = ((sweep_freqs)/conf.MIXER_CONST + bb_freqs[chan])/1.0e6 #era sweep_freqs/2

        Q = np.reshape(np.transpose(Qs), (len(Qs[0])*len(sweep_freqs)))
        I = np.reshape(np.transpose(Is), (len(Is[0])*len(sweep_freqs)))
        mag = self.ADC_to_dB(I, Q)
        
        rf_freqs = np.hstack(rf_freqs)
        rll_chanf_freqs = np.concatenate((rf_freqs[len(rf_freqs)/2:], rf_freqs[:len(rf_freqs)/2]))

        fig = plt.figure()
        ax = fig.gca()
        ax.plot(rf_freqs, mag)
        ax.set_title('VNA sweep')
        ax.set_xlabel('Frequency [MHz]')
        ax.set_ylabel('Magnitude [dB]')
        plt.tight_layout()
        plt.ion()
        plt.show()
        return

    def plot_targ(self, path):
        lo_freqs, Is, Qs = ri.open_stored(path)
        
        lo_freqs = np.loadtxt(path + '/sweep_freqs.dat')
        bb_freqs = np.loadtxt(path + '/bb_freqs.dat')
        #tt_freqs = np.loadtxt(path + '/target_freqs.dat')
        target_freqs_new = np.loadtxt(path + '/target_freqs_new.dat')
        indexmin = np.load(path + '/index_freqs_new.npy')

        channels = len(bb_freqs)
        mags = np.zeros((channels, len(lo_freqs))) 
        chan_freqs = np.zeros((channels, len(lo_freqs)))
        new_targs = np.zeros((channels))
        for chan in range(channels):
            mags[chan] = self.ADC_to_dB(Is[:, chan], Qs[:, chan])
            chan_freqs[chan] = (lo_freqs/conf.MIXER_CONST + bb_freqs[chan])/1.0e6
        
        new_targs = [chan_freqs[chan][np.argmin(mags[chan])] for chan in range(channels)]

        fig = plt.figure()
        ax = fig.gca()
        for chan in range(channels):
            color = np.random.rand(3)
            ax.text(chan_freqs[chan][0], mags[chan][0], chan, color=color)
            ax.plot(chan_freqs[chan], mags[chan], color=color)
            ax.plot(target_freqs_new[chan], mags[chan][indexmin[chan]], linestyle='', marker='o', color=color)

        ax.set_title('Target sweep')
        ax.set_xlabel('Frequency [MHz]')
        ax.set_ylabel('Magnitude [dB]')
        plt.tight_layout()
        plt.ion()
        plt.show()
        return


    def open_stored(self, save_path = None):
        files = sorted(os.listdir(save_path))
        sweep_freqs = np.array([np.float(filename[1:-4]) for filename in files if (filename.startswith('I'))])
        I_list = [os.path.join(save_path, filename) for filename in files if filename.startswith('I')]
        Q_list = [os.path.join(save_path, filename) for filename in files if filename.startswith('Q')]
        Is = np.array([np.load(filename) for filename in I_list])
        Qs = np.array([np.load(filename) for filename in Q_list])
        return sweep_freqs, Is, Qs
    
    def make_format_time(self, dirfile_directory):
        formatname = dirfile_directory +'/format_time'
        logging.info("formatname: " + formatname)
        format_file = open(formatname, "w")   
        
        while True:
            print("entering while")
            packet = self.s.recv(conf.UDP_PACKET_LENGHT)
            print(len(packet))
            if len(packet) < conf.UDP_PACKET_LENGHT:
                continue
            else:
                logging.info("Packet received len: " + str(len(packet)))
                
                packet_count_bin_0 = packet[-1]+packet[-2]+packet[-3]+packet[-4]#packet[-4:]    
                self.packet_count_bin_0 = packet_count_bin_0
                print(packet_count_bin_0)
                print(type(packet_count_bin_0))
                logging.info("Packet_count_bin_0: " + packet_count_bin_0)
                packet_count_0 = struct.unpack("<L", packet_count_bin_0)[0]
                print(packet_count_0)
                ts0 = time.time()
                logging.info("ts0: " + str(ts0))
                break

        dt = 1. / self.accum_freq
        
        b = (ts0 - packet_count_0 * dt)
        a = dt

        line = "time_packet_count LINCOM packet_count"+"  "+str(a)+" "+str(b)
        logging.info(line)

        format_file.write(line)
        format_file.close()
        
        logging.info("format_time closed")

    def make_format_complex(self):

        '''
        This function writes the format_complex_extra file containing resonance circle parameters.
        '''

        formatname = self.datadir.as_posix()+'/format_complex_extra'
        I_center = self.centers.real
        Q_center = self.centers.imag
        cosi = np.cos(-self.rotations)
        sini = np.sin(-self.rotations)	

        print("saving format_complex_extra in "+ formatname)
		#ftrunc = np.hstack(freqs.astype(int))
        format_file = open(formatname, 'w')
        for i in range(len(self.radii)):

			format_file.write( 'chC_'+str(i).zfill(3)+' LINCOM   ch_'+str(i).zfill(3)+' 1  '+str(-I_center[i])+';'+str(-Q_center[i])+' # centered \n')
			format_file.write('chCR_'+str(i).zfill(3)+' LINCOM  chC_'+str(i).zfill(3)+' '+str(cosi[i])+';'+str(sini[i])+'   0  # centered and rotated \n') 
			format_file.write('chCr_'+str(i).zfill(3)+' LINCOM chCR_'+str(i).zfill(3)+' '+str(1/self.radii[i])+'   0 #centered, rotated and scaled   \n')
			format_file.write( 'chi_'+str(i).zfill(3)+' PHASE  '+'chCr_'+str(i).zfill(3)+'.r   0  # I centered \n')
			format_file.write( 'chq_'+str(i).zfill(3)+' PHASE  '+'chCr_'+str(i).zfill(3)+'.i   0  # Q centered \n')
			format_file.write( 'chp_'+str(i).zfill(3)+' PHASE  '+'chCr_'+str(i).zfill(3)+'.a   0  # Phase \n')
			format_file.write( 'cha_'+str(i).zfill(3)+' PHASE  '+'chCr_'+str(i).zfill(3)+'.m   0  # Magnitude \n \n')

        format_file.close()
        return


    def make_format(self, path_current = False):
        logging.info("Running make_format")

        if path_current:
			logging.info("current selected")
			formatname = conf.path_format.as_posix()+'/format_complex'
			freqs = self.cold_array_bb/1.e6+self.center_freq/self.divconst
        else:
			
			file_resonances = raw_input('Absolute path to a list of resonances basebands (e.g. /data/mistral/setup/kids/sweeps/target/current/bb_freqs.npy) ? ')
			freqs = np.load(file_resonances)/1.e6+self.center_freq/self.divconst
			folder_dirfile = raw_input('Dirfile folder (e.g. /data/mistral/data_logger/log_kids/) ? ')
			formatname = os.path.join(conf.path_format.as_posix(),'format_complex')
        
        logging.info("format_name: "+formatname)
        
        ftrunc = np.hstack(freqs.astype(int))
        format_file = open(formatname, 'w')
        
        for i in range(len(freqs)):
			decimal = int(freqs[i]*1000 % ftrunc[i])
			format_file.write('/ALIAS  KID_'+str(ftrunc[i])+'_'+str(decimal).zfill(3)+' chQ_'+str(i).zfill(3)+'  \n'   )

        format_file.close()
        
        logging.info("format_file done")

    
    def programLO(self, freq=200.0e6, sweep_freq=0):
        self.vi.simple_set_freq(0, freq)
        return

    def menu(self, prompt, options):
        print('\t' + prompt + '\n')
        for i in range(len(options)):
            print('\t' + '\033[32m' + str(i) + ' ..... ' '\033[0m' + options[i] + '\n')
        logging.info("Waiting for option: ")
        opt = input()
        return opt

    def print_useful_paths(self):
        print("\nUseful paths:")
        print("- Roach test: \t\t\t/home/mew/parameters/test")
        print("- Target directory: \t\t" + self.targetdir.as_posix())
        print("- Target current directory: \t" + self.targetcurrentdir.as_posix())
        print("- VNA directory: \t\t" + self.vnadir.as_posix())
        print("- VNA current directory: \t" + self.vnacurrentdir.as_posix())

    def main_opt(self):
        while True:
            opt = self.menu(self.main_prompt, self.main_opts)
			
            if opt == 0:
                self.clearScreen()
                self.initialize() 
				
            elif opt == 1:
                print("Writing test comb ({:d} tones)".format(conf.NUMBER_OF_TEST_TONES))
                self.writeQDR(self.test_comb, transfunc = False)
			
            elif opt == 2:
                self.print_useful_paths()
                file_path = raw_input('Absolute path to bb_freqs.dat: ')
                self.cold_array_bb = np.loadtxt(file_path)
                self.cold_array_bb = self.cold_array_bb[self.cold_array_bb != 0]
                rf_tones = (self.cold_array_bb + ((self.center_freq/conf.MIXER_CONST)*1.0e6))/1.0e6
                print("Done")

            elif opt == 3:
                self.print_useful_paths()
                file_path = raw_input('Absolute path to sweep_freqs.dat: ')
                self.path_configuration = Path(file_path)
                self.array_configuration()
                self.writeQDR(self.cold_array_bb)

            elif opt == 4:
                Npackets = input('\nNumber of UDP packets to stream? ' )
                chan = input('chan = ? ')
                self.stream_UDP(chan, Npackets)
				
            elif opt == 5:
                prompt = raw_input('Do plot after sweep? (y/n) ')
                if prompt == 'y':
                    self.vna_sweep(do_plot=True)
                else:
					self.vna_sweep(do_plot=False)
                fk.main(path=self.vnacurrentdir.as_posix(), savefile=True)
					
            elif opt == 6:
                path = raw_input("Path to a VNA sweep (e.g. {:s})".format(self.vnacurrentdir.as_posix()))
                fk.main(path, savefile=True)

            elif opt == 7:
                prompt = raw_input('Do plot after sweep? (y/n) ')
                if prompt == 'y':
					self.target_sweep(do_plot=True)
                else:
					self.target_sweep(do_plot=False)
                print("Setting frequencies to the located values ")
                time.sleep(0.7)
                if self.do_transf == True:
					self.writeQDR(self.cold_array_bb, transfunc = True)
                else:
					self.writeQDR(self.cold_array_bb)
					
            elif opt == 8:
                self.global_attenuation=input("Insert global attenuation (decimal, <1.0, e.g 0.01)")

            elif opt == 9:
                self.path_configuration = raw_input("Absolute path to a folder with freqs, centers, radii and rotations (e.g. /data/mistral/setup/kids/sweeps/target/current)")
                self.array_configuration()

            elif opt == 10:
                if self.path_configuration=='':
					print("Array configuration (freqs, centers, radii and rotations) undefined")
					self.path_configuration = raw_input("Absolute path to a folder with freqs, centers, radii and rotations (e.g.  /data/mistral/setup/kids/sweeps/target/current)")
					self.array_configuration()
                else: 
					print("Using array configuration from" , self.path_configuration)
                if "current" in sys.argv:
                    self.radii = np.load(self.path_configuration.as_posix() + "/radii.npy")
                    self.centers = np.load(self.path_configuration.as_posix() + "/centers.npy")
                    self.rotations = np.load(self.path_configuration.as_posix()+"/rotations.npy")
                nchannel=len(self.radii)
                #nchannel=input('number of channels?')#aggiunto 09082017 AP
                print("nchannles:", nchannel)
                try:
					self.dirfile_complex(nchannel)
                except KeyboardInterrupt:
					pass 

            elif opt == 11:
                path_to_vna = raw_input("Absolute path to a VNA folder (e.g. /home/mew/data/setup/kids/sweeps/vna/current)")
                self.plot_vna(path_to_vna)
                
            elif opt == 12:
                path_to_target = raw_input("Absolute path to a Target folder (e.g. /home/mew/data/setup/kids/sweeps/target/current)")
                self.plot_targ(path_to_target)

            elif opt == 13:
                sys.exit()
                
            else:
                pass
        return

if __name__=='__main__':
	ri = roachInterface()
	ri.main_opt()