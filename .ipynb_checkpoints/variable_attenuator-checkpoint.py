import serial
import time

class Attenuator:
	def __init__(self, port = "/dev/ttyACM0"): #default port
		self.conn = serial.Serial(None,
					9600, 
					serial.EIGHTBITS, 
					serial.PARITY_NONE, 
					serial.STOPBITS_ONE,
					timeout=0.5)
					
		self.conn.setPort(port)
		
	def get_att(self):
		
		if self.conn.isOpen() == False:
			self.conn.open()
			time.sleep(0.1) #for some reason opening the port takes some time

		
		self.conn.write("1\n") #sending 1 to the arduino asks for the attenuation values

		att_values = self.conn.read(1000) #reading att values

		att_values = att_values.split("\r")[0].split(",")
		att1 = att_values[0]
		att2 = att_values[1].split("\r")[0]
		
		#self.conn.close()
		
		return float(att1),float(att2)
		
	def set_att(self,channel,attenuation):
		
		if self.conn.isOpen() == False:
			self.conn.open()
			time.sleep(0.1)
			
		if (channel == 1 or channel == "RF-OUT"):
			channel_ = 2
		elif (channel == 2 or channel == "RF-IN"):
			channel_ = 3
		elif (channel == 3 or channel == "RF-IO"):
			channel_ = 4
		
		if (attenuation > 31.75):
			print("WARNING: attenuation can't be larger than 31.75 dB. Setting to max value.")
			attenuation_ = 31.74
		if (attenuation < 0):
			print("WARNING: attenuation can't be negative. Setting to 0")
			attenuation_ = 0
		elif (attenuation >= 0.0 and attenuation <= 31.75):
			attenuation_ = attenuation
		
		data = str(channel_)+"\n"+str(attenuation_)+"\n"
		
		self.conn.write(data)
		
		result = self.conn.read(1000)
		
		#self.conn.close()
		
		return str(result.split("\r")[0])
