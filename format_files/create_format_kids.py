import numpy as np

n_channels = 1000

fp = open("format_complex","w")

fp.write("time    RAW d   1\n")
fp.write("/ALIAS  FRAMENUM    packet_count\n")

for linetype in ("I","Q"):
    for i in range(0, n_channels):
 
        if i == 0 and i%10 == 0:           
            fp.write("\n")
        chnum = str(i).zfill(3)
        string = "ch"+linetype+"_"+chnum+"\t"+"RAW"+"\t"+"INT32"+"\t1\n"
        fp.write(string)

for i in range(0, n_channels):
    
    chnum = str(i).zfill(3)
    #string = "ch"+linetype+"_"+chnum+"\t"+"RAW"+"\t"+"INT32"+"\t1\n"
    string = "ch"+"_"+chnum+"\tLINCOM\t2\t"+"chI_"+chnum+"\t1\t0\t"+"chQ_"+chnum+"\t0;1\t0\n"    
    fp.write(string)

fp.write("/INCLUDE format_complex_extra")


fp.close()

#np.savetxt("format_file_generated",format_file)

