import time, datetime

Timer_Start = time.time()
print('\nProgramme Starts at: {0}\n'.format(datetime.datetime.now()))

import numpy as np
# import healpy as hp
# import healpy.rotator as hpr
# import healpy.pixelfunc as hpf
# import healpy.visufunc as hpv
# import astropy as ap
# import matplotlib.pyplot as plt
# from astropy.io import fits
import glob
import sys
import os

sys.stdout.flush()

Frequency_Min = 110.
Frequency_Max = 120.
Frequency_Step = 5.

File_Start = 3
File_End = 21
File_Width = 18 # Number of files loaded into each iteration.
File_Step = 200 # Step to iterate Start_File

nside_start = 64  # starting point to calculate dynamic A
nside_standard = 64  # resolution of sky, dynamic A matrix length of a row before masking.
nside_beamweight = 16  # undynamic A matrix shape

Time_Average_preload = 1  # 12 # Number of Times averaged before loaded for each file (keep tails)'
Frequency_Average_preload = 1  # 16 # Number of Frequencies averaged before loaded for each file (remove tails)'

Valid_Threshold = 10**(-1.31)

for freq in np.arange(Frequency_Min, Frequency_Max, Frequency_Step):
    for id_file in range(File_Start, File_End, File_Step):
        sys.stdout.flush()
        print ('>>>>>>>>>>>>>>> Run Frequency: {0} for File(s): {1} <<<<<<<<<<<<<<<<<<<'.format(freq, id_file))
        try:
            os.system('ipython HERA-VisibilitySimulation-MapMaking.py {0} {1} {2} {3} {4} {5} {6} {7} {8}'.format(freq, File_Start, File_Start+File_Width, nside_start, nside_standard, nside_beamweight, Time_Average_preload, Frequency_Average_preload, Valid_Threshold))
        except:
            print ('>>>>>>>>>>>>>>> Error in Running Frequency: {0} for File(s): {1} <<<<<<<<<<<<<<<<<<<'.format(freq, id_file))

Timer_End = time.time()
try:
    print('\nProgramme Ends at: {0}'.format(datetime.datetime.now()))
    print('>>>>>>>>>>>>>>>>>>>>>> Total Used Time: {0} seconds. <<<<<<<<<<<<<<<<<<<<<<<<\n'.format(Timer_End - Timer_Start))
except:
    print('No Used Time Printed.')