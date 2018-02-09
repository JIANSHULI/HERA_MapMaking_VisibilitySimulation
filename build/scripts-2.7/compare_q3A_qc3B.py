__author__ = 'omniscope'

import numpy as np
import glob, os

vis_tags = ['q3AL', 'qC3B']
ubls = {}
tflists = {}
Ni = {}
vis_data = {}

for tag_i, tag in enumerate(vis_tags):
    datatag = '_2016_01_20_avg'
    vartag = '_2016_01_20_avg'
    datadir = '/home/omniscope/data/GSM_data/absolute_calibrated_data/'
    nf = 1

    data_filenames = glob.glob(datadir + tag + '_*_abscal_xx*' + datatag)
    for data_filename in data_filenames:
        for p, pol in enumerate(['xx', 'xy', 'yx', 'yy']):
            #tf file
            tf_basename = '_'.join(os.path.basename(data_filename).split('_')[:5]) + '_%i.tf'%nf
            tf_filename = os.path.dirname(data_filename) + '/' + tf_basename
            nt = len(np.fromfile(tf_filename, dtype='complex64')) / nf
            tflist = np.fromfile(tf_filename, dtype='complex64').reshape((nt,nf))
            tflists[data_filename] = tflist
            tlist = np.real(tflist[:, 0])
            flist = np.imag(tflist[0, :])    #will be later assuming flist only has 1 element
            vis_freq = flist[0] / 1e3
            print data_filename, vis_freq, nt, tlist[:3]

            #ubl file
            ubl_filename = glob.glob('_'.join(tf_filename.split('_')[:-2]) + '_*_3.ubl')[0]
            nUBL = len(np.fromfile(ubl_filename, dtype='float32')) / 3
            ubls[data_filename] = np.fromfile(ubl_filename, dtype='float32').reshape((nUBL, 3))
            print "%i UBLs to include"%len(ubls[data_filename])

            #get Ni (1/variance) and data
            var_filename = data_filename.replace(datatag, vartag+'.var').replace('xx', pol)
            if p == 0:
                Ni[data_filename] = np.zeros((4, nUBL, nt), dtype='float32')
                vis_data[data_filename] = np.zeros((4, nUBL, nt), dtype='complex64')
            Ni[data_filename][p] = 1./(np.fromfile(var_filename, dtype='float32').reshape((nt, nUBL)).transpose())
            vis_data[data_filename][p] = np.fromfile(data_filename.replace('xx', pol), dtype='complex64').reshape((nt, nUBL)).transpose()
