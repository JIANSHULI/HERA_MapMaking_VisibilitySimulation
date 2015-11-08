__author__ = 'omniscope'
import numpy as np

import simulate_visibilities.simulate_visibilities as sv
import time

bnside = 256
freqs = range(110, 200, 10)

raw_beam_data = np.concatenate([np.fromfile('/home/omniscope/data/mwa_beam/healpix_%i_%s.bin' % (bnside, p), dtype='complex64').reshape(
    (len(freqs), 12 * bnside ** 2, 2)) for p in ['x', 'y']], axis=-1).transpose(0, 2, 1) #freq by 4 by pix
sv.plot_jones(raw_beam_data[5])


vstest = sv.Visibility_Simulator()
vstest.initial_zenith = np.array([0, np.pi/4])
tm = time.time()
beam_heal_equ = np.array(
            [sv.rotate_healpixmap(beam_healpixi, 0, np.pi / 2 - vstest.initial_zenith[1], vstest.initial_zenith[0]) for
             beam_healpixi in raw_beam_data[5]])
print (time.time()-tm) / 60.
sv.plot_jones(beam_heal_equ)

