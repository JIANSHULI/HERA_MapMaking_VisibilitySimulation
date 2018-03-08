The beams here originate from Nicolas Fagnoni's simulations, found at:

https://github.com/Nicolas-Fagnoni/Simulations

These are power beams and the script `convert_to_healpix.py` was used to convert the directivity beams from the CST output to healpix. This was done on 12 May, 2017, using Simulations git has 587e076. Only "East" polarization is available from the simulation, but "North" can be created by rotating ninety degrees because there is nothing in the simulation that would break the symmetry.

The beams are centered at theta = 0, and East at phi = 0 in healpix coordinate. An example script to read in the data and plot is provided, `example_beam_read.py`.
