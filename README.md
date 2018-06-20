# HERA_MapMaking_VisibilitySimulation

## If you want to install the package:
1. conda create -n yourenvname python=2.x
2. source activate {path to your env}, (check which python) 
3. pre-install: 
* numpy >= 1.10
* scipy    
* matplotlib
* astropy >=1.2
* pyephem
* [aipy](https://github.com/HERA-Team/aipy/)
* [pyuvdata](https://github.com/HERA-Team/pyuvdata/)
* [omnical](https://github.com/HERA-Team/omnical/) >= 5.0.2
* [linsolve](https://github.com/HERA-Team/linsolve)
* [hera_qm](https://github.com/HERA-Team/hera_qm)
4. cd {path to HERA_MapMaking_VisibilitySimulation} 
5. python setup.py install
  
## If you want to look at the main script:
1. go to /scripts
2. open HERA-VisibilitySimulation-MapMaking.py
  
## If you want to use or look at the results:
1. go to /Maps_Blender
2. go to /Maps_Blender/SelectedPlots_IDR2.1_x for better results
3. there will be '.fits' file for results sky map (Data and GSM) with non-zero values on valid pixels
