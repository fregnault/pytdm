# pyTDm


Welcome to pyTDm ! This python module initializes a modified Titov-Démoulin
(Titov et al. 2014) flux rope and writes it in a
[PLUTO](http://plutocode.ph.unito.it/) (Mignone et al. 2014) data file. 

There is currently (October 2022) work to extend that module to be able to
add the TDm in the static version of PLUTO and also in EUHFORIA coronal model.

The code will read the `TDm.config` file (which contains the different parameter of the TDm)

# Installation

Put
```
git clone https://github.com/fregnault/pyTDm.git
export PATH=$PATH:/where/the/pyTDm/is
```

# Basic usage

Let `originals_dir` be the directory where you have the datafile to which to want
to add the TDm. 
    
    Put the original data (without the TDm) in the `originals_dir` directory

    Create a directory where you want the datafile with the TDm to be

    Run `init_TDm.py` to add the TDm


