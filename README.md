# pyTDm


Welcome to pyTDm ! 


This python module initializes a modified Titov-Démoulin
(Titov et al. 2014) flux rope and writes it in a
[PLUTO](http://plutocode.ph.unito.it/) AMR data file. (Mignone et al. 2014)

There is currently (October 2022) work to extend that module to be able to
add the TDm in the static version of PLUTO and also in EUHFORIA coronal model.

The code will read the `TDm.config` file (which contains the different parameters of the TDm) and add it to the corresponding datafile.

If anyone is interested in using this module please send an email to
fl.regnault@gmail.com so that you can be helped in how to use this module
in the most efficient way.

# Installation

```
git clone https://github.com/fregnault/pyTDm.git
export PATH=$PATH:/where/the/pyTDm/is
```

# Basic usage

## Bash executable
The `init_TDm.py` file is an executable that can be used in the following way
Let `originals_dir` be the directory where you have the datafile to which to want
to add the TDm. 
    
    - Put the original data (without the TDm) in a given directory

    - Create a directory where you want the datafile with the TDm to be

    - Run `init_TDm.py` to add the TDm

## Python (recommended)

A more flexible way would be to load the `pyTDm` module and then add the
magnetic structure in your data (that would have been already loaded) with python script.

To have a example you can have a look at the `init_TDm.py` file which does the
basic task.

