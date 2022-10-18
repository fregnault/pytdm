![version](https://img.shields.io/badge/version-1.0.0-green)
![pythonversion](https://img.shields.io/badge/Python-3.7+-green)


# pyTDm : Python tool to initialize a TDm flux rope in a simulation

Welcome to pytdm ! 


This python module initializes a modified Titov-Démoulin
[(Titov et al.
2014)](https://iopscience.iop.org/article/10.1088/0004-637X/790/2/163) flux rope and writes it in a
[PLUTO](http://plutocode.ph.unito.it/) AMR data file. [(Mignone et al.
2012)](https://iopscience.iop.org/article/10.1088/0067-0049/198/1/7)

There is currently (October 2022) work to extend that module to be able to
add the TDm in the static version of PLUTO and also in the [COCONUT](https://iopscience.iop.org/article/10.3847/1538-4357/ac7237) coronal model.

The code reads the `TDm.config` file (which contains the different parameters of the TDm) and add it to the corresponding datafile.

If anyone is interested in using this module please send an email to
fl.regnault@gmail.com so that you can be helped in how to use this module
in the most efficient way.

# Installation

```
git clone https://github.com/fregnault/pyTDm.git
cd pytdm
pip install .
```

# Basic usage

To see an example of the insertion of a TDm flux rope in a PLUTO AMR datafile
see the [pluto_amr](https://github.com/fregnault/pytdm/tree/main/pytdm/examples/pluto_amr) example directory.


# Incoming features


 - Right now the module can add a TDm flux rope in a COCONUT simulation with the `add_TDm_coconut` but this has not been fully validated yes.
 - There is ongoing work in inserting the TDm in a dbl file of a static grid with the
   PLUTO simulation

# Acknowlegdments

The `pyPLUTO.py` was written by A. Strugarek, B. Perri and V. Réville.
