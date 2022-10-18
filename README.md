![version](https://img.shields.io/badge/version-1.0.0-green)
![pythonversion](https://img.shields.io/badge/Python-3.7+-green)


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
pip install .
```

# Basic usage

To see an example of the insertion of a TDm flux rope in a PLUTO AMR datafile
see the [examples](https://github.com/fregnault/pytdm/tree/main/pytdm/examples/pluto_amr) directory.
