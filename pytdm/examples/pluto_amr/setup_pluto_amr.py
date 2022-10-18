from pytdm import pytdm
import numpy as np

T = pytdm.TDm('testing_pluto_amr')

# The TDm parameters are loaded in the `add_TDm_pluto_amr` function

# Path to the directory where the "original" data file is
originals_dir = '../data/pluto_amr/'

# Adding the flux rope to the simulation
T.add_TDm_pluto_amr(
        35,
        originals_dir=originals_dir
        )
