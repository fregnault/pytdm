from pytdm import pytdm
import numpy as np

T = pytdm.TDm('testing_pluto_amr')

# Read TDm.conf
T.read_TDm_parameter()

originals_dir = '../data/pluto_amr/'

T.add_TDm_pluto_amr(
        35,
        originals_dir=originals_dir
        )
