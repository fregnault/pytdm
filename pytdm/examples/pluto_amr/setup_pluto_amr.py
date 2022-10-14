import pytdm as pytdm
import numpy as np

T = pytdm.TDm('testing_pluto_amr')

# Read TDm.conf
T.read_TDm_parameter()

originals_dir = '../../data/'

T.add_TDm_pluto_amr(
        35,
        originals_dir=originals_dir
        )
