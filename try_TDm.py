import importlib as imp
import TDm as TDm_class
imp.reload(TDm_class)
import numpy as np
import os,sys
import pyPLUTO as pp


case = 'case_try'

TDm = TDm_class.TDm_object(case)


TDm.add_magnetic_structure(35,filetype='chk')


D = pp.pload(
        35,
        w_dir = '',
        datatype = 'hdf5',
        x1range=[1,1.5],
        level = -1,
        filetype = 'chk'
        )



D2 = pp.pload(
        35,
        w_dir = 'rthinner_z10/',
        datatype = 'hdf5',
        x1range=[1,1.5],
        level = -1,
        filetype = 'chk'
        )


diff1 = getattr(D,'X-magnfield') - getattr(D2,'X-magnfield')
diff2 = getattr(D,'Y-magnfield') - getattr(D2,'Y-magnfield')
diff3 = getattr(D,'Z-magnfield') - getattr(D2,'Z-magnfield')
