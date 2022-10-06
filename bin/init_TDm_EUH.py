#!/usr/bin/env python

import pyTDm as TDm_class
import numpy as np
import os,sys



# Get the name of the current directory
cwd = os.getcwd()
case = cwd.split('/')[-1]

# Definition of TDm class
TDm = TDm_class.TDm_object(case)


filename = 'corona-flow0-P' # prefix of CF files
filedir = '/home/fregnault/data/Simulation/TDm_EUHFORIA/originals/dipole_5/' # Location of CF files

print('Taking dipole as original')

# Adding attributes to the TDm object with values that are needed for the pyTDm
# setup
TDm.CF_param = dict({})

TDm.CF_param['nb_proc'] = 108
TDm.CF_param['nb_r'] = 200
TDm.CF_param['nb_th'] = 100
TDm.CF_param['nb_phi'] = 200
TDm.CF_param['eps'] = 0.01

TDm.add_flux_rope_EUHFORIA(
        2, # Dumb value for now
        filename,
        ori_dir=filedir,
        )










