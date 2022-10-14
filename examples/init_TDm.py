#!/usr/bin/env python

import TDm as TDm_class
import numpy as np
import os,sys


cwd = os.getcwd()
case = cwd.split('/')[-1]

narg = len(sys.argv)

print(narg,sys.argv)

if narg==1:
   SolO=False
   iteration = 35
   filetype='chk'

    
elif narg==2:
    args = sys.argv[1].split(',')

    SolO=args[0] 
    if SolO:
        iteration = 72
    else:
        iteration = 35

    filetype = 'chk'
    SolO=args[0]

else:    
    args = sys.argv[1].split(',')

    if len(args) == 1:
        iteration = int(args[1])
        filetype = 'chk'

    elif len(args) == 2:
        iteration = int(args[1])
        filetype = args[2]



print(f'TDm setup for {case} with first iteration = {iteration} and filetype = {filetype}')

# Definition of TDm class
TDm = TDm_class.TDm_object(case)


TDm.add_magnetic_structure(iteration,filetype=filetype)






