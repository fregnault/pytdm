from pytdm import pytdm
import numpy as np

x = np.linspace(-1,1,50)
y = np.linspace(-1,1,50)
z = np.linspace(0,1,25)


xx,yy,zz = np.meshgrid(x,y,z,indexing='ij')
