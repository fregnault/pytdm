''' File containing utility functions that are not related to the flux
rope itself '''

import numpy as np
from scipy.sparse import coo_matrix


def nufd1(x):
    ''' 
    Author : Antoine Strugarek (CEA)
    '''

    n = len(x)
    if (n == 1):
        return 1.
    h = x[1:]-x[:n-1]
    a0 = -(2*h[0]+h[1])/(h[0]*(h[0]+h[1]))
    ak = -h[1:]/(h[:n-2]*(h[:n-2]+h[1:]))
    an = h[-1]/(h[-2]*(h[-1]+h[-2]))
    b0 = (h[0]+h[1])/(h[0]*h[1])
    bk = (h[1:] - h[:n-2])/(h[:n-2]*h[1:])
    bn = -(h[-1]+h[-2])/(h[-1]*h[-2])
    c0 = -h[0]/(h[1]*(h[0]+h[1]))
    ck = h[:n-2]/(h[1:]*(h[:n-2]+h[1:]))
    cn = (2*h[-1]+h[-2])/(h[-1]*(h[-2]+h[-1]))
    val  = np.hstack((a0,ak,an,b0,bk,bn,c0,ck,cn))
    row = np.tile(np.arange(n),3)
    dex = np.hstack((0,np.arange(n-2),n-3))
    col = np.hstack((dex,dex+1,dex+2))
    D = coo_matrix((val,(row,col)),shape=(n,n))

    return D


def ComputeRot(x1,x2,x3,var1,var2,var3,geom='cartesian'):
    ''' Compute the rotation of the [var1,var2,var3] vector
    Author : Antoine Strugarek (CEA)
    '''

    Rot1=np.zeros_like(var1); Rot2=np.zeros_like(var1); Rot3=np.zeros_like(var1);
    if geom == 'cartesian':
        deriv_x = nufd1(x1) ; deriv_y = nufd1(x2) ; deriv_z = nufd1(x3)
        if len(x3) != 1:
            for ix,xx in enumerate(x1):
                for iz,zz in enumerate(x3):
                    Rot1[ix,:,iz] +=  (deriv_y*var3[ix,:,iz])
                    Rot3[ix,:,iz] += -(deriv_y*var1[ix,:,iz])
                for iy,yy in enumerate(x2):
                    Rot1[ix,iy,:] += -(deriv_z*var2[ix,iy,:])
                    Rot2[ix,iy,:] +=  (deriv_z*var1[ix,iy,:])
            for iy,yy in enumerate(x2):
                for iz,zz in enumerate(x3):
                    Rot2[:,iy,iz] += -(deriv_x*var3[:,iy,iz])
                    Rot3[:,iy,iz] +=  (deriv_x*var2[:,iy,iz])
        else:
            print("Rotational not coded yet in cartesian in 2.5D")
    elif geom == 'spherical':
        deriv_r = nufd1(x1) ; deriv_t = nufd1(x2) ; deriv_p = nufd1(x3)
        if len(x3) == 1:
            for ix,xx in enumerate(x1):
                Rot1[ix,:] +=  (deriv_t*(np.sin(x2)*var3[ix,:]))/(xx*np.sin(x2))
                Rot3[ix,:] += -(deriv_t*var1[ix,:])/xx
            for iy,yy in enumerate(x2):
                Rot2[:,iy] += -(deriv_r*(x1*var3[:,iy]))/x1
                Rot3[:,iy] +=  (deriv_r*(x1*var2[:,iy]))/x1
        else:
            for ix,xx in enumerate(x1):
                for iz,zz in enumerate(x3):
                    Rot1[ix,:,iz] +=  (deriv_t*(np.sin(x2)*var3[ix,:,iz]))/(xx*np.sin(x2))
                    Rot3[ix,:,iz] += -(deriv_t*var1[ix,:,iz])/xx
                for iy,yy in enumerate(x2):
                    Rot1[ix,iy,:] += -(deriv_p*var2[ix,iy,:])/(xx*np.sin(yy))
                    Rot2[ix,iy,:] +=  (deriv_p*var1[ix,iy,:])/(xx*np.sin(yy))
            for iy,yy in enumerate(x2):
                for iz,zz in enumerate(x3):
                    Rot2[:,iy,iz] += -(deriv_r*(x1*var3[:,iy,iz]))/x1
                    Rot3[:,iy,iz] +=  (deriv_r*(x1*var2[:,iy,iz]))/x1
    else:
        print('Rotation for '+geom+' not coded yet')

    return [Rot1,Rot2,Rot3]
