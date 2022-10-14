##
import sys
import numpy as np
from subprocess import call
from . import formula_TDm
import imp
from tqdm import tqdm
import os 
import pyPLUTO as pp
import struct

import h5py as h5

# For loading EUHFORIA 
from pyevtk.hl import gridToVTK
import pyvista as pyv
from vtk.util.numpy_support import vtk_to_numpy
from scipy.interpolate import griddata
from scipy.interpolate import Rbf
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

##

class TDm:
    ''' TODO : you may want to add a function that will create a model of your
    TDm.conf file
    '''

    def __init__(self,case):
        ''' Initialisation
        ''' 
        
        print('Welcome to pyTDm')
        # print('Warning this is an alpha release')
        print('If you have any questions regarding this please')

        print('contact : fl.regnault@gmail.com')


        self.is_setup = False
        self.param_loaded = False
        self.case = case

	def __repr__(self):
        description = f'TDm flux rope {name}'
            
        if self.param_loaded = True
            for it in self.setup.keys():
                description += f'{it} : {self.setup[it]}')

        return  description

    # Import setup parameter
    def read_TDm_parameter(self,conf_path = './'):
        ''' 
        Put the parameter of the TDm setup into a dictonnary 

        The conf file has to be named "TDm.config" and at the {conf_path}
        location which is by default the name of the case 
        '''
        
        with open('{}/TDm.config'.format(conf_path),'r') as f:
                lines = f.readlines()

        self.setup = dict({})
       
        # Loop over lines
        for l in lines:
            
            # This is not a interesting line
            if l[0] in ['[','\n']:
                continue

            # Sanity check
            if len(l.split(' ')) > 2:
                print(l)
                raise ValueError("Too much columns in conf file")

            name_var = l.split(' ')[0]

            if name_var in ['CASE_TDM','GEOMETRY']:
                var = str(l.split(' ')[1])
                var = var.strip('\n')
            else:
                var = float(l.split(' ')[1])


            self.setup[name_var] = var
        
        self.param_loaded = True
        
        print(f'Reading setup at {conf_path}')

        print(f'----------------------')
        print(f'-------TDm setup------')
        print(f'----------------------')

        for it in self.setup.keys():
            
            print(f'{it} : {self.setup[it]}')
            print(f'----------------------')

    
    def cart_to_sph(vcart,xx2,xx3): 

        sinth=sin(xx2);
        costh=cos(xx2);
        sinph=sin(xx3);
        cosph=cos(xx3);
        vsph = [0]*3

        vsph[0] =  sinth*cosph*vcart[0] + sinth*sinph*vcart[1] + costh*vcart[2] ;
        vsph[1] =  costh*cosph*vcart[0] + costh*sinph*vcart[1] - sinth*vcart[2] ;
        vsph[2] = -sinph*vcart[0] + cosph*vcart[1] ;

        return vsph

    def get_monopoles(self,x1,x2,x3):
        ''' Add two monopoles above and below the TDm FR with opposite polarity
        '''

        tTDm = self.setup['X_1'] 
        pTDm = self.setup['X_2'] 

        # Setting the grid
        xx1,xx2,xx3 = np.meshgrid(x1,x2,x3,indexing='ij')
        xx = xx1*np.sin(xx2)*np.cos(xx3);
        yy = xx1*np.sin(xx2)*np.sin(xx3);
        zz = xx1*np.cos(xx2);
            
        # initialisation
        Bcart = [0]*3
        r0=0.9;

        #First monopole

        th0=tTDm+0.2;
        ph0=pTDm;
        x0 = r0*np.sin(th0)*np.cos(ph0);
        y0 = r0*np.sin(th0)*np.sin(ph0);
        z0 = r0*np.cos(th0);
        rp = np.sqrt( pow(xx-x0,2)+pow(yy-y0,2)+pow(zz-z0,2));
        Bcart[0] = - (xx-x0)/pow(rp,3.);
        Bcart[1] = - (yy-y0)/pow(rp,3.);
        Bcart[2] = - (zz-z0)/pow(rp,3.);

        B_sph1 = formula_TDm.cart_to_sph(Bcart,xx2,xx3);

        #Second monopole

        th0=tTDm-0.2;
        ph0=pTDm;
        x0 = r0*np.sin(th0)*np.cos(ph0);
        y0 = r0*np.sin(th0)*np.sin(ph0);
        z0 = r0*np.cos(th0);
        rp = np.sqrt( pow(xx-x0,2)+pow(yy-y0,2)+pow(zz-z0,2));
        Bcart[0] =  (xx-x0)/pow(rp,3.);
        Bcart[1] =  (yy-y0)/pow(rp,3.);
        Bcart[2] =  (zz-z0)/pow(rp,3.);

        B_sph2 = formula_TDm.cart_to_sph(Bcart,xx2,xx3);

        B_sph = [0]*3
        for i in range(0,3):
            B_sph[i] = B_sph1[i] + B_sph2[i]
        
        return B_sph

    # =====================================================================
    def setup_bc(self,B_amb,d,a,R,Delta,case,zeta,
            alpha_0,theta_0,phi_0,it,bDirOut,ar_grid,B_grid):
        '''
        setup in binary files the boundary conditions (surface of the sun) for the
        magnetic field
        write it in a .bin file

        Sizes of array in each direction (3) are written in bc_sizes.bin and the
        data is in bc_data.bin
        '''

        # Estimation of the ambient magnetic field at the location of the FR

        r = ar_grid[0]
        theta = ar_grid[1]
        phi = ar_grid[2]

        # Getting the grid for the TDm setup
        dr = r[1] - r[0]

        # We are only interested by r -2dr and r-dr
        # We add 2 points around these value in order to avoid uncentered derivative
        # in the curl 
        ghost_r = np.array([r[0]-3*dr,r[0]-2*dr,r[0]-dr,r[0],r[1],r[2],r[3],r[4]])
        print(ghost_r)

        real_ghost = ghost_r[1:4]
        r_real_ghost = range(1,4)

        # Setting TDm on ghost cell 
        B = formula_TDm.TDm_setup(
                ghost_r,
                theta,
                phi,
                alpha_0,
                theta_0,
                phi_0,
                d,
                a,
                R,
                zeta,
                B_amb,
                Delta,
                case,
                geometry='spherical',
                )

        B_r = B[0]
        B_t = B[1]
        B_p = B[2]
    
        # In this model we define different boundary condition close to the
        # foot of the TDm FR. To define the contour of this boundary we compute
        # the magnetic energy of the B from the TD and of the solar wind. 
        # The boundary are changed in PLUTO if E_TD / E_sw > 0.5

        E_TD = np.sqrt(B_r[3,...]**2 + B_t[3,...]**2 + B_p[3,...]**2)
        E_sw = np.sqrt(B_grid[0][0,...]**2 + B_grid[1][0,...]**2 + B_grid[2][0,...]**2)

        ratio = E_TD / E_sw

        flag = (ratio > 0.5).astype(int)
        

        arrays = dict({
            'B_r':B_r,
            'B_t':B_t,
            'B_p':B_p,
            'r':r,
            'theta':theta,
            'phi':phi
            })

        print('Writing bin files')
        with open('{}/flag_bc.bin'.format(bDirOut),'wb') as flag_bc:      
            for j in range(len(theta)):
                for k in range(len(phi)):
                    flag_bc.write(struct.pack('i',flag[j,k]))

        # Saving the size in each direction in a binary arrays that contains
        # 3 int
        size_ar = np.array([len(real_ghost),len(theta),len(phi)])

        # Saving size
        with open('{}/bc_sizes.bin'.format(bDirOut),'wb') as f:      
            for int_value in size_ar:                      
                f.write(struct.pack('i', int_value))  

        # Saving the actual data
        with open('{}/bc_data.bin'.format(bDirOut),'wb') as f_array:

            for i in r_real_ghost:
                for j in range(len(theta)):
                    for k in range(len(phi)):
                        f_array.write(struct.pack('d',B_r[i,j,k]))

            for i in r_real_ghost:
                for j in range(len(theta)):
                    for k in range(len(phi)):
                        f_array.write(struct.pack('d',B_t[i,j,k]))

            for i in r_real_ghost:
                for j in range(len(theta)):
                    for k in range(len(phi)):
                        f_array.write(struct.pack('d',B_p[i,j,k]))

            for i in range(len(real_ghost)):
                f_array.write(struct.pack('d',real_ghost[i]))

            for i in range(len(theta)):
                f_array.write(struct.pack('d',theta[i]))
            
            for i in range(len(phi)):
                f_array.write(struct.pack('d',phi[i]))

        # with open('{}/surface.bin'.format(bDirOut),'wb') as f_array:

            # for j in range(len(theta)):
                # for k in range(len(phi)):
                    # f_array.write(struct.pack('d',B_r_s[j,k]))

            # for j in range(len(theta)):
                # for k in range(len(phi)):
                    # f_array.write(struct.pack('d',B_t_s[j,k]))

            # for j in range(len(theta)):
                # for k in range(len(phi)):
                    # f_array.write(struct.pack('d',B_p_s[j,k]))




    # -------------------------------------------------------------------
    def add_TDm_pluto(self,iteration,originals_dir,filetype='chk',geometry='spherical'):
        ''' 
        Add the magnetic structure to an existing hdf5 files

        Inputs
        ======
        iteration: iteration of the PLUTO file at which we want to add the TDm
        
        originals_dir: path of the directory containing the data to which we
        add the TDm (and the other necessary file to read it, ie grid.out for
        PLUTO AMR file)

        filetype: 'chk' (default) or 'data', type of PLUTO file to which we want to add
        the TDm
        
        geometry: 'spherical' (default) or 'cartesian', type of geometry to use
        as output of the magnetic field
        '''
        
        print('Spherical geometry is hard coded in formula_TDm.TDm_setup')

        originals_dir = os.environ['TDM_ORIGINALS']

        # Directory where the filed to be modified is
        bdirIn=originals_dir        
        print('ori_dir = ',bdirIn)

        # Directory where to put the modified file
        # Setting to current directory  
        cwd = os.getcwd()
        bdirOut=f'{cwd}/'


        self.read_TDm_parameter()
        
        alpha_0 = self.setup['ALPHA_0']
        theta_0 = self.setup['X1']
        phi_0   = self.setup['X2']

        # TDm parameters setup
        d = self.setup['D']
        a = self.setup['A']
        R = self.setup['R']
        Delta = self.setup['DELTA']
        case = self.setup['CASE_TDM'] 
        zeta = self.setup['ZETA'] 


        # In PLUTO the star always have a radius = 1
        R_star = 1

        # Gamma = 1.05

        ##

        ################## the rest is automated, see on the 'Modify PRIMITIVE variables here' flag for modification 

        # Dictionnary for the equivalence between the conservative and primitive
        # variable Cons

        dict_equi_ConsToPrim = dict({
            'Density':'rho',
            'X-momentum':'vx1',
            'Y-momentum':'vx2',
            'Z-momentum':'vx3',
            'X-magnfield':'bx1',
            'Y-magnfield':'bx2',
            'Z-magnfield':'bx3',
            'energy-density':'prs',
            'psi_glm':'psi_glm'
            })


        for loading_file in [filetype]:

            call("mkdir -p "+bdirOut,shell=True)
            FileIn  = bdirIn+"{}.{:04d}.hdf5".format(loading_file,iteration)
            FileOut = bdirOut+"{}.{:04d}.hdf5".format(loading_file,iteration)
            call("cp "+FileIn+" "+FileOut,shell=True)

            ### Primitive variables (with div_cleaning) are, in order of storage:
            ###      rho, vr, vt, vp, br, bt, bp, prs, psi_glm
            ###
            ### Conservative variables are, in order of storage:
            ###      rho, rho.vr, rho.vt, rho.vp, br, bt, bp, E, psi_glm
            ###
            ### with
            ###      E = (1/2).(vr**2+vt**2+vp**2) + (1/2).(br**2+bt**2+bp**2) + rho.e
            ###
            ### and rho.e depends on the equation of state. In our case, it is
            ###      rho.e = prs / (Gamma-1)
            ###
            ### with Gamma set in pluto.ini (usually Gamma=1.05 in our polytropic models)

            def PrimToCons(Prim,Gamma=1.05):
                ''' Primitive to Conservative variables
                '''

                Cons=np.zeros_like(Prim)
                Cons[:,:,:,0] = Prim[:,:,:,0] # rho

                if Prim.shape[-1] == 9:

                    Cons[:,:,:,-1] = Prim[:,:,:,-1] # psi_glm

                for i in range(3):

                    Cons[:,:,:,i+1] = Prim[:,:,:,0]*Prim[:,:,:,i+1] # rho.v
                    Cons[:,:,:,i+4] = Prim[:,:,:,i+4] # B
                    Cons[:,:,:,7] = 0.5*Prim[:,:,:,0]*\
                        (Prim[:,:,:,1]**2+Prim[:,:,:,2]**2+Prim[:,:,:,3]**2) + \
                                0.5*(Prim[:,:,:,4]**2+Prim[:,:,:,5]**2+Prim[:,:,:,6]**2) + \
                                Prim[:,:,:,7]/(Gamma-1)

                return Cons
            


            def ConsToPrim(Cons,Gamma=1.05):
                ''' Conservative to Primitives variables
                '''

                Prim=np.zeros_like(Cons)
                Prim[:,:,:,0] = Cons[:,:,:,0] # rho

                if Cons.shape[-1] == 9:

                    Prim[:,:,:,-1] = Cons[:,:,:,-1] # psi_glm

                for i in range(3):

                    Prim[:,:,:,i+1] = Cons[:,:,:,i+1]/Cons[:,:,:,0] # v
                    Prim[:,:,:,i+4] = Cons[:,:,:,i+4] # B
                    Prim[:,:,:,7] = (Cons[:,:,:,7] - 0.5*Prim[:,:,:,0]* \
                        (Prim[:,:,:,1]**2+Prim[:,:,:,2]**2+Prim[:,:,:,3]**2) -\
                        0.5*(Prim[:,:,:,4]**2+Prim[:,:,:,5]**2+Prim[:,:,:,6]**2))*(Gamma-1)

                return Prim
            
            print('Loading {}.{:04d}.hdf5'.format(loading_file,iteration))
            fp = h5.File(FileOut,'r+')

            Dvars = []
            for iv in range(fp.attrs.get('num_components')):
                Dvars.append(fp.attrs.get('component_'+str(iv)).decode())

            if loading_file == 'chk': 
                list_var = []

                for var in Dvars:
                    list_var.append(dict_equi_ConsToPrim[var])

            else: 
                list_var = Dvars

            
            dir_grid = '{}/'.format(case)
            path_grid = '{}/grid_{}.npy'.format(bdirIn,iteration)


            print(path_grid) 

            # In order to avoid huge file loading the grid can be saved in a npy file 
            # Checking if the grid of the original hdf5 file has been already
            # saved

            if os.path.exists(path_grid):

                ar_grid = np.load(path_grid,allow_pickle=True)
                case_loaded = False

            else:

                print('{} does not exists creating it'.format(path_grid))
                # Loading the grid of the case that we will modify
                # However, modifications are not not thanks to the D
                # (pyPLUTO) object. There are done directly in the hdf5
                # file later in the code 
                D = pp.pload(
                        iteration,
                        w_dir=bdirIn,
                        level=-1,
                        datatype='hdf5',
                        filetype='chk'
                        )

                case_loaded = True
                ar_grid = np.array([D.x1,D.x2,D.x3])




                np.save(path_grid,ar_grid)

            
            # Getting r, theta and phi from ar_grid
            r = ar_grid[0]
            theta = ar_grid[1]
            phi = ar_grid[2]

            # Getting the magnetic field of the dipole a the place of the flux rope
            pos_tube_r = np.argmin(np.abs(r - (R - d + a + R_star)))
            pos_tube_t = np.argmin(np.abs(theta - theta_0))
            pos_tube_p = np.argmin(np.abs(phi - phi_0))
            
            print('B amb is taken a R - d + a')
            # path_B_grid = 'grid_B_{}.npy'.format(iteration)
            path_B_grid = '{}/grid_B_{}.npy'.format(bdirIn,iteration)
            path_prs_grid = '{}/grid_prs_{}.npy'.format(bdirIn,iteration)
            
            print(path_B_grid) 

            # Checking if the B_grid already exists
            if os.path.exists(path_B_grid):

                print('{} already exists'.format(path_B_grid))

                B_grid = np.load(path_B_grid,allow_pickle=True)

            # If it does not then check if we have already loaded the
            # original case 
            else:
                
                print('{} does not exists creating it'.format(path_B_grid))
                if case_loaded:

                    # Case already loaded, no need to reload
                    B_grid = np.array([
                        getattr(D,'X-magnfield'),
                        getattr(D,'Y-magnfield'),
                        getattr(D,'Z-magnfield'),
                        ])
                    

                else:
                    # Loading the B grid of the case that we will modify
                    # However, modifications are not not thanks to the D
                    # (pyPLUTO) object. There are done directly in the hdf5
                    # file later in the code 
                    D = pp.pload(
                            iteration,
                            w_dir=bdirIn,
                            level=-1,
                            # x1range=[1,r_lim],
                            datatype='hdf5',
                            filetype='chk'
                            )
                        
                    B_grid = np.array([
                        getattr(D,'X-magnfield'),
                        getattr(D,'Y-magnfield'),
                        getattr(D,'Z-magnfield'),
                        ])
                    
                    print('Saving')
                    # Saving
                    np.save(
                            path_B_grid,
                            B_grid
                            )
                


            B_amb = np.sqrt(
                  B_grid[0][pos_tube_r,pos_tube_t,pos_tube_p]**2 \
                + B_grid[1][pos_tube_r,pos_tube_t,pos_tube_p]**2 \
                + B_grid[2][pos_tube_r,pos_tube_t,pos_tube_p]**2
                )
            

            print(f'B_amb is {B_amb}')
            # print(f'B_amb_D is at {B_amb_D}')


            # Setting boundary condition 
            self.setup_bc(B_amb,d,a,R,Delta,case,zeta,
                    alpha_0,theta_0,phi_0,
                    iteration,
                    bdirOut,
                    ar_grid,
                    B_grid,
                    )

            # Getting the index of B parameters
            Brvar = list_var.index('bx1')
            Btvar = list_var.index('bx2')
            Bpvar = list_var.index('bx3')
            Rhovar = list_var.index('rho')
            Prsvar = list_var.index('prs')

            print(list_var) 


            # Number of parameters
            nvar = len(list_var)

            # I can now modify the OutFile
            # Read the file
            # Loop over the data to be modified
            nlev = fp.attrs.get('num_levels')
            dim = fp['Chombo_global'].attrs.get('SpaceDim')
            
            # Loop over the different level
            for i in tqdm(range(nlev),desc='level'):

                fl = fp["level_{:d}".format(i)]
                data = fl['data:datatype=0']
                boxes = fl['boxes']
                nbox = len(boxes['lo_i'])
                x1b = fl.attrs.get('domBeg1')

                if (dim == 1):
                    x2b = 0
                else:
                    x2b = fl.attrs.get('domBeg2')

                if (dim == 1 or dim == 2):
                    x3b = 0
                else:
                    x3b = fl.attrs.get('domBeg3')

                dx = fl.attrs.get('dx')
                logr = fl.attrs.get('logr')
                ystr = 1. ; zstr = 1. 

                if (dim >= 2):
                    ystr = fl.attrs.get('g_x2stretch')

                if (dim == 3):
                    zstr = fl.attrs.get('g_x3stretch')
                
                ncount=0
                for j in tqdm(range(nbox),desc='box'): # loop on all boxes of a given level

                    ib = boxes[j]['lo_i'] ; ie = boxes[j]['hi_i'] ; nbx = ie-ib+1
                    jb = 0 ; je = 0 ; nby = 1
                    kb = 0 ; ke = 0 ; nbz = 1

                    if (dim > 1):
                        jb = boxes[j]['lo_j'] 
                        je = boxes[j]['hi_j'] 
                        nby = je-jb+1

                    if (dim > 2):
                        kb = boxes[j]['lo_k'] 
                        ke = boxes[j]['hi_k'] 
                        nbz = ke-kb+1

                    szb = nbx*nby*nbz*nvar
                    q=data[ncount:ncount+szb].reshape((nvar,nbz,nby,nbx)).T

                    # q has the variables in a table [nx_loc,ny_loc,nz_loc,nvar]
                    # x1,x2,x3 are the local space dimensions
                    if logr == 0:
                        x1 = x1b + (ib+np.array(range(nbx))+0.5)*dx

                    else:
                        x1 = x1b*(np.exp((ib+np.array(range(nbx))+1)*dx)\
                                +np.exp((ib+np.array(range(nbx)))*dx))*0.5

                    x2 = x2b + (jb+np.array(range(nby))+0.5)*dx*ystr
                    x3 = x3b + (kb+np.array(range(nbz))+0.5)*dx*zstr

                    # First get primitive variables
                    qold=q.copy()
                    
                    # Perform the transformation only if needed
                    if filetype == 'chk':
                        Prim = ConsToPrim(q) 

                    #######################################################################
                    ######################## Modify PRIMITIVE variables here
                    #######################################################################
                    # ivar  = list_var.index('rho')
                    # indx = np.where((x1 > 10.) & (x1 < 14.))[0]
                    # Prim[indx,:,:,ivar] = 10.*Prim[indx,:,:,ivar] # rho * 10 example

                    
                    
                    # We are taking mu*I instead of just I
                    x1_m = x1
                    n_x1 = len(x1_m)
                    n_x2 = len(x2)
                    n_x3 = len(x3)

                    # Inserting new points at the beginning and the end of
                    # the patch in order to compute well the curl because 
                    # of decentered derivatives
                    
                    # Compute dr,dtheta and dphi at each extremities of the patch
                    # Beginnings
                    drb     = x1_m[1]  - x1_m[0]
                    dthetab = x2[1]    - x2[0]
                    dphib   = x3[1]    - x3[0]

                    # Ends
                    dre     = x1_m[-1] - x1_m[-2]
                    dthetae = x2[-1]   - x2[-2]
                    dphie   = x3[-1]   - x3[-2]
                    
                    # Compute new points
                    rb_point     = [x1_m[0]  - 2*drb,     x1_m[0] - drb]
                    thetab_point = [x2[0]    - 2*dthetab, x2[0] - dthetab]
                    phib_point   = [x3[0]    - 2*dphib,   x3[0] - dphib]

                    
                    re_point     = [x1_m[-1] + dre,     x1_m[-1] + 2*dre]
                    thetae_point = [x2[-1]   + dthetae, x2[-1] + 2*dthetae]
                    phie_point   = [x3[-1]   + dphie,   x3[-1] + 2*dphie]

                    # Add them and prepare slices to remove these points before
                    # adding the TDm
                    slice_r = slice(2,-2)
                    slice_theta = slice(2,-2)
                    slice_phi = slice(2,-2)

                    new_r     = np.append(rb_point,x1_m)
                    new_r     = np.append(new_r,re_point)

                    new_theta = np.append(thetab_point,x2)
                    new_theta = np.append(new_theta,thetae_point)

                    new_phi   = np.append(phib_point,x3)
                    new_phi   = np.append(new_phi,phie_point)
                   

                    rho_sw = Prim[0:n_x1,0:n_x2,0:n_x3,Rhovar]

                    # Computing the magnetic structure
                    B_sph = formula_TDm.TDm_setup( 
                            new_r,
                            new_theta,
                            new_phi,
                            alpha_0,
                            theta_0,
                            phi_0,
                            d,
                            a,
                            R,
                            zeta,
                            B_amb,
                            Delta,
                            case,
                            geometry=geometry
                            )
                    
                    # Filling the magnetic field
                    Prim[0:n_x1,0:n_x2,0:n_x3,Brvar] += B_sph[0][slice_r,slice_theta,slice_phi]
                    Prim[0:n_x1,0:n_x2,0:n_x3,Btvar] += B_sph[1][slice_r,slice_theta,slice_phi]
                    Prim[0:n_x1,0:n_x2,0:n_x3,Bpvar] += B_sph[2][slice_r,slice_theta,slice_phi]
                        
                    
                    #######################################################################
                    # Go back to conservative
                    q=PrimToCons(Prim)
                    # print(np.nanmax(np.abs((qold-q)/qold)[:,:,:,7])) 
                    # This is validated up to machine precision
                    #######################################################################
                    # Back to data, this directly modifies the hdf5 file
                    data[ncount:ncount+szb] = q.T.flatten()
                    ncount = ncount+szb
                    
            # Close File
            fp.close()

    

    def add_tdm_coconut(self,iteration,filename,geometry='spherical',ori_dir=False):
        ''' 
        Add the TDm flux rope to the coconut simulation
        '''
        
        print('Spherical geometry is hard coded in formula_TDm.TDm_setup')


        print('')
        print('THIS BRANCH IS STILL EXPERIMENTAL')
        print('')
            
        if ori_dir:
            originals_dir = ori_dir
        else:
            originals_dir = os.environ['TDM_ORIGINALS_EUHFORIA']


        # Directory where the filed to be modified is
        bdirIn=originals_dir        
        print('ori_dir = ',bdirIn)

        # Directory where to put the modified file
        # Setting to current directory  
        cwd = os.getcwd()
        dirOut=f'{cwd}/'

        self.read_TDm_parameter()
        
        alpha_0 = self.setup['ALPHA_0']
        theta_0 = self.setup['X_1']
        phi_0   = self.setup['X_2']

        # TDm parameters setup
        d = self.setup['D']
        a = self.setup['A']
        R = self.setup['R']
        Delta = self.setup['DELTA']
        case = self.setup['CASE_TDM'] 
        zeta = self.setup['ZETA'] 
        
        nb_proc = self.CF_param['nb_proc']
        nb_r = self.CF_param['nb_r']
        nb_th = self.CF_param['nb_th']
        nb_phi = self.CF_param['nb_phi']
        eps = self.CF_param['eps']

        # Load files
        print('Loading files...')
        for i in tqdm(range(nb_proc)):

            fileloc = bdirIn + filename + str(i) + '.vtu'
            mesh = pyv.read(fileloc)

            # Grid
            points_loc = vtk_to_numpy(mesh.GetPoints().GetData())
            nb_pts_loc = np.shape(points_loc)[0]

            x_loc = points_loc[:,0]
            y_loc = points_loc[:,1]
            z_loc = points_loc[:,2]

            # Quantities
            rho_loc = mesh.get_array('rho')
            p_loc = mesh.get_array('p')
            bx_loc = mesh.get_array('Bx')
            by_loc = mesh.get_array('By')
            bz_loc = mesh.get_array('Bz')
            v_loc = mesh.get_array('v')
            vx_loc = v_loc[:,0]
            vy_loc = v_loc[:,1]
            vz_loc = v_loc[:,2]

            # Adding to global file
            if (i == 0):

                points = points_loc
                nb_pts = nb_pts_loc
                x = x_loc
                y = y_loc
                z = z_loc
                rho = rho_loc
                p = p_loc
                bx = bx_loc
                by = by_loc
                bz = bz_loc
                vx = vx_loc
                vy = vy_loc
                vz = vz_loc

            else:
                # Add to global arrays
                points = np.concatenate((points,points_loc))
                nb_pts = nb_pts + nb_pts_loc
                x = np.concatenate((x,x_loc))
                y = np.concatenate((y,y_loc))
                z = np.concatenate((z,z_loc))
                rho = np.concatenate((rho,rho_loc))
                p = np.concatenate((p,p_loc))
                bx = np.concatenate((bx,bx_loc))
                by = np.concatenate((by,by_loc))
                bz = np.concatenate((bz,bz_loc))
                vx = np.concatenate((vx,vx_loc))
                vy = np.concatenate((vy,vy_loc))
                vz = np.concatenate((vz,vz_loc))

        # Check for duplicates
        df = pd.DataFrame(points)
        df_drop = df.drop_duplicates()
        idx = df_drop[0].index.to_numpy()
        points = df_drop.to_numpy()
        nb_dup = len(idx)
        nb_pts = len(points)

        x = x[idx]
        y = y[idx]
        z = z[idx]
        rho = rho[idx]
        p = p[idx]
        bx = bx[idx]
        by = by[idx]
        bz = bz[idx]
        vx = vx[idx]
        vy = vy[idx]
        vz = vz[idx]

        print('Removing {} duplicates...'.format(nb_dup))
        
        r = np.sqrt(x**2+y**2+z**2)
        rxy = np.sqrt(x**2+y**2)

        r0 = np.unique(r)
        n1 = len(r0)
        theta = np.arccos(z/r)
        theta0 = np.unique(theta)
        n2 = len(theta0)
        phi = np.arctan2(y,x)
        phi2pi = (phi) % (2*np.pi)
        phi0 = np.unique(phi)
        n3 = len(phi0)
        
        print(x)
         
# From 3D to 2D
# Interpolation on regular grid
        print('Interpolation...')
        

        x1 = np.logspace(0, np.log10(25), nb_r)
        x2 = np.linspace(eps, np.pi-eps, nb_th)
        x3 = np.linspace(eps, 2.0*np.pi-eps, nb_phi)
        
        X1, X2, X3 = np.meshgrid(x1, x2, x3, indexing='ij')

        grid_x = X1*np.sin(X2)*np.cos(X3)
        grid_y = X1*np.sin(X2)*np.sin(X3)
        grid_z = X1*np.cos(X2)

        # https://en.wikipedia.org/wiki/Del_in_cylindrical_and_spherical_coordinates

        br = ( x*bx     +    y*by + z*bz                ) / r
        bt = ( x*z * bx + y*z* by - (x**2 + y**2) * bz)   / (r * rxy)
        bp = (-y*bx     +    x*by                     )   / rxy

        # interpolation
        brr = griddata(points, br, (grid_x, grid_y, grid_z), method='nearest')
        btr = griddata(points, bt, (grid_x, grid_y, grid_z), method='nearest')
        bpr = griddata(points, bp, (grid_x, grid_y, grid_z), method='nearest')

        print('Interpolation Done')

        # rhor = griddata(points, rho, (grid_x, grid_y, grid_z), method='nearest')
        # prsr = griddata(points, p, (grid_x, grid_y, grid_z), method='nearest')

        # Estimating the B_amb
         
        r0 = 1 + self.setup['R'] - self.setup['D'] # Height at which B_amb is measured
        t0 = self.setup['X_1']
        p0 = self.setup['X_2']
    

        # x0 = r0 * np.cos(p0) * np.sin(t0)
        # y0 = r0 * np.sin(p0) * np.sin(t0)
        # z0 = r0 * np.cos(t0)
        
        # x_loc = np.argmin(np.abs(x-x0))
        # y_loc = np.argmin(np.abs(x-x0))
        # z_loc = np.argmin(np.abs(x-x0))

        r_loc = np.argmin(np.abs(x1-r0))
        t_loc = np.argmin(np.abs(x2-t0))
        p_loc = np.argmin(np.abs(x3-p0))

        B_amb = np.sqrt(
                brr[r_loc,t_loc,p_loc]**2 +
                btr[r_loc,t_loc,p_loc]**2 +
                bpr[r_loc,t_loc,p_loc]**2 
                )
        


        # Computing the flux rope magnetic field
        B_cart = formula_TDm.TDm_setup( 
                x1,
                x2,
                x3,
                alpha_0,
                theta_0,
                phi_0,
                d,
                a,
                R,
                zeta,
                B_amb,
                Delta,
                case,
                geometry=geometry
                )

        # Now we add the magnetic field 
        brr += B_cart[0]
        btr += B_cart[1]
        bpr += B_cart[2]
 
        # Now putting everything in cartesian and save to vtk file

        #  coordinates
        # x = X1 * np.sin(X2) * np.cos(X3)
        # y = X1 * np.sin(X2) * np.sin(X3)
        # z = X1 * np.cos(X2)

        # vector
        bxr = np.sin(X2) * np.cos(X3) * brr + np.cos(X2) * np.cos(X3) * btr - np.sin(X3) * bpr
        byr = np.sin(X2) * np.sin(X3) * brr + np.cos(X2) * np.sin(X3) * btr + np.cos(X3) * bpr
        bzr = np.cos(X2)              * brr - np.sin(X2) * btr

        brr = np.asfortranarray(brr)
        btr = np.asfortranarray(btr)
        bpr = np.asfortranarray(bpr)

        bxr = np.asfortranarray(bxr)
        byr = np.asfortranarray(byr)
        bzr = np.asfortranarray(bzr)
        
        # Name of the vts file
        vtsfile = self.case 
        

        print('Saving to vts')

        gridToVTK(vtsfile,grid_x,grid_y,grid_z,
                pointData = {
                    'B_cart':(bxr,byr,bzr),
                    'B_sph' :(brr,btr,bpr),
                    'B_TDm' :(B_cart[0],B_cart[1],B_cart[2])
                    }
                )

    def convergent_flow(self,a,R,ar_grid,bdirOut):
        ''' 
        Compute a velocity field that converge to the PIL of the FR
        '''

        theta = ar_grid[1]
        phi = ar_grid[2]

        tt,pp = np.meshgrid(theta,phi,indexing='ij')

        def gauss_t(xx2,mu_t,s_t):
            ''' 
            Gaussian on theta
            '''
            
            gauss = np.sign(1.57 - xx2) * np.exp(-0.5 * ((xx2 - mu_t)/s_t)**2) \
                    * 1 / (s_t * np.sqrt(2 * np.pi))

            norm_t  = 1/(s_t * np.sqrt(2*np.pi))

            gauss = gauss / norm_t

            return gauss

        # we normalise the gaussien by it's value at the mean value so that it is
        # equal to 1 
    


        def gauss_p(xx3,mu_p,s_p):
            '''
            Gaussian on phi
            '''
            gauss =  np.exp(-0.5 * ((xx3 - mu_p)/s_p)**2) \
                    * 1 / (s_p * np.sqrt(2 * np.pi))

            # we normalise the gaussian by it's value at the mean value so that it is
            # equal to 1 
            norm_p  = 1/(s_p * np.sqrt(2*np.pi))

            gauss = gauss / norm_p

            return gauss

            # Params for both gaussian
            # ==========================

        theta_0 = self.setup["X_1"]
        phi_0 = self.setup["X_2"]

        mu_p = phi_0
        s_p = self.setup["R"] * 0.5

        mu_t =  theta_0
        s_t = self.setup["A"] * 1.3
    
        UNIT_VELOCITY = 4.367e7 # cm/s

        V0 = 200e5 / UNIT_VELOCITY # 100 km/s

        print(f'Velocity hard coded ({V0:.2f} PLUTO unit)')

        gauss_tot = V0 * gauss_t(tt,mu_t,s_t) * gauss_p(pp,mu_p,s_p)

        with open('{}/speed_bc.bin'.format(bdirOut),'wb') as speed_bc:      
            for j in range(len(theta)):
                for k in range(len(phi)):
                    speed_bc.write(struct.pack('d',gauss_tot[j,k]))



