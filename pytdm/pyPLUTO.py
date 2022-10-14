from __future__ import print_function
import os
import sys
import struct
import array
import numpy as np
import pdb
import linecache
try:
    import scipy as S
    import scipy.ndimage
    import scipy.interpolate
    from scipy.interpolate import UnivariateSpline,RectBivariateSpline,interp2d
    if (hasattr(scipy.interpolate,'RegularGridInterpolator')):
        from scipy.interpolate import RegularGridInterpolator
    noscipy=False
except ImportError:
    noscipy=True
from matplotlib.pyplot import *
from matplotlib.mlab import *
try:
    from mpl_toolkits.axes_grid1 import make_axes_locatable
except ImportError:
    print("<AS> Plotting is limited")
except OSError:
    print("<AS> Plotting is limited")
from subprocess import call
import time
from timeit import default_timer as timer
# This is to read data from remote repositories
try:
    from paramiko import SSHClient,SSHConfig,ProxyCommand 
    hasparamiko=True
except ImportError:
    hasparamiko=False
try:
    import h5py as h5
    hasH5 = True
except ImportError:
    hasH5 = False
#try:
#    import yt
#    hasYT=True
#except ImportError:
hasYT=False
try:
    from evtk.hl import gridToVTK # This can be downloaded there: https://bitbucket.org/pauloh/pyevtk
    hasVTK=True
except ImportError:
    try:
        from pyevtk.hl import gridToVTK # This can be install with pip install pyevtk
        hasVTK=True
    except ImportError:
        hasVTK=False
from tqdm import tqdm 
from numba import jit
from multiprocessing import Pool
from joblib import Parallel,delayed
from subprocess import call

def curdir():
    """ Get the current working directory.
    """
    curdir = os.getcwd()+'/'
    return curdir

def get_nstepstr(ns):
    """ Convert the float input *ns* into a string that would match the data file name.
    
    **Inputs**:
    
    ns -- Integer number that represents the time step number. E.g., The ns for data.0001.dbl is 1.\n
    
    **Outputs**:
    
    Returns the string that would be used to complete the data file name. E.g., for data.0001.dbl, ns = 1 and pyPLUTO.get_nstepstr(1) returns '0001'
    """
    nstepstr = str(ns)
    while len(nstepstr) < 4:
        nstepstr= '0'+nstepstr
    return nstepstr

def nlast_info(w_dir=None,datatype=None):
    """ Prints the information of the last step of the simulation as obtained from dbl.out or flt.out
    
    **Inputs**:
    
    w_dir -- path to the directory which has the dbl.out(or flt.out) and the data\n
    datatype -- If the data is of 'float' type then datatype = 'float' else by default the datatype is set to 'double'.
    
    **Outputs**:
    
    This function returns a dictionary with following keywords - \n
    
    nlast -- The ns for the last file saved.\n
    time -- The simulation time for the last file saved.\n
    dt -- The time step dt for the last file. \n
    Nstep -- The Nstep value for the last file saved.
    
    
    **Usage**:
    
    In case the data is 'float'.
    
    ``wdir = /path/to/data/directory``\n
    ``import pyPLUTO as pp``\n
    ``A = pp.nlast_info(w_dir=wdir,datatype='float')``
    
    """
    if w_dir is None: w_dir=curdir()
    if datatype == 'float':
        fname_v = w_dir+"flt.out"
    else:
        fname_v = w_dir+"dbl.out"
    last_line = open(fname_v,"r").readlines()[-1].split()
    nlast = int(last_line[0])
    SimTime =  float(last_line[1])
    Dt = float(last_line[2])
    Nstep = int(last_line[3])
        
    print ("------------TIME INFORMATION--------------")
    print ('nlast =',nlast)
    print ('time  =',SimTime)
    print ('dt    =', Dt)
    print ('Nstep =',Nstep)
    print ("-------------------------------------------")
        
    return {'nlast':nlast,'time':SimTime,'dt':Dt,'Nstep':Nstep}

class gridP(object):
    """ This Class is for reading only the grid """
    def __init__(self, gridfile,x1range=None,x2range=None,x3range=None):
        """ Read grid values from the grid.out file.
        **Inputs**:
        gridfile -- name of the grid.out file which has information about the grid. 
        """
        xL = []
        xR = []
        nmax = []
        #gfp = open(gridfile, "r")
        gfp = self.OpenFile(gridfile,"r")
        for i in gfp.readlines():
            if len(i.split()) == 1:
                try:
                    int(i.split()[0])
                    nmax.append(int(i.split()[0]))
                except:
                    pass
            if len(i.split()) == 3:
                try:
                    int(i.split()[0])
                    xL.append(float(i.split()[1]))
                    xR.append(float(i.split()[2]))
                except:
                    pass
            if len(i.split()) == 5:
                try:
                    int(i.split()[0])
                    xL.append(float(i.split()[1]))
                    xR.append(float(i.split()[3]))
                except:
                    pass
        self.n1, self.n2, self.n3 = nmax
        n1 = self.n1
        n1p2 = self.n1 + self.n2
        n1p2p3 = self.n1 + self.n2 + self.n3
        self.x1 = np.asarray([0.5*(xL[i]+xR[i]) for i in range(n1)])
        self.dx1 = np.asarray([(xR[i]-xL[i]) for i in range(n1)])
        self.x2 = np.asarray([0.5*(xL[i]+xR[i]) for i in range(n1, n1p2)])
        self.dx2 = np.asarray([(xR[i]-xL[i]) for i in range(n1, n1p2)])
        self.x3 = np.asarray([0.5*(xL[i]+xR[i]) for i in range(n1p2, n1p2p3)])
        self.dx3 = np.asarray([(xR[i]-xL[i]) for i in range(n1p2, n1p2p3)])
        self.n1_tot = self.n1 ; self.n2_tot = self.n2 ; self.n3_tot = self.n3
        if (x1range != None):
            self.x1range= x1range
            self.n1_tot = self.n1
            self.irange = range(abs(self.x1-self.x1range[0]).argmin(),abs(self.x1-self.x1range[1]).argmin()+1)
            self.n1  = len(self.irange)
            self.x1  = self.x1[self.irange]
            self.dx1 = self.dx1[self.irange]
        else:
            self.irange = range(self.n1)
        if (x2range != None):
            self.x2range=x2range
            self.n2_tot = self.n2
            self.jrange = range(abs(self.x2-self.x2range[0]).argmin(),abs(self.x2-self.x2range[1]).argmin()+1)
            self.n2  = len(self.jrange)
            self.x2  = self.x2[self.jrange]
            self.dx2 = self.dx2[self.jrange]
        else:
            self.jrange = range(self.n2)
        if (x3range != None):
            self.x3range=x3range
            self.n3_tot = self.n3
            self.krange = range(abs(self.x3-self.x3range[0]).argmin(),abs(self.x3-self.x3range[1]).argmin()+1)
            self.n3  = len(self.krange)
            self.x3  = self.x3[self.krange]
            self.dx3 = self.dx3[self.krange]
        else:
            self.krange = range(self.n3)
        self.Slice=(x1range != None) or (x2range != None) or (x3range != None)
        if self.n2_tot == 1 and self.n3_tot == 1:
            self.nshp = (self.n1)
        elif self.n3_tot == 1:
            self.nshp = (self.n2, self.n1)
            self.nshp_b1s = (self.n2, self.n1+1)
            self.nshp_b2s = (self.n2+1, self.n1)
            self.nshp_b3s = (self.n2, self.n1)
        else:
            self.nshp = (self.n3, self.n2, self.n1)
            self.nshp_b1s = (self.n3,self.n2, self.n1+1) if self.n1 > 1 else (self.n3,self.n2, self.n1)
            self.nshp_b2s = (self.n3,self.n2+1, self.n1) if self.n2 > 1 else (self.n3,self.n2, self.n1)
            self.nshp_b3s = (self.n3+1,self.n2, self.n1) if self.n3 > 1 else (self.n3,self.n2, self.n1)
        # Create the xr arrays containing the edges positions
        # Useful for pcolormesh which should use those
        self.x1r = np.zeros(self.n1+1) ; self.x1r[1:] = self.x1 + self.dx1/2.0 ; self.x1r[0] = self.x1r[1]-self.dx1[0]
        self.x2r = np.zeros(self.n2+1) ; self.x2r[1:] = self.x2 + self.dx2/2.0 ; self.x2r[0] = self.x2r[1]-self.dx2[0]
        self.x3r = np.zeros(self.n3+1) ; self.x3r[1:] = self.x3 + self.dx3/2.0 ; self.x3r[0] = self.x3r[1]-self.dx3[0]

class pload(object):
    """
    This Class has all the routines loading the data from the
    binary files output from PLUTO Simulations. Assign an object
    when the data is loaded for some *ns*.
    
    **Inputs:**
    
    ns -- Time step of the data, data.ns.dbl or data.ns.flt\n
    w_dir -- path to the directory which has the dbl.out(or flt.out) and the data\n
    datatype -- If the data is of 'float' type then datatype = 'float' else by default the datatype is set to 'double'.
    
    
    **Outputs:**
    
    A pyPLUTO.pload object  having all the relevant information of the corresponding data file 
    
    **Usage**:
    
    ``import pyPLUTO as pp``\n
    ``wdir = '/path/to/the data files/'``\n
    ``D = pp.pload(1,w_dir=wdir)``\n

    Now D is the pyPLUTO.pload object having all the relevant information
    of the corresponding data file - data.0001.dbl.
    
    
    """
    def __init__(self, ns, w_dir=None, machine='', datatype=None, level=0,
            x1range=None, x2range=None,
            x3range=None,dec=1,silent=False,centre=False,noload=False,debug_time=False,filetype='data'):
        """Loads the data.
        
        **Inputs**:
        
        ns -- Step Number of the data file\n
        w_dir -- path to the directory which has the data files\n
        datatype -- Datatype (default = 'double')\n
        level -- selected level in AMR (default = 0)\n
        x1range -- 2 elements list giving the x-range of the portion to be extracted\n
        x2range -- 2 elements list giving the y-range of the portion to be extracted\n
        x3range -- 2 elements list giving the z-range of the portion to be extracted 
        dec     -- decrease the resolution bu a factor dec when reading

        **Outputs**:
        
        pyPLUTO pload object whose keys are arrays of data values.
        
        """
        self.NStep = ns
        self.Simtime = 0.0
        self.Dt = 0.0
        self.debug_time=debug_time

        self.n1 = 0
        self.n2 = 0
        self.n3 = 0

        self.x1 = []
        self.x2 = []
        self.x3 = []
        self.dx1 = []
        self.dx2 = []
        self.dx3 = []

        self.x1range = x1range
        self.x2range = x2range
        self.x3range = x3range

        self.filetype = filetype

        self.dec = dec


        self.NStepStr = str(self.NStep)
        while len(self.NStepStr) < 4:
            self.NStepStr = '0'+self.NStepStr

        if datatype is None:
            datatype = "double"
        self.datatype = datatype

        if ((not hasH5) and (datatype == 'hdf5')):
            print ('--> Please install h5py to be able to read hdf5 files with python')
            return

        self.level = level

        if w_dir is None:
            w_dir = os.getcwd() + '/'
            
        self.wdir = w_dir

        if machine != '':
            self.RemoteDir=True
            self.ConfigClient(machine=machine)
        else:
            self.RemoteDir=False
        
        Data_dictionary = self.ReadDataFile(self.NStepStr,silent=silent,noload=noload,centre=centre,filetype=filetype)
        for keys in Data_dictionary:
            object.__setattr__(self, keys, Data_dictionary.get(keys))
            
        # Alias for older versions of PLUTO
        if (self.version == 3):
            self.vx1 = self.v1
            self.vx2 = self.v2
            self.vx3 = self.v3
            if hasattr(self,'pr'):
                self.prs = self.pr
            if hasattr(self,'b1'):
                self.bx1 = self.b1
                self.bx2 = self.b2
                self.bx3 = self.b3
        # to cope with version 4.3
        if hasattr(self,'Bx1'):
            self.bx1=self.Bx1
            self.bx2=self.Bx2
            self.bx3=self.Bx3
        # Personal variables
        if hasattr(self,'B1_bckg'):
            self.bx1 = self.bx1 + self.B1_bckg
            self.bx2 = self.bx2 + self.B2_bckg
            self.bx3 = self.bx3 + self.B3_bckg 
            
    def ConfigClient(self,machine='hubble'):
        config=SSHConfig()
        config.parse(open(os.path.expanduser('~/.ssh/config')))
        self.computer=config.lookup(machine)
        self.ssh_client=SSHClient()
        self.ssh_client.load_system_host_keys()

    def OpenFile(self,filename,option):
        if self.RemoteDir:
            self.ssh_client.load_system_host_keys()
            self.ssh_client.connect(self.computer['hostname'],22,self.computer['user'])
            sftp=self.ssh_client.open_sftp()
            fp=sftp.file(filename,option)
        else:
            fp=open(filename,option)
        return fp

            
    def ReadTimeInfo(self, timefile):
        """ Read time info from the outfiles.
        
        **Inputs**:
        
        timefile -- name of the out file which has timing information. 

        """

        if (self.datatype == 'hdf5'):
            fh5 = h5.File(timefile,'r') 
            self.SimTime = fh5.attrs.get('time')
            #self.Dt = 1.e-2 # Should be erased later given the level in AMR
            fh5.close()
        else:
            ns = self.NStep
            #f_var = open(timefile, "r")
            f_var = self.OpenFile(timefile, "r")
            tlist = []
            for line in f_var.readlines():
                tlist.append(line.split())
            try:
                self.SimTime = float(tlist[ns][1])
                self.Dt = float(tlist[ns][2])
            except:
                if self.aggressive_read:
                    print("*** Timing will be wrong ***")
                    self.SimTime = float(tlist[-1][1])
                    self.Dt = float(tlist[-1][2])
                else:
                    print("ERROR in ReadTimeInfo")

    def ReadVarFile(self, varfile,num,aggressive_read=True):
        """ Read variable names from the outfiles.

        **Inputs**:
        
        varfile -- name of the out file which has variable information. 
        
        """

        self.aggressive_read=aggressive_read
        if (self.datatype == 'hdf5'):
            fh5 = h5.File(varfile,'r') 
            self.fltype = 'single_file' 
            self.endianess = '>' # not used with AMR, kept for consistency
            self.vars = []
            for iv in range(fh5.attrs.get('num_components')):
                self.vars.append(fh5.attrs.get('component_'+str(iv)).decode())
            fh5.close()
        else:
            vfp = self.OpenFile(varfile, "r")
            aa = vfp.readlines()
            try:
                varinfo = aa[num].split()
            except:
                if not aggressive_read:
                    print("ERROR in ReadVarFile %s with it=%i" % (varfile,num))
                else:
                    print("*** Reading old varinfo from previous runs, be careful ***")
                    varinfo = aa[-1].split()

            self.fltype = varinfo[4]
            self.endianess = varinfo[5]
            self.vars = varinfo[6:]
            vfp.close()

    def ReadSysFile(self):
        """ Read the version of PLUTO in sysconf.out """

        sysfile = self.wdir+"sysconf.out" 
        if (os.path.exists(sysfile)):
            #fsys = open(sysfile)
            fsys = self.OpenFile(sysfile,"r")
            for line in fsys.readlines():
                ll = line.split('=')
                if (ll[0].strip() == 'PLUTO_VERSION'):
                    self.version = int(ll[1].strip()[1])
            fsys.close()
        else:
            # Default is version 4
            self.version = 4
        
    def ReadGridFile(self, gridfile, dtype):
        """ Read grid values from the grid.out file.
        
        **Inputs**:
      
        gridfile -- name of the grid.out file which has information about the grid. 
        
        """
        xL = []
        xR = []
        nmax = []
        #gfp = open(gridfile, "r")
        gfp = self.OpenFile(gridfile, "r")
        for i in gfp.readlines():
            if len(i.split()) == 1:
                try:
                    int(i.split()[0])
                    nmax.append(int(i.split()[0]))
                except:
                    pass
            if len(i.split()) == 3:
                try:
                    int(i.split()[0])
                    xL.append(float(i.split()[1]))
                    xR.append(float(i.split()[2]))
                except:
                    pass
            if len(i.split()) == 5:
                try:
                    int(i.split()[0])
                    xL.append(float(i.split()[1]))
                    xR.append(float(i.split()[3]))
                except:
                    pass

        self.n1, self.n2, self.n3 = nmax
        n1 = self.n1
        n1p2 = self.n1 + self.n2
        n1p2p3 = self.n1 + self.n2 + self.n3
        self.x1 = np.asarray([0.5*(xL[i]+xR[i]) for i in range(n1)],dtype=dtype)
        self.dx1 = np.asarray([(xR[i]-xL[i]) for i in range(n1)],dtype=dtype)
        self.x2 = np.asarray([0.5*(xL[i]+xR[i]) for i in range(n1, n1p2)],dtype=dtype)
        self.dx2 = np.asarray([(xR[i]-xL[i]) for i in range(n1, n1p2)],dtype=dtype)
        self.x3 = np.asarray([0.5*(xL[i]+xR[i]) for i in range(n1p2, n1p2p3)],dtype)
        self.dx3 = np.asarray([(xR[i]-xL[i]) for i in range(n1p2, n1p2p3)],dtype=dtype)

        self.n1_tot = self.n1 ; self.n2_tot = self.n2 ; self.n3_tot = self.n3
        if (self.x1range != None):
            self.n1_tot = self.n1
            self.irange = range(abs(self.x1-self.x1range[0]).argmin(),abs(self.x1-self.x1range[1]).argmin()+1)
            self.n1  = len(self.irange)
            self.x1  = self.x1[self.irange]
            self.dx1 = self.dx1[self.irange]
        else:
            self.irange = range(self.n1)
        if (self.x2range != None):
            self.n2_tot = self.n2
            self.jrange = range(abs(self.x2-self.x2range[0]).argmin(),abs(self.x2-self.x2range[1]).argmin()+1)
            self.n2  = len(self.jrange)
            self.x2  = self.x2[self.jrange]
            self.dx2 = self.dx2[self.jrange]
        else:
            self.jrange = range(self.n2)
        if (self.x3range != None):
            self.n3_tot = self.n3
            self.krange = range(abs(self.x3-self.x3range[0]).argmin(),abs(self.x3-self.x3range[1]).argmin()+1)
            self.n3  = len(self.krange)
            self.x3  = self.x3[self.krange]
            self.dx3 = self.dx3[self.krange]
        else:
            self.krange = range(self.n3)
        self.Slice=(self.x1range != None) or (self.x2range != None) or (self.x3range != None)

        #prodn = self.n1*self.n2*self.n3
        if self.n2_tot == 1 and self.n3_tot == 1:
            self.nshp = (self.n1)
            self.dec_slice = "[::self.dec]"
        elif self.n3_tot == 1:
            self.nshp = (self.n2, self.n1)
            self.nshp_b1s = (self.n2, self.n1+1)
            self.nshp_b2s = (self.n2+1, self.n1)
            self.nshp_b3s = (self.n2, self.n1)
            self.dec_slice = "[::self.dec,::self.dec]"
        else:
            self.nshp = (self.n3, self.n2, self.n1)
            self.nshp_b1s = (self.n3,self.n2, self.n1+1) if self.n1 > 1 else (self.n3,self.n2, self.n1)
            self.nshp_b2s = (self.n3,self.n2+1, self.n1) if self.n2 > 1 else (self.n3,self.n2, self.n1)
            self.nshp_b3s = (self.n3+1,self.n2, self.n1) if self.n3 > 1 else (self.n3,self.n2, self.n1)
            self.dec_slice = "[::self.dec,::self.dec,::self.dec]"

        # Create the xr arrays containing the edges positions
        # Useful for pcolormesh which should use those
        self.x1r = np.zeros(self.n1+1) ; self.x1r[1:] = self.x1 + self.dx1/2.0 ; self.x1r[0] = self.x1r[1]-self.dx1[0]
        self.x2r = np.zeros(self.n2+1) ; self.x2r[1:] = self.x2 + self.dx2/2.0 ; self.x2r[0] = self.x2r[1]-self.dx2[0]
        self.x3r = np.zeros(self.n3+1) ; self.x3r[1:] = self.x3 + self.dx3/2.0 ; self.x3r[0] = self.x3r[1]-self.dx3[0]


    def DataScanVTK(self, fp, n1, n2, n3, endian, dtype,noload=False):
        """ Scans the VTK data files. 
    
        **Inputs**:
        
        fp -- Data file pointer\n
        n1 -- No. of points in X1 direction\n
        n2 -- No. of points in X2 direction\n
        n3 -- No. of points in X3 direction\n
        endian -- Endianess of the data
        dtype -- datatype 
        
        **Output**:
        
        Dictionary consisting of variable names as keys and its values. 
        
        """
        ks = []
        vtkvar = []
        n1_tot = self.n1_tot ; n2_tot = self.n2_tot; n3_tot = self.n3_tot
        while True:
            l = fp.readline()
            try:
                l.split()[0]
            except IndexError:
                pass
            else:
                if l.split()[0] == 'SCALARS':
                    ks.append(l.split()[1])
                elif l.split()[0] == 'LOOKUP_TABLE':
                    if noload:
                        fp.seek(nb,1)
                        tmp=eval("np.zeros(self.nshp){}.transpose()".format(self.dec_slice))
                        vtkvar.append(tmp)
                    else:
                        A = array.array(dtype)
                        fmt = endian+str(n1_tot*n2_tot*n3_tot)+dtype
                        nb = np.dtype(fmt).itemsize 
                        A.fromstring(fp.read(nb))
                        if (self.Slice):
                            darr = np.zeros((n1*n2*n3))
                            indxx = np.sort([n3_tot*n2_tot*k + j*n2_tot + i for i in self.irange for j in self.jrange for k in self.krange])
                            if (sys.byteorder != self.endianess):
                                A.byteswap()
                            for ii,iii in enumerate(indxx):
                                darr[ii] = A[iii]
                                vtkvar_buf = [darr]
                        else:
                            vtkvar_buf = np.frombuffer(A,dtype=np.dtype(fmt))
                        #vtkvar.append(np.reshape(vtkvar_buf,self.nshp).transpose())
                        tmp=eval("np.reshape(vtkvar_buf,self.nshp){}.transpose()".format(self.dec_slice))
                        vtkvar.append(tmp)
                else:
                    pass
            if l == '':
                break
            
        vtkvardict = dict(zip(ks,vtkvar))
        return vtkvardict

    def InterpGrid(self,i,dim,il,fl,data,boxes,nbox,vars,AMRBoxes,AMRLevel,flagAMR,
            ncount,LevelDic,x1b,x2b,x3b,dx,
            freb,logr,ystr,zstr,noload,nvar,centre=False,silent=False):
        
        print(f'Level {i}')

        for j in range(nbox): # loop on all boxes of a given level

            ibeg=LevelDic['ibeg']
            jbeg=LevelDic['jbeg']
            kbeg=LevelDic['kbeg']
            iend=LevelDic['iend']
            jend=LevelDic['jend']
            kend=LevelDic['kend']

            AMRLevel[i]['box'].append({'x0':0.,'x1':0.,'ib':0,'ie':0,\
                                        'y0':0.,'y1':0.,'jb':0,'je':0,\
                                        'z0':0.,'z1':0.,'kb':0,'ke':0})
            # Box indexes
            ib = boxes[j]['lo_i'] ; ie = boxes[j]['hi_i'] ; nbx = ie-ib+1
            jb = 0 ; je = 0 ; nby = 1
            kb = 0 ; ke = 0 ; nbz = 1
            if (dim > 1):
                jb = boxes[j]['lo_j'] ; je = boxes[j]['hi_j'] ; nby = je-jb+1
            if (dim > 2):
                kb = boxes[j]['lo_k'] ; ke = boxes[j]['hi_k'] ; nbz = ke-kb+1
            szb = nbx*nby*nbz*nvar
            # Rescale to current level
            kb = kb*freb[i] ; ke = (ke+1)*freb[i] - 1
            jb = jb*freb[i] ; je = (je+1)*freb[i] - 1
            ib = ib*freb[i] ; ie = (ie+1)*freb[i] - 1

            # Skip boxes lying outside ranges
            if ((ib > iend) or (ie < ibeg) or \
                (jb > jend) or (je < jbeg) or \
                (kb > kend) or (ke < kbeg)):
                ncount = ncount + szb
            else:

                ### Read data
                q = data[ncount:ncount+szb].reshape((nvar,nbz,nby,nbx)).T
                
                ### Find boxes intersections with current domain ranges
                ib0 = max([ibeg,ib]) ; ie0 = min([iend,ie])
                jb0 = max([jbeg,jb]) ; je0 = min([jend,je])
                kb0 = max([kbeg,kb]) ; ke0 = min([kend,ke])
            
                ### Store box corners in the AMRLevel structure
                if logr == 0:
                    AMRLevel[i]['box'][j]['x0'] = x1b + dx*(ib0)
                    AMRLevel[i]['box'][j]['x1'] = x1b + dx*(ie0+1)
                else:
                    AMRLevel[i]['box'][j]['x0'] = x1b*np.exp(dx*(ib0))
                    AMRLevel[i]['box'][j]['x1'] = x1b*np.exp(dx*(ie0+1))
                    #AMRLevel[i]['box'][j]['x0'] = x1b*0.5*(np.exp(dx*(ib0))+np.exp(dx*(ib0+1)))
                    #AMRLevel[i]['box'][j]['x1'] = x1b*0.5*(np.exp(dx*(ie0))+np.exp(dx*(ie0+1)))
                AMRLevel[i]['box'][j]['y0'] = x2b + dx*(jb0)*ystr
                AMRLevel[i]['box'][j]['y1'] = x2b + dx*(je0+1)*ystr
                AMRLevel[i]['box'][j]['z0'] = x3b + dx*(kb0)*zstr
                AMRLevel[i]['box'][j]['z1'] = x3b + dx*(ke0+1)*zstr
                AMRLevel[i]['box'][j]['ib'] = ib0 ; AMRLevel[i]['box'][j]['ie'] = ie0 
                AMRLevel[i]['box'][j]['jb'] = jb0 ; AMRLevel[i]['box'][j]['je'] = je0 
                AMRLevel[i]['box'][j]['kb'] = kb0 ; AMRLevel[i]['box'][j]['ke'] = ke0
                AMRLevel[i]['box'][j]['dx'] = dx  ; AMRLevel[i]['box'][j]['logr'] = logr
                AMRLevel[i]['box'][j]['xstr'] = 1.
                AMRLevel[i]['box'][j]['ystr'] = ystr  ; AMRLevel[i]['box'][j]['zstr'] = zstr
                AMRBoxes[ib0-ibeg:ie0-ibeg+1, jb0-jbeg:je0-jbeg+1, kb0-kbeg:ke0-kbeg+1] = il
                        
                ### Extract the box intersection from data stored in q
                cib0 = int((ib0-ib)/freb[i]) ; cie0 = int((ie0-ib)/freb[i])
                cjb0 = int((jb0-jb)/freb[i]) ; cje0 = int((je0-jb)/freb[i])
                ckb0 = int((kb0-kb)/freb[i]) ; cke0 = int((ke0-kb)/freb[i])
                q1 = np.zeros((cie0-cib0+1, cje0-cjb0+1, cke0-ckb0+1,nvar))
                q1 = q[cib0:cie0+1,cjb0:cje0+1,ckb0:cke0+1,:]
                
                # if j == 0:
                    # print('q[0,0,0,0]',q[0,0,0,0])
                    # print(f'Center is {centre}')
                    # # print('q[0,0,0,1]',q[0,0,0,1])
                    # # print('q[0,0,0,2]',q[0,0,0,2])
                    # # print('q[0,0,0,3]',q[0,0,0,3])
                    # # print('q[0,0,0,4]',q[0,0,0,4])
                    # print('x0 = ',x1b+dx*(ib0))
                    # print('y0 = ',x2b+dx*(jb0))
                    # print('z0 = ',x3b+dx*(kb0))
                    # # breakpoint()
                    

                if not noload:
                    # Remap the extracted portion
                    if (dim == 1):
                        new_shape = (ie0-ib0+1,1)
                    elif (dim == 2):
                        new_shape = (ie0-ib0+1,je0-jb0+1)
                    else:
                        new_shape = (ie0-ib0+1,je0-jb0+1,ke0-kb0+1)
                    stmp = list(new_shape)
                    #while stmp.count(1) > 0:
                    #    stmp.remove(1)
                    

                    new_shape=tuple(stmp)
                    myT = Tools()
                    for iv in range(nvar):
                        vars[ib0-ibeg:ie0-ibeg+1,jb0-jbeg:je0-jbeg+1,kb0-kbeg:ke0-kbeg+1,iv] = \
                                                                                                myT.congrid(q1[:,:,:,iv],new_shape,method='linear',minusone=True,centre=centre).reshape((ie0-ib0+1,je0-jb0+1,ke0-kb0+1))
                 
                # if i == 4:
                    # breakpoint()

                ncount = ncount+szb
                flagAMR[ib0-ibeg:ie0-ibeg+1,jb0-jbeg:je0-jbeg+1,kb0-kbeg:ke0-kbeg+1]=i

        res = [AMRBoxes,AMRLevel,flagAMR,vars]
        
        return res



    
    def DataScanHDF5(self, fp, myvars,
            ilev,noload=False,silent=False,centre=False):
        """ Scans the Chombo HDF5 data files. (reading the hdf5 files in non AMR version is not supported yet)
        
        **Inputs**:
        
        fp -- Data file pointer
        myvars -- Names of the variables to read
        il -- required level
        
        **Output**:
        
        Dictionary consisting of variable names as keys and its values. 
        
        """
        # Read the grid information
        dim = fp['Chombo_global'].attrs.get('SpaceDim')
        nlev = fp.attrs.get('num_levels')
        if ilev < 0:
            il = nlev-1
        else:
            il = min(nlev-1,ilev)
        self.level = il
        if (il != ilev):
            print("Maximum level is {}, reading it".format(il))
        lev  = []
        for i in range(nlev):
            lev.append('level_'+str(i))        
        freb = np.zeros(nlev,dtype='int')
        for i in range(il+1)[::-1]:
            fl = fp[lev[i]]
            if (i == il):
                pdom = fl.attrs.get('prob_domain')
                dx = fl.attrs.get('dx')
                dt = fl.attrs.get('dt')
                ystr = 1. ; zstr = 1. ; logr = 0
                try:
                    geom = fl.attrs.get('geometry')
                    logr = fl.attrs.get('logr')
                    if (dim >= 2):
                        ystr = fl.attrs.get('g_x2stretch')
                    if (dim == 3):
                        zstr = fl.attrs.get('g_x3stretch')
                except:
                    print('Old HDF5 file, not reading stretch and logr factors')
                freb[i] = 1
                x1b = fl.attrs.get('domBeg1')
                if (dim == 1):
                    x2b = 0
                else:
                    x2b = fl.attrs.get('domBeg2')
                if (dim == 1 or dim == 2):
                    x3b = 0
                else:
                    x3b = fl.attrs.get('domBeg3')
                jbeg = 0 ; jend = 0 ; ny = 1
                kbeg = 0 ; kend = 0 ; nz = 1
                if (dim == 1):
                    ibeg = pdom[0] ; iend = pdom[1] ; nx = iend-ibeg+1
                elif (dim == 2):
                    ibeg = pdom[0] ; iend = pdom[2] ; nx = iend-ibeg+1
                    jbeg = pdom[1] ; jend = pdom[3] ; ny = jend-jbeg+1
                elif (dim == 3):
                    ibeg = pdom[0] ; iend = pdom[3] ; nx = iend-ibeg+1
                    jbeg = pdom[1] ; jend = pdom[4] ; ny = jend-jbeg+1
                    kbeg = pdom[2] ; kend = pdom[5] ; nz = kend-kbeg+1
            else:
                rat = fl.attrs.get('ref_ratio')
                freb[i] = rat*freb[i+1]
        dx0 = dx*freb[0]

        ## Allow to load only a portion of the domain
        if (self.x1range != None):
            if logr == 0:
                self.x1range = self.x1range-x1b
            else:
                self.x1range = [np.log(self.x1range[0]/x1b),np.log(self.x1range[1]/x1b)]
            ibeg0 = min(self.x1range)/dx0 ; iend0 = max(self.x1range)/dx0
            ibeg  = max([ibeg, int(ibeg0*freb[0])]) ; iend = min([iend,int(iend0*freb[0]-1)])
            nx = iend-ibeg+1
        if (self.x2range != None):
            self.x2range = (self.x2range-x2b)/ystr
            jbeg0 = min(self.x2range)/dx0 ; jend0 = max(self.x2range)/dx0
            jbeg  = max([jbeg, int(jbeg0*freb[0])]) ; jend = min([jend,int(jend0*freb[0]-1)])
            if jend < jbeg:
                jend=jbeg
            ny = jend-jbeg+1
        if (self.x3range != None):
            self.x3range = (self.x3range-x3b)/zstr
            kbeg0 = min(self.x3range)/dx0 ; kend0 = max(self.x3range)/dx0
            kbeg  = max([kbeg, int(kbeg0*freb[0])]) ; kend = min([kend,int(kend0*freb[0]-1)])
            if kend < kbeg:
                kend=kbeg
            nz = kend-kbeg+1
	    
        ## Create uniform grids at the required level
        if logr == 0:
            x1 = x1b + (ibeg+np.array(range(nx))+0.5)*dx
        else:
            x1 = x1b*(np.exp((ibeg+np.array(range(nx))+1)*dx)+np.exp((ibeg+np.array(range(nx)))*dx))*0.5
        ## print(" Level {} ".format(il))
        ## print("  >> x1b  = {}".format(x1b))
        ## print("  >> ibeg = {}".format(ibeg))
        ## print("  >> nx   = {}".format(nx))
        ## print("  >> dx   = {}".format(dx))
            
        x2 = x2b + (jbeg+np.array(range(ny))+0.5)*dx*ystr
        x3 = x3b + (kbeg+np.array(range(nz))+0.5)*dx*zstr
        if logr == 0:
            dx1 = np.ones(nx)*dx
        else:
            dx1 = x1b*(np.exp((ibeg+np.array(range(nx))+1)*dx)-np.exp((ibeg+np.array(range(nx)))*dx))
        dx2 = np.ones(ny)*dx*ystr
        dx3 = np.ones(nz)*dx*zstr

        # Create the xr arrays containing the edges positions
        # Useful for pcolormesh which should use those
        x1r = np.zeros(len(x1)+1) ; x1r[1:] = x1 + dx1/2.0 ; x1r[0] = x1r[1]-dx1[0]
        x2r = np.zeros(len(x2)+1) ; x2r[1:] = x2 + dx2/2.0 ; x2r[0] = x2r[1]-dx2[0]
        x3r = np.zeros(len(x3)+1) ; x3r[1:] = x3 + dx3/2.0 ; x3r[0] = x3r[1]-dx3[0]
        NewGridDict = dict([('n1',nx),('n2',ny),('n3',nz),\
                            ('x1',x1),('x2',x2),('x3',x3),\
                            ('x1r',x1r),('x2r',x2r),('x3r',x3r),\
                            ('dx1',dx1),('dx2',dx2),('dx3',dx3),\
                            ('Dt',dt)])
        
        # Variables table
        nvar = len(myvars)
        vars = np.zeros((nx,ny,nz,nvar))
        flagAMR = np.zeros((nx,ny,nz))
        
        LevelDic = {'nbox':0,'ibeg':ibeg,'iend':iend,'jbeg':jbeg,'jend':jend,'kbeg':kbeg,'kend':kend}
        AMRLevel = [] 
        AMRBoxes = np.zeros((nx,ny,nz))
        for i in range(il+1):

            AMRLevel.append(LevelDic.copy())
            fl = fp[lev[i]]
            data = fl['data:datatype=0']
            boxes = fl['boxes']
            nbox = len(boxes['lo_i'])
            AMRLevel[i]['nbox'] = nbox
            ncount = 0
            AMRLevel[i]['box']=[]



            # args = (i,dim,il,fl,data,boxes,nbox,vars,AMRBoxes,AMRLevel,flagAMR,
            # ncount,LevelDic,x1b,x2b,x3b,dx,freb,logr,ystr,zstr,noload,nvar)
            
            # res = Parallel(n_jobs=4)(delayed(self.InterpGrid)(j,*args)  
                            # for j in range(nbox))

            # AMRBoxes = res[0]
            # AMRLevel = res[1]
            # flagAMR = res[2]
            # vars = res[3]
            
            AMRBoxes,AMRLevel,flagAMR,vars=\
            self.InterpGrid(i,dim,il,fl,data,boxes,nbox,vars,AMRBoxes,AMRLevel,flagAMR,
                        ncount,LevelDic,x1b,x2b,x3b,dx,
                        freb,logr,ystr,zstr,noload,nvar,centre=centre,silent=silent)
        

        h5vardict={}
        for iv in range(nvar):
            h5vardict[myvars[iv]] = vars[:,:,:,iv].squeeze()
        AMRdict = dict([('AMRBoxes',AMRBoxes),('AMRLevel',AMRLevel),('flagAMR',flagAMR)])
        OutDict = dict(NewGridDict)
        OutDict.update(AMRdict)
        OutDict.update(h5vardict)

        return OutDict

    def DataScan(self, fp, n1, n2, n3, endian, dtype, outsh, off=None,noload=False):
        """ Scans the DBL and FLT data files. 
        
        **Inputs**:
        
        fp -- Data file pointer\n
        n1 -- No. of points in X1 direction\n
        n2 -- No. of points in X2 direction\n
        n3 -- No. of points in X3 direction\n
        endian -- Endianess of the data
        dtype -- datatype
        off -- offset (for avoiding staggered B fields) 
     
        **Output**:
        
        Dictionary consisting of variable names as keys and its values. 
        
        """
        #if off is not None:
        #    off_fmt = endian+str(off)+dtype
        #    nboff = np.dtype(off_fmt).itemsize
        #    fp.read(nboff)
            
        if noload:
            to_return=eval("np.zeros(outsh)%s.transpose()" % (self.dec_slice))
        else:
            n1_tot = self.n1_tot ; n2_tot = self.n2_tot ; n3_tot = self.n3_tot
            irange = self.irange ; jrange = self.jrange ; krange = self.krange
            if (off == 1):
                n1_tot = n1_tot + 1
            elif (off == 2):
                n2_tot = n2_tot + 1
            elif (off == 3):
                n3_tot = n3_tot + 1
            A = array.array(dtype)
            fmt = endian+str(n1_tot*n2_tot*n3_tot)+dtype
            nb = np.dtype(fmt).itemsize 
            if self.debug_time: start=timer()
            A.fromstring(fp.read(nb))
            if self.debug_time: end=timer() ; print('read+fromstring',end-start)
            if self.debug_time: start=timer()
            if (self.Slice):
                darr = np.zeros((n1*n2*n3))
                indxx = np.sort([n2_tot*n1_tot*k + j*n1_tot + i for i in irange for j in jrange for k in krange])
                if (sys.byteorder != self.endianess):
                    A.byteswap()
                for ii,iii in enumerate(indxx):
                    darr[ii] = A[iii]
                darr = [darr]
            else:
                darr = np.frombuffer(A,dtype=np.dtype(fmt)).astype(dtype)
            if self.debug_time: end=timer() ; print('frombuffer',end-start)
            if self.debug_time: start=timer()
            to_return=eval("np.reshape(darr[0],outsh){}.transpose()".format(self.dec_slice))
            if self.debug_time: end=timer() ; print('reshape',end-start)

        return to_return

    def ReadSingleFile(self, datafilename, myvars, n1, n2, n3, endian,
                       dtype, ddict,silent=False,noload=False,centre=False):
        """Reads a single data file, data.****.dtype.
    
        **Inputs**:
        
        datafilename -- Data file name\n
        myvars -- List of variable names to be read\n
        n1 -- No. of points in X1 direction\n
        n2 -- No. of points in X2 direction\n
        n3 -- No. of points in X3 direction\n
        endian -- Endianess of the data
        dtype -- datatype
        ddict -- Dictionary containing Grid and Time Information\n
        which is updated\n
        
        **Output**:
        
        Updated Dictionary consisting of variable names as keys and its values.
        

        """
        #if not noload:
        if self.datatype == 'hdf5':
            fp = h5.File(datafilename,'r')
        else:
            fp = self.OpenFile(datafilename, "rb")
        #else:
        #    #if not noload:
        #    #    fp = self.OpenFile(datafilename, "rb")
        #    #else:
        #    fp=None
        if not silent:
            if noload:
                print ("Initialize without reading data file %s"%datafilename)
            else:
                print ("Reading Data file %s"%datafilename)
        if self.datatype == 'vtk':
            vtkd = self.DataScanVTK(fp, n1, n2, n3, endian, dtype,noload=noload)
            ddict.update(vtkd)
        elif self.datatype == 'hdf5':
            h5d = self.DataScanHDF5(fp,myvars,self.level,noload=noload,silent=silent,centre=centre)
            ddict.update(h5d)
            if hasYT:
                print ('You have yt installed: the pyPLUTO object also contains the yt object.')
                self.get_caseparams()
                units_override = {"length_unit":(self.Ulen,"cm"),\
                                  "time_unit":(self.Ulen/self.Uvel,"s"),\
                                  "mass_unit":(self.Urho*(self.Ulen**3),"g")}
                self.Pyt = yt.load(datafilename,units_override=units_override,verbose=0)
        else:
            for i in range(len(myvars)):
                if (myvars[i] == 'bx1s') or (myvars[i] == 'b1s') or (myvars[i] == 'Bx1s'):
                    n1p1 = n1+1 if n1 > 1 else 1
                    ddict.update({myvars[i]: self.DataScan(fp, n1p1, n2, n3, endian, dtype, self.nshp_b1s,off=1,noload=noload)})
                elif (myvars[i] == 'bx2s') or (myvars[i] == 'b2s') or (myvars[i] == 'Bx2s'):
                    n2p1 = n2+1 if n2 > 1 else 1
                    ddict.update({myvars[i]: self.DataScan(fp, n1, n2p1, n3, endian, dtype, self.nshp_b2s,off=2,noload=noload)})
                elif (myvars[i] == 'bx3s') or (myvars[i] == 'b3s') or (myvars[i] == 'Bx3s'):
                    n3p1 = n3+1 if n3 > 1 else 1
                    ddict.update({myvars[i]: self.DataScan(fp, n1, n2, n3p1, endian, dtype, self.nshp_b3s,off=3,noload=noload)})
                else:
                    ddict.update({myvars[i]: self.DataScan(fp, n1, n2, n3, endian, dtype, self.nshp,noload=noload)})
            if (not silent) and (not noload):
                tmp=fp.read()
                if (len(tmp) > 0):
                    print ('Oh BOY, something went wrong becasue',len(tmp),'were not read')

        if not noload:
            fp.close()

    def ReadMultipleFiles(self, nstr, dataext, myvars, n1, n2, n3, endian,
                          dtype, ddict,noload=False):
        """Reads a  multiple data files, varname.****.dataext.
        
        **Inputs**:
        
        nstr -- File number in form of a string\n
        dataext -- Data type of the file, e.g., 'dbl', 'flt' or 'vtk' \n
        myvars -- List of variable names to be read\n
        n1 -- No. of points in X1 direction\n
        n2 -- No. of points in X2 direction\n
        n3 -- No. of points in X3 direction\n
        endian -- Endianess of the data
        dtype -- datatype
        ddict -- Dictionary containing Grid and Time Information\n
        which is updated.
        
        **Output**:
        
        Updated Dictionary consisting of variable names as keys and its values.
        
        
        """
        for i in range(len(myvars)):
            datafilename = self.wdir+myvars[i]+"."+nstr+dataext
            #fp = open(datafilename, "rb")
            fp = self.OpenFile(datafilename, "rb")
            if self.datatype == 'vtk':
                ddict.update(self.DataScanVTK(fp, n1, n2, n3, endian, dtype,noload=noload))
            else:
                ddict.update({myvars[i]: self.DataScan(fp, n1, n2, n3, endian,
                                                       dtype,noload=noload)})
            fp.close()

    def ReadDataFile(self, num, silent=False, noload=False,centre=False,filetype='data'):
        """Reads the data file generated from PLUTO code.

        **Inputs**:
        
        num -- Data file number in form of an Integer.
        
        **Outputs**:
        
        Dictionary that contains all information about Grid, Time and 
        variables.
        
        """
        gridfile = self.wdir+"grid.out"
        if self.datatype == "float":
            dtype = "f"
            varfile = self.wdir+"flt.out"
            dataext = ".flt"
        elif self.datatype == "vtk":
            dtype = "f"
            varfile = self.wdir+"vtk.out"
            dataext=".vtk"
        elif self.datatype == 'hdf5':
            dtype = 'd'
            dataext = '.hdf5'
            nstr = num
            varfile = self.wdir+self.filetype+"."+nstr+dataext
        else:
            dtype = "d"
            varfile = self.wdir+"dbl.out"
            dataext = ".dbl"
        
        self.ReadVarFile(varfile,int(num))
        self.ReadGridFile(gridfile,dtype)
        self.ReadTimeInfo(varfile)
        self.ReadSysFile()
        nstr = num
        if self.endianess == 'big':
            endian = ">"
        elif self.datatype == 'vtk':
            endian = ">"
        else:
            endian = "<"

        D = [('nstep', self.NStep), ('time', self.SimTime), ('dt', self.Dt),
             ('n1', self.n1), ('n2', self.n2), ('n3', self.n3),
             ('x1', self.x1), ('x2', self.x2), ('x3', self.x3),
             ('dx1', self.dx1), ('dx2', self.dx2), ('dx3', self.dx3),
             ('x1r', self.x1r), ('x2r', self.x2r), ('x3r', self.x3r),
             ('endian', self.endianess), ('datatype', self.datatype),
             ('filetype', self.fltype)]
        ddict = dict(D)

        if self.fltype == "single_file":
            datafilename = self.wdir+self.filetype+"."+nstr+dataext
            self.ReadSingleFile(datafilename, self.vars, self.n1, self.n2,
                                self.n3, endian, dtype,
                                ddict,silent=silent,centre=centre,noload=noload)
        elif self.fltype == "multiple_files":
            self.ReadMultipleFiles(nstr, dataext, self.vars, self.n1, self.n2,
                                   self.n3, endian, dtype, ddict,noload=noload)
        else:
            print ("Wrong file type : CHECK pluto.ini for file type.")
            print ("Only supported are .dbl and .flt")
            sys.exit()            
        
        # Update dict if the resolution was decreased
        if self.dec != 1:
            for idir in [1,2,3]:
                ddict.update({'x{}'.format(idir) : getattr(self,"x{}".format(idir))[::self.dec]})
                ddict.update({'n{}'.format(idir) : len(getattr(self,"x{}".format(idir))[::self.dec])})
                ddict.update({'x{}r'.format(idir): getattr(self,"x{}r".format(idir))[::self.dec]})
                ddict.update({'dx{}'.format(idir): getattr(self,"dx{}".format(idir))[::self.dec]*self.dec})

        return ddict


    def ToVTK(self,outfile='',additional_fields=[]):
        """
        
        This is used to tansform dbl files into vtk 
        Arguments:
           - outfile: name of the output file with the vtk extension
           - additional_fields: list of additional field names to output in the vtk file
                                -> this is typically useful when performing some operations 
                                   on the 3D fields 

        """

        if not hasVTK:
            print("You should install VTK python module, see https://bitbucket.org/pauloh/pyevtk")
            return

        # By default write with the same basename
        if len(outfile) == 0:
            call("mkdir -p "+self.wdir+"VTK",shell=True)
            outfile = self.wdir+"VTK/data."+self.NStepStr

        # List of fiels to output in vtk (vector can be created afterwards, no need to recopy things)
        outfields=self.vars+additional_fields
        dict_output={}
        for v in outfields:
            dict_output[v] = np.asfortranarray(getattr(self,v))

        # Perform the ouput
        x1=np.asfortranarray(self.x1r)
        x2=np.asfortranarray(self.x2r)
        x3=np.asfortranarray(self.x3r)
        print("Saving {}".format(outfile))
        gridToVTK(outfile,x1,x2,x3,cellData=dict_output)

    def get_caseparams(self):
        
        # Get the parameters from definitions.h
        fname = self.wdir+'/definitions.h'
        if (os.path.exists(fname)):        
            lines=[line.strip().split() for line in open(fname)]
            for line in lines:
                if len(line) > 0:
                    if line[0] == '#define':
                        try:
                            object.__setattr__(self,line[1],float(line[2]))
                        except ValueError:
                            object.__setattr__(self,line[1],line[2])
        if hasattr(self,"UNIT_DENSITY"):
            self.Urho = self.UNIT_DENSITY
            self.Ulen = self.UNIT_LENGTH
            self.Uvel = self.UNIT_VELOCITY
            self.default_norm=False
        else:
            self.Urho = 5.15e-17
            self.Ulen = 6.9599e10
            self.Uvel = 4.367e7
            self.default_norm=True
    
        fname = self.wdir+'/pluto.ini'
        if (os.path.exists(fname)):
            lines=[line.strip() for line in open(fname)]
            for ll in lines[lines.index('[Parameters]')+2:]:
                lll = ll.split()
                if (len(lll) > 0):
                    object.__setattr__(self,lll[0],float(lll[1]))
            # Compute adimensional paramaters
            if hasattr(self,'R_STAR') and self.default_norm:
                self.Urho = 1.e-16
                self.Ulen = self.R_STAR*6.9599e10
                GG = 6.6728e-8 ; Msun = 1.98855e33 
                if not hasattr(self,'M_STAR'):
                    self.M_STAR=1.0
                self.Uvel = np.sqrt(GG*self.M_STAR*Msun/self.Ulen)
                self.default_norm=False
        #else:
        #    print ('Err: I did not find '+self.wdir+'/pluto.ini')
    
        # Specific stuff (AS)
        if hasattr(self,'RHOC'):
            self.RHO_STAR=self.RHOC
        if not hasattr(self,'RHO_STAR'):
            self.RHO_STAR=1.
        if not hasattr(self,'GEOMETRY'):
            self.GEOMETRY='cylindrical'
        if (not hasattr(self,'VROT_VESC')) and (not hasattr(self,'VphiStar_VESC')):
            self.VROT_VESC=3.03e-03
        if not hasattr(self,'CS_VESC'):
            self.CS_VESC=2.599e-01
        if not hasattr(self,'VA_VESC'):
            self.VA_VESC=3.183e-01
        if not hasattr(self,'GAMMA'):
            self.GAMMA=1.05

        
class Tools(object):
    """
    
    This Class has all the functions doing basic mathematical
    operations to the vector or scalar fields.
    It is called after pyPLUTO.pload object is defined.
    
    """
    
    def deriv(self,Y,X=None):
        """
        Calculates the derivative of Y with respect to X.
        
        **Inputs:**
        
        Y : 1-D array to be differentiated.\n
        X : 1-D array with len(X) = len(Y).\n
        
        If X is not specified then by default X is chosen to be an equally spaced array having same number of elements
        as Y.
        
        **Outputs:**
        
        This returns an 1-D array having the same no. of elements as Y (or X) and contains the values of dY/dX.
        
        """
        n = len(Y)
        n2 = n-2
        if X==None : X = np.arange(n)
        Xarr = np.asarray(X,dtype='float')
        Yarr = np.asarray(Y,dtype='float')
        x12 = Xarr - np.roll(Xarr,-1)   #x1 - x2
        x01 = np.roll(Xarr,1) - Xarr    #x0 - x1
        x02 = np.roll(Xarr,1) - np.roll(Xarr,-1) #x0 - x2
        DfDx = np.roll(Yarr,1) * (x12 / (x01*x02)) + Yarr * (1./x12 - 1./x01) - np.roll(Yarr,-1) * (x01 / (x02 * x12))
        # Formulae for the first and last points:
        DfDx[0] = Yarr[0] * (x01[1]+x02[1])/(x01[1]*x02[1]) - Yarr[1] * x02[1]/(x01[1]*x12[1]) + Yarr[2] * x01[1]/(x02[1]*x12[1])
        DfDx[n-1] = -Yarr[n-3] * x12[n2]/(x01[n2]*x02[n2]) + Yarr[n-2]*x02[n2]/(x01[n2]*x12[n2]) - Yarr[n-1]*(x02[n2]+x12[n2])/(x02[n2]*x12[n2])
        
        return DfDx
    
    def Grad(self,phi,x1,x2,dx1,dx2,polar=False):
        """ This method calculates the gradient of the 2D scalar phi.
        
        **Inputs:**
        
        phi -- 2D scalar whose gradient is to be determined.\n
        x1 -- The 'x' array\n
        x2 -- The 'y' array\n
        dx1 -- The grid spacing in 'x' direction.\n
        dx2 -- The grid spacing in 'y' direction.\n
        polar -- The keyword should be set to True inorder to estimate the Gradient in polar co-ordinates. By default it is set to False.
        
        **Outputs:**
        
        This routine outputs a 3D array with shape = (len(x1),len(x2),2), such that [:,:,0] element corresponds to the gradient values of phi wrt to x1 and [:,:,1] are the gradient values of phi wrt to x2.
        
        """
        (n1, n2) = phi.shape 
        grad_phi = np.zeros(shape=(n1,n2,2))
        h2 = np.ones(shape=(n1,n2))
        if polar == True:
            for j in range(n2):
                h2[:,j] = x1
        
        for i in range(n1):
            scrh1 = phi[i,:]
            grad_phi[i,:,1] = self.deriv(scrh1,x2)/h2[i,:]
        for j in range(n2):
            scrh2 = phi[:,j]
            grad_phi[:,j,0] = self.deriv(scrh2,x1)

        return grad_phi

    def Div(self,u1,u2,x1,x2,dx1,dx2,geometry=None):
        """ This method calculates the divergence of the 2D vector fields u1 and u2.
        
        **Inputs:**
        
        u1 -- 2D vector along x1 whose divergence is to be determined.\n
        u2 -- 2D vector along x2 whose divergence is to be determined.\n
        x1 -- The 'x' array\n
        x2 -- The 'y' array\n
        dx1 -- The grid spacing in 'x' direction.\n
        dx2 -- The grid spacing in 'y' direction.\n
        geometry -- The keyword *geometry* is by default set to 'cartesian'. It can be set to either one of the following : *cartesian*, *cylindrical*, *spherical* or *polar*. To calculate the divergence of the vector fields, respective geometric corrections are taken into account based on the value of this keyword.
        
        **Outputs:**
        
        A 2D array with same shape as u1(or u2) having the values of divergence.
        
        """
        (n1, n2) = u1.shape
        Divergence = np.zeros(shape=(n1,n2))
        du1 = np.zeros(shape=(n1,n2))
        du2 = np.zeros(shape=(n1,n2))
        
        A1 = np.zeros(shape=n1)
        A2 = np.zeros(shape=n2)

        dV1 = np.zeros(shape=(n1,n2))
        dV2 = np.zeros(shape=(n1,n2))

        if geometry == None : geometry = 'cartesian'
        
        #------------------------------------------------
        #  define area and volume elements for the
        #  different coordinate systems
        #------------------------------------------------
        
        if geometry == 'cartesian' :
            A1[:] = 1.0
            A2[:] = 1.0
            dV1   = np.outer(dx1,A2)
            dV2   = np.outer(A1,dx2)

        if geometry == 'cylindrical' :
            A1 = x1
            A2[:] = 1.0
            dV1 = np.meshgrid(x1*dx1,A2)[0].T*np.meshgrid(x1*dx1,A2)[1].T
            for i in range(n1) : dV2[i,:] = dx2[:]
        
        if geometry == 'polar' :
            A1    = x1
            A2[:] = 1.0
            dV1   = np.meshgrid(x1,A2)[0].T*np.meshgrid(x1,A2)[1].T
            dV2   = np.meshgrid(x1,dx2)[0].T*np.meshgrid(x1,dx2)[1].T

        if geometry == 'spherical' :
            A1 = x1*x1
            A2 = np.sin(x2)
            for j in range(n2): dV1[:,j] = A1*dx1
            dV2   = np.meshgrid(x1,np.sin(x2)*dx2)[0].T*np.meshgrid(x1,np.sin(x2)*dx2)[1].T

        # ------------------------------------------------
        #              Make divergence
        # ------------------------------------------------
        for i in range(1,n1-1):
            du1[i,:] = 0.5*(A1[i+1]*u1[i+1,:] - A1[i-1]*u1[i-1,:])/dV1[i,:]
        for j in range(1,n2-1):
            du2[:,j] = 0.5*(A2[j+1]*u2[:,j+1] - A2[j-1]*u2[:,j-1])/dV2[:,j]

        Divergence = du1 + du2
        return Divergence

    def RTh2Cyl(self,R,Th,X1,X2):
        """ This method does the transformation from spherical coordinates to cylindrical ones.
        
        **Inputs:**
        
        R - 2D array of spherical radius coordinates.\n
        Th - 2D array of spherical theta-angle coordinates.\n
        X1 - 2D array of radial component of given vector\n
        X2 - 2D array of thetoidal component of given vector\n
        
        **Outputs:**
        
        This routine outputs two 2D arrays after transformation.
        
        **Usage:**
        
        ``import pyPLUTO as pp``\n
        ``import numpy as np``\n
        ``D = pp.pload(0)``\n
        ``ppt=pp.Tools()``\n
        ``TH,R=np.meshgrid(D.x2,D.x1)``\n
        ``Br,Bz=ppt.RTh2Cyl(R,TH,D.bx1,D.bx2)``
        
        D.bx1 and D.bx2 should be vectors in spherical coordinates. After transformation (Br,Bz) corresponds to vector in cilindrical coordinates.
        
        
        """
        Y1=X1*np.sin(Th)+X2*np.cos(Th)
        Y2=X1*np.cos(Th)-X2*np.sin(Th)
        return Y1,Y2

    def myInterpol(self,RR,N):
        """ This method interpolates (linear interpolation) vector 1D vector RR to 1D N-length vector. Useful for stretched grid calculations. 
        
        **Inputs:**
        
        RR - 1D array to interpolate.\n
        N  - Number of grids to interpolate to.\n
        
        **Outputs:**
        
        This routine outputs interpolated 1D array to the new grid (len=N).
        
        **Usage:**
        
        ``import pyPLUTO as pp``\n
        ``import numpy as np``\n
        ``D = pp.pload(0)``\n
        ``ppt=pp.Tools()``\n
        ``x=linspace(0,1,10) #len(x)=10``\n
        ``y=x*x``\n
        ``Ri,Ni=ppt.myInterpol(y,100) #len(Ri)=100``
        
        Ri - interpolated numbers;
        Ni - grid for Ri
        
        """    
        
        NN=np.linspace(0,len(RR)-1,len(RR))
        spline_fit=UnivariateSpline(RR,NN,k=3,s=0)
        
        RRi=np.linspace(RR[0],RR[-1],N)
        NNi=spline_fit(RRi)
        NNi[0]=NN[0]+0.00001
        NNi[-1]=NN[-1]-0.00001
        return RRi,NNi
        
    def getUniformGrid(self,r,th,rho,Nr,Nth):
        """ This method transforms data with non-uniform grid (stretched) to uniform. Useful for stretched grid calculations. 
        
        **Inputs:**
        
        r  - 1D vector of X1 coordinate (could be any, e.g D.x1).\n
        th - 1D vector of X2 coordinate (could be any, e.g D.x2).\n
        rho- 2D array of data.\n
        Nr - new size of X1 vector.\n
        Nth- new size of X2 vector.\n
        
        **Outputs:**
        
        This routine outputs 2D uniform array Nr x Nth dimension
        
        **Usage:**
        
        ``import pyPLUTO as pp``\n
        ``import numpy as np``\n
        ``D = pp.pload(0)``\n
        ``ppt=pp.Tools()``\n
        ``X1new, X2new, res = ppt.getUniformGrid(D.x1,D.x2,D.rho,20,30)``
        
        X1new - X1 interpolated grid len(X1new)=20
        X2new - X2 interpolated grid len(X2new)=30
        res   - 2D array of interpolated variable
        
        """    

        Ri,NRi=self.myInterpol(r,Nr)
        Ra=np.int32(NRi);Wr=NRi-Ra

        YY=np.ones([Nr,len(th)])
        for i in range(len(th)):
            YY[:,i]=(1-Wr)*rho[Ra,i] + Wr*rho[Ra+1,i]

        THi,NTHi=self.myInterpol(th,Nth)
        THa=np.int32(NTHi);Wth=NTHi-THa

        ZZ=np.ones([Nr,Nth])
        for i in range(Nr):
            ZZ[i,:]=(1-Wth)*YY[i,THa] + Wth*YY[i,THa+1]

        return Ri,THi,ZZ
    
    def sph2cyl(self,D,Dx,rphi=None,theta0=None):
        """ This method transforms spherical data into cylindrical applying interpolation. Works for stretched grid as well, transforms poloidal (R-Theta) data by default. Fix theta and set rphi=True to get (R-Phi) transformation.
        
        **Inputs:**
        
        D  - structure  from 'pload' method.\n
        Dx - variable to be transformed (D.rho for example).\n
        
        **Outputs:**
        
        This routine outputs transformed (sph->cyl) variable and grid.
        
        **Usage:**
        
        ``import pyPLUTO as pp``\n
        ``import numpy as np``\n
        ``D = pp.pload(0)``\n
        ``ppt=pp.Tools()``\n
        ``R,Z,res = ppt.sph2cyl(D,D.rho.transpose())``
        
        R - 2D array with cylindrical radius values
        Z - 2D array with cylindrical Z values
        res - 2D array of transformed variable
        
        """    
        
        if rphi is None or rphi == False:
            rx=D.x1
            th=D.x2            
        else:
            rx=D.x1*np.sin(theta0)
            th=D.x3
            
        rx,th,Dx=self.getUniformGrid(rx,th,Dx.T,200,200)
        Dx=Dx.T
        
        if rphi is None or rphi == False:
            
            r0=np.min(np.sin(th)*rx[0])
            rN=rx[-1]
            dr=rN-r0
            z0=np.min(np.cos(th)*rN)
            zN=np.max(np.cos(th)*rN)
            dz=zN-z0
            dth=th[-1]-th[0]
            rl=np.int32(len(rx)*dr/(rx[-1]-rx[0]))  
            zl=np.int32(rl* dz/dr)
            thl=len(th)
            r=np.linspace(r0, rN, rl)
            z=np.linspace(z0, zN, zl)
        else:
            r0=np.min([np.sin(th)*rx[0] , np.sin(th)*rx[-1]])
            rN=np.max([np.sin(th)*rx[0] , np.sin(th)*rx[-1]])
            dr=rN-r0
            z0=np.min(np.cos(th)*rN)
            zN=np.max(np.cos(th)*rN)
            dz=zN-z0
            dth=th[-1]-th[0]
            rl=np.int32(len(rx)*dr/(rx[-1]-rx[0]))  
            zl=np.int32(rl* dz/dr)
            thl=len(th)
            r=np.linspace(r0, rN, rl)
            z=np.linspace(z0, zN, zl)
            
        R,Z = np.meshgrid(r, z)
        Rs = np.sqrt(R*R + Z*Z)
        
        Th = np.arccos(Z/Rs)
        kv_34=find(R<0)
        Th.flat[kv_34]=2*np.pi - Th.flat[kv_34]
        
        ddr=rx[1]-rx[0]
        ddth=th[1]-th[0]
        
        Rs_copy=Rs.copy()
        Th_copy=Th.copy()
        
        nR1=find(Rs<rx[0])  
        Rs.flat[nR1]=rx[0] 
        nR2=find(Rs>rN)
        Rs.flat[nR2]=rN
        
        nTh1=find(Th>th[-1])
        Th.flat[nTh1]=th[-1]
        nTh2=find(Th<th[0])
        Th.flat[nTh2]=th[0]
        
        
        ra = ((len(rx)-1.001)/(np.max(Rs.flat)-np.min(Rs.flat)) *(Rs-np.min(Rs.flat)))  
        tha = ((thl-1.001)/dth *(Th-th[0]))  
        
        rn = np.int32(ra)
        thn = np.int32(tha)
        dra=ra-rn
        dtha=tha-thn
        w1=1-dra
        w2=dra
        w3=1-dtha
        w4=dtha
        lrx=len(rx)
        NN1=np.int32(rn+thn*lrx)
        NN2=np.int32((rn+1)+thn*lrx)
        NN3=np.int32(rn+(thn+1)*lrx)
        NN4=np.int32((rn+1)+(thn+1)*lrx)
        n=np.transpose(np.arange(0,np.product(np.shape(R))))
        DD=Dx.copy()
        F=R.copy()
        F.flat[n]=w1.flat[n]*(w3.flat[n]*Dx.flat[NN1.flat[n]] + w4.flat[n]*Dx.flat[NN3.flat[n]] )+\
            w2.flat[n]*(w3.flat[n]*Dx.flat[NN2.flat[n]] + w4.flat[n]*Dx.flat[NN4.flat[n]] )
        
        nR1=find(Rs_copy<rx[0]-ddr/1.5)
        nR2=find(Rs_copy>rN+ddr/1.5)
        nTh1=find(Th_copy>th[-1]+ddth/1.5)
        nTh2=find(Th_copy<th[0]-ddth/1.5)
        
        nmask=np.concatenate((nR1,nR2,nTh1,nTh2))
        F.flat[nmask]=np.nan;
        return R,Z,F
    
    def congrid(self,a, newdims, method='nearest',logr=False, centre=False, minusone=False):
        '''Arbitrary resampling of source array to new dimension sizes.
        Currently only supports maintaining the same number of dimensions.
        To use 1-D arrays, first promote them to shape (x,1).
        
        Uses the same parameters and creates the same co-ordinate lookup points
        as IDL''s congrid routine, which apparently originally came from a VAX/VMS
        routine of the same name.
        
        method:
        neighbour - closest value from original data
        nearest and linear - uses n x 1-D interpolations using
                             scipy.interpolate.interp1d
        (see Numerical Recipes for validity of use of n 1-D interpolations)
        spline - uses ndimage.map_coordinates
    
        centre:
        True - interpolation points are at the centres of the bins
        False - points are at the front edge of the bin
    
        minusone:
        For example- inarray.shape = (i,j) & new dimensions = (x,y)
        False - inarray is resampled by factors of (i/x) * (j/y)
        True - inarray is resampled by(i-1)/(x-1) * (j-1)/(y-1)
        This prevents extrapolation one element beyond bounds of input array.
        '''

        if not a.dtype in [np.float64, np.float32]:
            a = np.cast[float](a)
    
        m1 = np.cast[int](minusone)
        ofs = np.cast[int](centre) * 0.5
        old = np.array( a.shape )
        ndims = len( a.shape )

        if len( newdims ) != ndims:
            print ("[congrid] dimensions error. " \
                  "This routine currently only support " \
                   "rebinning to the same number of dimensions. Input dims are {} and output {}".format(a.shape,newdims))
            return None
        newdims = np.asarray( newdims, dtype=float )
        dimlist = []
       
        # if method == 'congrid2':

            # for i in range(ndims):
                
                # olddims = [np.arange(i, dtype = np.float) for i in list( a.shape )]
                
                # if ((i == 0) & (logr)):
                    # dimlist = np.logspace(np.log10(base[0]),np.log10(base[-1]),newdims[i])
                # else:
                    # dimlist = np.logspace(base[0],base[-1],newdims[i])

                    # mint = scipy.interpolate.interp1d(old

        if method == 'neighbour':
            for i in range( ndims ):
                base = np.indices(newdims)[i]
                dimlist.append( (old[i] - m1) / (newdims[i] - m1) \
                                * (base + ofs) - ofs )
            cd = np.array( dimlist ).round().astype(int)
            newa = a[list( cd )]
            return newa
    
        elif method in ['nearest','linear']:
            # calculate new dims
            for i in range( ndims ):
                base = np.arange( newdims[i] )
                dimlist.append( (old[i] - m1) / (newdims[i] - m1) \
                                * (base + ofs) - ofs )
            # specify old dims
            olddims = [np.arange(i, dtype = np.float) for i in list( a.shape )]
    
            # first interpolation - for ndims = any
            if (len(olddims[-1]) == 1):
                newa=np.tile(a,len(dimlist[-1]))
            else:
                mint = scipy.interpolate.interp1d(olddims[-1], a,
                        kind=method)
                newa = mint(dimlist[-1])
             
            trorder = [ndims - 1] + list(range( ndims - 1 ))
            # breakpoint()
            for i in range( ndims - 2, -1, -1 ):
                newa = newa.transpose( trorder )
    
                if (len(olddims[i]) == 1):
                    # Particular case
                    newa = np.tile(newa,len(dimlist[i]))
                else:
                    mint = scipy.interpolate.interp1d( olddims[i], newa,
                            kind=method)
                    newa = mint( dimlist[i] )

            # breakpoint() 

            if ndims > 1:
                # need one more transpose to return to original dimensions
                newa = newa.transpose( trorder )
    
            return newa
        elif method in ['spline']:
            oslices = [ slice(0,j) for j in old ]
            oldcoords = np.ogrid[oslices]
            nslices = [ slice(0,j) for j in list(newdims) ]
            newcoords = np.mgrid[nslices]
    
            newcoords_dims = range(np.rank(newcoords))
            #make first index last
            newcoords_dims.append(newcoords_dims.pop(0))
            newcoords_tr = newcoords.transpose(newcoords_dims)
            # makes a view that affects newcoords
    
            newcoords_tr += ofs
    
            deltas = (np.asarray(old) - m1) / (newdims - m1)
            newcoords_tr *= deltas
    
            newcoords_tr -= ofs
    
            newa = scipy.ndimage.map_coordinates(a, newcoords)
            return newa
        else:
            print ("Congrid error: Unrecognized interpolation type.\n", \
                  "Currently only \'neighbour\', \'nearest\',\'linear\',", \
                  "and \'spline\' are supported.")
            return None


class Image(object):
    ''' This Class has all the routines for the imaging the data
    and plotting various contours and fieldlines on these images.
    CALLED AFTER pyPLUTO.pload object is defined
    '''

    def pldisplay(self, D, var,**kwargs):
        """ This method allows the user to display a 2D data using the 
        matplotlib's pcolormesh.

        **Inputs:**

          D   -- pyPLUTO pload object.\n
          var -- 2D array that needs to be displayed.
        
        *Required Keywords:*

          x1 -- The 'x' array\n
          x2 -- The 'y' array
        
        *Optional Keywords:*

          vmin -- The minimum value of the 2D array (Default : min(var))\n
          vmax -- The maximum value of the 2D array (Default : max(var))\n
          title -- Sets the title of the image.\n
          label1 -- Sets the X Label (Default: 'XLabel')\n
          label2 -- Sets the Y Label (Default: 'YLabel')\n
          polar -- A list to project Polar data on Cartesian Grid.\n
            polar = [True, True] -- Projects r-phi plane.\n
            polar = [True, False] -- Project r-theta plane.\n
            polar = [False, False] -- No polar plot (Default)\n
          cbar -- Its a tuple to set the colorbar on or off. \n
            cbar = (True,'vertical') -- Displays a vertical colorbar\n
            cbar = (True,'horizontal') -- Displays a horizontal colorbar\n
            cbar = (False,'') -- Displays no colorbar.
         
        **Usage:**
          
          ``import pyPLUTO as pp``\n
          ``wdir = '/path/to/the data files/'``\n
          ``D = pp.pload(1,w_dir=wdir)``\n
          ``I = pp.Image()``\n
          ``I.pldisplay(D, D.v2, x1=D.x1, x2=D.x2, cbar=(True,'vertical'),\
          title='Velocity',label1='Radius',label2='Height')``
        """
        x1 = kwargs.get('x1')
        x2 = kwargs.get('x2')
        var = var.T

        fnum=kwargs.get('fignum',1)
        if fnum != -1:
            f1 = figure(fnum, figsize=kwargs.get('figsize',[10,10]),
                        dpi=80, facecolor='w', edgecolor='k')
            ax1 = f1.add_subplot(111)
        else:
            f1=gcf()
            ax1=gca()
        ax1.set_aspect(kwargs.get('aspect','equal'))        

        if kwargs.get('polar',[False,False])[0]:
            xx, yy = self.getPolarData(D,kwargs.get('x2'),rphi=kwargs.get('polar')[1])
            pcolormesh(xx,yy,var,vmin=kwargs.get('vmin',np.min(var)),vmax=kwargs.get('vmax',np.max(var)),cmap=kwargs.get('cmap','RdYlBu_r'),rasterized=kwargs.get('rasterized',True),shading=kwargs.get('shading','flat'))
        else:
            ax1.axis([np.min(x1),np.max(x1),np.min(x2),np.max(x2)])
            pcolormesh(x1,x2,var,vmin=kwargs.get('vmin',np.min(var)),vmax=kwargs.get('vmax',np.max(var)),cmap=kwargs.get('cmap','RdYlBu_r'),rasterized=kwargs.get('rasterized',True),shading=kwargs.get('shading','flat'))
        
        title(kwargs.get('title',"Title"),size=kwargs.get('size'))
        xlabel(kwargs.get('label1',"Xlabel"),size=kwargs.get('size'))
        ylabel(kwargs.get('label2',"Ylabel"),size=kwargs.get('size'))
        if kwargs.get('cbar',(False,''))[0] == True:
            colorbar(orientation=kwargs.get('cbar')[1])
        sca(ax1)
            
    def multi_disp(self,*args,**kwargs):
        mvar = []
        var_cart_list=[]
        for arg in args:
            mvar.append(arg.T)
            
        xmin = np.min(kwargs.get('x1'))
        xmax = np.max(kwargs.get('x1'))        
        ymin = np.min(kwargs.get('x2'))
        ymax = np.max(kwargs.get('x2'))
        mfig = figure(kwargs.get('fignum',1),figsize=kwargs.get('figsize',[10,10]))
        Ncols = kwargs.get('Ncols')
        Nrows = len(args)/Ncols
        mprod = Nrows*Ncols
        dictcbar=kwargs.get('cbar',(False,'','each'))
        for j in range(mprod):
            mfig.add_subplot(Nrows,Ncols,j+1)
            pcolormesh(kwargs.get('x1'),kwargs.get('x2'), mvar[j])
            axis([xmin,xmax,ymin,ymax])
            gca().set_aspect('equal')
            
            xlabel(kwargs.get('label1',mprod*['Xlabel'])[j])
            ylabel(kwargs.get('label2',mprod*['Ylabel'])[j])
            title(kwargs.get('title',mprod*['Title'])[j])
            if (dictcbar[0] == True) and (dictcbar[2] =='each'):
                colorbar(orientation=kwargs.get('cbar')[1])
            if dictcbar[0] == True and dictcbar[2]=='last':
                if (j == np.max(range(mprod))):colorbar(orientation=kwargs.get('cbar')[1])
                    
    def oplotbox(self, AMRLevel, lrange=[0,0], cval=['b','g','r','m','Orange','c','w','k'],\
                 islice=-1, jslice=-1, kslice=-1,geom='CARTESIAN',lw=1,alpha=1):
        """ 
        This method overplots the AMR boxes up to the specified level. 

        **Input:**

          AMRLevel -- AMR object loaded during the reading and stored in the pload object
        
        *Optional Keywords:*

          lrange     -- [level_min,level_max] to be overplotted. By default it shows all the loaded levels\n
          cval       -- list of colors for the levels to be overplotted.\n
          [ijk]slice -- Index of the 2D slice to look for so that the adequate box limits are plotted. 
                        By default oplotbox considers you are plotting a 2D slice of the z=min(x3) plane.\n
          geom       -- Specified the geometry. Currently, CARTESIAN (default), POLAR and SPHERICAL geometries are handled.
        """

        nlev = len(AMRLevel)
        lrange[1] = min(lrange[1],nlev-1)
        npl  = lrange[1]-lrange[0]+1
        lpls = [lrange[0]+v for v in range(npl)]  
        cols = cval[0:nlev]
        # Get the offset and the type of slice
        Slice = 0 ; inds = 'k'
        xx = 'x' ; yy ='y'
        if (islice >= 0):
            Slice = islice + AMRLevel[0]['ibeg'] ; inds = 'i'
            xx = 'y' ; yy ='z'
        if (jslice >= 0):
            Slice = jslice + AMRLevel[0]['jbeg'] ; inds = 'j'
            xx = 'x' ; yy ='z'
        if (kslice >= 0):
            Slice = kslice + AMRLevel[0]['kbeg'] ; inds = 'k'
            xx = 'x' ; yy ='y'
                
        # Overplot the boxes
        for il in lpls:
            level = AMRLevel[il]
            for ib in range(level['nbox']):
                box = level['box'][ib]
                if ((Slice-box[inds+'b'])*(box[inds+'e']-Slice) >= 0):
                    if (geom == 'CARTESIAN'):
                        x0 = box[xx+'0'] ; x1 = box[xx+'1'] 
                        y0 = box[yy+'0'] ; y1 = box[yy+'1'] 
                        plot([x0,x1,x1,x0,x0],[y0,y0,y1,y1,y0],color=cols[il],lw=lw)
                    elif (geom == 'POLAR'):
                        dn = np.pi/50.
                        x0 = box[xx+'0'] ; x1 = box[xx+'1'] 
                        y0 = box[yy+'0'] ; y1 = box[yy+'1']
                        if y0 == y1:
                            y1 = 2*np.pi+y0 - 1.e-3
                        xb = np.concatenate([
                            [x0*np.cos(y0),x1*np.cos(y0)],\
                            x1*np.cos(np.linspace(y0,y1,num=int(abs(y0-y1)/dn) )),\
                            [x1*np.cos(y1),x0*np.cos(y1)],\
                            x0*np.cos(np.linspace(y1,y0,num=int(abs(y0-y1)/dn)))])
                        yb = np.concatenate([
                            [x0*np.sin(y0),x1*np.sin(y0)],\
                            x1*np.sin(np.linspace(y0,y1,num=int(abs(y0-y1)/dn))),\
                            [x1*np.sin(y1),x0*np.sin(y1)],\
                            x0*np.sin(np.linspace(y1,y0,num=int(abs(y0-y1)/dn)))])
                        plot(xb,yb,color=cols[il],lw=lw,alpha=alpha)
                    elif (geom == 'SPHERICAL'):
                        dn = np.pi/100.
                        x0 = box[xx+'0'] ; x1 = box[xx+'1'] 
                        y0 = box[yy+'0'] ; y1 = box[yy+'1']
                        if y0 == y1:
                            y1 = 2*np.pi+y0 - 1.e-3
                        yb = np.concatenate([
                            [x0*np.cos(y0),x1*np.cos(y0)],\
                            x1*np.cos(np.linspace(y0,y1,num=int(abs(y0-y1)/dn) )),\
                            [x1*np.cos(y1),x0*np.cos(y1)],\
                            x0*np.cos(np.linspace(y1,y0,num=int(abs(y0-y1)/dn))),\
                            [x0*np.cos(y0),x1*np.cos(y0)]])
                        xb = np.concatenate([
                            [x0*np.sin(y0),x1*np.sin(y0)],\
                            x1*np.sin(np.linspace(y0,y1,num=int(abs(y0-y1)/dn))),\
                            [x1*np.sin(y1),x0*np.sin(y1)],\
                            x0*np.sin(np.linspace(y1,y0,num=int(abs(y0-y1)/dn))),\
                            [x0*np.sin(y0),x1*np.sin(y0)]])
                    # Need to change directions when cutting in the (r,phi) plane compared to the (r,theta) plane    
                    if (jslice >= 0): 
                        plot(yb,xb,color=cols[il],lw=lw,alpha=alpha)
                    else:
                        plot(xb,yb,color=cols[il],lw=lw,alpha=alpha)

    def FRoplotbox(self, AMRLevel, lrange=[0,0], cval=['b','g','r','m','Orange','c','w','k'],\
                 islice=-1, jslice=-1, kslice=-1,geom='CARTESIAN',lw=1,alpha=1):
        """ 
        This method overplots the AMR boxes up to the specified level. 

        **Input:**

          AMRLevel -- AMR object loaded during the reading and stored in the pload object
        
        *Optional Keywords:*

          lrange     -- [level_min,level_max] to be overplotted. By default it shows all the loaded levels\n
          cval       -- list of colors for the levels to be overplotted.\n
          [ijk]slice -- Index of the 2D slice to look for so that the adequate box limits are plotted. 
                        By default oplotbox considers you are plotting a 2D slice of the z=min(x3) plane.\n
          geom       -- Specified the geometry. Currently, CARTESIAN (default), POLAR and SPHERICAL geometries are handled.
        """

        nlev = len(AMRLevel)
        # lrange[1] = min(lrange[1],nlev-1)
        # npl  = lrange[1]-lrange[0]+1
        # lpls = [lrange[0]+v for v in range(npl)]  
        
        levels = np.arange(lrange[0],lrange[1]+1,1)

        # Getting the color 
        cols = cval[0:nlev]
        # Get the offset and the type of slice
        Slice = 0 ; inds = 'k'
        xx = 'x' ; yy ='y'
        if (islice >= 0):
            Slice = islice + AMRLevel[0]['ibeg'] ; inds = 'i'
            xx = 'y' ; yy ='z'
        if (jslice >= 0):
            Slice = jslice + AMRLevel[0]['jbeg'] ; inds = 'j'
            xx = 'x' ; yy ='z'
        if (kslice >= 0):
            Slice = kslice + AMRLevel[0]['kbeg'] ; inds = 'k'
            xx = 'x' ; yy ='y'
                
        # Overplot the boxes

        # Loop on the levels
        for il in levels:
            level = AMRLevel[il]

            # Loop on boxes
            for ib in range(level['nbox']):
                box = level['box'][ib]
                if ((Slice-box[inds+'b'])*(box[inds+'e']-Slice) >= 0):
                    if (geom == 'CARTESIAN'):
                        x0 = box[xx+'0'] ; x1 = box[xx+'1'] 
                        y0 = box[yy+'0'] ; y1 = box[yy+'1'] 
                        plot([x0,x1,x1,x0,x0],[y0,y0,y1,y1,y0],color=cols[il],lw=lw)
                    elif (geom == 'POLAR'):
                        dn = np.pi/50.
                        x0 = box[xx+'0'] ; x1 = box[xx+'1'] 
                        y0 = box[yy+'0'] ; y1 = box[yy+'1']
                        if y0 == y1:
                            y1 = 2*np.pi+y0 - 1.e-3
                        xb = np.concatenate([
                            [x0*np.cos(y0),x1*np.cos(y0)],\
                            x1*np.cos(np.linspace(y0,y1,num=int(abs(y0-y1)/dn) )),\
                            [x1*np.cos(y1),x0*np.cos(y1)],\
                            x0*np.cos(np.linspace(y1,y0,num=int(abs(y0-y1)/dn)))])
                        yb = np.concatenate([
                            [x0*np.sin(y0),x1*np.sin(y0)],\
                            x1*np.sin(np.linspace(y0,y1,num=int(abs(y0-y1)/dn))),\
                            [x1*np.sin(y1),x0*np.sin(y1)],\
                            x0*np.sin(np.linspace(y1,y0,num=int(abs(y0-y1)/dn)))])
                        plot(xb,yb,color=cols[il],lw=lw,alpha=alpha)

                    elif (geom == 'SPHERICAL'):
                        
                        #Resolution of the circle
                        dn = 2 *np.pi/512.
                        ncell = 16

                        x0 = box[xx+'0'] ; x1 = box[xx+'1'] 
                        y0 = box[yy+'0'] ; y1 = box[yy+'1']

                        if y0 == y1:
                            y1 = 2*np.pi+y0 - 1.e-3


                        xs = np.linspace(x0,x1,num = int(abs(x0-x1)/ncell))

                        ybox = np.concatenate([
                            xs*np.cos(y0),
                            x1*np.cos(np.linspace(y0,y1,num=int(abs(y0-y1)/dn) )),
                            xs[::-1]*np.cos(y1),
                            x0*np.cos(np.linspace(y1,y0,num=int(abs(y0-y1)/dn))),
                            xs*np.cos(y0)
                            ])

                        xbox = np.concatenate([
                            xs*np.sin(y0),
                            x1*np.sin(np.linspace(y0,y1,num=int(abs(y0-y1)/dn))),
                            xs[::-1]*np.sin(y1),
                            x0*np.sin(np.linspace(y1,y0,num=int(abs(y0-y1)/dn))),
                            xs*np.sin(y0)
                            ])
                        


                        if il == 4:
                            for i in range(0,len(xbox)-1,4):
                                ylist =\
                                np.array([ybox[i],ybox[i]])
                                xlist =\
                                np.array([xbox[i],xbox[i+1]])

                                    

                                # print(xlist)
                                # print(ylist)
                                # plot(xlist,ylist,color='black')

                    # Need to change directions when cutting in the (r,phi) plane compared to the (r,theta) plane    
                    if (jslice >= 0): 
                        plot(ybox,xbox,color=cols[il],lw=lw,alpha=alpha)
                    else:
                        plot(xbox,ybox,color=cols[il],lw=lw,alpha=alpha)

    def getPolarData(self, Data, ang_coord, rphi=False):
        """To get the Cartesian Co-ordinates from Polar.
        
        **Inputs:**
        
          Data -- pyPLUTO pload Object\n
          ang_coord -- The Angular co-ordinate (theta or Phi)
         
        *Optional Keywords:*
        
          rphi -- Default value FALSE is for R-THETA data, 
          Set TRUE for R-PHI data.\n

        **Outputs**:
        
          2D Arrays of X, Y from the Radius and Angular co-ordinates.\n
          They are used in pcolormesh in the Image.pldisplay functions.
        """
        D = Data
        if ang_coord is D.x2:
            x2r = D.x2r
        elif ang_coord is D.x3:
            x2r = D.x3r
        else:
            print("Angular co-ordinate must be given")
            
        rcos = np.outer(np.cos(x2r), D.x1r)
        rsin = np.outer(np.sin(x2r), D.x1r)        
        
        if rphi:
            xx = rcos
            yy = rsin
        else:
            xx = rsin
            yy = rcos
            
        return xx, yy
        
    def field_interp(self,var1,var2,x,y,dx,dy,xp,yp):
        """ This method interpolates value of vector fields (var1 and var2) on field points (xp and yp).
        The field points are obtained from the method field_line.

        **Inputs:**
        
          var1 -- 2D Vector field in X direction\n
          var2 -- 2D Vector field in Y direction\n
          x -- 1D X array\n
          y -- 1D Y array\n
          dx -- 1D grid spacing array in X direction\n
          dy -- 1D grid spacing array in Y direction\n
          xp -- field point in X direction\n
          yp -- field point in Y direction\n

        **Outputs:**

                  A list with 2 elements where the first element corresponds to the interpolate field point in 'x' direction and the second element is the field point in 'y' direction.  

        """
        q=[]
        U = var1
        V = var2
        i0 = np.abs(xp-x).argmin()
        j0 = np.abs(yp-y).argmin()
        scrhUx = np.interp(xp,x,U[:,j0])
        scrhUy = np.interp(yp,y,U[i0,:])
        q.append(scrhUx + scrhUy - U[i0,j0])
        scrhVx = np.interp(xp,x,V[:,j0])
        scrhVy = np.interp(yp,y,V[i0,:])
        q.append(scrhVx + scrhVy - V[i0,j0])
        return q

    def field_line(self,var1,var2,x,y,dx,dy,x0,y0,step_max=1.0,rp=0.1,rorb=0,rs=1.,rmax=10000.):
        """ This method is used to obtain field lines (same as fieldline.pro in PLUTO IDL tools).
        
        **Inputs:**
        
        var1 -- 2D Vector field in X direction\n
        var2 -- 2D Vector field in Y direction\n
        x -- 1D X array\n
        y -- 1D Y array\n
        dx -- 1D grid spacing array in X direction\n
        dy -- 1D grid spacing array in Y direction\n
        x0 -- foot point of the field line in X direction\n
        y0 -- foot point of the field line in Y direction\n
        
        **Outputs:**
        
        This routine returns a dictionary with keys - \n
        qx -- list of the field points along the 'x' direction.
        qy -- list of the field points along the 'y' direction.
        
        **Usage:**
        
        See the myfieldlines routine for the same. 
        
        
        """
        xbeg = x[0] - 0.5*dx[0]
        xend = x[-1] + 0.5*dx[-1]
        
        ybeg = y[0]  - 0.5*dy[0]
        yend = y[-1] + 0.5*dy[-1]
        
        
        MAX_STEPS = 10000
        # FWD
        inside_domain=self.in_my_domain(x0,y0,xbeg,ybeg,xend,yend,rs,rorb,rp,rmax)
        xln_fwd = [x0]
        yln_fwd = [y0]
        xln_bck = [x0]
        yln_bck = [y0]
        k = 0
        tt0 = time.time()
        t1 = 0. ; t2 = 0.; t3 = 0; t4 =0. ; t5 =0
        while (inside_domain == True):
            t0 = time.time()
            R1 = self.field_interp(var1,var2,x,y,dx,dy,xln_fwd[k],yln_fwd[k])
            t1 = t1 + (time.time()-t0)

            t0 = time.time()
            dl = 0.5*np.max(np.concatenate((dx,dy)))/(np.sqrt(R1[0]*R1[0] + R1[1]*R1[1] + 1.e-14))
            dl = min([dl,step_max])
            t2 = t2 + (time.time()-t0)

            xscrh = xln_fwd[k] + 0.5*dl*R1[0]
            yscrh = yln_fwd[k] + 0.5*dl*R1[1]
            t3 = t3 + (time.time()-t0)

            t0 = time.time()
            R2 = self.field_interp(var1,var2,x,y,dx,dy,xscrh,yscrh)
            t4 = t4 + (time.time()-t0)

            xln_one = xln_fwd[k] + dl*R2[0]
            yln_one = yln_fwd[k] + dl*R2[1]

            t0 = time.time()
            xln_fwd.append(xln_one)
            yln_fwd.append(yln_one)
            t4 = t4 + (time.time()-t0)

            t0 = time.time()
            inside_domain=self.in_my_domain(xln_one,yln_one,xbeg,ybeg,xend,yend,rs,rorb,rp,rmax)
            inside_domain = inside_domain and (k < MAX_STEPS-3)
            t5 = t5 + (time.time()-t0)

            k = k + 1
            if ((xln_fwd[k-1] == xln_fwd[k]) and (yln_fwd[k-1] == yln_fwd[k])):
                inside_domain = False

        ttot = time.time()-tt0 ; t1 = t1/ttot ; t2 = t2/ttot; t3 = t3/ttot; t4 = t4/ttot ; t5 = t5/ttot
        if (k == 9998):
            print ('Times',t1,t2,t3,t4,t5,ttot)
        k_fwd = k

        #BCK
        inside_domain=self.in_my_domain(x0,y0,xbeg,ybeg,xend,yend,rs,rorb,rp,rmax) and (k < MAX_STEPS-3)
        k=0
        while inside_domain == True:
            R1 = self.field_interp(var1,var2,x,y,dx,dy,xln_bck[k],yln_bck[k])
            dl = 0.5*np.max(np.concatenate((dx,dy)))/(np.sqrt(R1[0]*R1[0] + R1[1]*R1[1] + 1.e-14))
            dl = -min([abs(dl),step_max])
            xscrh = xln_bck[k] + 0.5*dl*R1[0]
            yscrh = yln_bck[k] + 0.5*dl*R1[1]
            R2 = self.field_interp(var1,var2,x,y,dx,dy,xscrh,yscrh)
            xln_one = xln_bck[k] + dl*R2[0]
            yln_one = yln_bck[k] + dl*R2[1]
            xln_bck.append(xln_one)
            yln_bck.append(yln_one)
            inside_domain=self.in_my_domain(xln_one,yln_one,xbeg,ybeg,xend,yend,rs,rorb,rp,rmax)
            inside_domain = inside_domain and (k < MAX_STEPS-3)
            k = k + 1
            if ((xln_bck[k-1] == xln_bck[k]) and (yln_bck[k-1] == yln_bck[k])):
                inside_domain = False
        k_bck = k

        qx = np.asarray(xln_bck[0:k_bck][::-1]+xln_fwd[0:k_fwd])
        qy = np.asarray(yln_bck[0:k_bck][::-1]+yln_fwd[0:k_fwd])
        flines={'qx':qx,'qy':qy}
        return flines
    
    def ASfield_line(self,var1,var2,x,y,dx,dy,x0,y0,step_max=100.,step_min=1.e-3,maxsteps=20000,rs=1,rp=0,rorb=0,rmax=10000.,fixed_step=-1,only_fwd=False,only_bck=False,verbose=False):
        """ This method is used to obtain field lines (same as fieldline.pro in PLUTO IDL tools).
        
        **Inputs:**
        
        var1 -- 2D Vector field in X direction\n
        var2 -- 2D Vector field in Y direction\n
        x -- 1D X array\n
        y -- 1D Y array\n
        dx -- 1D grid spacing array in X direction\n
        dy -- 1D grid spacing array in Y direction\n
        x0 -- foot point of the field line in X direction\n
        y0 -- foot point of the field line in Y direction\n
        
        **Outputs:**
        
        This routine returns a dictionary with keys - \n
        qx -- list of the field points along the 'x' direction.
        qy -- list of the field points along the 'y' direction.
        
        **Usage:**
        
        See the myfieldlines routine for the same. 
        
        
        """

        xbeg = x[0] - 0.5*dx[0]
        xend = x[-1] + 0.5*dx[-1]
        
        ybeg = y[0]  - 0.5*dy[0]
        yend = y[-1] + 0.5*dy[-1]

                
        MAX_STEPS = maxsteps

        # First interpolate to a new function for the two vectors
        new_var1 = RectBivariateSpline(x,y,var1)
        new_var2 = RectBivariateSpline(x,y,var2)
        
        ### defintions for the BS23 method
        b1  = 2.0/9.0  ; b2  = 1.0/3.0 ; b3  = 4.0/9.0 ; b4  = 0.0 
        bs1 = 7.0/24.0 ; bs2 = 1.0/4.0 ; bs3 = 1.0/3.0 ; bs4 = 1.0/8.0
        a21 = 0.5
        a31 = 0.0      ; a32 = 0.75
        a41 = 2.0/9.0  ; a42 = 1.0/3.0 ; a43 = 4.0/9.0
        tol = 1.e-6

        # FWD
        inside_domain=self.in_my_domain(x0,y0,xbeg,ybeg,xend,yend,rs,rorb,rp,rmax)
        xln_fwd = [x0]
        yln_fwd = [y0]
        k = 0
        dh = step_min
        not_attract = True
        R3_old = np.array([xln_fwd[k],yln_fwd[k]])
        while (inside_domain == True)and(not only_bck):

            dh = min([dh,step_max])
            dh = max([dh,step_min])
            # BS23 method
            R0 = np.array([xln_fwd[k],yln_fwd[k]])

            F1 = np.array([new_var1(R0[0],R0[1]),new_var2(R0[0],R0[1])]).ravel()
            R1 = R0 + dh*a21*F1

            F2 = np.array([new_var1(R1[0],R1[1]),new_var2(R1[0],R1[1])]).ravel()
            R1 = R0 + dh*(a31*F1+a32*F2)

            F3 = np.array([new_var1(R1[0],R1[1]),new_var2(R1[0],R1[1])]).ravel()
            R1 = R0 + dh*(a41*F1+a42*F2+a43*F3)

            F4 = np.array([new_var1(R1[0],R1[1]),new_var2(R1[0],R1[1])]).ravel()
            R3 = R0 + dh*(b1*F1 +b2*F2 +b3*F3 +b4*F4)

            R2 = R0 + dh*(bs1*F1+bs2*F2+bs3*F3+bs4*F4)

            if ((abs(R3_old[0]-R3[0])<1.e-8) and (abs(R3_old[1] - R3[1])<1.e-8)):
                not_attract = False
            R3_old = R3

            #print ('[FWD]',dh,R3)
            # compute error
            err = np.max(abs(R2-R3))/tol
            if ((err < 1.0) or (abs(dh)/step_min <= 1.0)):
                # accept step
                k = k + 1
                err = max([err,1.e-12])
                dhnext = 0.9*abs(dh)*err**(-0.3333)
                dhnext = min([dhnext,3.*abs(dh)])
                dh = dhnext
                xln_fwd.append(R3[0])
                yln_fwd.append(R3[1])
            else:
                dh = 0.1 * abs(dh) * err**(-0.5)

            inside_domain=self.in_my_domain(xln_fwd[k],yln_fwd[k],xbeg,ybeg,xend,yend,rs,rorb,rp,rmax)
            inside_domain = inside_domain and (k < MAX_STEPS-3) and not_attract


            #if (R0_old-R0 == 0).all():
            #    inside_domain=False
            R0_old = R0.copy()
            
            #if not not_attract:
            #    print ('I found an attractor around ',R3)


        k_fwd = k
        xln_bck = [x0]
        yln_bck = [y0]
        #BCK
        k=0
        inside_domain=self.in_my_domain(x0,y0,xbeg,ybeg,xend,yend,rs,rorb,rp,rmax) and (k < MAX_STEPS-3)
        dh = -step_min
        not_attract = True
        R3_old = np.array([xln_bck[k],yln_bck[k]])

        while (inside_domain == True)and(not only_fwd):


            dh = -min([abs(dh),step_max])
            dh = -max([abs(dh),step_min])
            # BS23 method
            R0 = np.array([xln_bck[k],yln_bck[k]])

            F1 = np.array([new_var1(R0[0],R0[1]),new_var2(R0[0],R0[1])]).ravel()
            R1 = R0 + dh*a21*F1

            F2 = np.array([new_var1(R1[0],R1[1]),new_var2(R1[0],R1[1])]).ravel()
            R1 = R0 + dh*(a31*F1+a32*F2)

            F3 = np.array([new_var1(R1[0],R1[1]),new_var2(R1[0],R1[1])]).ravel()
            R1 = R0 + dh*(a41*F1+a42*F2+a43*F3)

            F4 = np.array([new_var1(R1[0],R1[1]),new_var2(R1[0],R1[1])]).ravel()
            R3 = R0 + dh*(b1*F1 +b2*F2 +b3*F3 +b4*F4)

            R2 = R0 + dh*(bs1*F1+bs2*F2+bs3*F3+bs4*F4)

            if ((abs(R3_old[0]-R3[0])<1.e-8) and (abs(R3_old[1] - R3[1])<1.e-8)):
                not_attract = False
            R3_old = R3

            # compute error
            #print ('[BCK]',dh,R3)
            err = np.max(abs(R2-R3))/tol
            if ((err < 1.0) or (abs(dh)/step_min <= 1.0)):
                # accept step
                k = k + 1
                err = max([err,1.e-12])
                dhnext = 0.9*abs(dh)*err**(-0.3333)
                dhnext = min([dhnext,3.*abs(dh)])
                dh = -dhnext
                xln_bck.append(R3[0])
                yln_bck.append(R3[1])
            else:
                dh = -0.9 * abs(dh) * err**(-0.5)

            inside_domain=self.in_my_domain(xln_bck[k],yln_bck[k],xbeg,ybeg,xend,yend,rs,rorb,rp,rmax)
            inside_domain = inside_domain and (k < MAX_STEPS-3) and not_attract

            #if (R0_old-R0 == 0).all():
            #    inside_domain=False
            R0_old = R0.copy()

        k_bck = k

        qx = np.asarray(xln_bck[0:k_bck][::-1]+xln_fwd[0:k_fwd])
        qy = np.asarray(yln_bck[0:k_bck][::-1]+yln_fwd[0:k_fwd])
        flines={'qx':qx,'qy':qy}
        return flines

    def in_my_domain(self,x0,y0,xbeg,ybeg,xend,yend,rs,rorb,rp,rmax):
        inside_domain = x0 > xbeg and x0 < xend and y0 > ybeg and y0 < yend 
        rsph   = np.sqrt(x0**2+y0**2)
        if (rp == 0):
            rsph_p = 2.
        else:
            rsph_p = np.sqrt((x0-rorb)**2+y0**2)/rp
        inside_domain = inside_domain and (rsph > rs) and (rsph_p > 1) and (rsph < rmax)
        return inside_domain

    def ASfl3D(self,v1,v2,v3,x,y,z,dx,dy,dz,x0,y0,z0,step_max=0.1,step_min=1.e-3,maxsteps=20000,rs=1.,rp=0.1,rorb=0,fixed_step=-1,only_fwd=False,only_bck=False,rmax=1000):
        """ This method is used to obtain field lines coordinates in 3D. It is typically
        used for advanced analysis
        
        **Inputs:**
        
        v1 -- 3D Vector field in X direction\n
        v2 -- 3D Vector field in Y direction\n
        v3 -- 3D Vector field in Z direction\n
        x -- 1D X array\n
        y -- 1D Y array\n
        z -- 1D Z array\n
        dx -- 1D grid spacing array in X direction\n
        dy -- 1D grid spacing array in Y direction\n
        dz -- 1D grid spacing array in Z direction\n
        x0 -- foot point of the field line in X direction\n
        y0 -- foot point of the field line in Y direction\n
        z0 -- foot point of the field line in Z direction\n
        
        **Outputs:**
        
        This routine returns a dictionary with keys - \n
        qx -- list of the field points along the 'x' direction.
        qy -- list of the field points along the 'y' direction.
        qz -- list of the field points along the 'z' direction.
        
        """
        xbeg = x[0]  #+ 0.5*dx[0]
        xend = x[-1] #- 0.5*dx[-1]
        
        ybeg = y[0]  #+ 0.5*dy[0]
        yend = y[-1] #- 0.5*dy[-1]

        zbeg = z[0]  #+ 0.5*dz[0]
        zend = z[-1] #- 0.5*dz[-1]
                
        MAX_STEPS = maxsteps

        # First interpolate to a new function for the two vectors
        if (hasattr (scipy.interpolate, 'RegularGridInterpolator')):
            new_v1 = RegularGridInterpolator((x,y,z),v1)
            new_v2 = RegularGridInterpolator((x,y,z),v2)
            new_v3 = RegularGridInterpolator((x,y,z),v3)
        else:
            print ('I need a more recent version of scipy to trace field lines in 3D')
            return [0],[0],[0]
        
        ### defintions for the BS23 method
        b1  = 2.0/9.0  ; b2  = 1.0/3.0 ; b3  = 4.0/9.0 ; b4  = 0.0 
        bs1 = 7.0/24.0 ; bs2 = 1.0/4.0 ; bs3 = 1.0/3.0 ; bs4 = 1.0/8.0
        a21 = 0.5
        a31 = 0.0      ; a32 = 0.75
        a41 = 2.0/9.0  ; a42 = 1.0/3.0 ; a43 = 4.0/9.0
        tol = 1.e-6

        # FWD
        inside_domain=self.in_my_domain3D(x0,y0,z0,xbeg,ybeg,zbeg,xend,yend,zend,rorb,rp,rs)
        xln_fwd = [x0]
        yln_fwd = [y0]
        zln_fwd = [z0]
        k = 0
        dh = step_max
        R3_old = np.array([xln_fwd[k],yln_fwd[k]])
        R2_old = np.array([xln_fwd[k],yln_fwd[k]])
        R1_old = np.array([xln_fwd[k],yln_fwd[k]])
        while (inside_domain == True) and (not only_bck):

            not_attract = True
            dh = min([dh,step_max])
            dh = max([dh,step_min])
            # BS23 method
            R0 = np.array([xln_fwd[k],yln_fwd[k],zln_fwd[k]])

            pos = (R0[0],R0[1],R0[2])
            F1 = np.array([new_v1(pos),new_v2(pos),new_v3(pos)]).ravel()
            R1 = R0 + dh*a21*F1

            pos = (R1[0],R1[1],R1[2])
            if self.in_my_domain3D(R1[0],R1[1],R1[2],xbeg,ybeg,zbeg,xend,yend,zend,rorb,rp,rs):
                F2 = np.array([new_v1(pos),new_v2(pos),new_v3(pos)]).ravel()
            else:
                F2 = F1
            R1 = R0 + dh*(a31*F1+a32*F2)

            pos = (R1[0],R1[1],R1[2])
            if self.in_my_domain3D(R1[0],R1[1],R1[2],xbeg,ybeg,zbeg,xend,yend,zend,rorb,rp,rs):
                F3 = np.array([new_v1(pos),new_v2(pos),new_v3(pos)]).ravel()
            else:
                F3 = F2
            R1 = R0 + dh*(a41*F1+a42*F2+a43*F3)

            pos = (R1[0],R1[1],R1[2])
            if self.in_my_domain3D(R1[0],R1[1],R1[2],xbeg,ybeg,zbeg,xend,yend,zend,rorb,rp,rs):
                F4 = np.array([new_v1(pos),new_v2(pos),new_v3(pos)]).ravel()
            else:
                F4 = F3
            R3 = R0 + dh*(b1*F1 +b2*F2 +b3*F3 +b4*F4)

            R2 = R0 + dh*(bs1*F1+bs2*F2+bs3*F3+bs4*F4)

            if ((abs(R3_old[0]-R3[0])<1.e-8) and (abs(R3_old[1] - R3[1])<1.e-8)):
                not_attract = False
            if ((abs(R2_old[0]-R2[0])<1.e-8) and (abs(R2_old[1] - R2[1])<1.e-8)):
                not_attract = False
            if ((abs(R1_old[0]-R1[0])<1.e-8) and (abs(R1_old[1] - R1[1])<1.e-8)):
                not_attract = False
            R3_old = R3
            R2_old = R2
            R1_old = R1

            # compute error
            err = np.max(abs(R2-R3))/tol
            if ((err < 1.0) or (abs(dh)/step_min <= 1.0)):
                # accept step
                k = k + 1
                err = max([err,1.e-12])
                dhnext = 0.9*abs(dh)*err**(-0.3333)
                dhnext = min([dhnext,3.*abs(dh)])
                dh = dhnext
                xln_fwd.append(R3[0])
                yln_fwd.append(R3[1])
                zln_fwd.append(R3[2])
            else:
                dh = 0.9 * abs(dh) * err**(-0.5)

            inside_domain=self.in_my_domain3D(xln_fwd[k],yln_fwd[k],zln_fwd[k],xbeg,ybeg,zbeg,xend,yend,zend,rorb,rp,rs)
            inside_domain = inside_domain and (k < MAX_STEPS-3) and not_attract

        if (k >= MAX_STEPS -3):
            print("** Warning, I reached MAX_STEPS in ASfl3D (fwd) **")
        elif (not only_bck):
            if self.in_my_domain3D(xln_fwd[k],yln_fwd[k],zln_fwd[k],xbeg,ybeg,zbeg,xend,yend,zend,rorb,rp,rs):
                print("I should not have stopped, k=%i, x = [%.2f, %.2f, %.2f]" % (k,xln_fwd[k],yln_fwd[k],zln_fwd[k])) 

        k_fwd = k
        xln_bck = [x0]
        yln_bck = [y0]
        zln_bck = [z0]
        #BCK
        k=0
        inside_domain=self.in_my_domain3D(x0,y0,z0,xbeg,ybeg,zbeg,xend,yend,zend,rorb,rp,rs) and (k < MAX_STEPS-3)
        dh = -step_max
        while (inside_domain == True) and (not only_fwd):

            dh = -min([abs(dh),step_max])
            dh = -max([abs(dh),step_min])
            # BS23 method
            R0 = np.array([xln_bck[k],yln_bck[k],zln_bck[k]])

            pos = (R0[0],R0[1],R0[2])
            F1 = np.array([new_v1(pos),new_v2(pos),new_v3(pos)]).ravel()
            R1 = R0 + dh*a21*F1

            pos = (R1[0],R1[1],R1[2])
            if self.in_my_domain3D(R1[0],R1[1],R1[2],xbeg,ybeg,zbeg,xend,yend,zend,rorb,rp,rs):
                F2 = np.array([new_v1(pos),new_v2(pos),new_v3(pos)]).ravel()
            else:
                F2 = F1
            R1 = R0 + dh*(a31*F1+a32*F2)

            pos = (R1[0],R1[1],R1[2])
            if self.in_my_domain3D(R1[0],R1[1],R1[2],xbeg,ybeg,zbeg,xend,yend,zend,rorb,rp,rs):
                F3 = np.array([new_v1(pos),new_v2(pos),new_v3(pos)]).ravel()
            else:
                F3 = F2
            R1 = R0 + dh*(a41*F1+a42*F2+a43*F3)

            pos = (R1[0],R1[1],R1[2])
            # print (pos,np.sqrt(R1[0]**2+R1[1]**2+R1[2]**2),xbeg,xend)
            if self.in_my_domain3D(R1[0],R1[1],R1[2],xbeg,ybeg,zbeg,xend,yend,zend,rorb,rp,rs):
                F4 = np.array([new_v1(pos),new_v2(pos),new_v3(pos)]).ravel()
            else:
                F4 = F3
            R3 = R0 + dh*(b1*F1 +b2*F2 +b3*F3 +b4*F4)

            R2 = R0 + dh*(bs1*F1+bs2*F2+bs3*F3+bs4*F4)

            # compute error
            err = np.max(abs(R2-R3))/tol
            if ((err < 1.0) or (abs(dh)/step_min <= 1.0)):
                # accept step
                k = k + 1
                err = max([err,1.e-12])
                dhnext = 0.9*abs(dh)*err**(-0.3333)
                dhnext = min([dhnext,3.*abs(dh)])
                dh = -dhnext
                xln_bck.append(R3[0])
                yln_bck.append(R3[1])
                zln_bck.append(R3[2])
            else:
                dh = -0.9 * abs(dh) * err**(-0.5)

            inside_domain=self.in_my_domain3D(xln_bck[k],yln_bck[k],zln_bck[k],xbeg,ybeg,zbeg,xend,yend,zend,rorb,rp,rs)
            inside_domain = inside_domain and (k < MAX_STEPS-3)

        k_bck = k
        if (k >= MAX_STEPS -3):
            print("** Warning, I reached MAX_STEPS in ASfl3D (bck) **")

        qx = np.asarray(xln_bck[0:k_bck][::-1]+xln_fwd[0:k_fwd])
        qy = np.asarray(yln_bck[0:k_bck][::-1]+yln_fwd[0:k_fwd])
        qz = np.asarray(zln_bck[0:k_bck][::-1]+zln_fwd[0:k_fwd])
        return [qx,qy,qz]

    def in_my_domain3D(self,x0,y0,z0,xbeg,ybeg,zbeg,xend,yend,zend,rorb,rp,rs):
        inside_domain = x0 > xbeg and x0 < xend and y0 > ybeg and y0 < yend and z0 > zbeg and z0 < zend
        rsph   = np.sqrt(x0**2+y0**2+z0**2)
        rsph_p = np.sqrt((x0-rorb)**2+y0**2+z0**2)/rp
        inside_domain = inside_domain and (rsph > rs) and (rsph_p > 1)
        return inside_domain

    def ASfieldlinesSph(self,bx1,bx2,x1,x2,dx1,dx2,x0arr,y0arr,step_max=100.,step_min=1.e-3,fixed_step=-1,maxsteps=20000,rs=1,rp=0,rorb=0,rmax=1000,color='k',ls='-',only_fwd=False,only_bck=False,verbose=False):

        # Interpolate
        fb1 = RectBivariateSpline(x1,x2,bx1)
        fb2 = RectBivariateSpline(x1,x2,bx2)
        #xbeg=np.sin(x2[0])*(x1.max())
        xbeg=0.
        #x=np.arange(xbeg,x1[-1],0.1)
        #y=np.arange(-x1[-1],x1[-1],0.1)
        x = np.concatenate([np.arange(xbeg,3.,0.1),np.geomspace(3.,x1[-1],200)])
        if xbeg == 0:
            y = np.concatenate([-x[1:][::-1],x[1:]])
        else:
            y = np.concatenate([-x[::-1],x])
        dx = np.diff(x);dx=np.concatenate([dx,dx[-2:-1]])#dx1[0]*np.ones(len(x))
        dy = np.diff(y);dy=np.concatenate([dy,dy[-2:-1]])#dx1[0]*np.ones(len(y))
        bx = np.zeros((len(x),len(y)))
        by = np.zeros((len(x),len(y)))
        for i in range(len(x)):
            for j in range(len(y)):
                rcart  = np.sqrt(x[i]**2 + y[j]**2)
                thcart = np.arccos(y[j]/rcart)
                bx[i,j] = fb1(rcart,thcart)
                by[i,j] = fb2(rcart,thcart)

        # Call cartesian version
        return self.ASfieldlines(bx,by,x,y,dx,dy,x0arr,y0arr,step_max=step_max,step_min=step_min,fixed_step=fixed_step,maxsteps=maxsteps,rs=rs,rmax=rmax,\
                                     rp=rp,rorb=rorb,color=color,ls=ls,only_fwd=only_fwd,only_bck=only_bck,verbose=verbose)

    def ASfl3DSph(self,bx1,bx2,bx3,x1,x2,x3,dx1,dx2,dx3,x0arr,y0arr,z0arr,step_max=100.,step_min=1.e-3,fixed_step=-1,maxsteps=20000,rs=1,rp=0,rorb=0,rmax=1000,color='k',ls='-',only_fwd=False,only_bck=False,verbose=False):

        # Interpolate
        print("Interpolating... ",end="",flush=True)
        fbr  = RegularGridInterpolator((x1,x2,x3),bx1,bounds_error=False,fill_value=0.)
        fbth = RegularGridInterpolator((x1,x2,x3),bx2,bounds_error=False,fill_value=0.)
        fbph = RegularGridInterpolator((x1,x2,x3),bx3,bounds_error=False,fill_value=0.)
        x = np.arange(-x1[-1],x1[-1],dx1.min())
        y = np.arange(-x1[-1],x1[-1],dx1.min())
        z = np.arange(-x1[-1],x1[-1],dx1.min())
        dx = dx1[0]*np.ones(len(x))
        dy = dx1[0]*np.ones(len(y))
        dz = dx1[0]*np.ones(len(z))
        bx = np.zeros((len(x),len(y),len(z)))
        by = np.zeros((len(x),len(y),len(z)))
        bz = np.zeros((len(x),len(y),len(z)))
        for i in range(len(x)):
            for j in range(len(y)):
                for k in range(len(z)):
                    rcart  = np.sqrt(x[i]**2 + y[j]**2+ z[k]**2)
                    thcart = np.arccos(z[k]/rcart)
                    phcart = np.arctan2(y[j],x[i])
                    bx[i,j,k] =  fbr((rcart,thcart,phcart))
                    by[i,j,k] = fbth((rcart,thcart,phcart))
                    bz[i,j,k] = fbph((rcart,thcart,phcart))
        
        # Call cartesian version
        print("FL... ",end="",flush=True)
        return self.ASfl3D(bx,by,bz,x,y,z,dx,dy,dz,x0arr,y0arr,z0arr,step_max=step_max,step_min=step_min,fixed_step=fixed_step,maxsteps=maxsteps,rs=rs,rmax=rmax,\
                           rp=rp,rorb=rorb,only_fwd=only_fwd,only_bck=only_bck)

    def ASfieldlines(self,bx1,bx2,x1,x2,dx1,dx2,x0arr,y0arr,step_max=100.,step_min=1.e-3,fixed_step=-1,maxsteps=20000,rs=1,rp=0,rorb=0,rmax=10000.,color='k',ls='-',only_fwd=False,only_bck=False,verbose=False):
        """ This method overplots the magnetic field lines at the footpoints given by (x0arr[i],y0arr[i]).

        **Inputs:**
        
          Data -- pyPLUTO.pload object\n
          x0arr -- array of x co-ordinates of the footpoints\n
          y0arr -- array of y co-ordinates of the footpoints\n
          stream -- keyword for two different ways of calculating the field lines.\n
            True -- plots contours of rAphi (needs to store vector potential)\n
            False -- plots the fieldlines obtained from the field_line routine. (Default option)\n

        *Optional Keywords:*
        
          colors -- A list of matplotlib colors to represent the lines. The length of this list should be same as that of x0arr.\n
          lw -- Integer value that determines the linewidth of each line.\n
          ls -- Determines the linestyle of each line.\n

        **Usage:**

        Assume that the magnetic field is a given as **B** = B0$\hat{y}$. Then to show this field lines we have to define the x and y arrays of field foot points.

        ``x0arr = linspace(0.0,10.0,20)``\n
        ``y0arr = linspace(0.0,0.0,20)``\n
        ``import pyPLUTO as pp``\n
        ``D = pp.pload(45)``\n
        ``I = pp.Image()``\n
        ``I.myfieldlines(D,x0arr,y0arr,colors='k',ls='--',lw=1.0)``

           """
        if len(x0arr) != len(y0arr) : print ("Input Arrays should have same size")
        QxList=[]
        QyList=[]
        for i in range(len(x0arr)):
            tmp = self.ASfield_line(bx1,bx2,x1,x2,dx1,dx2,x0arr[i],y0arr[i],\
                                        step_max=step_max,step_min=step_min,fixed_step=fixed_step,maxsteps=maxsteps,\
                                        rs=rs,rorb=rorb,rp=rp,rmax=rmax,only_fwd=only_fwd,only_bck=only_bck,\
                                        verbose=verbose)
            QxList.append(tmp.get('qx'))
            QyList.append(tmp.get('qy'))

        return [QxList,QyList]

    def myfieldlines(self,Data,x0arr,y0arr,stream=False,rs=1,rp=0.1,rorb=0,**kwargs):
        """ This method overplots the magnetic field lines at the footpoints given by (x0arr[i],y0arr[i]).

        **Inputs:**
        
          Data -- pyPLUTO.pload object\n
          x0arr -- array of x co-ordinates of the footpoints\n
          y0arr -- array of y co-ordinates of the footpoints\n
          stream -- keyword for two different ways of calculating the field lines.\n
            True -- plots contours of rAphi (needs to store vector potential)\n
            False -- plots the fieldlines obtained from the field_line routine. (Default option)\n

        *Optional Keywords:*
        
          colors -- A list of matplotlib colors to represent the lines. The length of this list should be same as that of x0arr.\n
          lw -- Integer value that determines the linewidth of each line.\n
          ls -- Determines the linestyle of each line.\n

        **Usage:**

        Assume that the magnetic field is a given as **B** = B0$\hat{y}$. Then to show this field lines we have to define the x and y arrays of field foot points.

        ``x0arr = linspace(0.0,10.0,20)``\n
        ``y0arr = linspace(0.0,0.0,20)``\n
        ``import pyPLUTO as pp``\n
        ``D = pp.pload(45)``\n
        ``I = pp.Image()``\n
        ``I.myfieldlines(D,x0arr,y0arr,colors='k',ls='--',lw=1.0)``

           """
           
        if len(x0arr) != len(y0arr) : print ("Input Arrays should have same size")
        QxList=[]
        QyList=[]
        StreamFunction = []
        levels =[]
        if stream == True:
            X, Y = np.meshgrid(Data.x1,Data.x2.T)
            StreamFunction = X*(Data.Ax3.T)
            for i in range(len(x0arr)):
                nx = np.abs(X[0,:]-x0arr[i]).argmin()
                ny = np.abs(X[:,0]-y0arr[i]).argmin()
                levels.append(X[ny,nx]*Data.Ax3.T[ny,nx])
            
            contour(X,Y,StreamFunction,levels,colors=kwargs.get('colors'),linewidths=kwargs.get('lw',1),linestyles=kwargs.get('ls','solid'))
        else:
            for i in range(len(x0arr)):
                QxList.append(self.field_line(Data.bx1,Data.bx2,Data.x1,Data.x2,Data.dx1,Data.dx1,x0arr[i],y0arr[i],rs=rs,rorb=rorb,rp=rp).get('qx'))
                QyList.append(self.field_line(Data.bx1,Data.bx2,Data.x1,Data.x2,Data.dx1,Data.dx1,x0arr[i],y0arr[i],rs=rs,rorb=rorb,rp=rp).get('qy'))
                plot(QxList[i],QyList[i],color=kwargs.get('colors'))
            axis([min(Data.x1),max(Data.x1),min(Data.x2),max(Data.x2)])


    def getSphData(self,Data,w_dir=None,datatype=None,**kwargs):
        """This method transforms the vector and scalar  fields from Spherical co-ordinates to Cylindrical.
        
        **Inputs**:
        
          Data -- pyPLUTO.pload object\n
          w_dir -- /path/to/the/working/directory/\n
          datatype -- If the data is of 'float' type then datatype = 'float' else by default the datatype is set to 'double'.
          
        *Optional Keywords*:
        
          rphi -- [Default] is set to False implies that the r-theta plane is transformed. If set True then the r-phi plane is transformed.\n
              x2cut -- Applicable for 3D data and it determines the co-ordinate of the x2 plane while r-phi is set to True.\n
          x3cut -- Applicable for 3D data and it determines the co-ordinate of the x3 plane while r-phi is set to False.\n

          
        """
 
        Tool = Tools()
        key_value_pairs = []
        if w_dir is None: w_dir = curdir()            
        allvars = Data.get_varinfo(datatype).get('allvars')
            
            
        if kwargs.get('rphi',False)==True:
            R,TH = np.meshgrid(Data.x1,Data.x3)
            if Data.n3 != 1:
                for variable in allvars:
                    key_value_pairs.append([variable,getattr(Data,variable)[:,kwargs.get('x2cut',0),:].T])
                SphData = dict(key_value_pairs)
                if ('bx1' in allvars) or ('bx2' in allvars):
                    (SphData['b1c'],SphData['b3c']) = Tool.RTh2Cyl(R,TH,SphData.get('bx1'),SphData.get('bx3'))
                    allvars.append('b1c')
                    allvars.append('b3c')
                if ('vx1' in allvars) or ('vx2' in allvars):
                    (SphData['v1c'],SphData['v3c']) = Tool.RTh2Cyl(R,TH,SphData.get('vx1'),SphData.get('vx3'))
                    allvars.append('v1c')
                    allvars.append('v3c')
            else:
                print ("No x3 plane for 2D data")
        else:
            R,TH = np.meshgrid(Data.x1,Data.x2)
            if Data.n3 != 1:
                for variable in allvars:
                    key_value_pairs.append([variable,getattr(Data,variable)[:,:,kwargs.get('x3cut',0)].T])
                SphData = dict(key_value_pairs)
                if ('bx1' in allvars) or ('bx2' in allvars):
                    (SphData['b1c'],SphData['b2c']) = Tool.RTh2Cyl(R,TH,SphData.get('bx1'),SphData.get('bx2'))
                    allvars.append('b1c')
                    allvars.append('b2c')
                if ('vx1' in allvars) or ('vx2' in allvars):
                    (SphData['v1c'],SphData['v2c']) = Tool.RTh2Cyl(R,TH,SphData.get('vx1'),SphData.get('vx2'))
                    allvars.append('v1c')
                    allvars.append('v2c')
            else:
                for variable in allvars:
                    key_value_pairs.append([variable,getattr(Data,variable)[:,:].T])
                SphData = dict(key_value_pairs)
                if ('bx1' in allvars) or ('bx2' in allvars):
                    (SphData['b1c'],SphData['b2c']) = Tool.RTh2Cyl(R,TH,SphData.get('bx1'),SphData.get('bx2'))
                    allvars.append('b1c')
                    allvars.append('b2c')
                if ('vx1' in allvars) or ('vx2' in allvars):
                    (SphData['v1c'],SphData['v2c']) = Tool.RTh2Cyl(R,TH,SphData.get('vx1'),SphData.get('vx2'))
                    allvars.append('v1c')
                    allvars.append('v2c')
        
        for variable in allvars:
            if kwargs.get('rphi',False)==True:
                R,Z,SphData[variable]= Tool.sph2cyl(Data,SphData.get(variable),rphi=True,theta0=Data.x2[kwargs.get('x2cut',0)])
            else:
                if Data.n3 != 1:
                    R,Z,SphData[variable] = Tool.sph2cyl(Data,SphData.get(variable),rphi=False)
                else:
                    R,Z,SphData[variable] = Tool.sph2cyl(Data,SphData.get(variable),rphi=False)

        return R,Z,SphData

    def pltSphData(self,Data,w_dir=None,datatype=None,**kwargs):
        """This method plots the transformed data obtained from getSphData using the matplotlib's imshow
        
        **Inputs:**

        Data -- pyPLUTO.pload object\n
        w_dir -- /path/to/the/working/directory/\n
        datatype -- If the data is of 'float' type then datatype = 'float' else by default the datatype is set to 'double'.

        *Required Keywords*:
            
        plvar -- A string which represents the plot variable.\n

        *Optional Keywords*:
        rphi -- [Default = False - for plotting in r-theta plane] Set it True for plotting the variable in r-phi plane. 
        
        """
              
        if w_dir is None: w_dir=curdir()
        R,Z,SphData = self.getSphData(Data,w_dir=w_dir,datatype=datatype,**kwargs)
    
        extent=(np.min(R.flat),max(R.flat),np.min(Z.flat),max(Z.flat))
        dRR=max(R.flat)-np.min(R.flat)
        dZZ=max(Z.flat)-np.min(Z.flat)


        isnotnan=-np.isnan(SphData[kwargs.get('plvar')])
        maxPl=max(SphData[kwargs.get('plvar')][isnotnan].flat)
        minPl=np.min(SphData[kwargs.get('plvar')][isnotnan].flat)
        normrange=False
        if minPl<0:
            normrange=True
        if maxPl>-minPl:
            minPl=-maxPl
        else:
            maxPl=-minPl      
        if (normrange and kwargs.get('plvar')!='rho' and kwargs.get('plvar')!='prs'):
            SphData[kwargs.get('plvar')][-1][-1]=maxPl
            SphData[kwargs.get('plvar')][-1][-2]=minPl

        if (kwargs.get('logvar') == True):
            SphData[kwargs.get('plvar')] = np.log10(SphData[kwargs.get('plvar')])

        im1= imshow(SphData[kwargs.get('plvar')], aspect='equal', origin='lower', cmap=cm.jet,extent=extent, interpolation='nearest')

    def vecfield(self,x1,x2,v1,v2,dd=10,color='k'):
        """ Wrapper for pluto around streamplot and/or quiver
        TO BE DONE
        """
        Q=quiver(x1[::dd,::dd],x2[::dd,::dd],v1[::dd,::dd],v2[::dd,::dd],color=color,\
                 pivot='mid')
        #speed = np.sqrt(v1*v1+v2*v2)
        #lw = 5*speed/speed.max()
        #streamplot(x1,x2,v1,v2,linewidth=lw,density=density,color=color)
    
