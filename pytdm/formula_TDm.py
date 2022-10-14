##
import numpy as np 
from tqdm import tqdm 
from scipy.special import ellipk,ellipe
import pyPLUTO as pp
import math
import os
import matplotlib.pyplot as plt 
import imp
from . import utils
##

def compute_Is(B_p,R,a):
    ''' Compute the shafranov intensity according to Titov et al. 2014
    (Equation 14)

    Inputs
    ======
    B_p : ambiant magnetic field 
    R : major radius of the torus
    a : minor radius of the torus

    Output
    ======
    Is : Shafranov intensity
    '''

    # eq. 7 of Titov 2014
    Is = - (4 * np.pi * R * B_p) / (np.log(8 * R/a) - 3/2 + 1/2)

    return Is

def rotation_matrix(theta_0,phi_0):

    # https://en.wikipedia.org/wiki/Spherical_coordinate_system
    rot = np.array((
        (  np.cos(theta_0) * np.cos(phi_0) , np.cos(theta_0) * np.sin(phi_0) , - np.sin(theta_0)),
        (- np.sin(phi_0)                   , np.cos(phi_0)                   , 0                ),
        (  np.sin(theta_0) * np.cos(phi_0) , np.sin(theta_0) * np.sin(phi_0) ,   np.cos(theta_0))
        ))

    return rot


def rotation_matrix_axes(alpha,axe):
    ''' Compute the rotation of an angle alpha of a 3D vector around the axe
    'axe' which can be x,y,z
    '''

    rotation_matrix = dict({
        'x': np.array(( ( 1 , 0            , 0            ),
                        ( 0 , np.cos(alpha),-np.sin(alpha)),
                        ( 0 , np.sin(alpha), np.cos(alpha))  )),

        'y': np.array(( ( np.cos(alpha)  , 0  , np.sin(alpha)),
                        ( 0              , 1  , 0           ),
                        ( -np.sin(alpha) , 0  , np.cos(alpha))  )),

        'z': np.array(( ( np.cos(alpha)  , -np.sin(alpha) , 0  ),
                        ( np.sin(alpha)   , np.cos(alpha)  , 0 ),
                        ( 0               ,             0  , 1 )  ))  })
    
    return rotation_matrix[axe]

def base_rotation(X,Y,Z,alpha_0,theta_0,phi_0):
    ''' Perform the rotation of the base but in the inverse order with respect
    to the vector
    
    Inputs:
    =======
    
    X,Y,Z (3D meshgrid) coordinates in the local cartesian plane
    alpha_0,theta_0 and phi_0 are the angle of rotation to apply 


    Outputs:
    ========

    new_x,new_y,new_y new axes with the "derotation" applied


    '''

    # X,Y,Z = np.meshgrid(x,y,z,indexing = 'ij')
    
    rot = rotation_matrix(theta_0,phi_0)
    
    R = \
            rotation_matrix_axes(-alpha_0,'z') @ \
            rot

                                       
    new_x = R[0,0] * X + R[0,1] * Y +  R[0,2] * Z
    new_y = R[1,0] * X + R[1,1] * Y +  R[1,2] * Z
    new_z = R[2,0] * X + R[2,1] * Y +  R[2,2] * Z
    
    #for i,j,k in zip(range(len(x)),range(len(y)),range(len(z))):

        #vec = np.array([x[i],y[j],z[k]]) 

        #new_vec = rotation_matrix(-theta,'z') @ \
                    #rotation_matrix(-phi,'y') @   \
                    #rotation_matrix(-alpha,'z') @ \
                    #vec
        
        #new_x[i] = new_vec[0]
        #new_y[j] = new_vec[1]
        #new_z[k] = new_vec[2]

    return new_x,new_y,new_z



def cube_rotation(x_comp,y_comp,z_comp,alpha_0,theta_0,phi_0):
    '''
    Performs the (de)rotation of the components using 3 cubes for each components
    '''
    # Rotation matrix
    rot = rotation_matrix(theta_0,phi_0)


    R = \
            rotation_matrix_axes(-alpha_0,'z') @ \
            rot

    R = R.T
                    
    new_x_comp = R[0,0] * x_comp + R[0,1] * y_comp +  R[0,2] *z_comp 
    new_y_comp = R[1,0] * x_comp + R[1,1] * y_comp +  R[1,2] *z_comp 
    new_z_comp = R[2,0] * x_comp + R[2,1] * y_comp +  R[2,2] *z_comp 


    return new_x_comp,new_y_comp,new_z_comp

def cart_TDm_setup(x,y,z,d,a,R,I,zeta,Delta,case):
    ''' Set up of the flux-rope using TDm and its local coordinates

    Inputs:
    =======
    x,y,z are in global cartesian system
    F = 3 / ( 5 * np.sqrt(2)) * I * a



    Outputs
    =======
    A_tot : vector potential
    '''
    
    mu = 1

    # delta eq.8
    delta = Delta / a
    
    eps = a/R

    # # mu (Set as mu_0 for now)
    # mu = 4 * np.pi * 10**-7

    # r_perp eq.16
    r_p = np.sqrt(y**2 + (z + d)**2) 

    # rho eq.15
    rho = np.sqrt((x**2 + (r_p - R)**2))

    # mask_high_rho = (rho > 40)
    phi = np.arccos((R - r_p)/rho)


    # ksi eq.24
    ksi = (rho - a) / (delta * a)

    def logcosh(x):
        # Tricks to avoid overflow while computing the TDm structure.
        # Credits
        # : https://stackoverflow.com/questions/57785222/avoiding-overflow-in-logcoshx

        # s always has real part >= 0
        s = np.sign(x) * x
        p = np.exp(-2 * s)
        return s + np.log1p(p) - np.log(2)

    def h(ksi):
        return (ksi + logcosh(ksi) + np.log(2))/2

    def hp(ksi):
        return (1 + np.sinh(ksi) / np.cosh(ksi))/2


    # f(ksi) eq 61
    def f(ksi):
        f0 = -0.406982224701535
        M1 = -1.5464309982239
        M2 = -0.249947772314288
        return (h(ksi) + f0 * np.exp( M1 * h(ksi) + M2 * h(ksi)**2 ))

    # eq. 56
    def fp(ksi):
        # eq 55
        def Theta(ksi):
            return np.pi/4 *( 1 + np.tanh(ksi))

        return np.sin(Theta(ksi))


    # g(ksi) eq 62
    def g(ksi):
        f0 = -0.406982224701535
        M3 = -2.38261647628
        return (h(ksi) - f0 * np.exp(M3*h(ksi)))
    
    def gp(ksi):
        f0 = -0.406982224701535
        M3 = -2.38261647628
        return hp(ksi) - M3* hp(ksi) * f0 * np.exp(M3*h(ksi))

    # eq. 59
    def gp_bis(ksi):
        # eq 55
        def Theta(ksi):
            return np.pi/4 *( 1 + np.tanh(ksi))

        return 1 - np.cos(Theta(ksi))

    #eq. A1
    def h_bis(ksi,delta,h):
        return h(-1/delta) + (h(ksi) - h(-1/delta))*np.tanh((ksi+(1/delta))/3)

    #eq. A1 for f
    def f_bis(ksi,delta,f):
        return f(-1/delta) + (f(ksi) - f(-1/delta))*np.tanh((ksi+(1/delta))/3)

    #eq. A1 for g
    def g_bis(ksi,delta,g):
        return g(-1/delta) + (g(ksi) - g(-1/delta))*np.tanh((ksi+(1/delta))/3)

    # rho* eq. 24
    rho_star = a * ( 1 + delta * f_bis(ksi,delta,f))

    # F = 3 / ( 5 * np.sqrt(2)) * I * a

    # rho solid star eq.40
    rho_solid = a * ( 1 + delta * g_bis(ksi,delta,g))


    # k* eq. 23
    k_star = np.sqrt( (r_p * R) / ( r_p * R + rho_star**2 / 4))

    # k solid star eq. 39
    k_solid = np.sqrt( (r_p * R) / ( r_p * R + rho_solid**2 / 4))


    #k eq. 14
    k = np.sqrt( (r_p* R) / (r_p * R + rho**2 / 4))

    # Flux eq. 69
    # F = - 3 / ( 5 * np.sqrt(2)) * I * a
    F = - (1 / 2) * mu * I * a

    # ===============================
    # Function A and its derivative
    # ===============================

    # eq. 13
    def A(k):

        return ((2 - k**2) * ellipk(k**2) - 2 * ellipe(k**2)) / k  

    # eq. 41
    def Ap(k):

        return ((2 - k**2)/(k**2 * (1 - k**2)) * ellipe(k**2) - 2/k**2 * ellipk(k**2))  

    # eq. 48
    def App(k):

        T1 = (5 * k**2 - 4)/(k**3 * (k**2 - 1))
        T2 = (k**4 - 7 * k**2 + 4)/(k**3 *(k**2 - 1)**2)

        return  T1 * ellipk(k**2) -  T2 * ellipe(k**2)


    # Projections on cartersian axes

    proj_x = dict({'x':1,'y':0,'z':0}) 

    proj_r_p = dict({'x':0,'y':y/r_p,'z':(z+d)/r_p}) 

    proj_theta = dict({'x':0,'y':-(z+d)/r_p,'z':y/r_p})

    A_tot = dict({})
    BF = dict({})
    BI = dict({})
    B_tot = dict({})
    
    if case == 'first':
        
        AI_out = dict({'x':np.array([]),'y':np.array([]),'z':np.array([])})
        AF_out = dict({'x':np.array([]),'y':np.array([]),'z':np.array([])})

        for comp in ['x','y','z']:
            
            T1 = (k - k_solid) * (Ap(k_solid) + \
                    a**2 * k_solid**3 /( 4 * R * r_p) * App(k_solid))

            T2 = k**3 * (R **2 - r_p**2 + x**2) * Ap(k_solid)/ (4 * R * r_p)

            T3 = x * k**3 / (2*R) * Ap(k_solid)

            # eq. 47
            AF = zeta * F / (4 * np.pi * r_p) * np.sqrt(R/r_p) * \
                    (  (A(k_solid) + T1 + T2) * proj_x[comp] + T3 * proj_r_p[comp])

            # eq. 22
            AI =  zeta * I / ( 2 * np.pi) * np.sqrt(R / r_p) * \
                    A(k_star) * proj_theta[comp]
            
            A_tot[comp] = AI + AF


    elif case == 'second':
        # Intermediary steps

        # Redefining rho_star and k_star using h sewing function (paragraphs after eq. 70 and
        # before eq. 74)
        rho_star = a * (1 + delta * h(ksi))

        k_star = np.sqrt( (r_p * R) / ( r_p * R + rho_star**2 / 4))


        Apk_kstar = (3 + 4 * Ap(k_star) * (k-k_star))

        TF1 = (a**2 * k_solid**3)/(4 * R * r_p) * Ap(k_solid)

        TF2 = (np.sign(Apk_kstar) * np.abs(Apk_kstar)**(5/2))/ (30 * np.sqrt(3))

        TF3_1 = (np.sign(Apk_kstar) * np.abs(Apk_kstar)**(3/2))/(12 * np.sqrt(3) * R * r_p)

        TF3_2 =(k**3 * (R**2 - r_p**2 + x**2) - (a**2 * k_solid**3)) * Ap(k_solid) \
                + a**2 * k_solid**3 * App(k_solid) * (k-k_solid)

        TF3 = TF3_1 * TF3_2

        TF4 = np.sign(Apk_kstar) * np.abs(Apk_kstar)**(3/2) * (x * k**3 * Ap(k_star)) / (6 * np.sqrt(3) * R)

        #===============
        # Final formula
        #===============

        A_tot = dict({'x':np.array([]),'y':np.array([]),'z':np.array([])})
       
        for comp in ['x','y','z']:

            # eq. 70
            AF = zeta * F/(4 * np.pi * r_p) * np.sqrt(R/r_p) * \
                    ( (A(k_solid) + TF1 + TF2  - 3/10 + TF3) * proj_x[comp] 
                            + TF4 * proj_r_p[comp])
            


            # eq. 74
            AI =  zeta * I / (2 * np.pi) * np.sqrt(R / r_p) * \
                    (A(k_star) + Ap(k_star) * (k-k_star) + App(k_star)/2 * (k-k_star)**2) \
                    * proj_theta[comp]

            A_tot[comp] = AI + AF
    else:
        raise ValueError('Wrong name of case')

    return A_tot

def to_spheric(A,rr,tt,pph):
    ''' Transform each components of the vector potential in spherical system
    '''

    A_out = dict({}) 
    A_out['r'] = np.sin(tt)*np.cos(pph) * A['x'] \
            + np.sin(tt) * np.sin(pph) * A['y'] \
            + np.cos(tt) * A['z'] 
    A_out['theta'] = np.cos(tt)*np.cos(pph) * A['x'] \
            + np.cos(tt) * np.sin(pph) * A['y'] \
            -np.sin(tt) * A['z']
    A_out['phi'] = -np.sin(pph) * A['x'] \
            + np.cos(pph) * A['y']

    return A_out

def TDm_setup(x1,x2,x3,alpha_0,theta_0,phi_0,d,a,R,zeta,B_p,Delta,case,geometry):

    ''' Set up in spherical coordinates of the flux-rope 
    using TDm and its local coordinates.
    

    The TDm is built on a cartesian geometry and, if needed, converted in
    spherical geometry.

    Inputs:
    =======
    r : radial distance
    theta : polar angle 
    phi : azimuthal angle
    d : depth at which the torus is buried 
    a : minor radius of the torus
    R : major radius of the torus
    B_p : ambiant magnetic field
    zeta : Modulation of Shafranov intensity
    I : Intensity of the torus
    Delta : 
    case : TDm case ('first' or 'second')
    geometry : output geometry of B ('spherical' or 'cartesian')



    Outputs
    =======
    B : magnetic field in "geometry" coordinates
    '''

    R_star = 1

    if geometry == 'cartesian':


        # Setting the grid for cartesian setup
        xx,yy,zz = np.meshgrid(x1,x2,x3,indexing='ij')

        # Computing the Shafranov current
        Is = compute_Is(B_p,R,a)

        # Modulation of the relative intensity of the FR compared to the
        # shafranov (ie stable) FR
        # /!\ The multiplication by zeta is inside the  cart TDm now
        I = Is

        # Computing the potential vector A
        A = cart_TDm_setup(xx,yy,zz,d,a,R,I,zeta,Delta,case)

        B_out = utils.ComputeRot(x1,x2,x3,A['x'],A['y'],A['z'])

    elif geometry == 'spherical':
        
        # Spherical coordinates
        r = x1
        theta = x2
        phi = x3

        # Grid of geometrical coordinates
        rr,tt,pph = np.meshgrid(r,theta,phi,indexing = 'ij')
        
        # Building components in cartesian
        xx = rr * np.sin(tt)*np.cos(pph)
        yy = rr * np.sin(tt)*np.sin(pph)
        zz = rr * np.cos(tt)

        # Getting the position of the FR for the evaluation of the ambient
        # magnetic field
        pos_tube_r = np.argmin(np.abs( r - (R - d + R_star)))
        pos_tube_t = np.argmin(np.abs(theta - theta_0))
        pos_tube_p = np.argmin(np.abs(phi - phi_0))
    
        # Computing Shafranov current
        Is = compute_Is(B_p,R,a)

        # Modulation of the relative intensity of the FR compared to the
        # shafranov (ie stable) FR
        # /!\ The multiplication by zeta is inside the  cart TDm now
        I = Is
        
        # Spherical to cartesian transformation
        new_xx = xx - R_star * np.sin(theta_0) * np.cos(phi_0)
        new_yy = yy - R_star * np.sin(theta_0) * np.sin(phi_0)
        new_zz = zz - R_star * np.cos(theta_0)

        # Rotation of the cartesian base depending on the position of the FR
        new_x,new_y,new_z = base_rotation(new_xx,new_yy,new_zz,alpha_0,theta_0,phi_0)
        

        # Potential vector A in cartesian
        A = cart_TDm_setup(new_x,new_y,new_z,d,a,R,I,zeta,Delta,case)

        x_comp = A['x'] 
        y_comp = A['y'] 
        z_comp = A['z'] 

        # A in cartesian with the good orientation on the sphere
        new_A = dict({})
        new_A['x'],new_A['y'],new_A['z'] = \
                cube_rotation(x_comp,y_comp,z_comp,alpha_0,theta_0,phi_0)

        # Getting A in spherical
        A_sph = to_spheric(new_A,rr,tt,pph)
        
        # Getting the magnetic field
        B_sph = utils.ComputeRot(r,theta,phi,A_sph['r'],A_sph['theta'],A_sph['phi'],geom='spherical')

        B_out = B_sph

    return B_out

##
