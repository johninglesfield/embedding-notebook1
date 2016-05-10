# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 16:04:49 2016
@author: johninglesfield
Functions needed for surface density of states in Chulkov model.
"""
import numpy as np
from numpy.lib.scimath import sqrt,log,arccos
from scipy.integrate import quad, odeint
from scipy.linalg import solve

def pot_param(param):
    """
    Calculates potential parameters in Chulkov model.
    """
    global alat,a10,a20,a1,a2,a3,z1,z_im,g0,alph,beta,lamb
    alat,a10,a1,a2,beta=param
    h_to_ev=27.2116
    a10=a10/h_to_ev
    a1=a1/h_to_ev
    a2=a2/h_to_ev
    g0=2.0*np.pi/alat
    a20=a2-a1-a10
    z1=5.0*np.pi/(4.0*beta)
    a3=-a20+a2*np.cos(5.0*np.pi/4.0)
    alph=beta*a2*np.sin(5.0*np.pi/4.0)/a3
    lamb=2.0*alph
    z_im=z1-log(-lamb/(4.0*a3))/alph
    return -a10, a1, z_im
    
def potential(z):
    """
    Calculates potential at z in Chulkov model.
    """
    if z<0.0:
        v=a1*np.cos(g0*z)
    elif z<z1:
        v=-a10-a20+a2*np.cos(beta*z)
    elif z<z_im:
        v=-a10+a3*np.exp(-alph*(z-z1))
    else:
        v=-a10+(np.exp(-lamb*(z-z_im))-1.0)/(4.0*(z-z_im))
    return v
    
def overlap(m,n):
    """
    Calculates overlap integral and kinetic energy
    matrix element between states m and n. (d,D) are the width of the 
    well and the distance over which the basis functions are defined.
    Created on Thu Feb 18 16:39:13 2016, John Inglesfield.
    """
    if m==0 and n==0:
        overlap_int=d
        kinetic_en=0
    elif m==n:
        fac=0.5*(m*np.pi/D)**2
        if m%2==0:
            overlap_int=0.5*d+D*np.sin(m*np.pi*d/D)/(2.0*m*np.pi)
            kinetic_en=fac*(0.5*d-D*np.sin(m*np.pi*d/D)/(2.0*m*np.pi))
        else:
            overlap_int=0.5*d-D*np.sin(m*np.pi*d/D)/(2.0*m*np.pi)
            kinetic_en=fac*(0.5*d+D*np.sin(m*np.pi*d/D)/(2.0*m*np.pi))
    elif m%2==0 and n%2==0:
        fac=0.5*m*n*(np.pi/D)**2
        overlap_int=D*(np.sin(0.5*(m-n)*np.pi*d/D)/(m-n)\
        +np.sin(0.5*(m+n)*np.pi*d/D)/(m+n))/np.pi
        kinetic_en=fac*D*(np.sin(0.5*(m-n)*np.pi*d/D)/(m-n)\
        -np.sin(0.5*(m+n)*np.pi*d/D)/(m+n))/np.pi
    elif m%2==1 and n%2==1:
        fac=0.5*m*n*(np.pi/D)**2
        overlap_int=D*(np.sin(0.5*(m-n)*np.pi*d/D)/(m-n)\
        -np.sin(0.5*(m+n)*np.pi*d/D)/(m+n))/np.pi
        kinetic_en=fac*D*(np.sin(0.5*(m-n)*np.pi*d/D)/(m-n)\
        +np.sin(0.5*(m+n)*np.pi*d/D)/(m+n))/np.pi
    else:
        overlap_int=0.0
        kinetic_en=0.0
    return overlap_int,kinetic_en
    
def pot_mat_eval(m,n):
    """
    Evaluates matrix elements of potential by numerical quadrature.
    """
    if m%2==0 and n%2==0:
        v_mat=quad(lambda z:potential(z)*np.cos(m*np.pi*(z-z_orig)/D)\
        *np.cos(n*np.pi*(z-z_orig)/D),z_left,z_right)[0]
    elif m%2==0 and n%2==1:
        v_mat=quad(lambda z:potential(z)*np.cos(m*np.pi*(z-z_orig)/D)\
        *np.sin(n*np.pi*(z-z_orig)/D),z_left,z_right)[0]
    elif m%2==1 and n%2==0:
        v_mat=quad(lambda z:potential(z)*np.sin(m*np.pi*(z-z_orig)/D)\
        *np.cos(n*np.pi*(z-z_orig)/D),z_left,z_right)[0]
    else:
        v_mat=quad(lambda z:potential(z)*np.sin(m*np.pi*(z-z_orig)/D)\
        *np.sin(n*np.pi*(z-z_orig)/D),z_left,z_right)[0]
    return v_mat
        
def embedding(m):
    """
    Calculates matrix element of embedding potential.
    Created on Fri 19 Feb 2016, John Inglesfield
    """
    if m==0:
        embl=1.0
        embr=1.0
    elif m%2==0:
        embr=np.cos(0.5*m*np.pi*d/D)
        embl=embr
    else:
        embr=np.sin(0.5*m*np.pi*d/D)
        embl=-embr
    return embl,embr
    
def matrix_construct(param):
    """
    Constructs Hamiltonian matrix h, overlap matrix s, and left and
    right embedding potential matrices el and er.
    Created on Fri 19 Feb 2016, John Inglesfield
    """
    global n_max,z_orig,z_left,z_right,d,D
    n_max,z_orig,d,D=param
    z_left=z_orig-d/2.0
    z_right=z_orig+d/2.0
    msg='z_left = %5.3f, z_right = %5.3f, z_im = %5.3f' % (z_left,\
    z_right,z_im)
    assert (z_left<0.0 and z_right>z_im+1.0), msg                 
    kin_en=np.zeros((n_max,n_max))
    ovlp=np.zeros((n_max,n_max))
    emb_left=np.zeros((n_max))
    emb_right=np.zeros((n_max))
    pot_mat=np.zeros((n_max,n_max))
    for m in range(0,n_max):
        emb_left[m],emb_right[m]=embedding(m)
        for n in range(0,n_max):
            ovlp[m,n],kin_en[m,n]=overlap(m,n)
            pot_mat[m,n]=pot_mat_eval(m,n)
    embl=np.outer(emb_left,emb_left)
    embr=np.outer(emb_right,emb_right)
    return embl,embr,ovlp,kin_en,pot_mat
    
def schroed(y,z,e,eta):
    """
    Defines Schroedinger equation for numerical solution.
    """
    psi_r,psi_i,phi_r,phi_i=y
    dydz=[phi_r,phi_i,2.0*(a1*np.cos(g0*z)-e)*psi_r+2.0*eta*psi_i,\
    2.0*(a1*np.cos(g0*z)-e)*psi_i-2.0*eta*psi_r]
    return dydz
    
def crystal_embed(e,eta):
    """
    Evaluates crystal embedding potential by solving the one-dimensional
    Schroedinger equation through one unit cell in both directions.
    """
    y0=[1.0,0.0,0.0,0.0]
    z1=np.linspace(-z_left,-z_left+alat,101)
    sol1=odeint(schroed,y0,z1,args=(e,eta))
    z2=np.linspace(-z_left+alat,-z_left,101)
    sol2=odeint(schroed,y0,z2,args=(e,eta))
    psi1=complex(sol1[50,0],sol1[50,1])
    psi1_prime=complex(sol1[50,2],sol1[50,3])
    psi2=complex(sol2[50,0],sol2[50,1])
    psi2_prime=complex(sol2[50,2],sol2[50,3])
    wronskian=psi1*psi2_prime-psi2*psi1_prime
    psi1=complex(sol1[100,0],sol1[100,1])
    psi2=complex(sol2[100,0],sol2[100,1])
    cos_ka=0.5*(psi1+psi2)
    ka=arccos(cos_ka)
    if ka.imag<0: ka=np.conj(ka)
    exp_ka=cos_ka+1.0j*sqrt(1.0-cos_ka**2)
    emb_cryst=0.5*wronskian/(exp_ka-psi2)
    if emb_cryst.imag>0.0:
        exp_ka=cos_ka-1.0j*sqrt(1.0-cos_ka**2)
        emb_cryst=0.5*wronskian/(exp_ka-psi2)
    return emb_cryst
    
def vacuum_embed(e,eta):
    """
    Evaluates Coulomb embedding potential.
    """
    energy=complex(e,eta)
    rte=sqrt(2.0*(energy+a10))
    eta=1.0/(4.0*rte)
    rho=rte*(z_im-z_right)
    hh=cc=1.0e-20
    dd=0.0
    for n in xrange(1,200):
        dd=2.0*(rho-eta-n*1.0j)+(n-eta*1.0j)*(n-1-eta*1.0j)*dd        
        cc=2.0*(rho-eta-n*1.0j)+(n-eta*1.0j)*(n-1-eta*1.0j)/cc
        dd=1.0/dd
        dl=dd*cc
        hh=hh*dl
        if (abs(dl-1.0)<1.0e-12): break
    hh=1.0j*rte*((1.0-eta/rho)+hh/rho)
    emb_coulomb=-0.5*hh
    return emb_coulomb
    
def green(matrices,e,eta):
    """
    Constructs Green function and local density of states for the
    Chulkov surface model.
    Created on Fri 11 March 2016, John Inglesfield
    """
    energy=complex(e,eta)
    embl,embr,ovlp,kin_en,pot_mat=matrices
    sigl=crystal_embed(e,eta)
    sigr=vacuum_embed(e,eta)
    ham=kin_en+pot_mat+sigl*embl+sigr*embr-energy*ovlp
    lds=np.trace(solve(ham,ovlp)).imag/np.pi    
    return lds
