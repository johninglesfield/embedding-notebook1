"""
Module calculates integrals needed for surface model. Function
hamiltonian evaluates eigenvalues of bound states and resonances
for a trial energy e_try. Function green evaluates density of
states in embedded region as a function of complex energy(energy,eta).
"""
import numpy as np
from scipy.linalg import eig, solve
from numpy.lib.scimath import sqrt
from numpy import trace
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
    global n_max,v,w,d,D
    n_max,v,w,d,D=param
    kin_en=np.zeros((n_max,n_max))
    ovlp=np.zeros((n_max,n_max))
    emb_left=np.zeros((n_max))
    emb_right=np.zeros((n_max))
    for m in range(0,n_max):
        (emb_left[m],emb_right[m])=embedding(m)
        for n in range(0,n_max):
            (ovlp[m,n],kin_en[m,n])=overlap(m,n)
    embl=np.outer(emb_left,emb_left)
    embr=np.outer(emb_right,emb_right)
    return embl,embr,ovlp,kin_en
def hamiltonian(matrices,e_try):
    """
    Constructs generalized eigenvalue problem to find bound states
    of surface model.
    Created on Fri 19 Feb 2016, John Inglesfield
    """
    embl,embr,ovlp,kin_en=matrices
    kl=sqrt(2*e_try)
    if e_try.real<0 and kl.imag<0: kl=-kl 
    kr=sqrt(2*(e_try-w))
    if (e_try.real-w)<0 and kr.imag<0: kr=-kr
    sigl=-0.5*complex(0.0,kl)
    dsigl=-0.5*complex(0.0,1.0/kl)
    sigr=-0.5*complex(0.0,kr)
    dsigr=-0.5*complex(0.0,1.0/kr)
    ham=kin_en+v*ovlp+(sigl-e_try*dsigl)*embl+(sigr-e_try*dsigr)*embr
    ovr=ovlp-dsigl*embl-dsigr*embr
#    if e_try.imag==0.0:
#        eigenvalues=eigh(ham,ovr,eigvals_only=True).real
#    else
    eigenvalues=eig(ham,ovr,left=False,right=False)
    eigenvalues=sorted(eigenvalues)
    return eigenvalues
def green(matrices,energy,eta):
    """
    Constructs Green function and local density of states for the
    surface model.
    Created on Fri 11 March 2016, John Inglesfield
    """
    embl,embr,ovlp,kin_en=matrices
    energy=complex(energy,eta)
    kl=sqrt(2*energy)
    if energy.real<0 and kl.imag<0: kl=-kl 
    kr=sqrt(2*(energy-w))
    if (energy.real-w)<0 and kr.imag<0: kr=-kr
    sigl=-0.5*complex(0.0,kl)
    sigr=-0.5*complex(0.0,kr)
    ham=kin_en+v*ovlp+sigl*embl+sigr*embr-energy*ovlp
    lds=trace(solve(ham,ovlp)).imag/np.pi    
    return lds
