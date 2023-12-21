import numpy as np

Z = {"C":6, "H":1, "O": 8, "N":7} 
molec_q=0

def standarize_labels(labels):
    """
    A function to standarize the atomic labels to
    match the common chemical notation.
    """
    stdlbl=[]
    for elem in labels:
        stdlbl.append(elem.lower().capitalize())
    return stdlbl

def calc_q_errors(q,atom_labels):
    """
    A function to compute the atomic errors in the reconstruction
    of the atomic charges (Q).
    """
    q=np.array(q,float)
    atom_labels=standarize_labels(atom_labels)
    zsum=0
    natoms=int(len(atom_labels))
    for elem in atom_labels:
        try: 
           zsum += Z[elem]
        except KeyError: 
           raise AssertionError ("Atom symbol not in the reference dictionary",elem)
    nelec = zsum-molec_q
    s4aim_charge=q.sum()
    s4aim_nelec = zsum-s4aim_charge
    s4aim_error= abs(s4aim_nelec-nelec)
    atomic_error=s4aim_error/float(natoms)
    return atomic_error

def calc_li_di_errors(li,di,atom_labels):
    """
    A function to compute the atomic and pairwise errors in the
    reconstruction of the number of electrons from the LI and DI
    SchNet4AIM predictions.
    """
    zsum=0
    natoms=int(len(atom_labels))
    for elem in atom_labels:
        try: 
           zsum += Z[elem]
        except KeyError: 
           raise AssertionError ("Atom symbol not in the reference dictionary",elem)
    nelec = zsum-molec_q
    li=np.array(li,float)
    di=np.array(di,float)
    s4aim_nelec=li.sum() + di.sum()
    s4aim_error= abs(s4aim_nelec-nelec)
    local_error = s4aim_error/(natoms*(natoms+1)/2)
    return local_error
