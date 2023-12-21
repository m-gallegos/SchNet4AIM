print(
"""
##################################################################################

                               ~ SchNet4AIM ~
                                  (v 1.0)

     Theoretical and Computational Chemistry Group, Univeristy of Oviedo.
        Theoretical Chemical Physics Group, Univeristy of Luxembourg.

M. Gallegos, V. Vassilev-Galindo, I. Poltavsky, A. M. Pendas and A. Tkatchenko
##################################################################################

A code to train SchNet models on the prediction of atomic and pairwise properties.
In. Molecules. (AIM) properties. It relies on the SchNetPack suite:
 - J. Chem. Theory Comput. 2019, 15 (1): 448-455.
 - J. Chem. Phys. 2023, 158 (14): 144801.
##################################################################################

"""
)

class Properties:
    """
    Keys to access structure properties in `SchNet4AIM.data.AtomsData`
    """

    # geometry
    Z = "_atomic_numbers"
    charge = "_charge"
    atom_mask = "_atom_mask"
    position = "_positions"
    R = position
    cell = "_cell"
    pbc = "_pbc"
    neighbors = "_neighbors"
    neighbor_mask = "_neighbor_mask"
    cell_offset = "_cell_offset"
    neighbor_pairs_j = "_neighbor_pairs_j"
    neighbor_pairs_k = "_neighbor_pairs_k"
    neighbor_pairs_mask = "_neighbor_pairs_mask"
    neighbor_offsets_j = "_neighbor_offsets_j"
    neighbor_offsets_k = "_neighbor_offsets_k"

    neighbors_lr = "_neighbors_lr"
    neighbor_mask_lr = "_neighbor_mask_lr"
    cell_offset_lr = "_cell_offset_lr"

    # chemical properties
    energy = "energy"
    forces = "forces"
    stress = "stress"
    dipole_moment = "dipole_moment"
    total_dipole_moment = "total_dipole_moment"
    polarizability = "polarizability"
    iso_polarizability = "iso_polarizability"
    at_polarizability = "at_polarizability"
    charges = "charges"
    energy_contributions = "energy_contributions"
    shielding = "shielding"
    hessian = "hessian"
    dipole_derivatives = "dipole_derivatives"
    polarizability_derivatives = "polarizability_derivatives"
    electric_field = "electric_field"
    magnetic_field = "magnetic_field"
    dielectric_constant = "dielectric_constant"
    magnetic_moments = "magnetic_moments"

    properties = [
        energy,
        forces,
        stress,
        dipole_moment,
        polarizability,
        shielding,
        hessian,
        dipole_derivatives,
        polarizability_derivatives,
        electric_field,
        magnetic_field,
    ]

    external_fields = [electric_field, magnetic_field]

    electric_properties = [
        dipole_moment,
        dipole_derivatives,
        dipole_derivatives,
        polarizability_derivatives,
        polarizability,
    ]
    magnetic_properties = [shielding]

    required_grad = {
        energy: [],
        forces: [position],
        hessian: [position],
        dipole_moment: [electric_field],
        polarizability: [electric_field],
        dipole_derivatives: [electric_field, position],
        polarizability_derivatives: [electric_field, position],
        shielding: [magnetic_field, magnetic_moments],
    }

import os
import sys
# Retrieve information about the environment variables

# Pmode : type of target property (1p or 2p)
# DBmode: type of database (json or db)
# frset : frozenset gathering the chemical diversity (given by the atomic numbers of the elements in the database).

print(" # Retrieving information from environment variables.............done!")
if ("Pmode" in os.environ):
   Pmode = os.environ['Pmode']
else:
   raise AssertionError(" Pmode not set!")
if ("DBmode" in os.environ):
   DBmode  = os.environ['DBmode']
else:
   raise AssertionError(" DBmode not set!")
if ("frset" in os.environ):
   frset   = os.environ['frset'].split(os.pathsep)                  
   frset = [int(item) for item in frset]
else:
   # If the frozenset is not defined, the default (C,H,O,N) will be employed.
   print(" # INFO: frozen set is not defined, default will be used")
   elements = [1,6,7,8]
   os.environ['frset']   = os.pathsep.join(str(i) for i in sorted(elements))  
   frset   = os.environ['frset'].split(os.pathsep)                 
   frset = [int(item) for item in frset]

print("   - Type of target property :", Pmode)
print("   - Database format         :", DBmode)
print("   - Chemical diversity      :", frset)
print("   - SchNet4AIM path         :", sys.modules['SchNet4AIM'])

# Load the required dependencies and packages
from SchNet4AIM.atomistic import AtomisticModel
from SchNet4AIM.data import *
from SchNet4AIM import atomistic
from SchNet4AIM import data
from SchNet4AIM import environment
from SchNet4AIM.train import metrics, hooks
from SchNet4AIM import nn
from SchNet4AIM import representation
from SchNet4AIM import train
from SchNet4AIM import utils
from SchNet4AIM.representation import SchNet
from SchNet4AIM.utils import __init__
