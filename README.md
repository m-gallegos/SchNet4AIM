# SchNet4AIM

SchNet4AIM is a code designed to train SchNet [1] models on atomic (1p) and pair-wise (2p) 
properties, such as those formulated within the context of the Quantum Theory of Atoms In Molecules (QTAIM). [2]

* The architecture of SchNet4AIM is a slight modification of the original SchNetPack code, [2, III] the functions
and routines of the latter which are not involvd in the training and prediction of 1p or 2p properties have been removed
for the sake of simplicity.
* The code can run on both CPUs and GPUs. The use of the latter is generally advisable for the sake of computational
time.
* Before importing or initializing the SchNet4AIM module, three bash environment variables must be specified:
  * Pmode : defines the type of target property. It can take two different values: "1p", for atomic properties, or "2p", for pair-wise properties.
  * DBmode: defines format employed for the database. It can take the "json" or "db" values, corresponding to the JSON and ASE-SQLite [3] database formats, respectively.
  * frset : defines the list of possible elements found in the database employed for the training, given in terms of their atomic numbers.

The environment variables can be easily defined using a simple block of python code (which should be included in the heading of the training or prediction scripts):

    dbname          = "./database_q.db"   # name of the database used throughout the training (*.json or *.db file formats are allowed)
    elements        =  [1,8]              # possible atomic numbers of the elements found in the database (e.g H,O)
    
    # Setting the bash environment variables controlling the type of property and database used.
    import os
    os.environ['Pmode']   = mode          # 1p (atomic properties) or 2p (pairwise properties).
    os.environ['DBmode']  = dbmode
    os.environ['frset']   = os.pathsep.join(str(i) for i in sorted(elements))



# Installation

SchNet4AIM can be easily installed using the pip Python package manager:

    pip install git+https://github.com/m-gallegos/SchNet4AIM.git

Alternatively, one can download the zip file from the SchNet4AIM GitHub and run the following command:

    pip install SchNet4AIM-main.zip

The code requires other Python modules to run which will be installed along with the former (if not present in the local machine). To avoid incompatibilities, it is generally advisable to install and run SchNet4AIM under a Python environment:

    python -m venv venv
    source /venv/bin/activate  

# Examples

In addition to the SchNet4AIM source code, this repository also includes a set of prototype scripts to exemplify its use, collected in the /examples/scripts/ directory.In particular, we include two key templates, one aimed at predicting the electronic QTAIM properties (Q, LI and DI) from a molecular geoemtry (including uncertainty estimates) and the other devoted to the analysis of the pairwise components that dominate the electronic delocalization between two groups or fragments along a sequential process. We note in passing that both are based on the proof-of-concept models incorporated in SchNet4AIM and are therefore only suitable for treating neutral C,H,O,N molecules.

## Predicting QTAIM electronic properties.

The script /examples/scripts/predict_QTAIM_electron_metrics.py is a simple python template designed to ease the prediction of QTAIM electron metrics (localization index, LI, delocalization index, DI, and atomic charges, Q). It receives two command-line arguments:

    python predict_QTAIM_electron_metrics.py geom.xyz nproc

  * geom.xyz : file containing the molecular geometry in XYZ cartesian coordinates.
  * nproc    : number of CPUs to be employed in the calculation.

Executing the previous script will produce an output file, geom_s4aim.qtaim, which comprises the SchNet4AIM predictions along with their uncertainties.The output format used is the following: first, the atomic properties (charges and LIs) appear in increasing order of the atoms, 

    -----------------------------------------------------------------------------
    ATOMIC CHARGE (Q)
    -----------------------------------------------------------------------------
    Atom      1 (C )  q = -0.031 +- 0.000 : -0.031 to -0.031 electrons.
    Atom      2 (C )  q = -0.029 +- 0.000 : -0.029 to -0.029 electrons.
    Atom      3 (C )  q = -0.004 +- 0.000 : -0.004 to -0.004 electrons.
    Atom      4 (C )  q = -0.004 +- 0.000 : -0.004 to -0.004 electrons.
    . . . . . . . . 
    . . . . . . . . 
    -----------------------------------------------------------------------------
    ELECTRON LOCALIZATION INDEX (LI)
    -----------------------------------------------------------------------------
    Atom      1 (C )  li =  3.968 +- 0.000 :  3.968 to  3.968 electrons.
    Atom      2 (C )  li =  3.968 +- 0.000 :  3.968 to  3.968 electrons.
    Atom      3 (C )  li =  3.889 +- 0.000 :  3.889 to  3.889 electrons.
    Atom      4 (C )  li =  3.889 +- 0.000 :  3.889 to  3.889 electrons.
    . . . . . . . . 
    . . . . . . . . 

followed by the delocalization indices of each of the pairs. For the latter, the results for the single pairs (upper triangular part of the matrix) are presented:

    -----------------------------------------------------------------------------
    ELECTRON DELOCALIZATION INDEX (DI)
    -----------------------------------------------------------------------------
    Pair      1     2 (C ,C ) DI =  1.513 +- 0.000 :  1.513 to  1.513 electrons.
    Pair      1     3 (C ,C ) DI =  1.236 +- 0.000 :  1.236 to  1.237 electrons.
    Pair      1     4 (C ,C ) DI =  0.059 +- 0.000 :  0.059 to  0.059 electrons.
    Pair      1     5 (C ,C ) DI =  0.094 +- 0.000 :  0.094 to  0.094 electrons.
    Pair      1     6 (C ,C ) DI =  0.073 +- 0.000 :  0.073 to  0.074 electrons.
    Pair      1     7 (C ,C ) DI =  0.067 +- 0.000 :  0.067 to  0.067 electrons.
    Pair      1     8 (C ,C ) DI =  0.024 +- 0.000 :  0.024 to  0.024 electrons.
    Pair      1     9 (C ,C ) DI =  0.025 +- 0.000 :  0.024 to  0.025 electrons.
    Pair      1    10 (C ,C ) DI =  0.011 +- 0.000 :  0.011 to  0.011 electrons.
    Pair      1    11 (C ,H ) DI =  0.952 +- 0.000 :  0.952 to  0.952 electrons.
    . . . . . . . . 
    . . . . . . . . 
We note in passing that the uncertainties are based on the accuracy with which SchNet4AIM reconstructs the total number of electrons of the molecule (either from the atomic charges or the localized and delocalized electron populations).
    
## XCAI: Unravelling the dominant contributions to the group DI.

    python explain_group_DI.py 300K_trj.xyz

# References

[1] (I) K. Schütt, P.-J. Kindermans, H. E. Sauceda Felix, S. Chmiela, A. Tkatchenko and K.-R. Müller, Advances in Neural Information Processing Systems, 2017. (II) K. T. Schütt, H. E. Sauceda, P.-J. Kindermans, A. Tkatchenko and K.-R. Müller, J. Chem. Phys., 2018, 148, 241722. (III)  K. T. Schütt, P. Kessel, M. Gastegger, K. A. Nicoli, A. Tkatchenko and K.-R. Müller, J. Chem. Theory Comput., 2019, 15, 448–455.

[2] R. Bader, Atoms in Molecules: A Quantum Theory, Oxford University Press, Oxford, 1990.

[3] https://wiki.fysik.dtu.dk/ase/ase/db/db.html#ase-db
