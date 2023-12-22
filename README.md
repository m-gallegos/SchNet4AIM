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

    python predict_QTAIM_electron_metrics.py geom.xyz 6


    python explain_group_DI.py 300K_trj.xyz

# References

[1] (I) K. Schütt, P.-J. Kindermans, H. E. Sauceda Felix, S. Chmiela, A. Tkatchenko and K.-R. Müller, Advances in Neural Information Processing Systems, 2017. (II) K. T. Schütt, H. E. Sauceda, P.-J. Kindermans, A. Tkatchenko and K.-R. Müller, J. Chem. Phys., 2018, 148, 241722. (III)  K. T. Schütt, P. Kessel, M. Gastegger, K. A. Nicoli, A. Tkatchenko and K.-R. Müller, J. Chem. Theory Comput., 2019, 15, 448–455.

[2] R. Bader, Atoms in Molecules: A Quantum Theory, Oxford University Press, Oxford, 1990.

[3] https://wiki.fysik.dtu.dk/ase/ase/db/db.html#ase-db
