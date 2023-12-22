#----------------------------------------------------------------------------
# This script is designed to identify the most dominant components of group DI
# (Delocalization Index) over a process within a specific time window. It
# performs an analysis on the local components that influence the behavior of 
# a global feature across a set of frames.

# Execution: python ./explain_group_DI.py trj_file.xyz
# 
# trj_file.xyz: a trajectory file in XYZ format (only CHON molecules).
#
# For instance:
#
# python ./explain_group_DI.py 13P-CO2_trj.xyz
#
# This will produce the 13P-CO2_trj_group_di.txt and
# 13P-CO2_trj_dominant_di_0.8.txt output files, showing the most relevant
# contributions to the group DI between a CO2 and a NH2 group in the 13P-
# CO2 complexation (900 K).
#
# INPUT VARIABLES:
#----------------------------------------------------------------------------
group_a  =  [1,2,3]      # Define atom numbers of group A (starting at 1).
group_b  =  [20,23,27]   # Define atom numbers of group B (starting at 1).
threshold=  0.8          # Cumulative explainability threshold (0 to 1).
nproc    =  8            # Number of processors to employ.
#----------------------------------------------------------------------------
import sklearn
import os
import importlib
import sys
elements =  [1,6,7,8]
os.environ['Pmode']   = "2p"
os.environ['DBmode']  = "json"
os.environ['frset']   = os.pathsep.join(str(i) for i in sorted(elements))
import SchNet4AIM as s4aim
from SchNet4AIM.examples.utils import general as s4aim_general
s4aim_path = s4aim.__file__
s4aim_path = s4aim_path.split("__init__.py")[0]+"examples/models/electronic/"
di_model_path= s4aim_path + "model_di_ElementalPairAIMwise"
s4aim_general.set_cpu_num(nproc)

trj_file=sys.argv[1]
name=trj_file.split(".xyz")[0]
frame_names=s4aim_general.readtrj(trj_file)

di_frames=[]
for frame in frame_names:
    print(" # Running SchNet4AIM predictions for frame ", frame)
    dbname=s4aim_general.xyz2json(xyzname=frame,matmode="offdiag",path=os.getcwd())
    dataset_ase,atoms,properties = s4aim_general.load_json_database_as_ase(dbname)
    atom_labels=properties['ele']
    raw_dis = s4aim_general.predict_model_2P(dataset_ase,di_model_path)
    di_frames.append(s4aim_general.get_di_mat(raw_dis))
    s4aim_general.clean_trjfiles(frame)
s4aim_general.explain_group_di(di_frames,group_a,group_b,name,atom_labels,threshold)
