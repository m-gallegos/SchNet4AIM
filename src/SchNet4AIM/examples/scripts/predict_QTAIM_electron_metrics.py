#----------------------------------------------------------------------------
# This script estimates the QTAIM electron metrics (Q, LI and DI) of a neutral
# CHON molecule using the pre-trained SchNet4AIM models. Additionally, 
# the uncertainty in the predictions is estimated from the accuracy in the
# reconstruction of the total number of electrons.

# Execution: python ./predict_QTAIM_electron_metrics.py  file.xyz nproc
# 
# file.xyz: geometry file in XYZ format (only CHON molecules).
# nproc   : number of CPUs to be used in the process.
#----------------------------------------------------------------------------


#----------------------------------------------------------------
elements        =  [1,6,7,8] 
import os
import importlib
import sys
os.environ['Pmode']   = "1p"
os.environ['DBmode']  = "json"
os.environ['frset']   = os.pathsep.join(str(i) for i in sorted(elements))
import SchNet4AIM as s4aim
from SchNet4AIM.examples.utils import general as s4aim_general
from SchNet4AIM.examples.utils import uncertainty as s4aim_uncertainty
s4aim_path = s4aim.__file__
s4aim_path = s4aim_path.split("__init__.py")[0]+"examples/models/electronic/"
q_model_path = s4aim_path + "model_q_AIMwise"
li_model_path= s4aim_path + "model_li_ElementalAIMwise"
di_model_path= s4aim_path + "model_di_ElementalPairAIMwise"
#----------------------------------------------------------------
xyzname=sys.argv[1]
if xyzname.endswith('.xyz'):
   name=xyzname[:-4]
else:
   raise AssertionError("Only xyz files are accepted !")
dbname=s4aim_general.xyz2json(xyzname=xyzname,matmode="offdiag",path=os.getcwd())
nproc=int(sys.argv[2])
s4aim_general.set_cpu_num(nproc)
#----------------------------------------------------------------
dataset_ase,atoms,properties = s4aim_general.load_json_database_as_ase(dbname)
atom_labels=properties['ele']
#----------------------------------------------------------------
raw_q  = s4aim_general.predict_model_1P(dataset_ase,q_model_path)
raw_li = s4aim_general.predict_model_1P(dataset_ase,li_model_path)
os.environ['Pmode']   = "2p"
import SchNet4AIM as s4aim
s4aim = importlib.reload(s4aim)
raw_dis = s4aim_general.predict_model_2P(dataset_ase,di_model_path)
#----------------------------------------------------------------
q_error=s4aim_uncertainty.calc_q_errors(raw_q[:, -1],atom_labels)
li_di_error=s4aim_uncertainty.calc_li_di_errors(raw_li[:,-1],raw_dis[:,-1],atom_labels)
#----------------------------------------------------------------
with open(xyzname.split(".xyz")[0]+"_s4aim.qtaim",'w') as f:
     f.write(" -----------------------------------------------------------------------------"+"\n")
     f.write(" ATOMIC CHARGE (Q)"+"\n")
     f.write(" -----------------------------------------------------------------------------"+"\n")
     for i in range(len(atom_labels)):
         f.write(f" Atom {i+1:6d} ({atom_labels[i]:2s})  q = {raw_q[:,-1][i]:6.3f} +- {q_error:2.3f} : {raw_q[:,-1][i]-q_error:6.3f} to {raw_q[:,-1][i]+q_error:6.3f} electrons."+"\n")
     f.write(" -----------------------------------------------------------------------------"+"\n")
     f.write(" ELECTRON LOCALIZATION INDEX (LI)"+"\n")
     f.write(" -----------------------------------------------------------------------------"+"\n")
     for i in range(len(atom_labels)):
         f.write(f" Atom {i+1:6d} ({atom_labels[i]:2s})  li = {raw_li[:,-1][i]:6.3f} +- {li_di_error:2.3f} : {raw_li[:,-1][i]-li_di_error:6.3f} to {raw_li[:,-1][i]+li_di_error:6.3f} electrons."+"\n")
     f.write(" -----------------------------------------------------------------------------"+"\n")
     f.write(" ELECTRON DELOCALIZATION INDEX (DI)"+"\n")
     f.write(" -----------------------------------------------------------------------------"+"\n")
     for i in range(raw_dis.shape[0]):
         pair_i = int(raw_dis[:, 0][i])
         pair_j = int(raw_dis[:, 1][i])
         di_val = raw_dis[:, -1][i]
         atom_i_label = atom_labels[pair_i - 1]
         atom_j_label = atom_labels[pair_j - 1]
         f.write(f" Pair {pair_i:6d}{pair_j:6d} ({atom_i_label:2s},{atom_j_label:2s}) DI = {di_val:6.3f} +- {li_di_error:2.3f} : {di_val-li_di_error:6.3f} to {di_val+li_di_error:6.3f} electrons."+"\n")
