import os 
import numpy as np
import json
from SchNet4AIM.data import atoms as s4aim_atoms
import torch as torch
from SchNet4AIM import AtomsLoader as s4aim_AtomsLoader
from SchNet4AIM import create_subset as s4aim_create_subset
from sklearn.decomposition import PCA
import psutil

def set_cpu_num(num_processors):
    """
    Limit the number of CPUs
    """
    pid = os.getpid()
    p = psutil.Process(pid)
    available_cpus = psutil.cpu_count(logical=False)

    if num_processors > available_cpus:
        print(f"Error: Requested {num_processors} processors, but only {available_cpus} available.")
        return

    cpu_affinity = list(range(num_processors))
    p.cpu_affinity(cpu_affinity)
    print(f"Script restricted to {num_processors} processor(s): {cpu_affinity}")
    return None

def paircalc(n,mmode):
    if (mmode == "offdiag"):
       npair=n*(n-1)/2
    elif (mmode == "ondiag"):
       npair=n*(n+1)/2
    else:
       raise AssertionError("Error in paircalc")
    npair=int(npair)
    return npair

def xyz2json(xyzname=None,matmode="offdiag",path=None):
    """
    A function to create a JSON file (compatible with 1p, 2p and molecular properties)
    from a XYZ file.
    """
    if xyzname.endswith(".xyz"): name=xyzname.split(".xyz")[0]
    else: raise AssertionError("Only xyz files are accepted !")
    dbname  = os.path.join(path,name + ".json")

    if (matmode != "offdiag") and (matmode != "ondiag"):
       raise AssertionError("Unrecognized matmode ", matmode)
    try:
        os.remove(dbname)
    except OSError:
        pass

    dum=str("name:S:1:")
    dum=dum.split(':')
    molecprop_labels=[]
    molecprop_index=[]
    molecprop_type=[]
    molecprop_labels.append(dum[0]) 
    molecprop_type.append(dum[1]) 
    molecprop_index.append(dum[2])
    molecprop_col=np.asarray(molecprop_index,int)

    dum=str("ele:S:1:pos:R:3:")
    dum=dum.split(':')
    atomicprop_labels=[]
    atomicprop_index=[]
    atomicprop_type=[]
    contador=0
    for k in np.arange(0,2):
        atomicprop_labels.append(dum[contador]) 
        atomicprop_type.append(dum[contador+1]) 
        atomicprop_index.append(dum[contador+2])
        contador+=3
    atomicprop_col=np.asarray(atomicprop_index,int)

    dum=str("pari:I:1:parj:I:1:") 
    dum=dum.split(':')
    interatomicprop_labels=[]
    interatomicprop_index=[]
    interatomicprop_type=[]
    contador=0
    for k in np.arange(0,2):
        interatomicprop_labels.append(dum[contador]) 
        interatomicprop_type.append(dum[contador+1]) 
        interatomicprop_index.append(dum[contador+2])
        contador+=3
    interatomicprop_col=np.asarray(interatomicprop_index,int)

    molecprop_index=np.array(molecprop_index,int)
    atomicprop_index=np.array(atomicprop_index,int)
    interatomicprop_index=np.array(interatomicprop_index,int)
    key_list=["natoms"] + ["npair"] + molecprop_labels + atomicprop_labels + interatomicprop_labels
    database_json={}
    for i in key_list:
        database_json[i]=[] 
    contador=0
    with open(xyzname, "r") as f:
         while True:
            contador+=1
            first_line = f.readline()
            try:
                natoms = int(first_line.strip("\n"))
            except ValueError:
                print(" INFO: First line in the xyz file is empty, end of xyz file")
                break
            npair = paircalc(natoms,matmode) 
            database_json["natoms"].append(natoms)
            database_json["npair"].append(npair)
            line=f.readline().strip("/n").split()
            if (len(line)) != (sum(molecprop_index)) :
               line=[]
               line.clear()
               line.append(str(name))
            low=0
            for i in np.arange(len(molecprop_labels)):
                val=[]
                val.clear()
                if (molecprop_type[i] == "I"):
                   for p in line[low:low+molecprop_index[i]]:
                       val.append(int(p))
                elif (molecprop_type[i] == "R"):
                   for p in line[low:low+molecprop_index[i]]:
                       val.append(float(p))
                else:
                   for p in line[low:low+molecprop_index[i]]:
                       val.append(str(p))
                if len(val) == 1:
                   database_json[molecprop_labels[i]].append(val[0])
                else:
                   database_json[molecprop_labels[i]].append(val)
                low=low+molecprop_index[i]
            aux234=[]     
            aux234.clear()
            for j in np.arange(len(atomicprop_labels)):
                aux234.append([])
            for i in np.arange(natoms):
                line=f.readline().strip("/n").split()
                if (len(line)) != (sum(atomicprop_index)) :
                   raise AssertionError("The number of datapoints and atomic properties do not match!")
                low=0
                for j in np.arange(len(atomicprop_labels)):
                    val=[]
                    val.clear()
                    if (atomicprop_type[j] == "I"):
                       for p in line[low:low+atomicprop_index[j]]:
                           val.append(int(p))
                    elif (atomicprop_type[j] == "R"):
                       for p in line[low:low+atomicprop_index[j]]:
                           val.append(float(p))
                    else:
                       for p in line[low:low+atomicprop_index[j]]:
                           val.append(str(p))
                    if len(val) == 1:
                       aux234[j].append(val[0])
                    else:
                       aux234[j].append(val)
                    low=low+atomicprop_index[j]
            for j in np.arange(len(atomicprop_labels)):
                    database_json[atomicprop_labels[j]].append(aux234[j])
            aux234=[]    
            aux234.clear()
            paresi=[]
            paresj=[]
            for j in np.arange(len(interatomicprop_labels)):
                aux234.append([])
            for i in np.arange(0,natoms):
                for j in np.arange(i+1,natoms):
                    paresi.append(i+1)
                    paresj.append(j+1)
            for i in np.arange(npair):
                line=[]
                line.clear()
                line.append(paresi[i])
                line.append(paresj[i])
                low=0
                for j in np.arange(len(interatomicprop_labels)):
                    val=[]
                    val.clear()
                    if (interatomicprop_type[j] == "I"):
                       for p in line[low:low+interatomicprop_index[j]]:
                           val.append(int(p))
                    if len(val) == 1:
                       aux234[j].append(val[0])
                    else:
                       aux234[j].append(val)
                    low=low+interatomicprop_index[j]
            for j in np.arange(len(interatomicprop_labels)):
                    database_json[interatomicprop_labels[j]].append(aux234[j])
    with open(dbname, 'w') as fp:
       json.dump(database_json, fp)
    fp.close()
    return dbname

def read_from_json(databasename):
    """
    A function to read databases stored in a JSON-like datafile.
    """
    with open(databasename) as jFile:
        jObject = json.load(jFile)
    jFile.close()
    return jObject

def load_json_database_as_ase(dbname):
    """
    A function to load a json database as an ASE dataset.
    """
    dataset = read_from_json(dbname)
    dataset_ase = s4aim_atoms.AtomsData(dbname)
    atoms, properties = dataset_ase.get_properties(0)
    return dataset_ase,atoms,properties

def predict_model_1P(dataset_ase,nn_model_path):
    """
    A function to make model predictions with 1P SchNet4AIM models.
    """
    nn_model = torch.load(nn_model_path,map_location=torch.device('cpu'))
    cumul=0
    for count, batch in enumerate(s4aim_AtomsLoader(s4aim_create_subset(dataset_ase, np.arange(0,len(dataset_ase))),batch_size=1,shuffle=False)):
        cumul +=1
        pred = nn_model(batch)
        results=[]
        results.clear()
        atoms_list=[]
        for i in pred:
            prop=str(i)
        atom_count=1
        for i in np.arange(0,len((pred[prop].detach().cpu().numpy())[0,:,0])):
            results.append((pred[prop].detach().cpu().numpy())[0,i,0])
            atoms_list.append(atom_count)
            atom_count +=1
    return np.column_stack([atoms_list,results])

def predict_model_2P(dataset_ase,nn_model_path):
    """
    A function to make model predictions with 2P SchNet4AIM models.
    """
    nn_model = torch.load(nn_model_path,map_location=torch.device('cpu'))
    cumul=0
    for count, batch in enumerate(s4aim_AtomsLoader(s4aim_create_subset(dataset_ase, np.arange(0,len(dataset_ase))),batch_size=1,shuffle=False)):
        cumul +=1
        pred = nn_model(batch)
        pari = batch["pari"]
        parj = batch["parj"]
        results=[]
        results.clear()
        pari_list=[]
        pari_list.clear()
        parj_list=[]
        parj_list.clear()
        for i in pred:
            prop=str(i)
        for i in np.arange(0,len((pred[prop].detach().cpu().numpy())[0,:,0])):
            results.append((pred[prop].detach().cpu().numpy())[0,i,0])
            pari_list.append(int(pari[0,i].detach().cpu().numpy()))
            parj_list.append(int(parj[0,i].detach().cpu().numpy()))
    return np.column_stack([pari_list,parj_list,results])

def readtrj(trj_file):
    """
    A function to retrieve the individual frames of a trj file.
    """
    name=trj_file.strip().split('.')[0]
    f=open(trj_file,'r')
    lines=f.readlines()
    f.close()
    
    frame_names=[]
    counter=0
    geom_counter=0
    while counter < len(lines):
         line=lines[counter]
         if(len(line.strip().split())==1):
            geom_counter +=1
            fout=open((str(name)+"_"+str(geom_counter)+".xyz"),'w')
            frame_names.append(str(name)+"_"+str(geom_counter)+".xyz")
            natoms=line.strip().split()[0]
            fout.write(str(natoms) + "\n")
            fout.write("Frame " + str(geom_counter) + "\n")
            counter +=1
            for i in np.arange(int(natoms)):
                counter +=1
                line=lines[counter].strip().split()
                fout.write(line[0] + "   " + str(line[1]) + "  " + str(line[2]) + "  " + str(line[3]) + "\n")  
            fout.close()
            counter +=1
         elif (len(line.strip().split())==0): break

    return frame_names
def clean_trjfiles(frame):
    os.remove(frame)
    os.remove(frame.split(".xyz")[0]+".json")
    return None

def get_pair_labels(g1,g2):
    """
    A function to compute the pair labels of each of the contributions to the net DI
    """
    labels=[]
    for atom_i in g1:
        for atom_j in g2:
            labels.append("P"+str(atom_i)+"-"+str(atom_j))
    return labels

def get_group_di(mat,g1,g2):
    """
    A function to compute the group DI along with its components
    """
    di_loc=[]
    di_g=0
    for atom_i in g1:
        for atom_j in g2:
            di_loc.append(mat[atom_i-1,atom_j-1])
            di_g += mat[atom_i-1,atom_j-1]
    return di_g, np.array(di_loc,float)

def get_di_mat(raw_dis):
    """
    A function to get the DI matrix from the raw SchNet4AIM DI predictions
    """
    pair_i=raw_dis[:,0]
    pair_j=raw_dis[:,1]
    di_val=raw_dis[:,2]
    natoms=int(max(pair_j))
    di_mat=np.zeros((natoms,natoms),float)  
    for i in range(di_val.shape[0]):
        di_mat[int(pair_i[i])-1,int(pair_j[i])-1]=di_val[i]
        di_mat[int(pair_j[i])-1,int(pair_i[i])-1]=di_val[i]
    return di_mat

def explain_group_di(di_frames,group_a,group_b,name,atom_labels,threshold):
    """
    Search the most relevant components to the DI between groups A and B.
    """
    try: 
       float(threshold)
       if threshold > 1 or threshold < 0: raise AssertionError(" Invalid threshold value", threshold)
    except Exception: raise AssertionError(" Invalid threshold value", threshold)
    var_labels = get_pair_labels(group_a, group_b)
    num_var=len(var_labels)
    num_frames=len(di_frames)
    if num_frames >= num_var: 
       num_components=num_var
    else:
       num_components=num_frames
    di_group_seq = []
    di_component_seq = []
    with open(name+"_group_di.txt",'w') as fout:
     fout.write(" Frame  Total " + " ".join(var_labels) + "\n")
     for frame in range(num_frames):
         di_group, di_components = get_group_di(di_frames[frame], group_a, group_b)
         di_group_seq.append(di_group)
         di_component_seq.append(di_components)
         formatted_di_components = ", ".join([f"{x:8.4f}" for x in di_components])
         formatted_di_components = " ".join([f"{x:8.4f}" for x in di_components])
         fout.write(f"{frame+1:8.0f} {di_group:8.4f} {formatted_di_components}" + "\n")
    pca = PCA(n_components=num_components)
    pca_result = pca.fit_transform(np.vstack(di_component_seq))
    principal_components = pca.components_
    explained_variance=pca.explained_variance_ratio_
    print("----------------------------------------------------------------------------------")
    print("                          XCAI-GROUP ELECTRON DELOCALIZATION ")
    print("----------------------------------------------------------------------------------")
    print(" # GROUP A: ", group_a)
    print(" # GROUP B: ", group_b)
    print(" # Pair-wise contributions to A-B DI :")
    for var in var_labels: 
        atom_i=var.split("-")[0][1:]
        atom_j=var.split("-")[1]
        atom_i_label = atom_labels[int(atom_i) - 1]
        atom_j_label = atom_labels[int(atom_j) - 1]
        print(f"  - {var:6s} Atom {atom_i:4s} ({atom_i_label:2s}) Atom {atom_j:4s} ({atom_j_label:2s}).")
    print("----------------------------------------------------------------------------------")
    print(" # Weights of each pairwise DI to the group DI:")
    global_score=[]
    importance_score=[]
    for var in range(num_var):
        score=0
        for pc in range(num_components):
            score += explained_variance[pc]*principal_components[pc][var]
        global_score.append(score)
    global_score=np.array(global_score,float)
    importance_score=np.array(abs(global_score),float)
    importance_score/=np.sum(importance_score)
    global_score/=np.sum(global_score)
    global_score_sum=np.sum(global_score)
    global_score = sorted(zip(var_labels, global_score), key=lambda x: abs(x[1]), reverse=True)
    importance_score = sorted(zip(var_labels, importance_score), key=lambda x: abs(x[1]), reverse=True)
    for variable, score in global_score:
        idx=var_labels.index(variable) +1
        print(f"  - {variable:6s} (var num {idx:5d}): {score:8.4f}")
    print(" # Importance score of each pairwise DI to the group DI:")
    cum_score=0.0
    dominant_vars=[]
    for variable, score in importance_score:
        idx=var_labels.index(variable) +1
        if cum_score < threshold: dominant_vars.append(idx)
        cum_score += score
        print(f"  - {variable:6s} (var num {idx:5d}): {score:8.4f}")
    print(" # Variables ", dominant_vars, " explain (at least) ", threshold," of the group DI.")
    with open(name+"_dominant_di_"+str(threshold)+".txt",'w') as fout:
        formatted_vars="  ".join([var_labels[var-1] for var in dominant_vars])
        fout.write(" Frame    Group-DI   Reconstructed-DI  "+formatted_vars+"\n")
        for frame in range(num_frames):
            val=di_component_seq[frame]
            cumul_val=0.0
            pairwise_comp=[] 
            for var in dominant_vars:
                pairwise_comp.append(val[var-1])
                cumul_val += val[var-1] 
            formatted_pairwise_comp = " ".join([f"{x:8.4f}" for x in pairwise_comp])
            fout.write(f"{frame+1:8.0f} {di_group_seq[frame]:8.4f} {cumul_val:8.4f} {formatted_pairwise_comp}" + "\n")
    print("----------------------------------------------------------------------------------")
    print(" # Group DIs saved to           ", name+"_group_di.txt")
    print(" # Dominant components saved to ", name+"_dominant_di_"+str(threshold)+".txt")
    print("----------------------------------------------------------------------------------")
    return None
