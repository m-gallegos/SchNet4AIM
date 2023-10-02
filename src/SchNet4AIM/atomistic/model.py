from torch import nn as nn
import torch
import numpy as np
from SchNet4AIM import Properties
from SchNet4AIM.nn.neighbors import atom_distances 
import SchNet4AIM as s4aim

__all__ = ["AtomisticModel"]

class ModelError(Exception):
    pass

class AtomisticModel(nn.Module):
    """
    Join a representation model with output modules.

    Args:
        representation (torch.nn.Module): Representation block of the model.
        output_modules (list or nn.ModuleList or s4aim.output_modules.Atomwise): Output
            block of the model. Needed for predicting properties.

    Returns:
         dict: property predictions
    """

    def __init__(self, representation, output_modules):
        super(AtomisticModel, self).__init__()
        self.representation = representation
        if type(output_modules) not in [list, nn.ModuleList]:
            output_modules = [output_modules]
        if type(output_modules) == list:
            output_modules = nn.ModuleList(output_modules)
        self.output_modules = output_modules
        # For gradients
        self.requires_dr = any([om.derivative for om in self.output_modules])

    def forward(self, inputs):
        """
        Forward representation output through output modules.
        """
        if self.requires_dr:
            inputs[Properties.R].requires_grad_()

        # Retrieve the local representation
        # For 1P properties, the SchNet representation can be directly employed 
        if (s4aim.Pmode == "1p"):
            inputs["representation"] = self.representation(inputs)
        # For 2P properties, the representation is obtained from the concatenation of 
        # atomic environments and the pairwise distance
        elif (s4aim.Pmode == "2p"):
            # Get the SchNet atomic environments
            atomic_envs = self.representation(inputs)                            
            # Compute the inter-atomic distances
            r_ij = atom_distances(positions=inputs[Properties.R], neighbors=inputs[Properties.neighbors])
            # Create the pairwise descriptor 
            pair_rep=[]
            # Get the local device
            locdevice=inputs[Properties.R].device                                            
            r_ij=torch.Tensor(r_ij)                        
            # Send the interatomic_numbers to the device (if required)
            if (locdevice != "cpu"): r_ij=r_ij.to(locdevice)                        
            for i in np.arange(int(inputs["npair"])):
                index_i=int(inputs["pari"][0,i])-1
                index_j=int(inputs["parj"][0,i])-1
                r_pair=r_ij[0,index_i,int((inputs[Properties.neighbors][0,index_i,:]==index_j).nonzero().squeeze())]
                # Concatenation of atomic environments
                ij_envs=torch.cat((atomic_envs[:,index_i,:],atomic_envs[:,index_j,:]),1)
                # Include the distance
                ij_rep=torch.cat((torch.reshape(r_pair,(1,1)),ij_envs),1)
                pair_rep.append(ij_rep)                               
            pair_rep=torch.cat(pair_rep)
            pair_rep=torch.reshape(pair_rep,(1,pair_rep.size()[0],pair_rep.size()[1]))    
            # Update the final representation
            inputs["representation"] = pair_rep
    
        outs = {}
        for output_model in self.output_modules:
            outs.update(output_model(inputs))
        return outs
