import numpy as np
import torch
from torch import nn as nn
from torch.autograd import grad
import SchNet4AIM
from SchNet4AIM import nn as L, Properties

"""
 Define the three possible output modules to be used for the prediction 
 of AIM properties. All of them rely on the general structure of the 
 original AtomWise output module of SchNet.

 -AIMwise: universal model, valid for 1P or 2P properties.
 -ElementalAIMwise: particle-specific model, valid for 1P properties.
 -ElementalPairAIMwise: particle-specific model, valid for 2P properties.
"""

__all__ = [
    "AIMwise",
    "ElementalAIMwise",
    "ElementalPairAIMwise",     
]


class AIMwiseError(Exception):
    pass

class AIMwise(nn.Module):                                                                            
    """
    Predicts 1P or 2P properties using a general (universal) model. It is based on the
    original AtomWise output module of SchNetPack.

    Args:
        n_in (int): input dimension of representation
        n_out (int): output dimension of target property (default: 1)
        aggregation_mode (str): one of {atomic} (default: atomic)
        n_layers (int): number of nn in output network (default: 2)
        n_neurons (list of int or None): number of neurons in each layer of the output
            network. If `None`, divide neurons by 2 in each layer. (default: None)
        activation (function): activation function for hidden nn
            (default: s4aim.nn.activations.shifted_softplus)
        property (str): name of the output property (default: "y")
        mean (torch.Tensor or None): mean of property
        stddev (torch.Tensor or None): standard deviation of property (default: None)
        outnet (callable): Network used for atomistic outputs. Takes SchNet4AIM input
            dictionary as input. Output is not normalized. If set to None,
            a pyramidal network is generated automatically. (default: None)
    Returns:
        tuple: prediction for property

        If derivative is not None additionally returns derivative w.r.t. atom positions.
    """

    def __init__(
        self,
        n_in,
        n_out=1,
        aggregation_mode="atomic",
        n_layers=2,
        n_neurons=None,
        activation=SchNet4AIM.nn.activations.shifted_softplus,
        derivative=None,
        property="y",
        mean=None,
        stress=None,
        stddev=None,
        outnet=None,
        create_graph=False,
    ):
        super(AIMwise, self).__init__()

        self.n_layers = n_layers
        self.create_graph = create_graph
        self.property = property
        self.derivative = derivative
        self.stress = stress

        mean = torch.FloatTensor([0.0]) if mean is None else mean
        stddev = torch.FloatTensor([1.0]) if stddev is None else stddev

        # build output network
        if outnet is None:
            self.out_net = nn.Sequential(
                SchNet4AIM.nn.base.GetItem("representation"),
                SchNet4AIM.nn.blocks.MLP(n_in, n_out, n_neurons, n_layers, activation),
            )
        else:
            self.out_net = outnet

        # build standardization layer
        self.standardize = SchNet4AIM.nn.base.ScaleShift(mean, stddev)

        # build aggregation layer
        if aggregation_mode == "atomic":                                      
            self.atom_pool = SchNet4AIM.nn.base.DummyAggregate()               
        else:
            raise AIMwiseError(
                "{} is not a valid aggregation " "mode!".format(aggregation_mode)
            )

    def forward(self, inputs):
        """
        output the predictions
        """
        atomic_numbers = inputs[Properties.Z]
        atom_mask = inputs[Properties.atom_mask]

        # run prediction
        yi = self.out_net(inputs)
        yi = self.standardize(yi)

        y = self.atom_pool(yi, atom_mask)

        # collect results
        result = {self.property: y}

        create_graph = True if self.training else self.create_graph

        return result                                                                                         

class ElementalAIMwise(AIMwise):                                                                            
    """
    Predicts properties in atom-wise fashion using a separate network for every chemical
    element of the central atom. 

    Besides the main arguments of the AIMwise output module, it can receive the following:
    Args:
        elements (set of int): List of atomic numbers present in the training set.
        n_hidden (int): number of neurons in each layer of the output network.
            (default: 50)
    """

    def __init__(
        self,
        n_in=128,
        n_out=1,
        aggregation_mode="atomic",  
        n_layers=3,
        property="y",
        derivative=None,
        create_graph=True,          
        elements = frozenset((SchNet4AIM.frset)),
        n_hidden=50,
        activation=SchNet4AIM.nn.activations.shifted_softplus,
        mean=None,
        stddev=None,
    ):
        outnet = SchNet4AIM.nn.blocks.GatedNetwork(
            n_in,
            n_out,
            elements,
            n_hidden=n_hidden,
            n_layers=n_layers,
            activation=activation,
        )

        super(ElementalAIMwise, self).__init__(
            n_in=n_in,
            n_out=n_out,
            aggregation_mode=aggregation_mode,
            n_layers=n_layers,
            n_neurons=None,
            activation=activation,
            property=property,
            derivative=derivative,
            create_graph=create_graph,  
            mean=mean,
            stddev=stddev,
            outnet=outnet,
        )                                                                                          


class ElementalPairAIMwise(AIMwise):                                                                             
    """
    Predicts properties in atomic pair-wise fashion using a separate network for every chemical
    pair.

    Besides the main arguments of the AIMwise output module, it can receive the following:
    Args:
        elements (set of int): List of atomic numbers present in the training set.
        n_hidden (int): number of neurons in each layer of the output network.
            (default: 50)
    """

    def __init__(
        self,
        n_in=128,
        n_out=1,
        aggregation_mode="atomic",   
        n_layers=3,
        property="y",
        derivative=None,
        create_graph=True,          
        elements = frozenset((SchNet4AIM.frset)),
        n_hidden=50,
        activation=SchNet4AIM.nn.activations.shifted_softplus,
        mean=None,
        stddev=None,
        atomref=None,
    ):
        
        # Compute the atomic pair identifiers and store them in the 
        # pairmat matrix.
        indij=[]
        count=0
        for i in np.arange(0,max(elements)):
            indij.append((i)*(2*max(elements)-i-1))
        indij=np.array(indij,int)
        pairmat =  np.zeros((max(elements),max(elements)))
        for i in np.arange(0,max(elements)):        
            for j in np.arange(i,max(elements)):    
                valor = int(indij[i] + j+1)
                pairmat[i][j] = valor
                pairmat[j][i] = valor

        # Call the output network
        outnet = SchNet4AIM.nn.blocks.GatedNetwork2P(   
            n_in,
            n_out,
            elements,
            indij,
            pairmat,
            n_hidden=n_hidden,
            n_layers=n_layers,
            activation=activation,
        )

        super(ElementalPairAIMwise, self).__init__(
            n_in=n_in,
            n_out=n_out,
            aggregation_mode=aggregation_mode,
            n_layers=n_layers,
            n_neurons=None,
            activation=activation,
            property=property,
            derivative=derivative,
            create_graph=create_graph,  
            mean=mean,
            stddev=stddev,
            outnet=outnet,
        )                                                                                            

