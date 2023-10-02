import torch
import SchNet4AIM as s4aim


__all__ = ["build_mse_loss"]

class LossFnError(Exception):
    pass


def build_mse_loss(properties, loss_tradeoff=None):
    """
    Build the mean squared error loss function.

    Args:
        properties (list): mapping between the model properties and the
            dataset properties
        loss_tradeoff (list or None): multiply loss value of property with tradeoff
            factor

    Returns:
        mean squared error loss function

    """
    if loss_tradeoff is None:
        loss_tradeoff = [1] * len(properties)
    if len(properties) != len(loss_tradeoff):
        raise LossFnError("loss_tradeoff must have same length as properties!")

    # Define the basic loss function for 1P and 2P based predictions
    if (s4aim.Pmode == "1p"):                         
       def loss_fn(batch, result):
           loss = 0.0
           for prop, factor in zip(properties, loss_tradeoff):
               diff = batch[prop] - result[prop]
               diff = diff ** 2
               err_sq = factor * torch.mean(diff)
               loss += err_sq
           return loss
    elif (s4aim.Pmode == "2p"):                    
       def loss_fn(batch, result):
           loss = 0.0
           for prop, factor in zip(properties, loss_tradeoff):
               diff = torch.reshape(batch[prop],(batch[prop].size()[0],batch[prop].size()[1],1)) - result[prop]   
               diff = diff ** 2
               err_sq = factor * torch.mean(diff)
               loss += err_sq
           return loss

    return loss_fn
