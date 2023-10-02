import warnings

import numpy as np
import torch

from SchNet4AIM import Properties
import SchNet4AIM as s4aim 

class Metric:
    r"""
    Base class for all metrics.

    Metrics measure the performance during the training and evaluation.

    Args:
        target (str): name of target property
        model_output (int, str): index or key, in case of multiple outputs
            (Default: None)
        name (str): name used in logging for this metric. If set to `None`,
            `MSE_[target]` will be used (Default: None)
        element_wise (bool): set to True if the model output is an element-wise
            property (forces, positions, ...)
    """

    def __init__(self, target, model_output=None, name=None, element_wise=False):
        self.target = target
        self.model_output = target if model_output is None else model_output
        if name is None:
            self.name = self.__class__.__name__
        else:
            self.name = name
        self.element_wise = element_wise

    def add_batch(self, batch, result):
        """Add a batch to calculate the metric on"""
        raise NotImplementedError

    def aggregate(self):
        """Aggregate metric over all previously added batches."""
        raise NotImplementedError

    def reset(self):
        """Reset metric attributes after aggregation to collect new batches."""
        pass


class ModelBias(Metric):
    r"""
    Calculates the bias of the model. For non-scalar quantities, the mean of
    all components is taken.

    Args:
        target (str): name of target property
        model_output (int, str): index or key, in case of multiple outputs
            (Default: None)
        name (str): name used in logging for this metric. If set to `None`,
            `MSE_[target]` will be used (Default: None)
        element_wise (bool): set to True if the model output is an element-wise
            property (forces, positions, ...)
    """

    def __init__(self, target, model_output=None, name=None, element_wise=False):
        name = "Bias_" + target if name is None else name
        super(ModelBias, self).__init__(
            target=target,
            model_output=model_output,
            name=name,
            element_wise=element_wise,
        )

        self.l2loss = 0.0
        self.n_entries = 0.0

    def reset(self):
        """Reset metric attributes after aggregation to collect new batches."""
        self.l2loss = 0.0
        self.n_entries = 0.0

    def _get_diff(self, y, yp):
        return y - yp

    def add_batch(self, batch, result):
        y = batch[self.target]
        if (s4aim.DBmode == "json"):
            # Reshape the tensor to match the required dimensions
            y = torch.reshape(y,(y.size()[0],y.size()[1],1))  
        if self.model_output is None:
            yp = result
        else:
            if type(self.model_output) is list:
                for idx in self.model_output:
                    result = result[idx]
            else:
                result = result[self.model_output]
            yp = result

        diff = self._get_diff(y, yp)
        self.l2loss += torch.sum(diff.view(-1)).detach().cpu().data.numpy()
        if self.element_wise:
            self.n_entries += (
                torch.sum(batch[Properties.atom_mask]).detach().cpu().data.numpy()
                * y.shape[-1]
            )
        else:
            self.n_entries += np.prod(y.shape)

    def aggregate(self):
        return self.l2loss / self.n_entries


class MeanSquaredError(Metric):
    r"""
    Metric for mean square error. For non-scalar quantities, the mean of all
    components is taken.

    Args:
        target (str): name of target property
        model_output (int, str): index or key, in case of multiple outputs
            (Default: None)
        name (str): name used in logging for this metric. If set to `None`,
            `MSE_[target]` will be used (Default: None)
        element_wise (bool): set to True if the model output is an element-wise
            property (forces, positions, ...)
    """

    def __init__(
        self,
        target,
        model_output=None,
        bias_correction=None,
        name=None,
        element_wise=False,
    ):
        name = "MSE_" + target if name is None else name
        super(MeanSquaredError, self).__init__(
            target=target,
            model_output=model_output,
            name=name,
            element_wise=element_wise,
        )

        self.bias_correction = bias_correction

        self.l2loss = 0.0
        self.n_entries = 0.0

    def reset(self):
        """Reset metric attributes after aggregation to collect new batches."""
        self.l2loss = 0.0
        self.n_entries = 0.0

    def _get_diff(self, y, yp):
        diff = y - yp
        if self.bias_correction is not None:
            diff += self.bias_correction
        return diff

    def add_batch(self, batch, result):
        y = batch[self.target]
        if (s4aim.DBmode == "json"):
            # Reshape the tensor to match the required dimensions
            y = torch.reshape(y,(y.size()[0],y.size()[1],1))   
        if self.model_output is None:
            yp = result
        else:
            if type(self.model_output) is list:
                for idx in self.model_output:
                    result = result[idx]
            else:
                result = result[self.model_output]
            yp = result

        diff = self._get_diff(y, yp)
        self.l2loss += torch.sum(diff.view(-1) ** 2).detach().cpu().data.numpy()
        if self.element_wise:
            self.n_entries += (
                torch.sum(batch[Properties.atom_mask]).detach().cpu().data.numpy()
                * y.shape[-1]
            )
        else:
            self.n_entries += np.prod(y.shape)

    def aggregate(self):
        return self.l2loss / self.n_entries


class RootMeanSquaredError(MeanSquaredError):
    r"""
    Metric for root mean square error. For non-scalar quantities, the mean of
    all components is taken.

    Args:
        target (str): name of target property
        model_output (int, str): index or key, in case of multiple outputs
            (Default: None)
        name (str): name used in logging for this metric. If set to `None`,
            `RMSE_[target]` will be used (Default: None)
        element_wise (bool): set to True if the model output is an element-wise
            property (forces, positions, ...)
    """

    def __init__(
        self,
        target,
        model_output=None,
        bias_correction=None,
        name=None,
        element_wise=False,
    ):
        name = "RMSE_" + target if name is None else name
        super(RootMeanSquaredError, self).__init__(
            target, model_output, bias_correction, name, element_wise=element_wise
        )

    def aggregate(self):
        """Aggregate metric over all previously added batches."""
        return np.sqrt(self.l2loss / self.n_entries)


class MeanAbsoluteError(Metric):
    r"""
    Metric for mean absolute error. For non-scalar quantities, the mean of all
    components is taken.

    Args:
        target (str): name of target property
        model_output (int, str): index or key, in case of multiple outputs
            (Default: None)
        name (str): name used in logging for this metric. If set to `None`,
            `MAE_[target]` will be used (Default: None)
        element_wise (bool): set to True if the model output is an element-wise
            property (forces, positions, ...)
    """

    def __init__(
        self,
        target,
        model_output=None,
        bias_correction=None,
        name=None,
        element_wise=False,
    ):
        name = "MAE_" + target if name is None else name
        super(MeanAbsoluteError, self).__init__(
            target=target,
            model_output=model_output,
            name=name,
            element_wise=element_wise,
        )

        self.bias_correction = bias_correction

        self.l1loss = 0.0
        self.n_entries = 0.0

    def reset(self):
        """Reset metric attributes after aggregation to collect new batches."""
        self.l1loss = 0.0
        self.n_entries = 0.0

    def _get_diff(self, y, yp):
        diff = y - yp
        if self.bias_correction is not None:
            diff += self.bias_correction
        return diff

    def add_batch(self, batch, result):
        y = batch[self.target]
        if (s4aim.DBmode == "json"):     
            # Reshape the tensor to match the required dimensions
            y = torch.reshape(y,(y.size()[0],y.size()[1],1))      
        if self.model_output is None:
            yp = result
        else:
            if type(self.model_output) is list:
                for idx in self.model_output:
                    result = result[idx]
            else:
                result = result[self.model_output]
            yp = result

        diff = self._get_diff(y, yp)
        self.l1loss += (
            torch.sum(torch.abs(diff).view(-1), 0).detach().cpu().data.numpy()
        )
        if self.element_wise:
            self.n_entries += (
                torch.sum(batch[Properties.atom_mask]).detach().cpu().data.numpy()
                * y.shape[-1]
            )
        else:
            self.n_entries += np.prod(y.shape)

    def aggregate(self):
        """Aggregate metric over all previously added batches."""
        return self.l1loss / self.n_entries
