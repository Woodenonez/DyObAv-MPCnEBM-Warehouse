import math

import torch
from torch import Tensor


def get_weight(grid: Tensor, coords: Tensor, sigmas: tuple, rho=0, normalized=True) -> Tensor:
    """Create a stack of ground truth masks to compare with the generated (energy) grid.

    Args:
        grid: With size (BxTxHxW), the generated energy grid, used as the template.
        coords: With size (BxTxDo), T is the pred_len, Do is the output dimension (normally 2).
        sigmas: A tuple or list of sigma_x and sigma_y.
        rho: The correlation parameter, currently 0.
        normalized: Normalize the weight mask to 1 (then the weight mask is not a probability distribution anymore).

    Returns:
        weight: With size (BxTxHxW), the same size as the grid.
    """
    bs, T, H, W = grid.shape # batch_size, pred_offset, height, width
    coords = coords[:,:,:,None,None]

    sigma_x, sigma_y = sigmas[0], sigmas[1]
    x = torch.arange(0, W, device=grid.device)
    y = torch.arange(0, H, device=grid.device)
    try:
        x, y = torch.meshgrid(x, y, indexing='xy')
    except:
        y, x = torch.meshgrid(y, x) # indexing is 'ij', this is because the old torch version doesn't support indexing
    x, y = x.unsqueeze(0).repeat(bs,T,1,1), y.unsqueeze(0).repeat(bs,T,1,1)
    in_exp = -1/(2*(1-rho**2)) * ((x-coords[:,:,0])**2/(sigma_x**2) 
                                + (y-coords[:,:,1])**2/(sigma_y**2) 
                                - 2*rho*(x-coords[:,:,0])/(sigma_x)*(y-coords[:,:,1])/(sigma_y))
    weight = 1/(2*math.pi*sigma_x*sigma_y*torch.sqrt(torch.tensor(1-rho**2))) * torch.exp(in_exp)
    if normalized:
        weight = weight/(weight.amax(dim=(2,3))[:,:,None,None])
        weight[weight<0.1] = 0
    return weight

def loss_kl(data: Tensor, label: Tensor, sigmas:tuple=(10,10), weight=None, l2_factor:float=0.00):
    if weight is None:
        weight = get_weight(data, label, sigmas=sigmas, normalized=False) # Gaussian fashion [BxTxHxW]

    energy_sum = torch.logsumexp(-data, dim=(2,3)) # [BxT]
    numerator_in_log   = torch.sum(data*weight, dim=(2,3))
    denominator_in_log = torch.sum(weight*energy_sum[:,:,None,None], dim=(2,3))

    l2 = torch.sum(torch.pow(data,2),dim=(2,3)) / (data.shape[2]*data.shape[3])
    kl = numerator_in_log + denominator_in_log + l2_factor*l2
    if len(label.shape) == 3:
        kl = torch.sum(kl, dim=1)
    return torch.mean(kl)

def loss_nll(data: Tensor, label: Tensor, sigmas:tuple=(10,10), l2_factor:float=0.0):
    #TODO Double check the index is (i,j) or (x,y)
    '''
    Args:
        data: (BxTxHxW), the energy grid
        label: (BxTxDo), T:pred_len, Do: output dimension [label should be the index (i,j) meaning which grid cell to choose]
    '''

    weight = get_weight(data, label, sigmas=sigmas) # Gaussian fashion [BxTxHxW]

    numerator_in_log   = torch.logsumexp(-data+torch.log(weight), dim=(2,3))
    denominator_in_log = torch.logsumexp(-data, dim=(2,3))

    l2 = torch.sum(torch.pow(data,2),dim=(2,3)) / (data.shape[2]*data.shape[3])
    nll = - numerator_in_log + denominator_in_log + l2_factor*l2
    if len(label.shape) == 3:
        nll = torch.sum(nll, dim=1)
    return torch.mean(nll)

def loss_enll(data: Tensor, label: Tensor, sigmas:tuple=(10,10), l2_factor:float=0.0):
    """The energy-based negative log-likelihood loss. The data is already processed by the positive output layer.

    Args:
        data: (BxTxHxW), the energy grid
        label: (BxTxDo), T:pred_len, Do: output dimension [label should be the index (i,j) meaning which grid cell to choose]
    """

    weight = get_weight(data, label, sigmas=sigmas) # Gaussian fashion [BxTxHxW]

    numerator_in_log   = torch.log(torch.sum(data*weight, dim=(2,3)))
    denominator_in_log = torch.log(torch.sum(data, dim=(2,3)))

    l2 = torch.sum(torch.pow(data,2),dim=(2,3)) #/ (data.shape[2]*data.shape[3])
    nll = - numerator_in_log + denominator_in_log + l2_factor*l2
    if len(label.shape) == 3:
        nll = torch.sum(nll, dim=1)
    return torch.mean(nll)


def loss_mse(data: Tensor, labels: Tensor): # for batch
    # data, labels - BxMxC
    squared_diff = torch.square(data-labels)
    squared_sum  = torch.sum(squared_diff, dim=2) # BxM
    loss = squared_sum/data.shape[0] # BxM
    return loss

def loss_msle(data: Tensor, labels: Tensor): # for batch
    # data, labels - BxMxC
    squared_diff = torch.square(torch.log(data)-torch.log(labels))
    squared_sum  = torch.sum(squared_diff, dim=2) # BxM
    loss = squared_sum/data.shape[0] # BxM
    return loss

def loss_mae(data: Tensor, labels: Tensor): # for batch
    # data, labels - BxMxC
    abs_diff = torch.abs(data-labels)
    abs_sum  = torch.sum(abs_diff, dim=2) # BxM
    loss = abs_sum/data.shape[0] # BxM
    return loss
