from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_, constant_
from .backbones import *


class UNetExtra(nn.Module):
    """A modified UNet implementation with an output layer.

    Notes:
        The output layer can be 'softplus', 'poselu'. 
        If output layer is None, there is no positive output layer.
    """
    def __init__(self, in_channels: List[int], num_classes: int, with_batch_norm=True, bilinear=True, lite:bool=True, out_layer:Optional[str]=None):
        """
        Args:
            in_channels: The number of channels for input, which is the input channels for the encoder.
            num_classes: The number of classes for labels, which is the output channels for the decoder.
            with_batch_norm: If True, use batch normalization. Defaults to True.
            bilinear: If True, use bilinear interpolation. Defaults to True.
            lite: If True, use lite UNet. Defaults to True.
            out_layer: The output layer. Defaults to None.

        Raises:
            ValueError: _description_
            ValueError: _description_
        """
        super(UNetExtra,self).__init__()
        if out_layer is None:
            self.out_layer = 'none'
        else:
            if out_layer.lower() not in ['softplus', 'poselu', 'none']:
                raise ValueError(f'The output layer [{out_layer}] is not recognized.')
            self.out_layer = out_layer.lower()

        self.unet = UNet(in_channels, num_classes, with_batch_norm, bilinear, lite=lite)
        if self.out_layer == 'softplus':
            self.outl = torch.nn.Softplus()
        elif self.out_layer == 'poselu':
            self.outl = PosELU(1e-6) # type: ignore
        else:
            self.outl = None # type: ignore

    def forward(self, x):
        logits = self.unet(x)
        if self.outl is not None:
            logits = self.outl(logits)
        return logits

    def to_energy_grid(self, output:torch.Tensor):
        '''
        Notes:
            Softplus: output = log(exp(-E)+1)
            PosELU:   output = exp(-E) if E<0 else E+1
            Identity: output = E
        '''
        if self.out_layer == 'softplus':
            energy_grid = -torch.log(torch.exp(output)-1)
        elif self.out_layer == 'poselu':
            energy_grid = output.clone()
            energy_grid[output>=1] = 1-energy_grid[output>=1]
            energy_grid[output<1] = -torch.log(energy_grid[output<1])
        else:
            energy_grid = output
        return energy_grid

    def to_prob_map(self, output:torch.Tensor, threshold=0.99, temperature=1, ebm=True, smaller_grid=False):
        '''Convert NN output (BxTxHxW) to probablity maps. High energy means low probability.
        
        Args:
            output: The processed energy grid, i.e. after the positive output layer.
            threshold: Within (0,1], ignore too large energy (too small processed energy). If 1, accept all values.
            temperature: The temperature from energy grid to probability map.
        
        Notes:
            For a processed grid !E' and threshold !a, if e'_ij<e'_thre, e'_ij=0, where e'_thre=e'_max*(1-a).
        '''
        grid_proc = output.clone() # this is not grid, but the processed grid

        if not ebm:
            prob_map = self.sigmoid(grid_proc)
            prob_map /= torch.sum(prob_map.view(grid_proc.shape[0], grid_proc.shape[1], -1), dim=-1, keepdim=True).unsqueeze(-1).expand_as(grid_proc)
        elif (self.out_layer in ['softplus', 'poselu']):
            pos_energy_max = torch.amax(grid_proc, dim=(2,3))
            pos_energy_thre = (1-threshold) * pos_energy_max # shape (bs*T)
            grid_proc[grid_proc<pos_energy_thre[:,:,None,None]] = torch.tensor(0.0)

            numerator = torch.exp(torch.tensor(temperature))*grid_proc
            denominator = torch.sum(numerator.view(grid_proc.shape[0], grid_proc.shape[1], -1), dim=-1, keepdim=True).unsqueeze(-1).expand_as(grid_proc)
            prob_map = numerator / denominator
        else: # no positive output layer
            energy_min = torch.amin(grid_proc, dim=(2,3))
            energy_max = torch.amax(grid_proc, dim=(2,3))
            energy_thre = energy_min + threshold * (energy_max - energy_min) # shape (bs*T)
            grid_proc[grid_proc>energy_thre[:,:,None,None]] = torch.tensor(float('inf'))

            if smaller_grid:
                grid_proc = grid_proc / (0.1*torch.abs(torch.amin(grid_proc, dim=(2,3)))[:,:,None,None]) # avoid too large/small energy (large absoulte value)

            grid_exp = torch.exp(-temperature*grid_proc)
            prob_map = grid_exp / torch.sum(grid_exp.view(grid_proc.shape[0], grid_proc.shape[1], -1), dim=-1, keepdim=True).unsqueeze(-1).expand_as(grid_proc)

        if torch.any(torch.isnan(prob_map)):
            raise ValueError('Nan in probability map!')
        return prob_map
    
    def sigmoid(self, output: torch.Tensor):
        return torch.sigmoid(output)


class E3Net(nn.Module): # 
    '''
    Ongoing, the idea is to have an Early Exit Energy-based (E3) network.
    
    Comment
        :The input size is (batch x channel x height x width).
    '''
    def __init__(self, in_channels, num_classes, en_channels, de_channels, with_batch_norm=False, out_layer:str='softplus'):
        super(E3Net,self).__init__()
        if (out_layer is not None): 
            if (out_layer.lower() not in ['softplus', 'poselu']):
                raise ValueError(f'The output layer [{out_layer}] is not recognized.')

        self.encoder = UNetTypeEncoder(in_channels, en_channels, with_batch_norm)
        self.inc = DoubleConv(en_channels[-1], out_channels=en_channels[-1], with_batch_norm=with_batch_norm)

        up_in_chs  = [en_channels[-1]] + de_channels[:-1]
        up_out_chs = up_in_chs # for bilinear
        dec_in_chs  = [enc + dec for enc, dec in zip(en_channels[::-1], up_out_chs)] # add feature channels
        dec_out_chs = de_channels

        self.decoder = nn.ModuleList()
        for in_chs, out_chs in zip(dec_in_chs, dec_out_chs):
            self.decoder.append(UpBlock(in_chs, out_chs, bilinear=True, with_batch_norm=with_batch_norm))
        
        self.multi_out_cl = nn.ModuleList() # out conv layer
        for de_ch in de_channels:
            self.multi_out_cl.append(nn.Conv2d(de_ch, num_classes, kernel_size=1))

        if out_layer == 'softplus':
            self.outl = torch.nn.Softplus()
        elif out_layer == 'poselu':
            self.outl = PosELU(1e-6) # type: ignore
        else:
            self.outl = torch.nn.Identity() # type: ignore

    def forward(self, x):
        features:list = self.encoder(x)

        features = features[::-1]
        x = self.inc(features[0])

        multi_out = []
        for feature, dec, out in zip(features[1:], self.decoder, self.multi_out_cl):
            x = dec(x, feature)
            multi_out.append(self.outl(out(x)))
        return multi_out

