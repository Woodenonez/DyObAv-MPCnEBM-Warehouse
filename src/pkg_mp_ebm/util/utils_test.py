from typing import Optional

import numpy as np
import matplotlib.patches as patches # type: ignore
import torch


def get_traj_from_pmap(prob_map: torch.Tensor) -> np.ndarray:
    traj = [] # type: ignore
    prob_map = prob_map[0,:]
    if prob_map.device == 'cuda':
        prob_map = prob_map.cpu()
    for i in range(prob_map.shape[0]):
        index_ = torch.where(prob_map[i,:]==torch.max(prob_map[i,:]))
        try:
            index = [x.item() for x in index_[::-1]]
        except:
             index = traj[-1]
        traj.append(index)
    return np.array(traj)

