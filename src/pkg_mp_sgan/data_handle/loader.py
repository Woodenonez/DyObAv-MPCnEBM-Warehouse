from typing import Optional

from torch.utils.data import DataLoader

from sgan.data.trajectories import TrajectoryDataset, seq_collate


def data_loader(args, path, batch_size:Optional[int]=None):
    dset = TrajectoryDataset(
        path,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        skip=args.skip,
        delim=args.delim)

    if batch_size is None:
        batch_size = args.batch_size
    
    loader = DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.loader_num_workers,
        collate_fn=seq_collate)
    return dset, loader
