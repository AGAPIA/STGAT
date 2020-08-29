from torch.utils.data import DataLoader
import sys
import os
from data.trajectories import TrajectoryDataset, seq_collate

sys.path.append(os.getcwd())


def data_loader(args, path):
    dset = TrajectoryDataset(
        path,
        batchSize=args.batch_size,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        skip=args.skip,
        delim=args.delim)

    loader = DataLoader(
        dset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.loader_num_workers,
        collate_fn=seq_collate,
        pin_memory=True)
    return dset, loader
