"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

A prefetch loader to speedup data loading
Modified from Nvidia Deep Learning Examples
(https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch).
"""
import random
from typing import List, Dict, Tuple, Union, Iterator

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist


class MetaLoader:
    """wraps multiple data loaders"""

    def __init__(
        self, loaders, accum_steps: int = 1, distributed: bool = False, device=None
    ):
        assert isinstance(loaders, dict)
        self.name2loader = {}
        self.name2iter = {}
        self.name2pre_epoch = {}
        self.names: List[str] = []
        ratios: List[int] = []
        for n, l in loaders.items():
            if isinstance(l, tuple):
                l, r, p = l
            elif isinstance(l, DataLoader):
                r = 1
                p = lambda e: None
            else:
                raise ValueError()
            self.names.append(n)
            self.name2loader[n] = l
            self.name2iter[n] = iter(l)
            self.name2pre_epoch[n] = p
            ratios.append(r)

        self.accum_steps = accum_steps
        self.device = device
        self.sampling_ratios = torch.tensor(ratios).float().to(self.device)
        self.distributed = distributed
        self.step = 0

    def __iter__(self) -> Iterator[Tuple]:
        """this iterator will run indefinitely"""
        task_id = None
        epoch_id = 0
        while True:
            if self.step % self.accum_steps == 0:
                task_id = torch.multinomial(self.sampling_ratios, 1)
                if self.distributed:
                    # make sure all process is training same task
                    dist.broadcast(task_id, 0)
            self.step += 1
            task = self.names[task_id.cpu().item()]
            iter_ = self.name2iter[task]
            try:
                batch = next(iter_)
            except StopIteration:
                epoch_id += 1
                # In distributed mode, calling the set_epoch() method at the beginning of each epoch 
                # before creating the DataLoader iterator is necessary to make shuffling work properly 
                # across multiple epochs. Otherwise, the same ordering will be always used.
                self.name2pre_epoch[task](epoch_id)
                iter_ = iter(self.name2loader[task])
                batch = next(iter_)
                self.name2iter[task] = iter_

            yield task, batch


def move_to_cuda(batch: Union[List, Tuple, Dict, torch.Tensor], device: torch.device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    elif isinstance(batch, list):
        return [move_to_cuda(t, device) for t in batch]
    elif isinstance(batch, tuple):
        return tuple(move_to_cuda(t, device) for t in batch)
    elif isinstance(batch, dict):
        return {n: move_to_cuda(t, device) for n, t in batch.items()}
    return batch


class PrefetchLoader(object):
    """
    overlap compute and cuda data transfer
    """
    def __init__(self, loader, device: torch.device):
        self.loader = loader
        self.device = device

    def __iter__(self):
        loader_it = iter(self.loader)
        self.preload(loader_it)
        batch = self.next(loader_it)
        while batch is not None:
            yield batch
            batch = self.next(loader_it)

    def __len__(self):
        return len(self.loader)

    def preload(self, it):
        try:
            self.batch = next(it)
        except StopIteration:
            self.batch = None
            return
        self.batch = move_to_cuda(self.batch, self.device)

    def next(self, it):
        batch = self.batch
        self.preload(it)
        return batch

    def __getattr__(self, name):
        method = self.loader.__getattribute__(name)
        return method


def build_dataloader(task, dataset, collate_fn, is_train: bool, opts):

    batch_size = opts.train_batch_size if is_train else opts.val_batch_size
    # if task == 'itm': batch_size = max(1, batch_size // 2)

    if opts.local_rank == -1:
        if is_train:
            sampler: Union[
                RandomSampler, SequentialSampler, DistributedSampler
            ] = RandomSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)

        size = torch.cuda.device_count() if torch.cuda.is_available() else 1
        pre_epoch = lambda e: None

        # DataParallel: scale the batch size by the number of GPUs
        if size > 1:
            batch_size *= size

    else:
        size = dist.get_world_size()
        sampler = DistributedSampler(
            dataset, num_replicas=size, rank=dist.get_rank(), shuffle=is_train
        )
        pre_epoch = sampler.set_epoch

    loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=opts.n_workers,
        pin_memory=opts.pin_mem,
        collate_fn=collate_fn,
        drop_last=False,
    )

    return loader, pre_epoch
