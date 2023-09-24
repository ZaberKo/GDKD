import torch
from torch.utils.data import DistributedSampler, Dataset
from typing import TypeVar, Optional, Iterator


class DistributedEvalSampler(DistributedSampler):
    """ A distributed sampler that doesn't add duplicates. Arguments are the same as DistributedSampler """

    def __init__(self, dataset: Dataset, shuffle: bool = False, seed: int = 0):
        super().__init__(dataset, seed=seed, shuffle=shuffle, drop_last=False)
        if len(self.dataset) % self.num_replicas != 0:
            # some ranks may have less samples, that's fine
            if self.rank >= len(self.dataset) % self.num_replicas:
                self.num_samples -= 1
            self.total_size = len(self.dataset)
