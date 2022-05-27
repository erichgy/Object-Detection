import random

import torch.utils.data
from torch.utils.data import Sampler
import numpy as np


class CustomBatchSampler(Sampler):

    def __init__(self, num_positive, num_negative, batch_positive, batch_negative) -> None:
        self.num_positive = num_positive
        self.num_negative = num_negative
        self.batch_positive = batch_positive
        self.batch_negative = batch_negative

        length = num_positive + num_negative
        self.idx_list = list(range(length))
        self.batch = batch_positive + batch_negative
        self.num_iter = length // self.batch

    def __iter__(self):
        sampler_list = list()
        for i in range(self.num_iter):
            tmp = np.concatenate((random.sample(self.idx_list[:self.num_positive], self.batch_positive),
                                  random.sample(self.idx_list[self.num_positive:], self.batch_negative)))
            random.shuffle(tmp)
            sampler_list.extend(tmp)
        return iter(sampler_list)

    def __len__(self) -> int:
        return self.num_iter * self.batch

    def get_num_batch(self) -> int:
        return self.num_iter

