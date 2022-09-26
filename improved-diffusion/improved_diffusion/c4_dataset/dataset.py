import torch
from itertools import cycle
from copy import copy
import numpy as np


class IterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, ex_iterables: dict, max_length, config, weights_sampling_mode="size"):
        self.ex_iterables: list = []
        self.weights: list = []
        for benchmark_name, dt in ex_iterables.items():
            if weights_sampling_mode == "size":
                if hasattr(dt, "num_rows"):
                    self.weights.append(dt.num_rows)
                else:
                    self.weights.append(config["data"][benchmark_name]["size"])
            else:
                self.weights.append(1)
            self.ex_iterables.append(dt)
        self.config = config
        self.max_length = max_length

    def __iter__(self):
        self.iterators = self._make_iters()
        self.current_weights = copy(self.weights)
        while True:
            index = torch.multinomial(input=torch.Tensor([self.current_weights]), num_samples=1).item()
            try:
                x = next(self.iterators[index])['inputs']
                yield x
            except StopIteration:
                self.current_weights[index] = 0
                if np.sum(self.current_weights) == 0:
                    self.iterators = self._make_iters()
                    self.current_weights = copy(self.weights)
                    return

    def _make_iters(self):
        return [iter(ex_iterable.shuffle()) for ex_iterable in self.ex_iterables]
