import numpy as np
import torch as tc

from torch.types import Device
from typing import Iterable, Any

IndexType = int | slice | Any # just anything that can be passed to a pytorch tensor. I can't be bothered to write it all out. Have you seen the type hint of torch.Tensor.__getitem__ ??


class ReplayBuffer:
    '''Simple replay buffer designed for applications in reinforcement learning. Internal buffers are pytorch tensors.'''
    def __init__(self, maxlen: int, device: Device):
        '''
        * maxlen: Maximum buffer length
        * device: Device of the buffer tensors
        '''
        if maxlen < 1:
            raise ValueError(f"maxlen must be at least 1 but is {maxlen}.")

        self.maxlen = maxlen
        self.device = device

        self._i = 0
        self._size = 0


    def _initialise_buffers(self, *args: tc.Tensor):
        '''Initialises buffers and overwrites itself'''
        # initialise buffers
        self.buffers = [tc.zeros([self.maxlen] + list(s.shape), device=self.device, dtype=s.dtype) for s in args]

        # overwrite this function with a dummy function
        self._initialise_buffers = lambda *args: None


    def add(self, *args: tc.Tensor):
        '''Add an entry'''
        self._initialise_buffers(*args)

        for buffer, value in zip(self.buffers, args, strict=True):
            buffer[self._i] = value
        
        self._size = max(self._size, self._i + 1)
        self._i = (self._i + 1) % self.maxlen


    def add_iter(self, *args: Iterable[tc.Tensor]):
        '''
        Add multiple entries at once.
        * args: Iterables of tensors, one for each field and all with the same length
        '''
        for x in zip(*args, strict=True):
            self.add(*x)


    def sample(self, n_samples: int, random: bool = True) -> tuple[tc.Tensor, ...]:
        '''Sample from buffer, either from the top or at random. Returns one tensor per field'''
        if n_samples > len(self):
            raise ValueError(f"Number of samples exceeds buffer length ({n_samples} > {len(self)}).")

        idc = tc.tensor(
                np.random.choice(len(self), n_samples, replace=False)
            if random else
                list(range(len(self) - n_samples, len(self))),
            device = self.device
        )

        out = self[idc]

        return out


    def __getitem__(self, i: IndexType) -> tuple[tc.Tensor, ...]:
        return tuple(b[i] for b in self.buffers)
    

    def __len__(self) -> int:
        return self._size
