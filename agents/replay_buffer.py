from collections import deque
import numpy as np

# TODO right now the replay buffer returns lists that need to be stacked to tensors. This is highly inefficient.
# Change buffer to store tensors (or arrays) for massive speed boost
class ReplayBuffer:
    '''Simple replay buffer designed for the application in reinforcement learning'''
    def __init__(self, n_fields: int, maxlen: int | None = None):
        '''* n_fields: Number of fields that are stored in each step (e.g. 4 for SARS)'''
        self.n_fields = n_fields
        self.buffer = deque(maxlen=maxlen)


    def add(self, *args):
        '''Add an entry'''
        assert len(args) == self.n_fields, "Wrong number of fields!"
        self.buffer.append(args)


    def add_iter(self, *args):
        '''
        Add multiple entries at once.
        * args: tuple of lists, one for each field and all with the same length
        '''
        for x in zip(*args, strict=True):
            self.add(*x)


    def __getitem__(self, i: int):
        return self.buffer[i]


    def sample(self, n_samples: int, random: bool = True):
        '''Sample from buffer, either from the top or at random. Returns one list per field'''
        out = tuple(list() for _ in range(self.n_fields))

        idc = (
                np.random.choice(len(self.buffer), n_samples, replace=False)
            if random else
                list(range(len(self.buffer) - n_samples, len(self.buffer)))
        )

        for i in idc:
            sample = self[i]
            for l, v in zip(out, sample):
                l.append(v)

        return out
    

    def __len__(self) -> int:
        return len(self.buffer)
